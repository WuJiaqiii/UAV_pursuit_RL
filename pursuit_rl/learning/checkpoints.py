"""Versioned weights, fresh initialization and strict current-version continuation."""

import hashlib
from pathlib import Path
import torch
from pursuit_rl.config import config_hash, migrate_checkpoint_config
from pursuit_rl.learning.contracts import policy_contract, require_compatible_policy
from pursuit_rl.simulation.observations import observation_dimensions
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.simulation.rewards import REWARD_VERSION
from pursuit_rl import __version__

SCHEMA = "pursuit-rl-checkpoint-v6"
ARCHIVED_SCHEMA = "pursuit-rl-v2-ned-frd-obs31-critic43-v1"


def validate_checkpoint(data):
    if data.get("schema") not in (SCHEMA, ARCHIVED_SCHEMA):
        raise ValueError(f'Unsupported checkpoint schema: {data.get("schema")}')


def load_model(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(data)
    data["config"] = migrate_checkpoint_config(data["config"])
    if data.get("policy_contract") is not None and data[
        "policy_contract"
    ] != policy_contract(data["config"]):
        raise ValueError("Checkpoint policy contract disagrees with configuration")
    model = AsymmetricPPO(
        *observation_dimensions(data["config"]),
        data["config"]["training"],
        device=device,
        create_optimizers=False,
    )
    model.load_state_dict(data["actor_critic"])
    model.eval()
    return model, data


def save_checkpoint(trainer, name):
    state = {
        "schema": SCHEMA,
        "reward_version": REWARD_VERSION,
        "policy_contract": policy_contract(trainer.config),
        "config": trainer.config,
        "config_hash": config_hash(trainer.config),
        "steps": trainer.steps,
        "episode_count": trainer.episode_count,
        "stage": trainer.stage,
        "stage_since": trainer.stage_since,
        "promotion_streak": trainer.promotion_streak,
        "best_score": trainer.best_score,
        "best_stage_score": trainer.best_stage_score,
        "parent_checkpoint": trainer.parent_checkpoint,
        "package_version": __version__,
        "source_sha256": trainer.source_sha256,
        "actor_critic": trainer.model.state_dict(),
        "actor_optimizer": trainer.model.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.model.critic_optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": trainer.env.rng.bit_generator.state if not trainer.tensor_backend else None,
        "torch_env_rng": trainer.env.gen.get_state() if trainer.tensor_backend else None,
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if trainer.model.device.type == "cuda"
            else None
        ),
    }
    tmp = trainer.ckpt / (name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(trainer.ckpt / (name + ".pt"))
    artifact = trainer.ckpt / (name + ".pt")
    trainer.exp_logger.event(
        "checkpoint",
        name=name,
        path=str(artifact.resolve()),
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        steps=trainer.steps,
        stage=trainer.stage,
    )
    trainer.exp_logger.logger.debug(
        "Saved checkpoint %s at step %s", name, trainer.steps
    )


def initialize_weights(trainer, path):
    """Start a NEW run from an existing policy, not a config-locked resume.

    Resets optimizers/step count/best score. Checks the observation/action
    contract, so fine-tuning cannot silently reinterpret policy channels.
    """
    if any(trainer.ckpt.glob("*.pt")):
        raise FileExistsError(
            "Fresh --init-from run requires an empty checkpoint directory"
        )
    source_path = Path(path).expanduser().resolve()
    source = torch.load(
        source_path, map_location=trainer.model.device, weights_only=False
    )
    validate_checkpoint(source)
    old = migrate_checkpoint_config(source["config"])
    require_compatible_policy(old, trainer.config, topology=True)
    trainer.model.load_state_dict(source["actor_critic"], strict=True)
    trainer.parent_checkpoint = {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "parent_steps": int(source["steps"]),
        "parent_config_hash": str(source.get("config_hash", "unknown")),
    }


def resume_training(trainer, path):
    saved = torch.load(path, map_location=trainer.model.device, weights_only=False)
    if (
        saved.get("schema") != SCHEMA
        or saved.get("reward_version") != REWARD_VERSION
        or saved.get("config_hash") != config_hash(trainer.config)
    ):
        raise ValueError(
            "Resume requires an identical current training/config version; use --init-from for older models."
        )
    if saved.get("policy_contract") != policy_contract(trainer.config):
        raise ValueError("Resume policy contract mismatch")
    trainer.model.load_state_dict(saved["actor_critic"])
    trainer.model.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    trainer.model.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    trainer.episode_count = saved.get("episode_count", 0)
    trainer.steps = saved["steps"]
    trainer.stage = saved["stage"]
    trainer.stage_since = saved["stage_since"]
    trainer.promotion_streak = saved["promotion_streak"]
    trainer.best_score = tuple(saved["best_score"])
    trainer.best_stage_score = {
        int(k): tuple(v) for k, v in saved["best_stage_score"].items()
    }
    trainer.parent_checkpoint = saved.get("parent_checkpoint")
    torch.set_rng_state(saved["torch_rng"].cpu())
    if trainer.tensor_backend:
        if saved.get('torch_env_rng') is None:
            raise ValueError('GPU tensor resume requires a matching tensor-backend checkpoint')
        trainer.env.gen.set_state(saved['torch_env_rng'].cpu())
    else:
        if saved.get('numpy_rng') is None:
            raise ValueError('NumPy resume requires matching NumPy backend checkpoint')
        trainer.env.rng.bit_generator.state = saved['numpy_rng']
    if trainer.model.device.type == "cuda" and saved.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    trainer.env.set_stage(trainer.stage)
    trainer.obs, trainer.priv = trainer.env.observe()
    # NOTE: explicit resampling of unfinished episodes; not bit-identical resume.
    # Preserve old best checkpoints if resuming to a different output folder.
    src = Path(path).parent
    for name in ["best.pt", *[f"best_stage{k}.pt" for k in trainer.best_stage_score]]:
        old = src / name
        new = trainer.ckpt / name
        if old.exists() and not new.exists():
            import shutil

            shutil.copy2(old, new)
    if trainer.best_score[0] >= 0 and not (trainer.ckpt / "best.pt").exists():
        raise FileNotFoundError(
            "Historical best.pt missing; refusing to silently replace it with latest."
        )

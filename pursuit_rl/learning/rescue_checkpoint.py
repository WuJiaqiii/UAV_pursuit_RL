"""Strict checkpoint boundary for rescue action units and observations."""

import copy
import math
from pathlib import Path
import torch
from pursuit_rl.config.rescue import rescue_contract, rescue_fingerprint
from pursuit_rl.learning.ppo import AsymmetricPPO

CHECKPOINT_SCHEMA = "pursuit-rescue-checkpoint-v1"


def rescue_model_config(cfg, profile):
    config = copy.deepcopy(cfg["training"])
    dt = cfg["physics"]["dt"] * profile["decision_interval_steps"]
    config.update(
        action_dim=4 if profile["action_mode"] == "ctbr" else 3,
        hidden_actor=profile["hidden_actor"],
        hidden_critic=profile["hidden_critic"],
        log_std_init=math.log(profile["initial_action_std"]),
        log_std_min=math.log(profile["min_action_std"]),
        exploration_rho=(
            math.exp(-dt / profile["exploration_time_s"])
            if profile["exploration_time_s"]
            else 0.0
        ),
        exploration_memory_start=32,
        actor_lr=profile["ppo_actor_lr"],
        critic_lr=profile["ppo_critic_lr"],
        batch_size=profile["ppo_batch_size"],
        epochs=profile["ppo_epochs"],
        target_kl=profile["ppo_target_kl"],
        ent_coef=profile["ppo_entropy_coef"],
        critic_loss="huber",
        critic_huber_delta=1.0,
        separate_critic_updates=True,
    )
    return config


def new_rescue_model(cfg, profile, optimizers=True):
    config = rescue_model_config(cfg, profile)
    dim = 32 + config["action_dim"]
    return AsymmetricPPO(
        dim, dim + 8, config, device=cfg["device"], create_optimizers=optimizers
    )


def save_rescue_checkpoint(path, model, cfg, profile, progress, origin):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": CHECKPOINT_SCHEMA,
        "base_config": cfg,
        "rescue_profile": profile,
        "contract": rescue_contract(cfg, profile),
        "fingerprint": rescue_fingerprint(cfg, profile),
        "model": model.state_dict(),
        "model_config": model.config,
        "origin": origin,
        "progress": progress,
        "actor_optimizer": (
            model.actor_optimizer.state_dict() if model.actor_optimizer else None
        ),
        "critic_optimizer": (
            model.critic_optimizer.state_dict() if model.critic_optimizer else None
        ),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if str(model.device).startswith("cuda")
            else None
        ),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def load_rescue_checkpoint(path, device=None, expected_contract=None, optimizers=False):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "This is not a rescue checkpoint; legacy CTBR networks are incompatible"
        )
    cfg = copy.deepcopy(saved["base_config"])
    profile = copy.deepcopy(saved["rescue_profile"])
    if saved["fingerprint"] != rescue_fingerprint(cfg, profile):
        raise ValueError("Checkpoint configuration fingerprint mismatch")
    actual = rescue_contract(cfg, profile)
    if saved["contract"] != actual or (
        expected_contract is not None and actual != expected_contract
    ):
        raise ValueError(
            "Wrong action units/rate/observation/physics for this rescue checkpoint"
        )
    if device is not None:
        cfg["device"] = device
    model = new_rescue_model(cfg, profile, optimizers=optimizers)
    if saved["model_config"] != model.config:
        raise ValueError("Rescue model configuration mismatch")
    model.load_state_dict(saved["model"])
    if optimizers:
        model.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        model.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    return model, cfg, profile, saved

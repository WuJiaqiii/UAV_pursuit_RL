"""Single-version checkpoint contract for high-speed first-entry CTBR."""

import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from pursuit_rl.learning.ppo import AsymmetricPPO
from .config import (
    DEFAULT,
    merge,
    fingerprint,
    interface,
    validate,
    decision_dt,
    observation_dims,
)
from .environment import PRIVILEGED_DIM

SCHEMA = "pursuit-first-entry-ctbr-checkpoint-v1"


def model_config(c):
    p, t = c["policy"], c["training"]
    return dict(
        action_dim=4,
        position_error_cap_m=p["position_error_cap_m"],
        hidden_actor=p["hidden_actor"],
        hidden_critic=p["hidden_critic"],
        log_std_init=math.log(p["initial_std"]),
        log_std_min=math.log(p["min_std"]),
        log_std_max=math.log(p["max_std"]),
        exploration_rho=math.exp(-decision_dt(c) / p["exploration_time_s"]),
        exploration_memory_start=32,
        actor_lr=t["actor_lr"],
        critic_lr=t["critic_lr"],
        batch_size=t["batch_size"],
        epochs=t["epochs"],
        target_kl=0.015,
        clip=0.15,
        ent_coef=0.0,
        max_grad_norm=0.5,
        critic_loss="huber",
        critic_huber_delta=c["critic"]["huber_delta"],
        value_clip=c["critic"]["value_clip"],
        value_norm=c["critic"]["value_norm"],
        norm_beta=c["critic"]["norm_beta"],
        norm_min_variance=c["critic"]["norm_min_variance"],
        prediction=copy.deepcopy(c["prediction"]),
        base_obs_dim=observation_dims(c)[0],
        separate_critic_updates=True,
    )


def new_model(c, optimizers=True):
    _, dim = observation_dims(c)
    return AsymmetricPPO(
        dim,
        dim + PRIVILEGED_DIM,
        model_config(c),
        device=c["device"],
        create_optimizers=optimizers,
    )


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = [
        root / "main.py",
        *(root / "pursuit_rl").rglob("*.py"),
        *(root / "tests").rglob("*.py"),
    ]
    for p in sorted(paths):
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def save(path, model, c, progress, origin):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(
        schema=SCHEMA,
        config=copy.deepcopy(c),
        fingerprint=fingerprint(c),
        interface=interface(c),
        model=model.state_dict(),
        model_config=model.config,
        progress=copy.deepcopy(progress),
        origin=origin,
        source_fingerprint=source_fingerprint(),
        actor_optimizer=(
            model.actor_optimizer.state_dict() if model.actor_optimizer else None
        ),
        critic_optimizer=(
            model.critic_optimizer.state_dict() if model.critic_optimizer else None
        ),
        predictor_optimizer=(
            model.predictor_optimizer.state_dict()
            if model.predictor_optimizer
            else None
        ),
        torch_rng=torch.get_rng_state(),
        cuda_rng=(
            torch.cuda.get_rng_state_all()
            if str(model.device).startswith("cuda")
            else None
        ),
    )
    temp = path.with_name(path.name + ".tmp")
    torch.save(data, temp)
    temp.replace(path)


def load(path, device=None, optimizers=False):
    """Only v8 first-entry checkpoints are accepted.

    The old sustained-flight v3/v4 formats and migration command are intentionally
    removed. Resume is additionally guarded by the source fingerprint in train().
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(
            "Unsupported checkpoint: this first-entry-only build does not load old versions"
        )
    c = validate(merge(copy.deepcopy(DEFAULT), data["config"]))
    config_fingerprint_matches = fingerprint(c) == data["fingerprint"]
    # v8 checkpoints created before bc_rounds was introduced omit that field.
    # The new default is one round, so accept only the exact legacy fingerprint
    # obtained by removing this single newly-added default field.
    if not config_fingerprint_matches and "bc_rounds" not in data["config"].get("training", {}):
        legacy_config = copy.deepcopy(c)
        legacy_config["training"].pop("bc_rounds", None)
        config_fingerprint_matches = fingerprint(legacy_config) == data["fingerprint"]
    if not config_fingerprint_matches or interface(c) != data["interface"]:
        raise ValueError("Corrupt or mismatched first-entry checkpoint contract")
    if device is not None:
        c["device"] = device
    model = new_model(c, optimizers)
    if model.config != data["model_config"]:
        raise ValueError("Checkpoint model architecture mismatch")
    model.load_state_dict(data["model"])
    model.critic_policy_match = not data["progress"].get(
        "critic_needs_calibration", False
    )
    if optimizers:
        if data["actor_optimizer"] is None or data["critic_optimizer"] is None:
            raise ValueError("Checkpoint has no trainable optimizer state")
        model.actor_optimizer.load_state_dict(data["actor_optimizer"])
        model.critic_optimizer.load_state_dict(data["critic_optimizer"])
        if model.predictor_optimizer is not None:
            model.predictor_optimizer.load_state_dict(data["predictor_optimizer"])
    return model, c, data


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)

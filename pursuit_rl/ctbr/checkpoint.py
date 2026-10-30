"""Strict new model contract: old rescue/legacy weights are rejected."""

import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from pursuit_rl.learning.ppo import AsymmetricPPO
from .config import fingerprint, interface, validate
from .environment import PRIVILEGED_DIM

SCHEMA = "pursuit-direct-ctbr-checkpoint-v2"


def model_config(c):
    p, t = c["policy"], c["training"]
    return dict(
        action_dim=4,
        hidden_actor=p["hidden_actor"],
        hidden_critic=p["hidden_critic"],
        log_std_init=math.log(p["initial_std"]),
        log_std_min=math.log(p["min_std"]),
        log_std_max=math.log(p["max_std"]),
        exploration_rho=math.exp(-c["physics"]["dt"] / p["exploration_time_s"]),
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
        critic_huber_delta=1.0,
        separate_critic_updates=True,
    )


def new_model(c, optimizers=True):
    dim = 32 + 4 * c["policy"]["history_steps"] + 2
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
    for p in sorted(root.rglob("*.py")):
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
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(
            "Expected new direct-CTBR v2 checkpoint; old velocity/CTBR checkpoints are incompatible"
        )
    c = validate(copy.deepcopy(data["config"]))
    if fingerprint(c) != data["fingerprint"] or interface(c) != data["interface"]:
        raise ValueError("Corrupt or mismatched checkpoint contract")
    if device is not None:
        c["device"] = device
    model = new_model(c, optimizers)
    if model.config != data["model_config"]:
        raise ValueError("Model configuration mismatch")
    model.load_state_dict(data["model"])
    if optimizers:
        model.actor_optimizer.load_state_dict(data["actor_optimizer"])
        model.critic_optimizer.load_state_dict(data["critic_optimizer"])
    return model, c, data


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)

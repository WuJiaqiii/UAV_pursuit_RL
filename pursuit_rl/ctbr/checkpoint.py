"""Strict new model contract: old rescue/legacy weights are rejected."""

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

SCHEMA = "pursuit-direct-ctbr-checkpoint-v4"
LEGACY_SCHEMA = "pursuit-direct-ctbr-checkpoint-v3"


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
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError(
            "Expected direct-CTBR v4 or v700/v3 checkpoint; v690/v691 and velocity checkpoints are incompatible"
        )
    legacy = data["schema"] == LEGACY_SCHEMA
    if legacy and optimizers:
        raise ValueError(
            "v700 checkpoints are inference-only here; use migrate before training, not resume"
        )
    c = copy.deepcopy(data["config"])
    expected_schema = "pursuit-ctbr-v3" if legacy else DEFAULT["schema"]
    if c.get("schema") != expected_schema:
        raise ValueError("Mismatched checkpoint/config schema")
    if fingerprint(c) != data["fingerprint"] or interface(c) != data["interface"]:
        raise ValueError("Corrupt or mismatched checkpoint contract")
    if legacy:
        c["schema"] = DEFAULT["schema"]
        # v700 inference MUST preserve its original unbounded Actor input.
        c["policy"]["position_error_cap_m"] = 0.0
    c = validate(merge(copy.deepcopy(DEFAULT), c))
    if device is not None:
        c["device"] = device
    model = new_model(c, optimizers)
    expected = copy.deepcopy(model.config)
    if legacy:
        expected.pop("position_error_cap_m")
    if expected != data["model_config"]:
        raise ValueError("Model configuration mismatch")
    model.load_state_dict(data["model"])
    model.critic_policy_match = not data["progress"].get(
        "critic_needs_calibration", False
    )
    if optimizers:
        model.actor_optimizer.load_state_dict(data["actor_optimizer"])
        model.critic_optimizer.load_state_dict(data["critic_optimizer"])
        if model.predictor_optimizer is not None:
            model.predictor_optimizer.load_state_dict(data["predictor_optimizer"])
    return model, c, data


def migration_contract(c):
    contract = interface(c)
    contract.pop("schema")
    contract.pop("observation")
    contract["policy"].pop("position_error_cap_m", None)
    return contract


def migrate(source, config, out, reset_std=True):
    """Explicitly transfer compatible v700/v701 weights into a new experiment.

    Only task/training settings, Actor position encoding and initial exploration
    may change. No optimizer or RNG continuation; the new critic needs fresh MC.
    """
    path = Path(out)
    if path.exists():
        raise FileExistsError("Migration output already exists")
    parent, old, saved = load(source, config["device"])
    if migration_contract(old) != migration_contract(config):
        raise ValueError(
            "Migration changes dynamics, reward, feature layout or control timing"
        )
    model = new_model(config)
    model.load_state_dict(parent.state_dict())
    with torch.no_grad():
        if reset_std:
            model.log_std.fill_(math.log(config["policy"]["initial_std"]))
        model.log_std.clamp_(
            math.log(config["policy"]["min_std"]), math.log(config["policy"]["max_std"])
        )
    origin = dict(
        source_path=str(Path(source).resolve()),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        source_schema=saved["schema"],
        source_origin=saved["origin"],
        source_progress=copy.deepcopy(saved["progress"]),
        old_position_error_cap_m=old["policy"]["position_error_cap_m"],
        position_error_cap_m=config["policy"]["position_error_cap_m"],
        reset_std=reset_std,
        initial_std=model.log_std.detach().exp().cpu().tolist(),
        actor_and_predictor_weights_unchanged=True,
        optimizers_reset=True,
    )
    progress = dict(
        physics_steps=0,
        ppo_updates=0,
        cumulative_ppo_physics_steps=saved["progress"].get(
            "cumulative_ppo_physics_steps", 0
        ),
        critic_needs_calibration=True,
        migration=origin,
    )
    model.critic_policy_match = False
    save(path, model, config, progress, "explicit_migration")
    write_json(path.with_suffix(".migration.json"), origin)
    return model, origin


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)

"""YAML inheritance, legacy metadata migration, CLI resolution and Config API."""

import copy
import hashlib
import json
from pathlib import Path
import yaml
from .values import ConfigNode, as_dict
from .schema import DEFAULT, validate


def _document(path, seen=None):
    path = Path(path).expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    seen.add(path)
    doc = yaml.safe_load(path.read_text(encoding="utf8"))
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ValueError("Configuration must be a map")
    parent = doc.pop("extends", None)
    documents = _document(path.parent / parent, seen) if parent else []
    return documents + [doc]


def load_config(path=None):
    config = copy.deepcopy(DEFAULT)
    for document in _document(path) if path else []:
        _apply_overrides(config, document)
    _resolve_device(config)
    validate(config)
    return config


def _apply_overrides(config, document):
    if not isinstance(document, dict):
        raise ValueError("Configuration overrides must be a map")
    document = copy.deepcopy(document)
    stages = document.pop("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must map stage names to overrides")
    _merge(config, document)
    for name, changes in stages.items():
        matches = [s for s in config["curriculum"] if s["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous stage: {name}")
        matches[0].setdefault("initial_geometry", "random")
        _merge(matches[0], changes, f"stages.{name}.")


def _resolve_device(config):
    if config["device"] == "auto":
        import torch

        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"


def migrate_checkpoint_config(old):
    """Read old model metadata only; retired experiments are not executable APIs."""
    old = copy.deepcopy(old)
    old.setdefault("simulation_backend", "numpy")
    old.setdefault("observation", {"use_cpa": True, "boundary_features": False})
    old["observation"].setdefault(
        "target_accel_scale", max(old.get("target", {}).get("max_accel", 7.0), 1.0)
    )
    for section, key in [
        ("target", "min_speed"),
        ("training", "test_seed"),
        ("reward", "success_radius"),
    ]:
        old.get(section, {}).pop(key, None)
    baseline = copy.deepcopy(DEFAULT)
    # Fill optional metadata of archived 31D models with their original no-op values.
    baseline["reward"].update(
        w_action_delta=0.005,
        w_speed_risk=0.0,
        w_tilt_risk=0.0,
        w_precision_potential=0.0,
    )
    baseline["training"]["max_stage"] = 3
    baseline["curriculum"] = baseline["curriculum"][:4]
    _merge(baseline, old)
    validate(baseline)
    return baseline


def _merge(dst, src, path=""):
    if not isinstance(src, dict):
        raise ValueError(f'Expected configuration map: {path or "root"}')
    for k, v in src.items():
        if k not in dst:
            raise ValueError(f"Unknown config key: {path}{k}")
        if isinstance(dst[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"Expected map: {path}{k}")
            _merge(dst[k], v, path + k + ".")
        else:
            if isinstance(dst[k], list) and not isinstance(v, list):
                raise ValueError(f"Expected list: {path}{k}")
            dst[k] = v


def config_hash(c):
    return hashlib.sha256(json.dumps(as_dict(c), sort_keys=True).encode()).hexdigest()


def dump_config(c, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump(as_dict(c), f, allow_unicode=True, sort_keys=False)


class Config(ConfigNode):
    """Resolved simulation settings; runtime paths/arguments are stored separately."""

    __slots__ = ()

    def __init__(self, values=None):
        values = (
            copy.deepcopy(DEFAULT) if values is None else copy.deepcopy(as_dict(values))
        )
        _resolve_device(values)
        validate(values)
        super().__init__(values)

    @classmethod
    def load(cls, path=None):
        return cls(load_config(path))

    @classmethod
    def from_args(cls, args, fallback=None):
        path = getattr(args, "config", None)
        if not path and getattr(args, "resume", None):
            candidate = (
                Path(args.resume).expanduser().resolve().parent.parent / "config.yaml"
            )
            if candidate.is_file():
                path = candidate
        if not path and fallback is None and getattr(args, "run", None):
            candidate = Path(args.run).expanduser().resolve() / "config.yaml"
            if candidate.is_file():
                # Old experiment snapshots are data, not supported old CLI schemas.
                fallback = migrate_checkpoint_config(
                    yaml.safe_load(candidate.read_text(encoding="utf8"))
                )
        data = (
            load_config(path)
            if path
            else (
                migrate_checkpoint_config(as_dict(fallback))
                if fallback is not None
                else load_config()
            )
        )
        for expression in getattr(args, "set_values", None) or []:
            if "=" not in expression:
                raise ValueError("--set requires dotted.path=YAML_value")
            dotted, value = expression.split("=", 1)
            keys = dotted.split(".")
            if not all(keys):
                raise ValueError("Invalid --set path")
            override = yaml.safe_load(value)
            for key in reversed(keys):
                override = {key: override}
            _apply_overrides(data, override)
        if getattr(args,'device',None) is not None:
            data['device']=getattr(args,'device')
            _resolve_device(data)
        if getattr(args,'simulation_backend',None) is not None:
            data['simulation_backend']=args.simulation_backend
        elif getattr(args,'device',None)=='cuda':
            # Explicit CUDA CLI request selects device-resident physics by default.
            data['simulation_backend']='torch'
        if getattr(args, "command", "train") == "train":
            for name in ("seed", "device"):
                if getattr(args, name, None) is not None:
                    data[name] = getattr(args, name)
            _resolve_device(data)
            for name in ("total_steps", "num_envs"):
                if getattr(args, name, None) is not None:
                    data["training"][name] = getattr(args, name)
        return cls(data)

    def with_overrides(self, changes):
        data = self.to_dict()
        _apply_overrides(data, changes)
        return Config(data)

    @property
    def hash(self):
        return config_hash(self)

    def save(self, path):
        dump_config(self, path)

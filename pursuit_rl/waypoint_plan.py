"""Configuration for the static virtual-waypoint experiment, independent of Torch."""

from dataclasses import asdict, dataclass, fields, replace
import json
import math
import re


PLAN_SCHEMA = "virtual-waypoint-experiments-v1"
ARMS = ("zero", "pd", "ppo", "bc_ppo", "aux_ppo")
BASELINES = frozenset(("zero", "pd"))
EVAL_SEED = 9_000_000


def _sequence(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty array")
    return tuple(value)


def _positive_counts(config, *names):
    for name in names:
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _finite(*values):
    try:
        return all(type(value) in (int, float) and math.isfinite(value) for value in values)
    except OverflowError:
        return False


@dataclass(frozen=True)
class WaypointConfig:
    radius_m: float = 0.5
    horizontal_distance_m: tuple[float, float] = (2.0, 5.0)
    reference_speed_mps: float = 8.0
    max_speed_mps: float = 10.0
    timeout_s: float = 6.0
    decision_ticks: int = 20

    def __post_init__(self):
        distance = _sequence(self.horizontal_distance_m, "horizontal_distance_m")
        if len(distance) != 2:
            raise ValueError("horizontal_distance_m must have two bounds")
        object.__setattr__(self, "horizontal_distance_m", distance)
        low, high = distance
        if not _finite(self.radius_m, low, high, self.reference_speed_mps,
                       self.max_speed_mps, self.timeout_s):
            raise ValueError("Waypoint settings must be finite numbers")
        if not 0 < self.radius_m < low <= high <= 20:
            raise ValueError("Require 0 < radius_m < initial horizontal distance <= 20 m")
        if not 0 < self.reference_speed_mps < self.max_speed_mps <= 10:
            raise ValueError("Require 0 < reference speed < maximum speed <= 10 m/s")
        _positive_counts(self, "decision_ticks")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    device: str = "cpu"
    arms: tuple[str, ...] = ARMS
    seeds: tuple[int, ...] = (42,)
    episodes: int = 64
    evaluation_seed: int = EVAL_SEED
    cpu_threads: int = 1

    def __post_init__(self):
        arms = _sequence(self.arms, "arms")
        seeds = _sequence(self.seeds, "seeds")
        if any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
            raise ValueError(f"arms must be unique and chosen from {', '.join(ARMS)}")
        if any(type(seed) is not int or not 0 <= seed < 1_000_000 for seed in seeds):
            raise ValueError("Training seeds must be integers from 0 to 999999")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Training seeds must be unique")
        if type(self.evaluation_seed) is not int or not 3_000_000 <= self.evaluation_seed < 2**63:
            raise ValueError("evaluation_seed must be an integer from 3000000 to 2^63 - 1")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "seeds", seeds)
        _positive_counts(self, "episodes", "cpu_threads")


@dataclass(frozen=True)
class PPOConfig:
    envs: int = 64
    rollout_steps: int = 128
    updates: int = 120
    batch_size: int = 512
    epochs: int = 3
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.97

    def __post_init__(self):
        _positive_counts(self, "envs", "rollout_steps", "updates", "batch_size", "epochs")
        if not _finite(self.actor_lr, self.critic_lr, self.gamma, self.gae_lambda):
            raise ValueError("PPO rates and discounts must be finite numbers")
        if min(self.actor_lr, self.critic_lr) <= 0:
            raise ValueError("PPO learning rates must be positive")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Require 0 < gamma <= 1 and 0 <= gae_lambda <= 1")


@dataclass(frozen=True)
class TeacherConfig:
    bc_steps: int = 256
    bc_epochs: int = 12
    aux_coef: float = 0.1

    def __post_init__(self):
        _positive_counts(self, "bc_steps", "bc_epochs")
        if not _finite(self.aux_coef) or self.aux_coef <= 0:
            raise ValueError("aux_coef must be finite and positive")


@dataclass(frozen=True)
class Plan:
    task: WaypointConfig = WaypointConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    ppo: PPOConfig = PPOConfig()
    teacher: TeacherConfig = TeacherConfig()

    def document(self):
        return {"schema": PLAN_SCHEMA, **asdict(self)}


DEFAULT_PLAN = Plan()
SECTIONS = {
    section.name: {field.name for field in fields(getattr(DEFAULT_PLAN, section.name))}
    for section in fields(Plan)
}


def _sections(overrides, label, ignored=()):
    if not isinstance(overrides, dict):
        raise ValueError(f"{label} must be an object")
    unknown = overrides.keys() - SECTIONS.keys() - set(ignored)
    if unknown:
        raise ValueError(f"Unknown settings in {label}: {', '.join(sorted(unknown))}")
    for section, known_keys in SECTIONS.items():
        values = overrides.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{label}.{section} must be an object")
        unknown = values.keys() - known_keys
        if unknown:
            raise ValueError(f"Unknown settings in {label}.{section}: {', '.join(sorted(unknown))}")
    return {section: overrides.get(section, {}) for section in SECTIONS}


def load_plan(path=None, stage=None, overrides=None):
    """Resolve built-in defaults, file settings, a named stage, then CLI flags."""
    raw = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
    shared = _sections(raw, "plan", ignored=("schema", "stages", "default_stage"))
    if raw.get("schema", PLAN_SCHEMA) != PLAN_SCHEMA:
        raise ValueError(f"Expected plan schema {PLAN_SCHEMA}")
    stages = raw.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must be an object of named stages")
    for name, definition in stages.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"Invalid stage name: {name}")
        _sections(definition, f"stages.{name}")
    default_stage = raw.get("default_stage")
    if default_stage is not None and (not isinstance(default_stage, str) or default_stage not in stages):
        raise ValueError("default_stage must name a defined stage")
    name = stage if stage is not None else default_stage
    if name is not None and name not in stages:
        available = ", ".join(stages) or "none"
        raise ValueError(f"Unknown stage {name!r}. Available stages: {available}")
    selected = _sections(stages[name], f"stages.{name}") if name is not None else {}
    explicit = _sections(overrides or {}, "overrides")
    values = {}
    for section in SECTIONS:
        settings = shared[section] | selected.get(section, {}) | explicit[section]
        try:
            values[section] = replace(getattr(DEFAULT_PLAN, section), **settings)
        except ValueError as error:
            raise ValueError(f"{section}: {error}") from error
    return Plan(**values), name

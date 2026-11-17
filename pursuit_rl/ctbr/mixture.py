"""Fresh on-policy sampling of current, recent and retained curriculum tasks."""

from types import SimpleNamespace
import torch
from .config import interface, fingerprint
from .environment import CTBRBatch


class TaskMixtureBatch:
    def __init__(self, c, recent, anchor, seed):
        n = c["num_envs"]
        tr = c["training"]
        recent_fraction = tr["replay_recent_fraction"] if recent else 0.0
        anchor_fraction = tr["replay_anchor_fraction"] if anchor else 0.0
        # A task can arrive through both the immediately previous phase and a
        # retained-skill anchor. Identical profiles represent one replay source,
        # not two independent allocations.
        if recent and anchor and fingerprint(recent) == fingerprint(anchor):
            recent_fraction = max(recent_fraction, anchor_fraction)
            anchor_fraction = 0.0
        nr = round(n * recent_fraction)
        na = round(n * anchor_fraction)
        if nr + na >= n:
            raise ValueError("Task mixture must include current-task environments")
        groups = [(c, n - nr - na), (recent, nr), (anchor, na)]
        self.groups = []
        self.task_counts = {}
        for index, (profile, size) in enumerate(groups):
            if not size:
                continue
            if interface(profile) != interface(c):
                raise ValueError("Replay task changes the policy interface")
            env = CTBRBatch(profile, n=size, seed=seed + index * 100003)
            self.groups.append(env)
            name = profile["task"]["name"]
            self.task_counts[name] = self.task_counts.get(name, 0) + size
        self.config = c
        first = self.groups[0]
        for key in (
            "device",
            "dt",
            "decision_dt",
            "decision_steps",
            "gamma",
            "gae_lambda",
            "actor_dim",
            "critic_dim",
            "base_actor_dim",
        ):
            setattr(self, key, getattr(first, key))
        self.n = n
        self.last_physics_steps = 0
        self.reward_components = {}
        self.reward_components_unscaled = {}

    @property
    def steps(self):
        return torch.cat([e.steps for e in self.groups])

    @property
    def target_positions(self):
        return torch.cat([e.plant.target.pos for e in self.groups])

    @property
    def plant(self):
        return SimpleNamespace(
            pursuer=SimpleNamespace(
                thrust=torch.cat([e.plant.pursuer.thrust for e in self.groups]),
                rate=torch.cat([e.plant.pursuer.rate for e in self.groups]),
            )
        )

    def observe(self):
        observations = [e.observe() for e in self.groups]
        return tuple(torch.cat([pair[k] for pair in observations]) for k in (0, 1))

    def reset(self, mask=None):
        start = 0
        for env in self.groups:
            env.reset(None if mask is None else mask[start : start + env.n])
            start += env.n
        return self.observe()

    def step(self, action, raw):
        start = 0
        results = []
        records = []
        for env in self.groups:
            result = env.step(action[start : start + env.n], raw[start : start + env.n])
            for rec in result[-1]:
                records.append(dict(rec, slot=rec["slot"] + start))
            results.append(result)
            start += env.n
        self.last_physics_steps = sum(e.last_physics_steps for e in self.groups)
        # Diagnostic keys differ across objectives and may be extended by future rewards.
        # Always concatenate in physical environment order, filling missing keys with zero.
        keys = sorted(set().union(*(e.reward_components.keys() for e in self.groups)))
        self.reward_components = {
            key: torch.cat([
                e.reward_components.get(key, torch.zeros(e.n, device=e.device))
                for e in self.groups
            ])
            for key in keys
        }
        raw_keys = sorted(
            set().union(*(e.reward_components_unscaled.keys() for e in self.groups))
        )
        self.reward_components_unscaled = {
            key: torch.cat([
                e.reward_components_unscaled.get(
                    key, torch.zeros(e.n, device=e.device)
                )
                for e in self.groups
            ])
            for key in raw_keys
        }
        return (
            self.observe(),
            *[torch.cat([r[k] for r in results]) for k in (1, 2, 3)],
            records,
        )


def training_env(c, replay, seed):
    if replay and (replay.get("recent") or replay.get("anchor")):
        return TaskMixtureBatch(c, replay.get("recent"), replay.get("anchor"), seed)
    env = CTBRBatch(c, seed=seed)
    env.task_counts = {c["task"]["name"]: c["num_envs"]}
    return env


def target_positions(env):
    return (
        env.target_positions
        if isinstance(env, TaskMixtureBatch)
        else env.plant.target.pos
    )

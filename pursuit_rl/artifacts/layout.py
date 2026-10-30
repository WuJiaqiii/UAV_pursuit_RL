"""Experiment identity and operation paths, separate from algorithm configuration."""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pursuit_rl.artifacts.io import write_json

SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")


def identifier(kind, name=None, seed=None):
    def slug(value):
        cleaned = re.sub(r"[^\w-]+", "-", str(value), flags=re.UNICODE).strip("-_")
        return cleaned[:48] or "experiment"

    parts = [datetime.now(SHANGHAI).strftime("%Y%m%d_%H%M%S"), slug(kind)]
    if name:
        parts.append(slug(name))
    if seed is not None:
        parts.append(f"s{seed}")
    return "_".join([*parts, uuid.uuid4().hex[:8]])


def find_experiment(path):
    """Use explicit identity markers; directory names alone do not prove ownership."""
    path = Path(path).expanduser().resolve()
    start = path if path.is_dir() else path.parent
    for candidate in [start, *start.parents]:
        if (candidate / "experiment.json").is_file() or (
            candidate / "run.json"
        ).is_file():
            return Experiment.open(candidate)
        operation = candidate / "operation.json"
        if operation.is_file():
            metadata = json.loads(operation.read_text(encoding="utf8"))
            owner = metadata.get("experiment")
            if owner and any(
                (Path(owner) / marker).is_file()
                for marker in ("experiment.json", "run.json")
            ):
                return Experiment.open(owner)
    return None


@dataclass(frozen=True)
class Experiment:
    directory: Path

    def __post_init__(self):
        object.__setattr__(
            self, "directory", Path(self.directory).expanduser().resolve()
        )

    @property
    def results(self):
        return self.directory / "results"

    @property
    def checkpoints(self):
        return self.directory / "checkpoints"

    @property
    def logs(self):
        return self.directory / "logs"

    @property
    def config_file(self):
        return self.directory / "config.yaml"

    @classmethod
    def plan(cls, kind, root="runs", name=None, seed=None, directory=None):
        return cls(
            Path(directory) if directory else Path(root) / identifier(kind, name, seed)
        )

    @classmethod
    def open(cls, directory):
        experiment = cls(directory)
        if not experiment.directory.is_dir() or not any(
            (experiment.directory / marker).is_file()
            for marker in ("experiment.json", "run.json")
        ):
            raise ValueError(f"Not a registered experiment: {experiment.directory}")
        marker = experiment.directory / "experiment.json"
        if marker.is_file():
            metadata = json.loads(marker.read_text(encoding="utf8"))
            if metadata.get("schema") != "pursuit-experiment-v1":
                raise ValueError(f"Unsupported experiment identity: {marker}")
        else:
            metadata = json.loads(
                (experiment.directory / "run.json").read_text(encoding="utf8")
            )
            if not str(metadata.get("schema", "")).startswith(
                "pursuit-"
            ) or not metadata.get("config_hash"):
                raise ValueError(
                    f"Invalid training run identity: {experiment.directory}"
                )
        return experiment

    def register(self, kind="train", name=None, seed=None, sources=()):
        """Register explicitly; planning or constructing Config never creates files."""
        marker = self.directory / "experiment.json"
        if marker.exists():
            return
        legacy = (self.directory / "run.json").is_file()
        metadata = (
            json.loads((self.directory / "run.json").read_text(encoding="utf8"))
            if legacy
            else {}
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(
            marker,
            {
                "schema": "pursuit-experiment-v1",
                "experiment_id": self.directory.name,
                "kind": kind,
                "name": name,
                "seed": metadata.get("seed", seed),
                "created_at": (
                    metadata.get("created_at")
                    if legacy
                    else datetime.now(SHANGHAI).isoformat(timespec="seconds")
                ),
                "registered_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
                "sources": [str(Path(path).expanduser().resolve()) for path in sources],
                "training": legacy,
            },
        )

    def operation(self, kind, name=None, directory=None, sources=()):
        if kind in (".", "..") or "/" in kind or "\\" in kind:
            raise ValueError("Operation kind must be one path component")
        path = (
            Path(directory)
            if directory
            else self.results / kind / identifier(kind, name)
        )
        return Operation(
            self,
            kind,
            path,
            tuple(str(Path(p).expanduser().resolve()) for p in sources),
        )


@dataclass(frozen=True)
class Operation:
    experiment: Experiment
    kind: str
    directory: Path
    sources: tuple = ()

    def __post_init__(self):
        object.__setattr__(
            self, "directory", Path(self.directory).expanduser().resolve()
        )

    def register(self, status, output):
        # A failed guard must not write into an existing result directory.
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(
            self.directory / "operation.json",
            {
                "schema": "pursuit-operation-v1",
                "kind": self.kind,
                "status": status,
                "experiment": str(self.experiment.directory),
                "sources": list(self.sources),
                "output": str(Path(output).expanduser().resolve()),
                "recorded_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            },
        )


def select_experiment(args, sources):
    if getattr(args, "run", None):
        return Experiment.open(args.run)
    inferred = [find_experiment(path) for path in sources]
    owners = {owner.directory for owner in inferred if owner is not None}
    if inferred and all(owner is not None for owner in inferred) and len(owners) == 1:
        return Experiment(owners.pop())
    return Experiment.plan(
        args.command, args.runs_dir, args.name, getattr(args, "seed", None)
    )

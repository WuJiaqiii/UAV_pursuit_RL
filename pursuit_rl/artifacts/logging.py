"""Per-invocation experiment logs with human output and structured lifecycle data."""

import copy
import hashlib
import importlib.metadata
import logging
import os
import platform
import shlex
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from pursuit_rl import __version__
from pursuit_rl.config import config_hash, dump_config, as_dict
from pursuit_rl.artifacts.io import log_jsonl, write_json
from pursuit_rl.artifacts.provenance import source_fingerprint

SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")


def timestamp():
    return datetime.now(SHANGHAI).isoformat(timespec="milliseconds")


class ColorFormatter(logging.Formatter):
    """Color a copied record so file handlers never see terminal escape codes."""

    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;41m",
    }

    def __init__(self, color=False):
        super().__init__("%(asctime)s | %(levelname)-8s | %(message)s")
        self.color = color

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, SHANGHAI).isoformat(
            timespec="seconds"
        )

    def format(self, record):
        record = copy.copy(record)
        if self.color:
            color = self.COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}\033[0m"
        return super().format(record)


def runtime_metadata():
    versions = {}
    for name in ("numpy", "torch", "PyYAML", "matplotlib", "tensorboard"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "versions": versions,
        "source_sha256": source_fingerprint(),
    }


def scalar_metrics(values, prefix=""):
    """Flatten reward components and action saturation for optional TensorBoard."""
    for key, value in values.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from scalar_metrics(value, tag)
        elif isinstance(value, (list, tuple)):
            yield from scalar_metrics(dict(enumerate(value)), tag)
        elif isinstance(value, (int, float)):
            yield tag, value


class ExpLogger:
    """One session per command/resume; no global logger class or root handlers."""

    def __init__(
        self, directory, mode, config=None, invocation=None, tensorboard=False
    ):
        self.directory = Path(directory)
        self.session_id = (
            datetime.now(SHANGHAI).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.session_dir = self.directory / "logs" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.logger = logging.Logger(f"pursuit.{mode}.{self.session_id}", logging.DEBUG)
        self.logger.propagate = False
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            ColorFormatter(
                color=bool(getattr(console.stream, "isatty", lambda: False)())
            )
        )
        file_handler = logging.FileHandler(
            self.session_dir / "console.log", encoding="utf8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ColorFormatter())
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.writer = None
        self.use_tensorboard = tensorboard
        self._clock = time.monotonic()
        self._closed = False
        self.sequence = 0
        self.state = {
            "schema": "pursuit-session-v1",
            "session_id": self.session_id,
            "mode": mode,
            "status": "initialized",
            "created_at": timestamp(),
            "timezone": "Asia/Shanghai",
            "output_directory": str(self.directory.resolve()),
            "cwd": str(Path.cwd()),
            "command": shlex.join([sys.executable, *sys.argv]),
            "argv": [sys.executable, *sys.argv],
            "arguments": invocation or {},
            "runtime": runtime_metadata(),
        }
        try:
            if config is not None:
                self.set_config(config)
            self._save_state()
        except BaseException:
            self.close()
            raise

    def metadata(self, **values):
        self.state.setdefault("metadata", {}).update(values)
        self._save_state()

    def record_inputs(self, paths):
        records = []
        for name, values in paths.items():
            for value in values if isinstance(values, (list, tuple)) else [values]:
                path = Path(value).expanduser().resolve()
                record = {"name": name, "path": str(path), "exists": path.exists()}
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    record.update(
                        size_bytes=path.stat().st_size, sha256=digest.hexdigest()
                    )
                records.append(record)
        self.state["inputs"] = records
        self._save_state()
        self.event("inputs", files=records)

    def _save_state(self):
        temporary = self.session_dir / "session.tmp"
        write_json(temporary, self.state)
        temporary.replace(self.session_dir / "session.json")

    def set_config(self, config):
        config = as_dict(config)
        dump_config(config, self.session_dir / "resolved_config.yaml")
        self.state["config_hash"] = config_hash(config)
        self.logger.info("Experiment settings | %s", self.state["mode"])
        for section, value in config.items():
            self.logger.info(
                "[%s]\n%s",
                section,
                yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip(),
            )
        self._save_state()

    def __enter__(self):
        if self._closed or self.state["status"] != "initialized":
            raise RuntimeError("An experiment session can only run once")
        self.state.update(status="running", started_at=self.state["created_at"])
        self._save_state()
        self.event("started", mode=self.state["mode"])
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.directory / "tensorboard"))
            except ImportError:
                self.logger.warning(
                    "TensorBoard unavailable; JSONL and text logs remain active"
                )
            except BaseException as error:
                self.fail(error)
                self.close()
                raise
        return self

    def record(self, filename, values, tensorboard=None):
        self.sequence += 1
        record = {
            **values,
            "timestamp": timestamp(),
            "session_id": self.session_id,
            "sequence": self.sequence,
        }
        log_jsonl(self.directory / filename, record)
        if tensorboard and self.writer:
            for tag, value in scalar_metrics(values):
                self.writer.add_scalar(
                    f"{tensorboard}/{tag}", value, values.get("steps", 0)
                )
            self.writer.flush()
        return record

    def event(self, event, **values):
        return self.record("events.jsonl", {**values, "event": event})

    def complete(self, result=None):
        self.state.update(
            status="completed",
            finished_at=timestamp(),
            elapsed_s=time.monotonic() - self._clock if self._clock else 0.0,
        )
        if result is not None:
            self.state["result"] = result
        self._save_state()
        self.event("completed", elapsed_s=self.state["elapsed_s"])
        self.logger.info(
            "Completed | session=%s | elapsed=%.2fs",
            self.session_id,
            self.state["elapsed_s"],
        )

    def fail(self, error):
        self.state.setdefault("started_at", self.state["created_at"])
        self.state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at=timestamp(),
            elapsed_s=time.monotonic() - self._clock if self._clock else 0.0,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            },
        )
        self._save_state()
        self.event(
            self.state["status"], error_type=type(error).__name__, message=str(error)
        )
        self.logger.error(
            "%s: %s",
            self.state["status"],
            error,
            exc_info=(type(error), error, error.__traceback__),
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.writer:
                self.writer.close()
        finally:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
                handler.close()

    def __exit__(self, error_type, error, tb):
        try:
            if error is not None:
                self.fail(error)
            elif self.state["status"] != "completed":
                self.complete()
        finally:
            self.close()
        return False

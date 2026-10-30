"""Unified entry points and observable success, failure and recovery lifecycles."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from main import build_parser, main
from pursuit_rl.config import load_config, dump_config
from pursuit_rl.artifacts.logging import ColorFormatter, ExpLogger
from pursuit_rl.training.trainer import Trainer


def states(directory):
    return [
        json.loads(path.read_text(encoding="utf8"))
        for path in sorted(Path(directory).glob("logs/*/session.json"))
    ]


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def test_color_formatter_preserves_record_and_file_format():
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "value=%s", (3,), None
    )
    colored = ColorFormatter(True).format(record)
    plain = ColorFormatter(False).format(record)
    assert "\033[" in colored and "\033[" not in plain
    assert "value=3" in plain and "+08:00" in plain
    assert (
        record.levelname == "INFO" and record.msg == "value=%s" and record.args == (3,)
    )


def test_independent_sessions_do_not_change_global_logging(tmp_path):
    logger_class = logging.getLoggerClass()
    root_handlers = list(logging.getLogger().handlers)
    first = ExpLogger(
        tmp_path,
        "analysis",
        config={"physics": {"dt": 0.05}},
        invocation={"test": True},
    )
    with first:
        first.logger.info("one message %s", 1)
        first.record("metrics.jsonl", {"steps": 2, "value": 0.5})
        first.complete({"outputs": ["result.json"]})
    second = ExpLogger(tmp_path, "analysis")
    with second:
        second.logger.info("second message")
    assert len(states(tmp_path)) == 2
    assert first.logger.handlers == [] and second.logger.handlers == []
    assert logging.getLoggerClass() is logger_class
    assert list(logging.getLogger().handlers) == root_handlers
    assert all(state["status"] == "completed" for state in states(tmp_path))
    text = (first.session_dir / "console.log").read_text(encoding="utf8")
    assert text.count("one message 1") == 1 and "second message" not in text
    assert "\033[" not in text
    row = records(tmp_path / "metrics.jsonl")[0]
    assert row["session_id"] == first.session_id and row["timestamp"].endswith("+08:00")
    with pytest.raises(RuntimeError):
        with first:
            pass


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("injected failure"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_exception_and_interrupt_are_persisted_and_closed(tmp_path, error, status):
    experiment = ExpLogger(tmp_path, "test")
    with pytest.raises(type(error)):
        with experiment:
            raise error
    state = states(tmp_path)[0]
    assert state["status"] == status
    assert state["error"]["type"] == type(error).__name__
    assert "started_at" in state and "finished_at" in state
    assert state["elapsed_s"] >= 0 and experiment.logger.handlers == []
    assert "Traceback" in (experiment.session_dir / "console.log").read_text(
        encoding="utf8"
    )


def test_training_records_and_interrupted_rollout_resume(tmp_path):
    config = load_config("configs/smoke.yaml")
    output = tmp_path / "train"
    trainer = Trainer(config, output)
    trainer.exp_logger.use_tensorboard = False
    update = trainer.model.update
    calls = 0

    def failing_update(buffer):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("optimizer test failure")
        return update(buffer)

    trainer.model.update = failing_update
    with pytest.raises(RuntimeError, match="optimizer test failure"):
        trainer.run()
    identity = (output / "run.json").read_bytes()
    assert states(output)[0]["status"] == "failed"
    assert records(output / "train.jsonl")[0]["reward_components"]
    assert records(output / "validation_details.jsonl")[0]["stage_report"][
        "episodes_detail"
    ]
    completed = records(output / "episodes.jsonl")
    assert completed and all(row["phase"] == "collected" for row in completed)
    previous_ids = {row["episode_id"] for row in completed}
    resumed = Trainer(config, output, resume=output / "checkpoints/latest.pt")
    assert resumed.episode_count > 0
    resumed.exp_logger.use_tensorboard = False
    resumed.run()
    assert (output / "run.json").read_bytes() == identity
    assert {state["status"] for state in states(output)} == {"failed", "completed"}
    all_episodes = records(output / "episodes.jsonl")
    assert len({row["episode_id"] for row in all_episodes}) == len(all_episodes)
    assert previous_ids < {row["episode_id"] for row in all_episodes}
    checkpoints = [
        row for row in records(output / "events.jsonl") if row["event"] == "checkpoint"
    ]
    assert checkpoints and all(len(row["sha256"]) == 64 for row in checkpoints)


def test_invalid_initialization_logs_failure_after_directory_guard(tmp_path):
    config = load_config("configs/smoke.yaml")
    output = tmp_path / "missing_weights"
    with pytest.raises(FileNotFoundError):
        Trainer(config, output, init_from=tmp_path / "missing.pt")
    assert states(output)[0]["status"] == "failed"
    before = sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
    with pytest.raises(FileExistsError):
        Trainer(config, output)
    assert (
        sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
        == before
    )


def test_tensorboard_optional_fallback_keeps_jsonl(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    with ExpLogger(tmp_path, "train", tensorboard=True) as logger:
        logger.record("train.jsonl", {"steps": 1, "loss": 0.3}, tensorboard="train")
        assert logger.writer is None
    assert records(tmp_path / "train.jsonl")[0]["loss"] == 0.3
    assert "TensorBoard unavailable" in (logger.session_dir / "console.log").read_text(
        encoding="utf8"
    )


def test_main_resume_reads_saved_configuration(tmp_path):
    config = load_config("configs/smoke.yaml")
    config["training"]["total_steps"] = 32
    output = tmp_path / "resume_cli"
    trainer = Trainer(config, output)
    trainer.exp_logger.use_tensorboard = False
    trainer.run()
    assert (
        main(
            [
                "train",
                "--resume",
                str(output / "checkpoints/latest.pt"),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert all(state["status"] == "completed" for state in states(output))
    assert len(states(output)) == 2


def test_main_failure_has_sidecar_and_nonzero_exit(tmp_path):
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--out",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    assert states(tmp_path / "result.run")[0]["status"] == "failed"


def test_unified_pipeline_generates_and_reads_outputs(tmp_path):
    config = load_config("configs/smoke.yaml")
    config_path = tmp_path / "config.yaml"
    dump_config(config, config_path)
    result = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(config_path),
                "--algorithms",
                "PN",
                "APN",
                "--stage",
                "0",
                "--episodes",
                "2",
                "--out",
                str(result),
                "--no-plots",
            ]
        )
        == 0
    )
    assert states(result)[0]["status"] == "completed"
    trace = result / "PN/configured_stage/episode_0000.npz"
    metrics = tmp_path / "metrics.json"
    assert main(["trace-metrics", str(trace), "--out", str(metrics)]) == 0
    assert json.loads(metrics.read_text())["reason"] == "timeout"
    figure = tmp_path / "trajectory.png"
    assert main(["visualize", "--trace", str(trace), "--out", str(figure)]) == 0
    assert figure.stat().st_size > 0
    assert (
        main(
            [
                "compare",
                str(result / "PN/configured_stage"),
                str(result / "APN/configured_stage"),
                "--out",
                str(tmp_path / "paired.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "report",
                "--input",
                str(result / "summary.json"),
                "--out",
                str(tmp_path / "report.csv"),
            ]
        )
        == 0
    )
    assert (
        main(["feasibility", str(result), "--out", str(tmp_path / "feasibility.csv")])
        == 0
    )
    assert (
        main(["export", "--source", str(result), "--out", str(tmp_path / "light.zip")])
        == 0
    )
    assert (tmp_path / "light.zip").stat().st_size > 0
    assert len(records(result / "benchmark.jsonl")) == 2


def test_unified_parser_is_mutually_exclusive_and_import_safe():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--resume", "a", "--init-from", "b"])
    result = subprocess.run(
        [sys.executable, "main.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "trace-metrics" in result.stdout and "benchmark" in result.stdout

"""Configuration resolution and ownership of complete experiment artifacts."""

import copy
import json
import re
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from main import build_parser, main
from pursuit_rl.config import Config, config_hash, dump_config, load_config
from pursuit_rl.artifacts.layout import Experiment, find_experiment, select_experiment


def test_config_is_deeply_read_only_and_serializable(tmp_path):
    config = Config.load("configs/smoke.yaml")
    assert config.training.num_envs == config["training"]["num_envs"] == 4
    assert isinstance(config.curriculum, tuple)
    with pytest.raises(AttributeError):
        config.seed = 100
    with pytest.raises(AttributeError):
        config.extra_field = "not allowed"
    with pytest.raises(AttributeError):
        config.training.num_envs = 100
    with pytest.raises(TypeError):
        config.curriculum[0].modes[0] = "other"
    with pytest.raises(TypeError):
        config["seed"] = 100
    detached = config.to_dict()
    detached["training"]["num_envs"] = 999
    assert config.training.num_envs == 4 and copy.deepcopy(config) is config
    changed = config.with_overrides({"training": {"num_envs": 2}})
    assert changed.training.num_envs == 2 and config.training.num_envs == 4
    output = tmp_path / "config.yaml"
    config.save(output)
    assert yaml.safe_load(output.read_text(encoding="utf8")) == config.to_dict()
    assert "!!python" not in output.read_text(encoding="utf8")
    assert Config.load(output).hash == config.hash == config_hash(config.to_dict())


def test_config_priority_and_runtime_paths_do_not_affect_hash(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("seed: 12\ntraining:\n  total_steps: 32\n", encoding="utf8")
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(path),
            "--set",
            "seed=21",
            "--set",
            "training.actor_lr=0.0001",
            "--seed",
            "43",
            "--out",
            str(tmp_path / "a"),
        ]
    )
    config = Config.from_args(args)
    assert config.seed == 43 and config.training.total_steps == 32
    assert config.training.actor_lr == 0.0001
    args.out = str(tmp_path / "b")
    args.name = "different"
    args.runs_dir = str(tmp_path / "c")
    assert Config.from_args(args).hash == config.hash
    assert (
        not (tmp_path / "a").exists()
        and not (tmp_path / "b").exists()
        and not (tmp_path / "c").exists()
    )
    args.command = "evaluate"
    args.seed = 999
    assert (
        Config.from_args(args).seed == 21
    )  # evaluation seed is not a training config override


def test_checkpoint_fallback_yaml_and_set_values(tmp_path):
    fallback = Config.load("configs/smoke.yaml").to_dict()
    fallback["seed"] = 17
    args = Namespace(
        command="benchmark",
        config=None,
        set_values=["stages.stage0.time=0.2"],
        seed=84000,
    )
    config = Config.from_args(args, fallback=fallback)
    assert config.seed == 17 and config.curriculum[0].time == 0.2
    for expression in [
        "missing=3",
        "training.actor_lr=-1",
        "not.an.option=2",
        "seed",
        "stages.unknown.time=1",
    ]:
        args.set_values = [expression]
        with pytest.raises(ValueError):
            Config.from_args(args, fallback=fallback)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("false", encoding="utf8")
    with pytest.raises(ValueError):
        Config.load(invalid)


def test_device_auto_is_explicit_and_serialized_as_string(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    automatic = Config().with_overrides({"device": "auto"})
    assert automatic.device == "cuda" and isinstance(automatic.to_dict()["device"], str)
    args = Namespace(
        command="train",
        config=None,
        seed=None,
        device="cpu",
        set_values=["device=auto"],
        num_envs=None,
        total_steps=None,
    )
    assert Config.from_args(args).device == "cpu"


def test_planning_is_side_effect_free_and_names_are_unique(tmp_path):
    first = Experiment.plan("train", tmp_path, "precision", 43)
    second = Experiment.plan("train", tmp_path, "precision", 43)
    assert first.directory != second.directory
    assert re.fullmatch(
        r"\d{8}_\d{6}_train_precision_s43_[0-9a-f]{8}", first.directory.name
    )
    assert list(tmp_path.iterdir()) == []
    assert first.results == first.directory / "results"
    first.register("train", "precision", 43)
    assert Experiment.open(first.directory) == first
    operation = first.operation("evaluate")
    assert operation.directory.is_relative_to(first.results / "evaluate")
    with pytest.raises(ValueError):
        first.operation("../bad")


def test_cross_experiment_inputs_get_their_own_comparison(tmp_path):
    a = Experiment.plan("train", tmp_path)
    a.register()
    b = Experiment.plan("train", tmp_path)
    b.register()
    args = Namespace(command="compare", run=None, runs_dir=str(tmp_path), name="paired")
    selected = select_experiment(
        args, [a.directory / "report.json", b.directory / "report.json"]
    )
    assert selected.directory not in (a.directory, b.directory)
    assert selected.directory.parent == tmp_path
    assert (
        select_experiment(args, [a.directory / "one.json", a.directory / "two.json"])
        == a
    )
    assert (
        select_experiment(args, [a.directory / "one.json", tmp_path / "unowned.json"])
        != a
    )
    args.run = str(b.directory)
    assert select_experiment(args, [a.directory / "one.json"]) == b


def test_default_pipeline_keeps_every_operation_with_its_training_run(tmp_path):
    root = tmp_path / "runs"
    assert (
        main(
            [
                "train",
                "--config",
                "configs/smoke.yaml",
                "--runs-dir",
                str(root),
                "--name",
                "precision",
            ]
        )
        == 0
    )
    runs = list(root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert "_train_precision_s12_" in run.name
    identity = (run / "experiment.json").read_bytes()
    checkpoint = run / "checkpoints/best.pt"
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--stage",
                "0",
                "--episodes",
                "2",
                "--save-traces",
            ]
        )
        == 0
    )
    evaluation = list((run / "results/evaluate").iterdir())[0]
    assert (evaluation / "report.json").is_file() and (
        evaluation / "traces/episode_0000.npz"
    ).is_file()
    assert (
        main(
            [
                "benchmark",
                "--checkpoint",
                str(checkpoint),
                "--algorithms",
                "PPO",
                "PN",
                "APN",
                "--stage",
                "0",
                "--episodes",
                "2",
                "--no-plots",
            ]
        )
        == 0
    )
    benchmark = list((run / "results/benchmark").iterdir())[0]
    before = (benchmark / "summary.json").read_bytes()
    assert (
        main(
            [
                "benchmark",
                "--checkpoint",
                str(checkpoint),
                "--algorithms",
                "PN",
                "--stage",
                "0",
                "--episodes",
                "2",
                "--no-plots",
            ]
        )
        == 0
    )
    assert len(list((run / "results/benchmark").iterdir())) == 2
    assert (benchmark / "summary.json").read_bytes() == before
    assert main(["report", "--input", str(benchmark / "summary.json")]) == 0
    assert main(["trace-metrics", str(evaluation / "traces/episode_0000.npz")]) == 0
    assert main(["export", "--run", str(run)]) == 0
    export = list((run / "results/export").iterdir())[0]
    with zipfile.ZipFile(export / "records.zip") as archive:
        assert "config.yaml" in archive.namelist()
        relative = export.relative_to(run).as_posix()
        assert not any(name.startswith(relative + "/") for name in archive.namelist())
    assert main(["train", "--resume", str(run / "checkpoints/latest.pt")]) == 0
    assert (run / "experiment.json").read_bytes() == identity
    assert len(list(root.iterdir())) == 1
    for operation in (run / "results").glob("*/*/operation.json"):
        meta = json.loads(operation.read_text(encoding="utf8"))
        assert meta["experiment"] == str(run) and meta["status"] == "completed"
    assert not (tmp_path / "results").exists()


def test_standalone_evaluation_creates_an_independent_experiment(tmp_path):
    root = tmp_path / "runs"
    assert (
        main(
            [
                "evaluate",
                "--algorithm",
                "PN",
                "--config",
                "configs/smoke.yaml",
                "--stage",
                "0",
                "--episodes",
                "2",
                "--runs-dir",
                str(root),
                "--name",
                "baseline",
            ]
        )
        == 0
    )
    run = list(root.iterdir())[0]
    assert "_evaluate_baseline_" in run.name
    metadata = json.loads((run / "experiment.json").read_text(encoding="utf8"))
    assert not metadata["training"]
    assert list((run / "results/evaluate").glob("*/report.json"))


def test_explicit_output_can_still_be_followed_to_owner(tmp_path):
    run = Experiment.plan("benchmark", tmp_path / "runs")
    run.register("benchmark")
    external = tmp_path / "custom_report"
    external.mkdir()
    operation = run.operation("benchmark", directory=external)
    operation.register("completed", external)
    assert find_experiment(external / "PN/case/episode_0000.npz") == run
    with pytest.raises(ValueError):
        Experiment.open(tmp_path / "missing")


def test_export_custom_log_root_does_not_exclude_experiment(tmp_path):
    run = Experiment.plan("train", tmp_path / "runs")
    run.register()
    Config.load("configs/smoke.yaml").save(run.config_file)
    assert (
        main(["export", "--run", str(run.directory), "--log-dir", str(run.directory)])
        == 0
    )
    archive_path = next(run.results.glob("export/*/records.zip"))
    with zipfile.ZipFile(archive_path) as archive:
        assert "config.yaml" in archive.namelist()
        assert not any(
            name.startswith("results/export/") for name in archive.namelist()
        )


def test_run_context_supplies_baseline_config_without_changing_hash(tmp_path):
    experiment = Experiment.plan("train", tmp_path)
    experiment.register()
    config = Config.load("configs/smoke.yaml")
    config.save(experiment.config_file)
    args = build_parser().parse_args(
        ["evaluate", "--algorithm", "PN", "--run", str(experiment.directory)]
    )
    assert Config.from_args(args).hash == config.hash


def test_failed_new_training_is_registered_but_guard_does_not_mutate_existing(tmp_path):
    root = tmp_path / "runs"
    assert (
        main(
            [
                "train",
                "--config",
                "configs/smoke.yaml",
                "--init-from",
                str(tmp_path / "missing.pt"),
                "--runs-dir",
                str(root),
            ]
        )
        == 1
    )
    run = list(root.iterdir())[0]
    assert (run / "experiment.json").is_file()
    state = json.loads(
        next((run / "logs").glob("*/session.json")).read_text(encoding="utf8")
    )
    assert state["status"] == "failed"
    before = {
        str(path.relative_to(run)): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert main(["train", "--config", "configs/smoke.yaml", "--out", str(run)]) == 1
    assert before == {
        str(path.relative_to(run)): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }

"""Paired A/B/C precision experiments with frozen baselines and bounded budgets."""

import copy
import hashlib
import json
from pathlib import Path
from .config import load as config_load, fingerprint, interface, validate
from .checkpoint import load, migrate, write_json, source_fingerprint
from .curriculum import gate
from .evaluation import evaluate
from .training import train
from .selection import passed

SCHEMA = "pursuit-ctbr-experiment-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cached_evaluate(model, c, checkpoint, path, episodes, seed, stochastic=False):
    key = dict(
        config=fingerprint(c),
        checkpoint_sha256=sha(checkpoint),
        device=c["device"],
        episodes=episodes,
        seed=seed,
        stochastic=stochastic,
    )
    path = Path(path)
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("experiment_evaluation") != key:
            raise ValueError(
                f"Cached evaluation changed: {path}; use a new output directory"
            )
        return report
    print(
        json.dumps(
            dict(
                event="experiment_evaluate",
                output=str(path),
                seed=seed,
                episodes=episodes,
                stochastic=stochastic,
            )
        ),
        flush=True,
    )
    report, _ = evaluate(model, c, episodes, seed, stochastic=stochastic)
    report["experiment_evaluation"] = key
    write_json(path, report)
    print(
        json.dumps(
            dict(
                event="experiment_evaluated",
                output=str(path),
                success_rate=report["success_rate"],
                tail_rmse=report["mean_tail_position_rmse"],
            )
        ),
        flush=True,
    )
    return report


def compare(reference, candidate):
    """Separate safety from survivor-only tracking errors."""
    if reference["initial_state_digest"] != candidate["initial_state_digest"]:
        raise ValueError("Unpaired evaluation initial states")
    before = {r["episode"]: r for r in reference["episodes_detail"]}
    pairs = [(before[r["episode"]], r) for r in candidate["episodes_detail"]]
    complete = [(a, b) for a, b in pairs if a["tail_complete"] and b["tail_complete"]]
    return dict(
        success_rate_before=reference["success_rate"],
        success_rate_after=candidate["success_rate"],
        success_delta=candidate["success_rate"] - reference["success_rate"],
        lost_successes=sum(a["success"] and not b["success"] for a, b in pairs),
        recovered_successes=sum(not a["success"] and b["success"] for a, b in pairs),
        paired_complete_tail_episodes=len(complete),
        paired_tail_rmse_delta=(
            sum(b["tail_position_rmse"] - a["tail_position_rmse"] for a, b in complete)
            / len(complete)
            if complete
            else None
        ),
    )


def run(
    plan_path, source_checkpoint, out, device="cpu", seeds=(42,), test_episodes=None
):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA or set(plan["variants"]) != {"A", "B", "C"}:
        raise ValueError("Expected an A/B/C experiment plan")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Experiment seeds must be distinct nonnegative integers")
    source = Path(source_checkpoint).resolve()
    source_sha = sha(source)
    if plan.get("source_sha256") and source_sha != plan["source_sha256"]:
        raise ValueError(
            "Source checkpoint differs from the audited v700 best; use an explicit new plan"
        )
    episodes = plan["test_episodes"] if test_episodes is None else test_episodes
    if (
        type(episodes) is not int
        or episodes < 1
        or plan["stochastic_episodes"] < 1
        or plan["validation_episodes"] < 1
    ):
        raise ValueError("Evaluation episode counts must be positive")
    for key in ("gate_seed_offsets", "retention_seed_offsets"):
        values = plan[key]
        if (
            len(values) < 2
            or len(set(values)) != len(values)
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError("Two distinct validation seed groups are required")
    root = Path(out).resolve()
    profiles = {
        name: config_load(path.parent / profile, device)
        for name, profile in plan["variants"].items()
    }
    if profiles["A"]["training"]["physics_steps"] != 0:
        raise ValueError("A must be a zero-training frozen baseline")
    if (
        profiles["B"]["training"]["physics_steps"] <= 0
        or profiles["B"]["training"]["physics_steps"]
        != profiles["C"]["training"]["physics_steps"]
    ):
        raise ValueError("B and C require identical positive training budgets")
    canonical = []
    for c in profiles.values():
        if (
            c["training"]["prediction_updates_enabled"]
            or c["training"]["bc_updates"]
            or c["training"]["transfer_dagger_updates"]
        ):
            raise ValueError(
                "First-wave trials freeze the predictor and do not add imitation updates"
            )
        entry = copy.deepcopy(c)
        entry["training"].pop("physics_steps")
        entry["policy"].pop("initial_std")
        canonical.append(entry)
    if any(c != canonical[0] for c in canonical[1:]):
        raise ValueError(
            "A/B/C may differ only in training budget and initial exploration std"
        )
    if profiles["A"]["policy"]["initial_std"] != profiles["B"]["policy"]["initial_std"]:
        raise ValueError("A and B must share the same initial exploration")
    evaluation_profiles = {
        name: config_load(path.parent / profile, device)
        for name, profile in plan["evaluation_profiles"].items()
    }
    replay_profiles = {
        name: config_load(path.parent / profile, device)
        for name, profile in plan["replay_profiles"].items()
    }
    if not set(plan["retention_tasks"]).issubset(evaluation_profiles):
        raise ValueError("Unknown retention task")
    if (
        plan.get("primary_task", next(iter(evaluation_profiles)))
        not in evaluation_profiles
    ):
        raise ValueError("Unknown primary evaluation task")
    if plan["test_seed_offset"] in set(
        plan["gate_seed_offsets"] + plan["retention_seed_offsets"]
    ):
        raise ValueError("Final-test seeds must be separate from validation")
    manifest = dict(
        schema=SCHEMA,
        plan=plan,
        source_sha256=source_sha,
        device=device,
        test_episodes=episodes,
        source_fingerprint=source_fingerprint(),
        profiles={k: fingerprint(v) for k, v in profiles.items()},
        evaluation_profiles={k: fingerprint(v) for k, v in evaluation_profiles.items()},
        replay_profiles={k: fingerprint(v) for k, v in replay_profiles.items()},
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Experiment contract changed; use a new output root")
    write_json(manifest_path, manifest)
    results = {}
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        configs = {}
        transfers = {}
        for name, profile in profiles.items():
            c = copy.deepcopy(profile)
            c["seed"] = seed
            c["training"]["validation_seed"] = (
                plan["training_validation_seed_offset"] + seed
            )
            validate(c)
            configs[name] = c
            transfer = seed_root / "initializations" / f"{name}.pt"
            if transfer.exists():
                _, old, saved = load(transfer, device)
                if (
                    fingerprint(old) != fingerprint(c)
                    or saved["progress"]["migration"]["source_sha256"] != source_sha
                ):
                    raise ValueError("Migration input changed")
            else:
                migrate(source, c, transfer, reset_std=True)
            transfers[name] = transfer

        def task_config(base, profile):
            if interface(base) != interface(profile):
                raise ValueError(
                    "Evaluation/replay profile changes the policy interface"
                )
            c = copy.deepcopy(base)
            c["task"] = copy.deepcopy(profile["task"])
            return validate(c)

        baseline, _, _ = load(transfers["A"], device)
        anchors = []
        for name in plan["retention_tasks"]:
            c = task_config(configs["A"], evaluation_profiles[name])
            refs = []
            for offset in plan["retention_seed_offsets"]:
                report = cached_evaluate(
                    baseline,
                    c,
                    transfers["A"],
                    seed_root / "references" / f"{name}_{offset+seed}.json",
                    plan["validation_episodes"],
                    offset + seed,
                )
                if report["mean_tail_position_rmse"] is None:
                    raise ValueError(
                        "Frozen baseline has no complete tail for a retention anchor"
                    )
                refs.append(
                    dict(
                        seed=offset + seed, tail_rmse=report["mean_tail_position_rmse"]
                    )
                )
            anchors.append(dict(name=name, config=c, reference=refs))

        selected = {"A": transfers["A"]}
        progress = {"A": dict(ppo_updates=0, physics_steps=0)}
        for name in ("B", "C"):
            c = configs[name]
            directory = seed_root / name

            def promotion(model, current, steps):
                return gate(
                    model,
                    current,
                    anchors,
                    directory,
                    plan["validation_episodes"],
                    seed,
                    c["training"]["minimum_success"],
                    plan,
                    f"selection_{steps['ppo_updates']:05d}",
                    fail_fast=True,
                )["passed"]

            replay = {k: task_config(c, v) for k, v in replay_profiles.items()}
            latest = directory / "checkpoints/latest.pt"
            result = train(
                c,
                directory,
                resume=latest if latest.exists() else None,
                init_from=transfers[name] if not latest.exists() else None,
                replay=replay,
                promotion_check=promotion,
            )
            feasible = directory / "checkpoints/best_feasible.pt"
            selected[name] = (
                feasible
                if feasible.exists()
                else directory / "checkpoints/best_current.pt"
            )
            progress[name] = result

        seed_results = {}
        evaluations = {}
        for name in ("A", "B", "C"):
            model, _, saved = load(selected[name], device)
            reports = {}
            for task, profile in evaluation_profiles.items():
                c = task_config(configs[name], profile)
                reports[task] = {}
                for mode, n in (
                    ("deterministic", episodes),
                    ("stochastic", plan["stochastic_episodes"]),
                ):
                    report = cached_evaluate(
                        model,
                        c,
                        selected[name],
                        seed_root / name / "test" / f"{task}_{mode}.json",
                        n,
                        plan["test_seed_offset"] + seed,
                        stochastic=mode == "stochastic",
                    )
                    reports[task][mode] = report
            evaluations[name] = reports
            seed_results[name] = dict(
                selected_checkpoint=str(selected[name]),
                selected_sha256=sha(selected[name]),
                role=(
                    "frozen_baseline"
                    if name == "A"
                    else (
                        "best_feasible"
                        if selected[name].stem == "best_feasible"
                        else "best_current_not_promoted"
                    )
                ),
                selected_origin=saved["origin"],
                local_ppo_updates=progress[name]["ppo_updates"],
                selected_ppo_updates=saved["progress"].get("ppo_updates", 0),
                precision_validation_passed=(
                    name != "A" and selected[name].stem == "best_feasible"
                ),
                primary_test_passed=passed(
                    reports[plan.get("primary_task", next(iter(evaluation_profiles)))][
                        "deterministic"
                    ],
                    task_config(
                        configs[name],
                        evaluation_profiles[
                            plan.get("primary_task", next(iter(evaluation_profiles)))
                        ],
                    ),
                ),
                local_physics_steps=progress[name]["physics_steps"],
                predictor_frozen=True,
                initial_std=configs[name]["policy"]["initial_std"],
                summary={
                    task: {
                        mode: dict(
                            success_rate=r["success_rate"],
                            tail_rmse=r["mean_tail_position_rmse"],
                            full_tail_episodes=r["full_tail_episodes"],
                            episodes=r["episodes"],
                        )
                        for mode, r in modes.items()
                    }
                    for task, modes in reports.items()
                },
            )
        # Isolate noise-only gains from PPO gains in C's stochastic evaluation.
        c_initial, _, _ = load(transfers["C"], device)
        c_initial_reports = {}
        for task, profile in evaluation_profiles.items():
            c = task_config(configs["C"], profile)
            c_initial_reports[task] = cached_evaluate(
                c_initial,
                c,
                transfers["C"],
                seed_root / "C_initial" / f"{task}_stochastic.json",
                plan["stochastic_episodes"],
                plan["test_seed_offset"] + seed,
                stochastic=True,
            )
        for name in ("B", "C"):
            seed_results[name]["vs_frozen_A_deterministic"] = {
                task: compare(
                    evaluations["A"][task]["deterministic"],
                    evaluations[name][task]["deterministic"],
                )
                for task in evaluation_profiles
            }
            reference = (
                evaluations["A"]
                if name == "B"
                else {k: dict(stochastic=v) for k, v in c_initial_reports.items()}
            )
            seed_results[name]["vs_own_initial_stochastic"] = {
                task: compare(
                    reference[task]["stochastic"], evaluations[name][task]["stochastic"]
                )
                for task in evaluation_profiles
            }
        write_json(seed_root / "comparison.json", seed_results)
        print(
            json.dumps(
                dict(
                    event="experiment_seed_complete",
                    seed=seed,
                    variants={
                        name: dict(
                            role=row["role"],
                            ppo_updates=row["local_ppo_updates"],
                            selected_ppo_updates=row["selected_ppo_updates"],
                            primary_test_passed=row["primary_test_passed"],
                        )
                        for name, row in seed_results.items()
                    },
                )
            ),
            flush=True,
        )
        results[str(seed)] = seed_results
        # Preserve completed other seeds when a launcher resumes just one seed.
        summary_path = root / "experiment_summary.json"
        prior = (
            json.loads(summary_path.read_text())["seeds"]
            if summary_path.exists()
            else {}
        )
        prior.update(results)
        write_json(
            summary_path,
            dict(
                schema=SCHEMA,
                seeds=prior,
                source_sha256=source_sha,
                note="Common v700 initialization; final-test scenarios are separate from selection. Failed precision gates are results, not runtime failures.",
            ),
        )
    return results

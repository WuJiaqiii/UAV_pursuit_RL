"""Run the paired survey protocol and write its experiment artifacts."""

import csv
import hashlib
import json
import subprocess

from scripts.route_ppo_plan import (
    ARMS, ROOT, VALIDATION_ROUTES, aggregate, check_frozen_scores, eligible,
    planned_updates, rank, write_json,
)
from scripts.survey_route_profiles import PHASES


def repository_head():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_suite(args, plan, source, engine):
    checkpoint = args.checkpoint.resolve()
    teacher, original = engine.load_teacher(checkpoint, args.device, source)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ev = plan["evaluation"]
    configs = {
        name: engine.route_config(original, name, args.device, original["seed"], plan)
        for name in PHASES
    }
    write_json(out / "plan.json", {
        **plan, "git_commit": repository_head(), "source_fingerprint": source,
        "device": args.device, "routes": PHASES, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "preflight": args.preflight, "final_test": args.final_test,
    })
    states = {}
    rows = []
    selections = []
    status = dict(completed_updates=0, planned_updates=planned_updates(plan, args.preflight),
                  arm=None, seed=None, update=None)

    def progress(phase, **detail):
        status.update(phase=phase, **detail)
        write_json(out / "progress.json", status)
        print(f"[{status['completed_updates']}/{status['planned_updates']} updates] "
              f"{phase} {status['arm'] or ''} seed={status['seed']} update={status['update']}", flush=True)

    def assess(model, routes, seed_list, folder):
        folder.mkdir(parents=True, exist_ok=True)
        summary = {}
        for name in routes:
            reports = []
            for seed in seed_list:
                report = engine.evaluate_policy(model, configs[name], ev["episodes_per_seed"], seed)
                digest = report["initial_state_digest"]
                if states.setdefault((name, seed), digest) != digest:
                    raise RuntimeError(f"Evaluation initial states differ: {name}, seed {seed}")
                write_json(folder / f"{name}_{seed}.json", report)
                reports.append(report)
            summary[name] = aggregate(reports)
        return summary

    def record(arm, seed, update, phase, summary, candidate=None):
        rows.extend(dict(phase=phase, arm=arm, seed=seed, update=update, route=name,
                         eligible=candidate, **result) for name, result in summary.items())
        temporary = out / "summary.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(out / "summary.csv")

    validation = ev["validation_seeds"]
    try:
        progress("baseline")
        baseline = assess(teacher, VALIDATION_ROUTES, validation, out / "baseline" / "validation")
        record("mixed_bc", None, 0, "validation", baseline)
        if not eligible(baseline, baseline, ev):
            raise RuntimeError("Baseline BC did not meet the precision gate on both validation routes")
        if args.preflight:
            progress("completed")
            return

        for seed in plan["seeds"]:
            progress("critic_warmup", seed=seed, arm=None, update=None)
            config, state, optimizer, warmup = engine.warm_critic(
                original, teacher, plan, args.device, seed,
            )
            write_json(out / f"critic_warmup_s{seed}.json", warmup)
            for arm in ARMS:
                directory = out / f"{arm}_s{seed}"
                directory.mkdir()
                best_path, best_update, best_rank = checkpoint, 0, rank(baseline)
                progress("training", arm=arm, seed=seed, update=0)
                with (directory / "metrics.jsonl").open("w", encoding="utf-8") as log:
                    for model, receipt in engine.train_updates(config, state, optimizer, teacher, plan, arm):
                        update = receipt["update"]
                        log.write(json.dumps(receipt, allow_nan=False) + "\n")
                        log.flush()
                        status["completed_updates"] += 1
                        progress("training", update=update)
                        if update % plan["evaluate_every"] and update != plan["updates"]:
                            continue
                        progress("validation")
                        result = assess(model, VALIDATION_ROUTES, validation, directory / f"update_{update:04d}")
                        candidate = eligible(result, baseline, ev)
                        record(arm, seed, update, "validation", result, candidate)
                        if arm == "critic_only":
                            check_frozen_scores(result, baseline)
                        if candidate and rank(result) > best_rank:
                            best_rank = rank(result)
                            best_path, best_update = directory / "best_validation.pt", update
                            engine.save_candidate(best_path, model, config, receipt, arm)
                engine.save_candidate(directory / "final.pt", model, config, receipt, arm)
                selections.append(dict(arm=arm, seed=seed, update=best_update, checkpoint=str(best_path)))
                write_json(out / "selection.json", selections)

        if args.final_test:
            progress("final_test", arm=None, seed=None, update=None)
            test_seeds = ev["test_seeds"]
            reference = assess(teacher, PHASES, test_seeds, out / "baseline" / "test")
            record("mixed_bc", None, 0, "test", reference)
            for selected in selections:
                arm, seed, update = (selected[key] for key in ("arm", "seed", "update"))
                progress("final_test", arm=arm, seed=seed, update=update)
                model = engine.load_candidate(selected["checkpoint"], args.device) if update else teacher
                result = assess(model, PHASES, test_seeds, out / f"{arm}_s{seed}" / "test")
                record(arm, seed, update, "test", result)
    except KeyboardInterrupt:
        progress("interrupted")
        raise
    except Exception as error:
        progress("failed", error=str(error))
        raise
    progress("completed", arm=None, seed=None, update=None)

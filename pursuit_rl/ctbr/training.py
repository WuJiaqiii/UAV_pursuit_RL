"""Optional geometric BC initialization, MC critic warm-up and on-policy PPO."""

import copy
import hashlib
import json
import math
import numbers
import sys
import time
from pathlib import Path
import torch
from pursuit_rl.learning.buffers import TensorRolloutBuffer
from .config import fingerprint, transfer_interface, decision_dt
from .checkpoint import new_model, save, load, write_json, source_fingerprint
from .environment import CTBRBatch
from .control import teacher
from .evaluation import evaluate, score, improved
from .errors import ExperimentGateError
from .selection import passed
from .value_learning import fit_critic, calibrate_critic
from .mixture import training_env, target_positions
from .progress import TrainingProgress


_PROGRESS_ON_TTY = False
from pursuit_rl.learning.prediction import supervised_windows, fit_predictor


def _assert_finite_json(value, path="event"):
    """Fail with the exact nested metric path instead of a generic JSON error."""
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite JSON number at {path}: {number!r}")


def emit(root, event, **fields):
    row = dict(event=event, **fields)
    _assert_finite_json(row)
    serialized = json.dumps(row, allow_nan=False)
    with (root / "metrics.jsonl").open("a") as f:
        f.write(serialized + "\n")
    # Interactive terminals use the progress bar as the live display. Keep
    # machine-readable event lines in metrics.jsonl; redirected/nohup logs
    # retain both JSON events and periodic human-readable progress snapshots.
    if not sys.stdout.isatty() or not _PROGRESS_ON_TTY:
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("diagnostics", "parent")},
                allow_nan=False,
            ),
            flush=True,
        )


@torch.no_grad()
def imitation_data(c, model=None, seed=10000):
    env = CTBRBatch(c, seed=seed)
    obs, priv = env.observe()
    states = []
    labels = []
    tick = 0
    count = 0
    segment_ticks = round(c["training"]["bc_segment_seconds"] / env.decision_dt)
    interactions = 0
    while count < c["training"]["bc_samples"]:
        target = teacher(env)
        if tick % c["training"]["bc_stride"] == 0:
            # Preserve body rates, thrust and prior CTBR actions used by Actor.
            states.append(obs.clone())
            labels.append(target.clone())
            count += env.n
        action = (
            target
            if model is None
            else model.sample_tensor(obs, priv, deterministic=True)[0]
        )
        raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        (obs, priv), _, done, _, _ = env.step(action, raw)
        interactions += env.last_physics_steps
        if bool(done.any()):
            obs, priv = env.reset(done)
        tick += 1
        if tick % segment_ticks == 0:
            obs, priv = env.reset()
    return (
        torch.cat(states)[: c["training"]["bc_samples"]],
        torch.cat(labels)[: c["training"]["bc_samples"]],
        interactions,
    )


def fit_actor(model, data, updates, batch_size):
    """Regress the ACTUALLY EXECUTED deterministic CTBR from full Actor state.

    Frozen predictor features and the autoregressive raw-action mix match the
    policy used by PPO. Normalization is folded into the Actor's first layer
    and removed afterwards without changing its direct-CTBR architecture.
    """
    observations, y = data
    first = model.actor[0]
    # DAgger aggregates many thousands of full trajectory-history samples.
    # Passing every sample to an LSTM predictor in one CUDA forward requires
    # an enormous temporary cuDNN workspace (can exceed 24 GB even when the
    # retained feature tensor itself is small). Precompute *frozen* predictor
    # features in bounded chunks, identical to the PPO update preprocessing.
    feature_batch = max(1, min(int(batch_size), 256))
    with torch.no_grad():
        x = torch.cat([
            model.bound_actor_input(model.actor_features(part)).detach()
            for part in observations.split(feature_batch)
        ], dim=0)
    if x.shape[-1] != first.in_features:
        raise ValueError("Imitation features do not match Actor input")
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(0.01)
    with torch.no_grad():
        original = first.weight.clone()
        first.weight.mul_(scale)
        first.bias.add_(original @ center)
    normalized = (x - center) / scale
    previous_raw = (
        x[:, model.memory_start : model.memory_start + model.action_dim]
        if model.rho else None
    )
    output_scale = y.std(0, unbiased=False).clamp_min(0.02)
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    loss = None
    try:
        for _ in range(updates):
            ids = torch.randint(len(x), (min(batch_size, len(x)),),
                                device=model.device)
            mean = model.actor(normalized[ids])
            if previous_raw is not None:
                mean = model.rho * previous_raw[ids] + (1.0 - model.rho) * mean
            executed = torch.tanh(mean)
            loss = ((executed - y[ids]) / output_scale).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite executed-action imitation loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.actor.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
    finally:
        with torch.no_grad():
            first.weight.div_(scale)
            first.bias.sub_(first.weight @ center)
    return float(loss.detach()) if loss is not None else None


def initialization_validation(model, c, root, progress, label):
    report, _ = evaluate(
        model,
        c,
        c["training"]["initialization_episodes"],
        c["training"]["validation_seed"],
    )
    progress["validation_physics_steps"] += sum(
        round(e["time"] / c["physics"]["dt"]) for e in report["episodes_detail"]
    )
    write_json(root / f"initialization_{label}.json", report)
    emit(
        root,
        "initialization_validation",
        label=label,
        success_rate=report["success_rate"],
        unsafe_entry_rate=report["unsafe_entry_rate"],
        failures=report["failures"],
    )
    return report



def replay_fingerprints(replay):
    """Stable task contracts, including mixture fractions, for safe resume."""
    result = {}
    for key, value in (replay or {}).items():
        if key == "tasks":
            result[key] = [
                dict(profile=fingerprint(item["config"]), fraction=float(item["fraction"]))
                for item in value
            ]
        else:
            result[key] = fingerprint(value) if value is not None else None
    return result


def generalization_contract(distillation=None, bc_profiles=None, validation_profiles=None):
    import hashlib
    teachers = {}
    if distillation is not None:
        if not 0 <= float(distillation.get("coefficient", 0)) <= 10:
            raise ValueError("Invalid frozen-policy distillation coefficient")
        for task, path in sorted(distillation.get("teachers", {}).items()):
            data = Path(path)
            if not data.is_file():
                raise FileNotFoundError(f"Frozen teacher checkpoint missing: {data}")
            teachers[task] = hashlib.sha256(data.read_bytes()).hexdigest()
        if float(distillation.get("coefficient", 0)) > 0 and not teachers:
            raise ValueError("Distillation needs at least one teacher checkpoint")
    return dict(
        distill_coefficient=float((distillation or {}).get("coefficient", 0)),
        teacher_sha256=teachers,
        bc_profiles=[fingerprint(c) for c in (bc_profiles or [])],
        validation_profiles=[fingerprint(c) for c in (validation_profiles or [])],
    )


def evaluate_generalization(model, profiles, episodes, seed, root, label):
    """Hold-out all tasks; select by worst-task performance, not tail only."""
    reports = {}
    steps = 0
    for index, profile in enumerate(profiles):
        task_name = profile["task"]["name"]
        report, _ = evaluate(model, profile, episodes, seed + index * 10007)
        reports[task_name] = dict(
            successes=report["successes"],
            episodes=report["episodes"],
            success_rate=report["success_rate"],
            unsafe_entry_rate=report["unsafe_entry_rate"],
            mean_minimum_distance_m=report["mean_minimum_distance_m"],
            failures=report["failures"],
        )
        steps += sum(round(ep["time"] / profile["physics"]["dt"])
                     for ep in report["episodes_detail"])
    rates = [x["success_rate"] for x in reports.values()]
    unsafe = [x["unsafe_entry_rate"] for x in reports.values()]
    outcome = dict(
        label=label, tasks=reports,
        worst_task_success_rate=min(rates),
        macro_success_rate=sum(rates) / len(rates),
        mean_unsafe_entry_rate=sum(unsafe) / len(unsafe),
        selection_score=[
            min(rates),
            sum(rates) / len(rates),
            -sum(unsafe) / len(unsafe),
        ],
    )
    write_json(root / f"generalization_{label}.json", outcome)
    return outcome, steps


def run_bc_initialization(c, model, root, ckpt, progress, transfer, seed_offset, bc_profiles=None):
    """Run one or more DAgger rounds; BC updates are divided across rounds."""
    tr = c["training"]
    total_updates = tr["bc_updates"]
    rounds = tr.get("bc_rounds", 1)
    if total_updates <= 0:
        return ("transfer_initial" if transfer else "random_initial"), None
    updates_per_round = [
        total_updates // rounds + int(index < total_updates % rounds)
        for index in range(rounds)
    ]
    total_samples = 0
    total_interactions = 0
    aggregated_states = []
    aggregated_labels = []
    last_loss = None
    report = None
    for index, round_updates in enumerate(updates_per_round):
        profiles = bc_profiles if bc_profiles is not None else [c]
        round_states, round_labels, interactions = [], [], 0
        samples_per_profile = max(1, tr["bc_samples"] // len(profiles))
        for profile_index, profile in enumerate(profiles):
            profile_copy = copy.deepcopy(profile)
            profile_copy["training"]["bc_samples"] = (
                tr["bc_samples"] - samples_per_profile * (len(profiles) - 1)
                if profile_index == len(profiles) - 1 else samples_per_profile
            )
            states, labels, used = imitation_data(
                profile_copy,
                model=model if transfer else None,
                seed=c["seed"] + seed_offset + index * 100003
                + profile_index * 131071,
            )
            round_states.append(states)
            round_labels.append(labels)
            interactions += used
        states = torch.cat(round_states)
        labels = torch.cat(round_labels)
        aggregated_states.append(states)
        aggregated_labels.append(labels)
        train_states = torch.cat(aggregated_states)
        train_labels = torch.cat(aggregated_labels)
        last_loss = fit_actor(
            model, (train_states, train_labels), round_updates, tr["batch_size"]
        )
        total_samples += len(states)
        total_interactions += interactions
        progress["bc_updates"] += round_updates
        progress["accepted_bc_updates"] += round_updates
        progress["imitation_physics_steps"] += interactions
        progress["bc_rounds_completed"] += 1
        save(
            ckpt / f"bc_round_{index + 1:02d}.pt",
            model,
            c,
            progress,
            "transfer_bc_initial" if transfer else "bc_initial",
        )
        label = "bc" if index == rounds - 1 else f"bc_round_{index + 1:02d}"
        report = initialization_validation(model, c, root, progress, label)
        emit(
            root,
            "bc_round",
            mode="dagger_actor_states" if transfer else "teacher_rollout_states",
            teacher_method=tr["teacher_method"],
            round=index + 1,
            rounds=rounds,
            loss=last_loss,
            samples=len(states),
            aggregated_training_samples=len(train_states),
            interactions_physics_steps=interactions,
            updates=round_updates,
            initialization_success_rate=report["success_rate"],
            loss_space="standardized_action_mse",
        )
        del states, labels, train_states, train_labels
    origin = "transfer_bc_initial" if transfer else "bc_initial"
    save(ckpt / "bc_initial.pt", model, c, progress, origin)
    emit(
        root,
        "bc",
        mode="dagger_actor_states" if transfer else "teacher_rollout_states",
        teacher_method=tr["teacher_method"],
        rounds=rounds,
        loss=last_loss,
        samples=total_samples,
        interactions_physics_steps=total_interactions,
        updates=total_updates,
        initialization_success_rate=report["success_rate"],
        loss_space="standardized_action_mse",
    )
    return origin, report


def train(c, out, resume=None, init_from=None, replay=None, promotion_check=None,
          run_manifest=None, bc_profiles=None, distillation=None,
          validation_profiles=None, show_progress=True):
    global _PROGRESS_ON_TTY
    _PROGRESS_ON_TTY = bool(
        show_progress and sys.stdout.isatty() and sys.stderr.isatty()
    )
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    if resume and run_manifest is not None:
        raise ValueError("A resumed experiment must preserve its existing run manifest")
    root = Path(out).resolve()
    tr = c["training"]
    contract = generalization_contract(distillation, bc_profiles, validation_profiles)
    if validation_profiles:
        from .config import interface
        for profile in validation_profiles:
            if interface(profile) != interface(c):
                raise ValueError("Validation task changes the policy interface")
    if bc_profiles:
        from .config import interface
        for profile in bc_profiles:
            if interface(profile) != interface(c):
                raise ValueError("DAgger task changes the policy interface")
    if resume and Path(resume).resolve().parent.parent != root:
        raise ValueError("Resume inside the checkpoint original run directory")
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError("New run requires empty output")
    # The trainer owns creation of its run directory. In particular, the
    # experiment launcher must not write run_manifest.json *before* this
    # strict emptiness check, or every fresh experiment fails immediately.
    root.mkdir(parents=True, exist_ok=True)
    if not resume:
        if run_manifest is not None:
            write_json(root / "run_manifest.json", run_manifest)
        write_json(root / "config.json", c)
    ckpt = root / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    torch.manual_seed(c["seed"])
    progress = dict(
        physics_steps=0,
        decision_transitions=0,
        ppo_updates=0,
        bc_updates=0,
        accepted_bc_updates=0,
        imitation_physics_steps=0,
        mc_physics_steps=0,
        validation_physics_steps=0,
        completed_episodes=0,
        successful_episodes=0,
        parent_physics_steps=0,
        cumulative_ppo_physics_steps=0,
        parent=None,
        replay_fingerprints=replay_fingerprints(replay),
        generalization_contract=contract,
        prediction_updates=0,
        mc_fit_updates=0,
        mc_calibration_rounds=0,
        last_mc_calibration_update=0,
        promotion_passed=False,
        best_score=[-1.0, -1.0, -1e9],
        best_origin=None,
        initial_score=None,
        selected_improved_over_initial=False,
        best_feasible_score=None,
        best_feasible_origin=None,
        best_feasible_improved_over_initial=False,
        critic_needs_calibration=False,
        bc_rounds_completed=0,
    )
    if resume:
        model, old, saved = load(resume, c["device"], optimizers=True)
        comparison = copy.deepcopy(c)
        comparison["training"]["physics_steps"] = old["training"]["physics_steps"]
        if fingerprint(comparison) != fingerprint(old):
            raise ValueError("Resume allows only total budget/device changes")
        if saved["source_fingerprint"] != source_fingerprint():
            raise ValueError(
                "Resume source code changed; start a new run with explicit init-from"
            )
        if (
            tr["physics_steps"] < saved["progress"]["physics_steps"]
            and tr["physics_steps"] != old["training"]["physics_steps"]
        ):
            raise ValueError("Resume budget below completed work")
        expected_replay = progress["replay_fingerprints"]
        progress.update(saved["progress"])
        if progress["replay_fingerprints"] != expected_replay:
            raise ValueError(
                "Resume task mixture changed; use the same curriculum launcher"
            )
        if progress.get("generalization_contract") != contract:
            raise ValueError("Resume generalization/DAgger/distillation settings changed")
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and str(model.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if (
            tr["physics_steps"] <= progress["physics_steps"]
            or (promotion_check is not None and progress["promotion_passed"])
        ) and (root / "training_summary.json").exists():
            return json.loads((root / "training_summary.json").read_text())
        emit(
            root,
            "resume",
            physics_steps=progress["physics_steps"],
            note="Unfinished environments restart; completed updates/optimizers retained",
        )
        if promotion_check is None:
            progress["promotion_passed"] = False
    else:
        model = new_model(c)
        if init_from:
            parent, old, saved = load(init_from, c["device"])
            if transfer_interface(old) != transfer_interface(c):
                raise ValueError(
                    "Warm start must preserve action, observations, dynamics, predictor and Critic architecture"
                )
            transferred = model.load_state_dict(parent.state_dict(), strict=False)
            expected_missing = (
                {"lagrange_multiplier"} |
                {key for key in model.state_dict() if key.startswith("cost_critic.")}
            ) if c["safety"]["constraint_enabled"] else set()
            if set(transferred.missing_keys) != expected_missing or transferred.unexpected_keys:
                raise ValueError(
                    "Unexpected model mismatch during V11 warm start: "
                    f"{transferred}"
                )
            model.critic_policy_match = False
            progress["critic_needs_calibration"] = True
            progress["parent_physics_steps"] = saved["progress"]["cumulative_ppo_physics_steps"]
            progress["parent"] = dict(
                path=str(Path(init_from).resolve()),
                sha256=hashlib.sha256(Path(init_from).read_bytes()).hexdigest(),
                origin=saved["origin"],
                ppo_updates=saved["progress"]["ppo_updates"],
                critic="warm weights, fresh optimizers + new MC calibration",
            )
            origin = "transfer_initial"
            del parent
            if tr["bc_updates"]:
                origin, bc_report = run_bc_initialization(
                    c, model, root, ckpt, progress, transfer=True,
                    seed_offset=20000, bc_profiles=bc_profiles
                )
                if bc_report["success_rate"] < tr["initialization_success_floor"]:
                    details = dict(
                        event="initialization_gate_failed",
                        phase=c["task"]["name"],
                        source="transfer_bc",
                        success_rate=bc_report["success_rate"],
                        minimum_success=tr["initialization_success_floor"],
                        failures=bc_report["failures"],
                        report=str(root / "initialization_bc.json"),
                        checkpoint=str(ckpt / "bc_initial.pt"),
                        message="Transferred Actor failed current-task BC validation; PPO was not started.",
                    )
                    write_json(root / "gate_failure.json", details)
                    raise ExperimentGateError(details)
        else:
            origin = "random_initial"
            # The teacher uses only current kinematics (first 21 features).
            with torch.no_grad():
                model.actor[0].weight[:, 21:] = 0
                if tr.get("zero_initial_actor_mean", False):
                    model.actor[-1].weight.zero_()
                    model.actor[-1].bias.zero_()
            if tr["bc_updates"]:
                origin, bc_report = run_bc_initialization(
                    c, model, root, ckpt, progress, transfer=False,
                    seed_offset=10000, bc_profiles=bc_profiles
                )
                if bc_report["success_rate"] < tr["initialization_success_floor"]:
                    details = dict(
                        event="initialization_gate_failed",
                        phase=c["task"]["name"],
                        source="fresh_bc",
                        success_rate=bc_report["success_rate"],
                        minimum_success=tr["initialization_success_floor"],
                        failures=bc_report["failures"],
                        report=str(root / "initialization_bc.json"),
                        checkpoint=str(ckpt / "bc_initial.pt"),
                        message="Teacher BC initialization failed validation; PPO was not started.",
                    )
                    write_json(root / "gate_failure.json", details)
                    raise ExperimentGateError(details)
        if tr["skip_mc_diagnostics"]:
            emit(root, "critic_warmup_skipped", reason="pure_ppo", critic_updates=0)
        else:
            if validation_profiles:
                # Match the initial MC critic regression to the *joint* PPO task
                # distribution, rather than fitting only on the tail task before
                # starting a four-task rollout.
                gathered = []
                initial_calibrations = {}
                per_task = max(2, math.ceil(tr["mc_episodes"] / len(validation_profiles)))
                for profile_idx, profile in enumerate(validation_profiles):
                    report, pair = evaluate(
                        model, profile, per_task,
                        810000 + c["seed"] + 10007 * profile_idx,
                        stochastic=True, collect_mc=True,
                    )
                    name = profile["task"]["name"]
                    initial_calibrations[name] = dict(
                        critic_mc_rmse=report["critic_mc_rmse"],
                        success_rate=report["success_rate"],
                    )
                    if name == c["task"]["name"]:
                        calibration = report
                    gathered.append(pair)
                    progress["mc_physics_steps"] += sum(
                        round(ep["time"] / c["physics"]["dt"])
                        for ep in report["episodes_detail"]
                    )
                data = (
                    torch.cat([item[0] for item in gathered]),
                    torch.cat([item[1] for item in gathered]),
                )
                emit(root, "joint_critic_warmup_samples",
                     episodes_per_task=per_task, tasks=initial_calibrations)
            else:
                calibration, data = evaluate(
                    model, c, tr["mc_episodes"], 810000 + c["seed"],
                    stochastic=True, collect_mc=True,
                )
                progress["mc_physics_steps"] += sum(
                    round(e["time"] / c["physics"]["dt"])
                    for e in calibration["episodes_detail"]
                )
            fit = fit_critic(model, data, tr["critic_updates"], tr["batch_size"])
            if tr["critic_updates"] > 0:
                model.critic_policy_match = True
                progress["critic_needs_calibration"] = False
            del data
            heldout, _ = evaluate(
                model, c, tr["mc_episodes"], 910000 + c["seed"], stochastic=True
            )
            write_json(root / "critic_warmup_heldout.json", heldout)
            progress["mc_physics_steps"] += sum(
                round(e["time"] / c["physics"]["dt"]) for e in heldout["episodes_detail"]
            )
            emit(
                root,
                "critic_warmup",
                fit=fit,
                mc_rmse_before_fit=calibration["critic_mc_rmse"],
                heldout_rmse=heldout["critic_mc_rmse"],
                heldout_bias=heldout["critic_bias"],
            )
        initial, _ = evaluate(
            model, c, tr["validation_episodes"], tr["validation_seed"]
        )
        write_json(root / "initial_validation.json", initial)
        progress["validation_physics_steps"] += sum(
            round(e["time"] / c["physics"]["dt"]) for e in initial["episodes_detail"]
        )
        if validation_profiles:
            joint, extra_steps = evaluate_generalization(
                model, validation_profiles, tr["validation_episodes"],
                tr["validation_seed"], root, "initial",
            )
            progress["validation_physics_steps"] += extra_steps
            initial_score = joint["selection_score"]
        else:
            initial_score = list(score(initial, mode=tr["checkpoint_selection"]))
        progress["best_score"] = list(initial_score)
        progress["initial_score"] = list(initial_score)
        progress["best_origin"] = origin
        progress["cumulative_ppo_physics_steps"] = progress["parent_physics_steps"]
        feasible = (
            joint["worst_task_success_rate"] >= tr["minimum_success"]
            and joint["mean_unsafe_entry_rate"] == 0
            if validation_profiles else passed(initial, c)
        )
        if feasible and promotion_check is not None:
            feasible = promotion_check(model, c, progress)
        if feasible:
            progress["best_feasible_score"] = list(initial_score)
            progress["best_feasible_origin"] = origin
            progress["promotion_passed"] = promotion_check is not None
            save(ckpt / "best_feasible.pt", model, c, progress, origin)
        for name in ("initial", "best", "best_current", "latest"):
            save(ckpt / f"{name}.pt", model, c, progress, origin)
        emit(
            root,
            "initial_validation",
            origin=origin,
            success_rate=initial["success_rate"],
            unsafe_entry_rate=initial["unsafe_entry_rate"],
        )
    write_json(root / "config.json", c)
    frozen_teachers = {}
    distill_coef = float((distillation or {}).get("coefficient", 0))
    aux_pd_coef = float(tr["aux_pd_coef"])
    if distill_coef > 0:
        for task, path in distillation["teachers"].items():
            previous, previous_config, _ = load(path, c["device"])
            if transfer_interface(previous_config) != transfer_interface(c):
                raise ValueError(f"Frozen teacher {task} has incompatible architecture")
            previous.eval().requires_grad_(False)
            frozen_teachers[task] = previous
    env = training_env(c, replay, seed=c["seed"] + 7919 * progress["ppo_updates"])
    safety_active = model.cost_critic is not None
    cost_gamma = math.exp(-env.decision_dt / c["safety"]["discount_time_s"])
    cost_lam = math.exp(-env.decision_dt / c["safety"]["advantage_time_s"]) / cost_gamma
    curriculum_groups = env.groups if hasattr(env, "groups") else [env]
    if distill_coef > 0:
        missing = set(frozen_teachers) - set(env.task_slices)
        if missing:
            raise ValueError(f"Frozen policy task(s) absent from rollout: {sorted(missing)}")
    emit(
        root,
        "training_distribution",
        task_environment_counts=env.task_counts,
        physics_hz=1 / env.dt,
        decision_hz=1 / env.decision_dt,
    )
    obs, priv = env.observe()
    episode_start = torch.ones(env.n, dtype=torch.bool, device=env.device)
    next_validation = (progress["physics_steps"] // tr["validation_every"] + 1) * tr[
        "validation_every"
    ]
    training_wall_start = time.perf_counter()
    progress_display = TrainingProgress(
        total_steps=tr["physics_steps"],
        initial_steps=progress["physics_steps"],
        initial_updates=progress["ppo_updates"],
        completed_episodes=progress["completed_episodes"],
        successful_episodes=progress["successful_episodes"],
        enabled=show_progress,
    )
    progress_display.update(
        progress["physics_steps"], progress["ppo_updates"],
        progress["completed_episodes"], progress["successful_episodes"],
        phase="starting", force=True,
    )
    while (
        progress["physics_steps"] < tr["physics_steps"]
        and not progress["promotion_passed"]
    ):
        update_wall_start = time.perf_counter()
        # The curriculum changes only the distribution of NEW on-policy
        # resets. Validation and heldout profiles are never modified.
        curriculum = c["safety"]
        progress_fraction = progress["physics_steps"] / max(tr["physics_steps"], 1)
        critical_prob = curriculum["critical_reset_fraction"] * min(
            1.0, progress_fraction / curriculum["curriculum_ramp_fraction"]
        )
        for subenv in curriculum_groups:
            subenv.critical_reset_fraction = critical_prob
        ticks = min(
            tr["rollout_steps"],
            math.ceil(
                (tr["physics_steps"] - progress["physics_steps"])
                / (env.n * env.decision_steps)
            ),
        )
        buf = TensorRolloutBuffer(
            ticks,
            env.n,
            env.actor_dim,
            env.critic_dim,
            env.device,
            action_dim=4,
            diagnostics={
                "value_abs_threshold": 100.0,
                "target_abs_threshold": 100.0,
                "advantage_abs_threshold": 100.0,
            },
        )
        if distill_coef > 0:
            buf.retention_mask = torch.zeros(
                (ticks, env.n), device=env.device, dtype=torch.bool,
            )
            buf.retention_targets = torch.zeros(
                (ticks, env.n, 4), device=env.device,
            )
        buf.physics = copy.deepcopy(c["physics"])
        buf.action_semantics = "direct-ctbr-first-entry-v1"
        buf.decision_dt = env.decision_dt
        records = []
        rollout_physics_steps = 0
        component_scaled_sums = {}
        component_unscaled_sums = {}
        component_slots = 0
        reward_component_residual_max = torch.zeros((), device=env.device)
        reward_scale_residual_max = torch.zeros((), device=env.device)
        prediction_positions = (
            torch.empty(ticks, env.n, 3, device=env.device)
            if model.predictor_optimizer is not None and tr["prediction_updates_enabled"]
            else None
        )
        for tick in range(ticks):
            try:
                if aux_pd_coef > 0:
                    # Teacher supervises sampled states only; no runtime mixing.
                    buf.teacher[tick] = (
                        torch.cat([teacher(part) for part in env.groups], 0)
                        if hasattr(env, "groups") else teacher(env)
                    )
                action, raw, lp, value_pred = model.sample_tensor(obs, priv)
                if distill_coef > 0:
                    with torch.no_grad():
                        for task_name, teacher_model in frozen_teachers.items():
                            slot = env.task_slices[task_name]
                            buf.retention_targets[tick, slot] = (
                                teacher_model.predict_tensor(obs[slot])
                            )
                            buf.retention_mask[tick, slot] = True
                (nobs, npriv), reward, term, trunc, completed = env.step(action, raw)
                if prediction_positions is not None:
                    prediction_positions[tick] = target_positions(env).float()
                rollout_physics_steps += env.last_physics_steps
                component_slots += env.n
                scaled_total = torch.zeros_like(reward)
                unscaled_total = torch.zeros_like(reward)
                for component in env.reward_components.values():
                    scaled_total += component
                for component in env.reward_components_unscaled.values():
                    unscaled_total += component
                reward_component_residual_max = torch.maximum(
                    reward_component_residual_max, (scaled_total - reward).abs().max()
                )
                reward_scale_residual_max = torch.maximum(
                    reward_scale_residual_max,
                    (scaled_total - unscaled_total * c["reward"]["scale"]).abs().max(),
                )
                for key, component in env.reward_components.items():
                    component_scaled_sums[key] = (
                        component_scaled_sums.get(
                            key, torch.zeros((), device=env.device)
                        )
                        + component.detach().sum()
                    )
                for key, component in env.reward_components_unscaled.items():
                    component_unscaled_sums[key] = (
                        component_unscaled_sums.get(
                            key, torch.zeros((), device=env.device)
                        )
                        + component.detach().sum()
                    )
            except Exception as exc:
                if not (root / "first_anomaly.pt").exists():
                    snapshot = buf.diagnostic_snapshot()
                    snapshot["tensors"] = {
                        k: v[:tick] for k, v in snapshot["tensors"].items()
                    }
                    torch.save(
                        dict(
                            phase="collection",
                            error=repr(exc),
                            config=c,
                            progress=progress,
                            model=model.state_dict(),
                            obs=obs.cpu(),
                            priv=priv.cpu(),
                            buffer=snapshot,
                        ),
                        root / "first_anomaly.pt",
                    )
                raise
            for name, data in [
                ("obs", obs),
                ("priv", priv),
                ("raw", raw),
                ("logprob", lp),
                ("values", value_pred),
                ("reward", reward),
                ("terminated", term),
                ("truncated", trunc),
                ("next_values", model.values_tensor(npriv)),
                ("episode_start", episode_start),
                ("elapsed", env.steps.float() * env.dt),
            ]:
                getattr(buf, name)[tick] = data
            if safety_active:
                buf.cost[tick] = (
                    torch.cat([part.last_safety_cost for part in env.groups])
                    if hasattr(env, "groups") else env.last_safety_cost
                )
                buf.cost_values[tick] = model.cost_values_tensor(priv)
                buf.cost_next_values[tick] = model.cost_values_tensor(npriv)
            buf.actuator_state[tick, :, 0] = env.plant.pursuer.thrust
            buf.actuator_state[tick, :, 1:] = env.plant.pursuer.rate
            for key, data in env.reward_components.items():
                if key not in buf.reward_components:
                    buf.reward_components[key] = torch.zeros_like(buf.reward)
                buf.reward_components[key][tick] = data
            for rec in completed:
                buf.event_code[tick, rec["slot"]] = {
                    "success": 1,
                    "speed": 2,
                    "outside": 3,
                    "tilt": 4,
                    "invalid_target": 6,
                    "timeout": 7,
                    "unsafe_entry": 8,
                }[rec["reason"]]
            buf.diagnostic_steps = tick + 1
            records.extend(completed)
            progress_display.update(
                progress["physics_steps"] + rollout_physics_steps,
                progress["ppo_updates"],
                progress["completed_episodes"] + len(records),
                progress["successful_episodes"] + sum(
                    record["success"] for record in records
                ),
                phase="rollout",
            )
            episode_start = term | trunc
            obs, priv = env.reset(episode_start) if completed else (nobs, npriv)
        rollout_wall_s = time.perf_counter() - update_wall_start
        buf.compute_gae(env.gamma, env.gae_lambda)
        if safety_active:
            buf.compute_cost_gae(cost_gamma, cost_lam)
        before = dict(
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            actor_optimizer=copy.deepcopy(model.actor_optimizer.state_dict()),
            critic_optimizer=copy.deepcopy(model.critic_optimizer.state_dict()),
            cost_optimizer=(
                copy.deepcopy(model.cost_optimizer.state_dict())
                if model.cost_optimizer else None
            ),
            torch_rng=torch.get_rng_state(),
            progress=copy.deepcopy(progress),
        )
        progress_display.update(
            progress["physics_steps"] + rollout_physics_steps,
            progress["ppo_updates"],
            progress["completed_episodes"] + len(records),
            progress["successful_episodes"] + sum(
                record["success"] for record in records
            ),
            phase="optimizer",
            force=True,
        )
        try:
            stats = model.update(buf, aux_pd_coef=aux_pd_coef, distill_coef=distill_coef)
            optimizer_wall_s = time.perf_counter() - update_wall_start - rollout_wall_s
        except Exception as exc:
            if not (root / "first_anomaly.pt").exists():
                torch.save(
                    dict(
                        before=before,
                        buffer=buf.diagnostic_snapshot(),
                        error=repr(exc),
                        config=c,
                    ),
                    root / "first_anomaly.pt",
                )
            raise
        progress["physics_steps"] += rollout_physics_steps
        progress["decision_transitions"] += ticks * env.n
        progress["ppo_updates"] += 1
        progress["cumulative_ppo_physics_steps"] = (
            progress["parent_physics_steps"] + progress["physics_steps"]
        )
        progress["completed_episodes"] += len(records)
        progress["successful_episodes"] += sum(r["success"] for r in records)
        if model.predictor_optimizer is not None and tr["prediction_updates_enabled"]:
            history = buf.obs[:, :, env.base_actor_dim :].reshape(
                ticks, env.n, c["prediction"]["history_steps"], 10
            )
            data = supervised_windows(
                history,
                prediction_positions,
                buf.terminated | buf.truncated,
                [round(h / env.decision_dt) for h in c["prediction"]["horizons_s"]],
                c["prediction"]["max_samples"],
            )
            prediction_fit = fit_predictor(model, data)
            progress["prediction_updates"] += prediction_fit["updates"]
            emit(
                root,
                "prediction_fit",
                ppo_update=progress["ppo_updates"],
                **prediction_fit,
            )
            del data
        diagnostics = stats.get("diagnostics")
        if (
            diagnostics
            and diagnostics.get("anomaly_reasons")
            and not (root / "first_anomaly.pt").exists()
        ):
            torch.save(
                dict(
                    before=before,
                    buffer=buf.diagnostic_snapshot(),
                    diagnostics=diagnostics,
                    config=c,
                ),
                root / "first_anomaly.pt",
            )
        emit(
            root,
            "ppo_update",
            physics_steps=progress["physics_steps"],
            target_physics_steps=tr["physics_steps"],
            progress_fraction=min(
                progress["physics_steps"] / max(tr["physics_steps"], 1), 1.0
            ),
            ppo_updates=progress["ppo_updates"],
            completed_episodes_total=progress["completed_episodes"],
            successful_episodes_total=progress["successful_episodes"],
            success_rate_total=(
                progress["successful_episodes"] / progress["completed_episodes"]
                if progress["completed_episodes"] else None
            ),
            nominal_seconds_per_env=progress["physics_steps"] * env.dt / env.n,
            completed_episodes=len(records),
            successes=sum(r["success"] for r in records),
            completed_success_rate=(
                sum(r["success"] for r in records) / len(records) if records else None
            ),
            completed_tilt_failure_rate=(
                sum(r["reason"] == "tilt" for r in records) / len(records)
                if records else None
            ),
            curriculum_critical_reset_fraction=critical_prob,
            predicted_tilt_peak_mean_rad=(
                float(torch.stack([
                    part.last_predicted_peak.mean() for part in curriculum_groups
                ]).mean().item())
                if c["safety"]["predictive_enabled"] else None
            ),
            metrics={k: v for k, v in stats.items() if k != "diagnostics"},
            diagnostics=diagnostics,
            task_results={
                name: dict(
                    episodes=sum(r["task"] == name for r in records),
                    successes=sum(r["task"] == name and r["success"] for r in records),
                )
                for name in env.task_counts
            },
            rollout_wall_s=rollout_wall_s,
            optimizer_wall_s=optimizer_wall_s,
            training_wall_s=time.perf_counter() - training_wall_start,
            physics_steps_per_rollout_second=(
                rollout_physics_steps / max(rollout_wall_s, 1e-9)
            ),
            reward_scale=c["reward"]["scale"],
            reward_component_residual_max=float(reward_component_residual_max.item()),
            reward_scale_residual_max=float(reward_scale_residual_max.item()),
            reward_components={
                name: dict(
                    scaled_sum=float(component_sum.item()),
                    scaled_mean_per_allocated_transition=(
                        float(component_sum.item()) / max(component_slots, 1)
                    ),
                    unscaled_sum=float(component_unscaled_sums[name].item()),
                    unscaled_mean_per_allocated_transition=(
                        float(component_unscaled_sums[name].item())
                        / max(component_slots, 1)
                    ),
                )
                for name, component_sum in component_scaled_sums.items()
            },
        )
        progress_display.update(
            progress["physics_steps"], progress["ppo_updates"],
            progress["completed_episodes"], progress["successful_episodes"],
            phase="training",
        )
        interval = tr["mc_calibration_every_updates"]
        if interval and (
            progress["ppo_updates"] - progress["last_mc_calibration_update"] >= interval
            or progress["physics_steps"] >= tr["physics_steps"]
        ):
            correction = calibrate_critic(model, c, root, progress)
            emit(
                root,
                "critic_correction",
                ppo_update=progress["ppo_updates"],
                accepted=correction["accepted"],
                before=correction["before"],
                after=correction["final"],
                fit=correction["fit"],
            )
        validated_this_update = (
            progress["physics_steps"] >= next_validation
            or progress["physics_steps"] >= tr["physics_steps"]
        )
        if validated_this_update:
            progress_display.update(
                progress["physics_steps"], progress["ppo_updates"],
                progress["completed_episodes"], progress["successful_episodes"],
                phase="validation", force=True,
            )
            report, _ = evaluate(
                model, c, tr["validation_episodes"], tr["validation_seed"]
            )
            write_json(root / f"validation_{progress['physics_steps']}.json", report)
            progress["validation_physics_steps"] += sum(
                round(e["time"] / env.dt) for e in report["episodes_detail"]
            )
            if tr["skip_mc_diagnostics"]:
                emit(root, "validation", success_rate=report["success_rate"],
                     unsafe_entry_rate=report["unsafe_entry_rate"],
                     stochastic_mc_rmse=None, stochastic_bias=None)
            else:
                cal, _ = evaluate(
                    model,
                    c,
                    tr["mc_episodes"],
                    810000 + c["seed"] + progress["ppo_updates"],
                    stochastic=True,
                )
                write_json(root / f"calibration_{progress['physics_steps']}.json", cal)
                progress["mc_physics_steps"] += sum(
                    round(e["time"] / env.dt) for e in cal["episodes_detail"]
                )
                emit(
                    root,
                    "validation",
                    success_rate=report["success_rate"],
                    unsafe_entry_rate=report["unsafe_entry_rate"],
                    stochastic_mc_rmse=cal["critic_mc_rmse"],
                    stochastic_bias=cal["critic_bias"],
                )
            if validation_profiles:
                joint, extra_steps = evaluate_generalization(
                    model, validation_profiles, tr["validation_episodes"],
                    tr["validation_seed"], root, str(progress["physics_steps"]),
                )
                progress["validation_physics_steps"] += extra_steps
                candidate_score = joint["selection_score"]
                candidate_feasible = (
                    joint["worst_task_success_rate"] >= tr["minimum_success"]
                    and joint["mean_unsafe_entry_rate"] == 0
                )
                emit(
                    root, "generalization_validation",
                    physics_steps=progress["physics_steps"],
                    **{k: v for k, v in joint.items() if k != "tasks"},
                    task_results=joint["tasks"],
                )
            else:
                candidate_score = list(score(report, mode=tr["checkpoint_selection"]))
                candidate_feasible = passed(report, c)
            if (
                tuple(candidate_score) > tuple(progress["best_score"])
                if validation_profiles
                else improved(
                    report, progress["best_score"],
                    mode=tr["checkpoint_selection"],
                )
            ):
                progress["best_score"] = list(candidate_score)
                progress["best_origin"] = "ppo"
                progress["selected_improved_over_initial"] = (
                    tuple(candidate_score) > tuple(progress["initial_score"])
                    if validation_profiles
                    else improved(
                        report, progress["initial_score"],
                        mode=tr["checkpoint_selection"],
                    )
                )
                save(ckpt / "best.pt", model, c, progress, "ppo")
                save(ckpt / "best_current.pt", model, c, progress, "ppo")
            feasible = candidate_feasible
            if feasible and promotion_check is not None:
                feasible = promotion_check(model, c, progress)
            if feasible:
                progress["promotion_passed"] = promotion_check is not None
                if progress["best_feasible_score"] is None or (
                    tuple(candidate_score) > tuple(progress["best_feasible_score"])
                    if validation_profiles
                    else improved(
                        report, progress["best_feasible_score"],
                        mode=tr["checkpoint_selection"],
                    )
                ):
                    progress["best_feasible_score"] = list(candidate_score)
                    progress["best_feasible_origin"] = "ppo"
                    progress["best_feasible_improved_over_initial"] = (
                        tuple(candidate_score) > tuple(progress["initial_score"])
                        if validation_profiles
                        else improved(
                        report, progress["initial_score"],
                        mode=tr["checkpoint_selection"],
                    )
                    )
                    save(ckpt / "best_feasible.pt", model, c, progress, "ppo")
            next_validation = (
                progress["physics_steps"] // tr["validation_every"] + 1
            ) * tr["validation_every"]
        progress_display.update(
            progress["physics_steps"], progress["ppo_updates"],
            progress["completed_episodes"], progress["successful_episodes"],
            phase="training",
        )
        if (
            progress["ppo_updates"] % tr["checkpoint_every_updates"] == 0
            or progress["physics_steps"] >= tr["physics_steps"]
            or validated_this_update
        ):
            save(ckpt / "latest.pt", model, c, progress, "ppo")
        if progress["ppo_updates"] % tr["save_every_updates"] == 0:
            save(
                ckpt / f"update_{progress['ppo_updates']:05d}.pt",
                model,
                c,
                progress,
                "ppo",
            )
        if progress["promotion_passed"]:
            break
    progress_display.finish(
        status=(
            "completed" if progress["physics_steps"] >= tr["physics_steps"]
            else "early-stop" if progress["promotion_passed"]
            else "stopped"
        )
    )
    result = dict(
        schema="pursuit-first-entry-training-v1",
        **progress,
        total_local_interactions=sum(
            progress[k]
            for k in (
                "physics_steps",
                "imitation_physics_steps",
                "mc_physics_steps",
                "validation_physics_steps",
            )
        ),
        unfinished_episodes=int((env.steps > 0).sum()),
        maximum_unfinished_elapsed_s=float(env.steps.max()) * env.dt,
        source_fingerprint=source_fingerprint(),
        training_wall_s=time.perf_counter() - training_wall_start,
        physics_hz=1 / env.dt,
        decision_hz=1 / env.decision_dt,
        task_environment_counts=env.task_counts,
        distillation_coefficient=distill_coef,
        generalization_profiles=(
            [p["task"]["name"] for p in validation_profiles]
            if validation_profiles else None
        ),
        stop_reason=(
            "curriculum_gate_passed"
            if progress["promotion_passed"]
            else "budget_reached"
        ),
        note="Initial/best/latest must be compared; completed training episodes exclude unfinished trajectories.",
    )
    write_json(root / "training_summary.json", result)
    return result

"""Raw-unit critic diagnostics and fresh-policy complete-return corrections."""

import copy
import torch
import torch.nn.functional as F
from .evaluation import evaluate
from .checkpoint import write_json


@torch.no_grad()
def value_errors(model, data, batch_size=2048):
    x, y = data
    prediction = torch.cat(
        [model.values_tensor(batch) for batch in x.split(batch_size)]
    )
    error = prediction - y
    starts = x[:, 31] >= x[:, 31].max() - 1e-8
    return dict(
        samples=len(y),
        rmse=float(error.square().mean().sqrt()),
        bias=float(error.mean()),
        initial_rmse=float(error[starts].square().mean().sqrt()),
        initial_bias=float(error[starts].mean()),
        units="scaled_environment_reward",
    )


def fit_critic(model, data, updates, batch_size):
    x, y = data
    before = value_errors(model, data, batch_size)["rmse"]
    model.update_value_statistics(y)
    target = model.value_norm.normalize(y)
    # Predictor is frozen, and this bounded preprocessing is reused during fit.
    features = torch.cat(
        [model.critic_features(batch) for batch in x.split(batch_size)]
    )
    starts = (x[:, 31] >= x[:, 31].max() - 1e-8).nonzero(as_tuple=True)[0]
    size = min(batch_size, len(x))
    n_start = max(1, size // 8)
    for _ in range(updates):
        ids = torch.cat(
            (
                torch.randint(len(x), (size - n_start,), device=model.device),
                starts[torch.randint(len(starts), (n_start,), device=model.device)],
            )
        )
        loss = F.huber_loss(
            model.critic(features[ids]).squeeze(-1),
            target[ids],
            delta=model.config["critic_huber_delta"],
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite critic fit")
        model.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.critic.parameters(), 0.5, error_if_nonfinite=True
        )
        model.critic_optimizer.step()
    after = value_errors(model, data, batch_size)["rmse"]
    return dict(
        samples=len(y),
        updates=updates,
        fit_rmse_before=before,
        fit_rmse_after=after,
        initial_state_samples=len(starts),
        initial_state_batch_fraction=n_start / size,
    )


def calibrate_critic(model, c, root, progress):
    """Only critic/normalizer change; both MC datasets use the same frozen actor.

    Hold-out returns are generated BEFORE fitting. Before/after errors therefore
    refer to identical states and stochastic outcomes, rather than two noisy
    evaluations. Hold-out data select correction acceptance, never fit weights.
    """
    tr = c["training"]
    update = progress["ppo_updates"]
    seed = 1810000 + c["seed"] + 1009 * update
    training, data = evaluate(
        model, c, tr["mc_episodes"], seed, stochastic=True, collect_mc=True
    )
    heldout, test_data = evaluate(
        model, c, tr["mc_episodes"], seed + 1000000, stochastic=True, collect_mc=True
    )
    progress["mc_physics_steps"] += sum(
        round(e["time"] / c["physics"]["dt"])
        for report in (training, heldout)
        for e in report["episodes_detail"]
    )
    before = value_errors(model, test_data, tr["batch_size"])
    backup = dict(
        critic=copy.deepcopy(model.critic.state_dict()),
        normalizer=copy.deepcopy(model.value_norm.state_dict()),
        optimizer=copy.deepcopy(model.critic_optimizer.state_dict()),
    )
    fit = fit_critic(model, data, tr["mc_calibration_updates"], tr["batch_size"])
    after = value_errors(model, test_data, tr["batch_size"])
    tolerance = tr["mc_max_regression_fraction"]
    accepted = after["rmse"] <= before["rmse"] * (1 + tolerance) + 1e-6
    # Also protect the rare episode-start population explicitly.
    accepted &= after["initial_rmse"] <= before["initial_rmse"] * (1 + tolerance) + 0.02
    if not accepted:
        model.critic.load_state_dict(backup["critic"])
        model.value_norm.load_state_dict(backup["normalizer"])
        model.critic_optimizer.load_state_dict(backup["optimizer"])
    elif tr["mc_calibration_updates"] > 0:
        model.critic_policy_match = True
        progress["critic_needs_calibration"] = False
    progress["mc_fit_updates"] += tr["mc_calibration_updates"]
    progress["mc_calibration_rounds"] += 1
    progress["last_mc_calibration_update"] = update
    report = dict(
        schema="pursuit-critic-correction-v1",
        ppo_update=update,
        physics_steps=progress["physics_steps"],
        train_seed=seed,
        heldout_seed=seed + 1000000,
        before=before,
        candidate_after=after,
        accepted=bool(accepted),
        final=after if accepted else before,
        fit=fit,
        actor_updated=False,
        predictor_updated=False,
        holdout_use="correction acceptance validation, not final test",
        initial_training_report=training,
        initial_heldout_report=heldout,
    )
    write_json(root / f"critic_correction_{update:05d}.json", report)
    return report

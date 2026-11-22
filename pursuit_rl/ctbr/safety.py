"""V11 action-conditioned attitude risk under the *actual CTBR rate actuator*.

Not a CBF proof or flight-safety guarantee. First execute the candidate command
for a decision interval, then apply a bounded level-attitude recovery policy
for a short horizon; evaluate the peak resulting tilt. No privileged dynamics
parameters are exposed to the Actor: the risk model is a training-only critic
target and is deliberately conservative under uncertainty.
"""
import math
import torch
from pursuit_rl.simulation.rigid_body import exp_so3


@torch.no_grad()
def predict_attitude_risk(env, action):
    """Return (risk[N] in [0, 1], peak_tilt_rad[N], margin_rad[N]).

    Reproduces the BodyBatch rate lag: rate += (cmd-rate)*(1-exp(-h/tau)).
    The predictor advances rotation at 10 ms increments using a midpoint
    integration approximation (the simulator uses 1 ms). The first decision
    interval executes action; all future intervals assume an ideal available
    *rate-command* recovery towards level. This is a recovery-proxy, NOT a
    guaranteed bound or a full future Actor rollout.
    """
    cfg = env.config["safety"]
    p = env.config["physics"]
    h = float(cfg["prediction_step_s"])
    interval = env.decision_dt
    horizon = float(cfg["horizon_s"])
    count = int(round(horizon / h))
    first_ticks = int(round(interval / h))
    u = env.plant.pursuer
    rot = u.rot.detach().clone()
    rate = u.rate.detach().clone()
    tau = u.tau_rate.detach().clone()
    limits = env.plant.rate_limits
    first = action[:, 1:].detach().clamp(-1, 1) * limits
    peak = torch.acos(rot[:, 2, 2].clamp(-1, 1))
    decay = (1.0 - torch.exp(-h / tau)).unsqueeze(1)
    gain = float(cfg["recovery_rate_gain"])
    for k in range(count):
        if k < first_ticks:
            cmd = first
        else:
            # dR33/dt = R[2,0]*wy - R[2,1]*wx;
            # choosing wx < 0 when R[2,1]>0 and wy > 0 when
            # R[2,0]>0 increases R33, recovering toward the level set.
            target = torch.stack((
                -gain * rot[:, 2, 1],
                 gain * rot[:, 2, 0],
                 torch.zeros_like(rot[:, 2, 0]),
            ), 1)
            cmd = torch.maximum(torch.minimum(target, limits), -limits)
        updated_rate = rate + (cmd - rate) * decay
        # 10 ms midpoint SO(3) integration gives bounded cost at ~1/10
        # the 1 ms simulator workload; validate its error separately.
        rot = rot.bmm(exp_so3((rate + updated_rate) * (0.5*h)))
        rate = updated_rate
        peak = torch.maximum(peak, torch.acos(rot[:, 2, 2].clamp(-1, 1)))
    margin = math.radians(float(p["max_tilt_deg"])) - peak
    warning = math.radians(float(cfg["warning_deg"]))
    limit = math.radians(float(p["max_tilt_deg"]))
    conservative_peak = peak + math.radians(float(cfg["uncertainty_deg"]))
    risk = ((conservative_peak - warning) / (limit - warning)).clamp(0, 1).square()
    return risk, peak, margin

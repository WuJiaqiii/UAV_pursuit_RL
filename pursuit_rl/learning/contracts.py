"""Policy input/action encoding is separate from the simulated task distribution."""

from pursuit_rl.simulation.observations import observation_dimensions

POLICY_VERSION = "ned-frd-asymmetric-v1"


def policy_contract(cfg):
    return {
        "version": POLICY_VERSION,
        "dimensions": list(observation_dimensions(cfg)),
        "observation": dict(cfg["observation"]),
        "scales": {
            key: cfg["reward"][key]
            for key in ("distance_scale", "speed_scale", "cpa_horizon")
        },
        "actions": {
            key: cfg["physics"][key]
            for key in (
                "hover_thrust",
                "thrust_delta",
                "thrust_min",
                "thrust_max",
                "rate_xy",
                "rate_yaw",
            )
        },
    }


def require_compatible_policy(source, target, topology=False):
    a, b = policy_contract(source), policy_contract(target)
    for key in a:
        if a[key] != b[key]:
            raise ValueError(f"Incompatible policy observation/action contract: {key}")
    if topology:
        for key in ("hidden_actor", "hidden_critic"):
            if source["training"][key] != target["training"][key]:
                raise ValueError(f"Incompatible network topology: {key}")

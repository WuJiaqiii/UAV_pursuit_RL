"""First safe-entry curriculum gate, independent of reward ranking."""


def passed(report, config, minimum_success=None):
    minimum = (
        config["training"]["minimum_success"]
        if minimum_success is None else minimum_success
    )
    return report["success_rate"] >= minimum


def summary(report, config, minimum_success=None):
    return dict(
        seed=report["seed"],
        success_rate=report["success_rate"],
        unsafe_entry_rate=report["unsafe_entry_rate"],
        failures=report["failures"],
        passed=passed(report, config, minimum_success),
    )

"""Precision-aware feasibility, independent of score ranking and stopping."""


def passed(report, config, minimum_success=None):
    minimum = (
        config["training"]["minimum_success"]
        if minimum_success is None
        else minimum_success
    )
    error = report["mean_tail_position_rmse"]
    return (
        report["success_rate"] >= minimum
        and error is not None
        and error <= config["task"]["maximum_tail_rmse"]
    )


def summary(report, config, minimum_success=None):
    return dict(
        seed=report["seed"],
        success_rate=report["success_rate"],
        tail_rmse=report["mean_tail_position_rmse"],
        maximum_tail_rmse=config["task"]["maximum_tail_rmse"],
        failures=report["failures"],
        passed=passed(report, config, minimum_success),
    )

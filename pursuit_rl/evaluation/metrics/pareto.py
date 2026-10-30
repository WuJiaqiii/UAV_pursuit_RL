"""Non-dominated sorting for *matched task* multi-objective comparisons.

Minimize `cost_key` while maximizing success rate. Cross-scenario values must
never be compared without first constructing a matched-task aggregation.
"""
import math


def pareto_front(rows, cost_key='lateral_accel_cost_mean', benefit_key='success_rate'):
    valid = []
    for row in rows:
        cost, benefit = row.get(cost_key), row.get(benefit_key)
        if cost is None or benefit is None:
            continue
        if not math.isfinite(float(cost)) or not math.isfinite(float(benefit)):
            continue
        valid.append(row)
    front = []
    for i, row in enumerate(valid):
        x, y = float(row[cost_key]), float(row[benefit_key])
        dominated = any(
            j != i and float(other[cost_key]) <= x and
            float(other[benefit_key]) >= y and
            (float(other[cost_key]) < x or float(other[benefit_key]) > y)
            for j, other in enumerate(valid)
        )
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: float(r[cost_key]))

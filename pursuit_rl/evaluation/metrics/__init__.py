"""Canonical metric entry point; validate a single episode once."""

from pursuit_rl.artifacts.trace import normalize_trace
from pursuit_rl.evaluation.metrics.mission import mission_metrics, _mission_metrics
from pursuit_rl.evaluation.metrics.maneuver import maneuver_metrics, _maneuver_metrics
from pursuit_rl.evaluation.metrics.envelope import envelope_metrics, _envelope_metrics
from pursuit_rl.evaluation.metrics.prediction import cpa_metrics, _cpa_metrics


def evaluate_trace(trace, capture_radius=None):
    trace = normalize_trace(trace)
    return {
        **_mission_metrics(trace, capture_radius),
        **_maneuver_metrics(trace),
        **_envelope_metrics(trace),
        **_cpa_metrics(trace),
    }

"""Shared profiles for the single-aircraft virtual survey curriculum."""

STATIC = "01_waypoint"
ARC = "02_survey_arc"

PHASES = {
    STATIC: dict(kind="goto", duration=10, tail_seconds=4, speed=[0, 0],
                 max_accel=0, distance=[0.5, 1.25]),
    ARC: dict(kind="circle", duration=12, tail_seconds=4, speed=[0.7, 0.7],
              max_accel=0.2, distance=[0.5, 1.25]),
    "03_long_arc": dict(kind="circle", duration=20, tail_seconds=4, speed=[0.7, 0.7],
                        max_accel=0.2, distance=[0.5, 1.25]),
    "04_offset_start": dict(kind="circle", duration=20, tail_seconds=4, speed=[0.7, 0.7],
                            max_accel=0.2, distance=[0.5, 1.25], initial_tilt_deg=3,
                            initial_velocity_noise=0.05),
}


def route_task(name):
    return dict(
        name=name, objective="sustained", radius=0.45, velocity_tolerance=0.6,
        maximum_tail_rmse=0.32, minimum_tail_fraction=0.90, initial_tilt_deg=0,
        initial_velocity_noise=0, match_initial_velocity=True, randomization=0,
    ) | PHASES[name]

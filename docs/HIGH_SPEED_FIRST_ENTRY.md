# High-speed first-entry CTBR curriculum (2.5g)

This is an **experimental flight envelope**, not a validated real drone.

- Maximum nominal specific thrust: 2.5 × 9.81 = **24.525 m/s²**.
- Maximum pursuer speed: **60 m/s**; target speed up to **50 m/s**.
- Entry sphere radius **0.5 m**, relative-speed limit **15 m/s**.
- Direct 4-D CTBR Actor, 100 Hz policy, 1 kHz physics.
- Nominal quadratic drag 0.005; actuator time constants and drag must be calibrated against flight logs before real-world deployment.
- Every phase shares one physics/observation/reward contract. Config names and task objectives differ.
- The agent's true relative pose/velocity are available as oracle observations. No sensor model is claimed.

## Tasks

The thirteen progressive stages cover stationary, low speed, medium speed (15–20 m/s), high speed (40–45 m/s), and extreme speed (45–50 m/s). Within each band, tail, head-on and random geometries are exercised. Head-on phases start farther away to allow braking before the 0.5 m sphere.

Initial pursuer velocity can be generated using `initial_pursuer_heading`, `initial_pursuer_speed`, and `trimmed_initialization`. Tail means the pursuer is behind the target on its current travel heading; head-on means it is in front and initially moving opposite the target. Random generates randomized geometry and heading. These are initial conditions, not a guarantee that every trajectory is dynamically reachable. Auditing reachable trajectories and failure causes is required.

## Commands

```bash
# Run reference guidance on a selected phase.
python main.py evaluate --reference --config configs/first_entry_highspeed/01_static.json \
  --episodes 64 --out runs/highspeed/01_static_teacher.json --device cuda

# Train an independent phase or the ordered curriculum.
python main.py train --config configs/first_entry_highspeed/01_static.json \
  --out runs/highspeed/01_static --device cuda
python main.py curriculum --plan configs/first_entry_highspeed/plan.json \
  --out runs/highspeed/curriculum --device cuda --episodes 64 --minimum-success 0.8

# Final evaluation: reserve unseen seeds and compare PN, PD, policy.
```

The phase budgets are **upper bounds**, not measured convergence values. They range from 12 million physics ticks for simple initial states to 220 million for the most difficult randomized stage; curriculum gates may promote earlier. Compare three or more seeds, report wall time, PPO update count, decision transitions, success and unsafe-entry rates.

## Transfer and reproducibility

Changing drag, thrust, rewards, position caps or policy inputs changes the direct-CTBR checkpoint interface. Do **not** `--resume` or `--init-from` a low-speed checkpoint into these profiles. Start a new model in the shared high-speed contract; later introduce an explicit Actor-only weight migration if wanted. Preserve all existing low-speed config files.

The reference controller supports `training.teacher_method` = `pd`, `pn`, or `pn_hybrid`. No guidance commands are mixed into the learned action during PPO. A successful PN baseline does not establish learnability or generalization.

## Physics limits

2.5g is defined here as total specific thrust divided by gravitational acceleration, not 2.5g *horizontal acceleration*. Maximum speed and tilt remain separate safety checks. Mass randomization also clips commanded actual load. Steady cruise feasibility depends on drag and tilt; the trimmed initializer raises if its requested steady state cannot be achieved.

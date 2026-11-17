# First-entry CTBR experiments (v8)

## Task and terminal semantics

The pursuer must intersect the first-entry sphere of radius `R=0.5 m`
before the episode deadline, with relative velocity norm no greater than
`15 m/s`, without exceeding the platform speed/tilt envelope or invalidating
the target workspace. The very first crossing is terminal: a crossing faster
than 15 m/s becomes `unsafe_entry` (failure), even if the pursuer could
decelerate after flying into the sphere. A 1 ms relative-segment/sphere check
avoids missed crossings between integration samples.

The Actor outputs four normalized CTBR actions at 100 Hz: normalized
mass-specific collective thrust and normalized three-axis body-rate
setpoints. Commands remain held through ten 1 ms dynamics steps. First-order
thrust/rate dynamics and SO(3) kinematics model the inner-loop response; no
motor-level control or AirSim execution is claimed.

The 2.5g limit applies to total specific thrust:
`(T/m)/g <= 2.5`, **not** to horizontal acceleration. Drag is quadratic with
nominal coefficient 0.005; at 50 m/s, modeled drag acceleration is 12.5
m/s². Level cruise with that drag requires a total specific force of roughly
15.9 m/s², within the nominal 24.525 m/s² thrust cap. Actual feasibility
also depends on transient tilt, mass, actuator lag and flight boundary.

## Stages and geometry

`configs/first_entry_highspeed/plan.json` contains 13 stages. Bands:
static (1), 3–8 m/s low (2–4), 15–20 m/s medium (5–7),
40–45 m/s high (8–10), 45–50 m/s extreme (11–13).
Each moving band covers tail, head-on and randomized geometry.

A `tail` reset places the pursuer behind the moving target, initially
aligned in heading; `head_on` places it ahead with opposing velocities.
Random produces randomized relative geometry and target acceleration
patterns. The initial-speed bands, initial distances and horizons differ
deliberately; **some sampled high-speed encounters may still be dynamically
unreachable**, and a physics feasibility prescreen must precede claims of
algorithmic failure. The speed range is enforced in the random target's
latent velocity reference; actual velocity may lag slightly during turns.

The high-speed profiles use `trimmed_initialization`: initial pose and
mass-normalized thrust compensate modeled drag at the sampled speed. The
initializer raises if sampled cruise thrust exceeds the 2.5g/nominal-thrust
envelope. Domain randomization varies drag, mass ratio and actuator
response in a limited range.

## Learning method

PPO uses an asymmetric Actor/Critic, with current oracle relative target
state and current CTBR actuator state as Actor features. A causal LSTM
predictor is supported and trained online. There is **no perception noise
model yet**. Critic may see environment parameters, random target-command
latents, and task parameters.

Reference teachers are `pd`, `pn` and `pn_hybrid`. PN uses line-of-sight
angular rate plus a longitudinal speed schedule, transitioning toward PD
close to the terminal sphere. The teacher provides BC samples or a **separate
evaluation baseline**; PPO's commands are never blended with teacher output.

The first-entry reward contains bounded distance progress, distance-dependent
braking, near-target relative-speed regularization, time penalty, action
smoothness, attitude/rate regularization, terminal success and failure.
Selection/gates rely on **actual first-entry success rate**, not reward
alone or sustained-position tail RMSE.

## Reproducibility / evaluation

Use three or more training seeds. Preserve checkpoints, configurations,
source SHA, rollout/physics-step counts, PPO update counts and wall-clock
statistics. Always evaluate teacher/BC/PPO on equal test seeds and the
same physics+task contract. Report full reason categories, mean entry
velocity, entry time, minimum distance and fraction of unsafe crossings.
Use held-out target trajectories/initial conditions for final reporting.
The budgets in the plan (12 million to 220 million physics ticks) are
**candidate upper limits** and not evidence of convergence.

For smoke testing, reduce physics budget with `train --physics-steps`
in a newly created output directory. The `curriculum` launcher enforces a
minimum budget of two full deadlines per environment.

## Compatibility

Only checkpoints created by this v8 first-entry code can be loaded.
There is no `migrate` command, no sustained-tail goal, no old v700/v701
code path, and no old Survey experiment. All branch history remains in Git.

The first complete high-speed training experiments have **not** been run as
part of this refactor, and simulated parameters must be calibrated before
real-flight use.

## Reward accounting and ablation

The environment constructs raw per-physics reward components, multiplies each
component by `reward.scale`, and returns their scaled sum to PPO. The
decision-step reward components are aggregated using the same within-decision
physics discount as the returned reward. Therefore
`sum(env.reward_components.values())` equals the reward tensor stored in the
PPO rollout. Raw component diagnostics are exposed separately as
`reward_components_unscaled`. Per-episode scaled and unscaled accumulators
are undiscounted physics-rate sums; evaluation Monte Carlo return is discounted
at the decision rate and is a different quantity.

Reward ablations may zero soft shaping weights such as braking, near-entry
velocity, tilt, rate, action smoothness or time. They must not disable the
first-entry radius, 15 m/s terminal relative-speed threshold, 2.5g thrust cap,
60 m/s pursuer speed limit, configured attitude/workspace/altitude limits,
or terminal completion/failure semantics. See
`docs/LOW_TAIL_REWARD_ABLATION.md` for the matched experiment matrix.

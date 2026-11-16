# Direct CTBR first-entry architecture

```text
configs/first_entry_highspeed/*.json → ctbr.config.validate
             │
             ▼
environment.CTBRBatch (first entry, 1 ms physics substeps)
  ├─ simulation.rigid_body.BodyBatch (thrust, body rates, attitude, drag)
  ├─ task reset: tail, head-on, random, and cruise trim
  ├─ goal check: sphere crossing + speed threshold + flight constraints
  ├─ reward: bounded progress / braking / speed / control / terminal
  └─ observations: Actor oracle, Critic privileged, causal target history
             │
             ▼
learning.ppo.AsymmetricPPO (4-D CTBR command every 10 physics steps)
  ├─ learning.prediction.TrajectoryPredictor (optional LSTM)
  ├─ learning.buffers.TensorRolloutBuffer (GAE and event diagnostics)
  ├─ learning.valuenorm.ValueNorm
  └─ ctbr.value_learning (held-out Monte Carlo critic calibration)
             │
             ▼
ctbr.training.train ──→ ctbr.checkpoint (strict v8-only format)
             │
             ▼
ctbr.evaluation.evaluate ──→ reports, traces, comparison scores
             │
             ▼
ctbr.curriculum.run ──→ promotion & retained-success gates
```

**Teacher isolation:** `ctbr.control.teacher` offers PD, PN, PN/PD hybrid
commands for BC initialization and an independent evaluation baseline. PPO
rollouts use only Actor commands.

**Contract isolation:** first-entry schema, observation geometry and the
2.5g flight model form one interface shared by all thirteen profiles;
stage-specific task parameters and budgets vary. Full model transfer
requires an exact interface; prior version checkpoint loading is unsupported.

**Metrics:** first successful safe sphere entry and unsafe crossings, time,
velocity, distance, failure reason, control saturation and prediction RMSE.
No sustained dwell-time or trailing-trajectory acceptance metrics remain.

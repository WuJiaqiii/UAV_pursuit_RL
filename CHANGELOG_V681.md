# V6.8.1

- Add explicit frozen-policy OOD challenge evaluation with paired reference cases.
- Record actual target speed/acceleration and target boundary activation at the first terminal event.
- Add actor-only task transfer, fresh critic/optimizers, MC warm-up and auditable parent lineage.
- Keep strict same-task resume and backward-compatible V6.8.0 checkpoint loading.
- Add nine gradual task profiles and gates on current success, original-task success and capture time.
- Preserve base PPO/action/time parameters; speed-interface upgrades remain separate experiments.

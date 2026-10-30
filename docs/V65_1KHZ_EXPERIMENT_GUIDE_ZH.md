# V6.5 真正 1 kHz 外部 PPO → CTBR：训练、频率对照及 E1–E5 试验说明

## 0. 本次修改的准确含义

本配置要求 PPO **每 1 ms 执行一次 Actor 推理，并产生一组新 CTBR 设定值**。仿真器也以 1 ms 积分；`physics.policy_interval_steps=1` 为真实 1000 Hz 策略更新。它不是 20 Hz 预测值的重复发送，也不是将 50 ms 积分拆成 50 次子步。训练不允许设置 `policy_interval_steps>1`（避免将保持的命令当成每一步新采样进行 PPO 更新）。评估允许设置为 5、10、20、50，配合同一 1 ms 积分构成 200、100、50、20 Hz 决策的**同一策略变频执行**消融。

仿真一阶推力与角速度响应时间常数仍是 0.10/0.13 s，并**未**把执行器改造成 1 ms 即时响应。实际飞控 1 kHz 接口、网络抖动与任务调度不包含在此模拟环境中；需要 HIL 单独核实。当前 1 kHz Actor 仍接收仿真真值，不能代替 1 kHz 传感器量测。

## 1. 关键参数及时间尺度

| 项目 | 旧 V6.4 | 新配置 `ppo_1khz.yaml` |
|---|---:|---:|
| PPO CTBR 新命令周期 | 50 ms | **1 ms** |
| 物理积分 | 10 ms 子步 | **1 ms**，每决策步 1 子步 |
| `physics.dt` / `substeps` | `.05` / `5` | `.001` / `1` |
| 策略保持步数 | 1 | **1**（禁止在训练中保持） |
| `gamma` | .999 | `.999**(1/50)` ≈ .99997999 |
| `gae_lambda` | .95 | `.95**(1/50)` ≈ .99897466 |
| `n_steps` | 256 | 2048（每环境 2.048 s） |
| batch size | 1024 | 2048 |
| Actor/critic LR | 3e-4 / 9e-4 | 1e-4 / 3e-4（建议起点，需消融） |
| `log_std_init` | -0.8 | -1.3（减少毫秒级随机指令抖动） |
| trace 保存周期 | 50 ms | **20 ms**（记录降采样，不影响 1 ms 实际闭环） |

**绝不能仅更改 `physics.dt` 而沿用原 gamma、lambda、Rollout**。`0.99998` 与 `0.998975` 用于保持连续物理时间尺度附近的折扣/GAE 衰减。连续型奖励原有代码多数乘 `dt`，势函数包含 `gamma`，因此整体无需逐项乘 50；但统计上必须重新调节奖励权重以及 stochastic policy 的频率相关抖动。

原始 20 Hz 训练 1600 万环境 transition，若 64 个环境、dt=0.05 s，则累计约 **12,500 模拟秒/环境**；1 kHz 下等模拟时间需要约 **8 亿 transition**。新 `ppo_1khz.yaml` 的 5000 万步仅相当于 **781.25 模拟秒/环境**，定位是初期验证及预训练权重微调，**不是 1600 万旧步的等价训练预算**。

`configs/high_frequency/curriculum_1khz.yaml` 给出了 8 亿步（Stage 0→5）的等物理时间预算示例；`ppo_1khz.yaml` 固定 Stage 5，推荐从已有 31D checkpoint 微调；`hard_1khz.yaml` 对应 2 m 高速 Hybrid 分布，适合第二阶段高难度微调；`smoke_1khz.yaml` 仅测试接口，绝不用于性能评价。

## 2. 安装与最小可运行验证（CPU 也可）

```bash
unzip pursuit_rl_v6_5_1khz_ctbr.zip
cd pursuit_rl_v6_5_1khz_ctbr
python -m pip install -e '.[dev,plots]'
python -m pytest -q

# 极短测试：1ms 推理、PPO训练、模型保存、验证
python main.py train --config configs/high_frequency/smoke_1khz.yaml \
  --out runs/smoke_1khz

# 从测试checkpoint执行1 kHz PPO/PN/APN闭环测试
python main.py benchmark --checkpoint runs/smoke_1khz/checkpoints/best.pt \
  --config configs/high_frequency/smoke_1khz.yaml --stage 0 \
  --algorithms PPO PN APN --episodes 2 --no-plots --out runs/smoke_benchmark_1khz
```

**短测中成功率没有研究意义**。检验 `protocol.json` 的 `physics_dt_s=0.001`、`policy_frequency_hz=1000`，以及各 `config.yaml` 的相同参数。固定场景过去会把 `physics.dt` 静默恢复为 0.05；V6.5 已专门修复这一点。

## 3. 从旧 Hard checkpoint 初始化 1 kHz PPO（建议首选）

```bash
# 50M decision steps，是初期微调预算；网络仍是旧版 31D actor / 43D critic。
python main.py train --config configs/high_frequency/ppo_1khz.yaml \
  --init-from /absolute/path/to/v64_hard_seed42/checkpoints/best.pt \
  --device cuda --seed 42 --name ppo_1khz_precision_seed42 \
  --out runs/ppo_1khz_precision_seed42

# 进一步在困难Hybrid上训练，切换任务配置必须 init-from，不能resume
python main.py train --config configs/high_frequency/hard_1khz.yaml \
  --init-from runs/ppo_1khz_precision_seed42/checkpoints/best.pt \
  --device cuda --seed 42 --name ppo_1khz_hard_seed42 \
  --out runs/ppo_1khz_hard_seed42
```

不带 `--init-from` 的从零训练入口：

```bash
python main.py train --config configs/high_frequency/curriculum_1khz.yaml \
  --device cuda --seed 42 --name full_curriculum_1khz \
  --out runs/full_curriculum_1khz
```

完整等时间预算成本非常高；正式训练前先运行 benchmark，并估计服务器完成 1 万次 1 ms 动力学和模型前向计算的真实吞吐量。`--resume` 只能用于相同 V6.5 配置的中断恢复，升级旧模型时使用 `--init-from`。

## 4. 1 kHz 在线 Benchmark（只用新训练 checkpoint）

```bash
# 先每场景8个回合评估逻辑与资源占用
python main.py benchmark --config configs/high_frequency/ppo_1khz.yaml \
  --checkpoint runs/ppo_1khz_hard_seed42/checkpoints/best.pt \
  --algorithms PPO PN APN \
  --scenarios precision_2m ood_pulse_2m ood_switchback_2m \
  --episodes 8 --seed 970000 --no-plots --out runs/benchmark_1khz_pilot

# 先确认仿真定义、基线调参及模型选择冻结，再做正式 256 回合比较
python main.py benchmark --config configs/high_frequency/ppo_1khz.yaml \
  --checkpoint runs/ppo_1khz_hard_seed42/checkpoints/best.pt \
  --algorithms PPO PN APN \
  --scenarios precision_2m singer_fast rapid_hybrid ood_pulse_2m ood_switchback_2m \
  --episodes 256 --seed 980000 --out runs/benchmark_1khz_formal
```

每个 150 s 任务在 1 kHz 可能运行 **150,000 次动作/物理循环**。务必先基于 8 个任务估计耗时和存储，确认可接受后再运行正式规模；正式评估最好三训练随机种子，且避免使用测试结果调参。

## 5. 频率扫描：相同 1 ms 物理积分，仅改变策略更新间隔

```bash
# 不需重新训练六个checkpoint；这是固定权重的部署频率消融（零阶保持）
python scripts/frequency_study_1khz.py \
  --checkpoint runs/ppo_1khz_hard_seed42/checkpoints/best.pt \
  --config configs/high_frequency/ppo_1khz.yaml \
  --hz 20 50 100 200 500 1000 \
  --scenarios precision_2m ood_pulse_2m ood_switchback_2m \
  --algorithms PPO PN APN --episodes 16 --seed 990000 \
  --out runs/frequency_study_pilot
```

输出每个频率独立 Benchmark 的 `protocol.json`、每回合报告、`frequency_summary.csv` 和 `frequency_protocol.json`。**频率消融使用相同环境种子、1 ms 物理积分和同一目标随机过程**；但它比较的是同一 checkpoint 的运行频率敏感性，不是不同频率分别训练的最终能力。后一种研究应在每个频率下分别训练模型，并根据*模拟物理秒数*而非仅 transition 数匹配预算。

需要单独手动测试特定频率可以：

```bash
# 1ms动力学，PPO每10ms更新一次（100Hz）
python main.py benchmark --config configs/high_frequency/ppo_1khz.yaml \
  --set physics.policy_interval_steps=10 \
  --checkpoint /absolute/path/to/best.pt --algorithms PPO PN APN \
  --scenarios ood_pulse_2m --episodes 8 --no-plots --out runs/100hz_pilot
```

## 6. E1–E5 at 1 kHz

```bash
python main.py experiments --config configs/high_frequency/ppo_1khz.yaml \
  --checkpoint /absolute/path/to/best.pt --suites E1 E3 E5 \
  --quick --episodes 2 --calibration-episodes 2 \
  --seed 950000 --out runs/e1e3_1khz_quick

# 通过quick测试后，再开展E1/E2/E3/E4/E5单独批量测试
python main.py experiments --config configs/high_frequency/ppo_1khz.yaml \
  --checkpoint /absolute/path/to/best.pt --suites E1 E2 E3 E4 E5 \
  --episodes 128 --calibration-episodes 32 \
  --seed 960000 --out runs/e1e5_1khz_formal
```

E3 需要迭代评估大量经典制导参数，在 1 kHz 规模下耗时明显；建议先完成 E1 及最关键的 E4 动力学失配，后续单独标定 E3。在 E4 中，观测延迟和命令延迟以 1 ms 整数格实现，例如 100 ms =100 个循环；数值与实际总线实时抖动不同。

## 7. 论文指标：防止降采样混淆

1. `evaluation.trace_stride=20` 表示每 20 ms 保存一个 NPZ 状态样本，**并非每 20 ms 才向飞控下发新指令**。每个 `PursuitBatch.step()` 都产生新的 PPO 动作（除非频率对照明确设置间隔）。
2. 在环境中每 1 ms 累积全频率的 `control_variation_full`、`lateral_accel_cost_full`、`high_envelope_any_ratio_full`，结束时写入 `terminal_record_json`。离线评估优先使用这些完整积分指标，避免把动作抖动因保存降采样而低估。
3. `terminal_success` 和 `terminal_reach_time` 是物理仿真事件记录，首次穿越插值点来自**最后 1 ms 仿真步**，不依据降采样的远距离点推断成功。
4. 其他轨迹依赖的统计（如部分 CPA、姿态分位数、最大角速度、速度变化）依然会受到 20 ms 保存采样率影响。若需精细论文控制时序，使用 `--set evaluation.trace_stride=1`，**只对少量典型回合使用**，不要用于成千上万个完整回合。
5. 正式实验中，目标状态观测仍是假设的高频仿真真值；实际 1 kHz 飞行中是否存在这样的新信息取决于状态估计模块的更新和外推能力。

## 8. 运行时延检查与硬件部署限制

```bash
python scripts/measure_actor_1khz.py \
  --checkpoint runs/ppo_1khz_hard_seed42/checkpoints/best.pt \
  --device cuda --warmup 500 --iterations 5000 \
  --out runs/actor_1khz_latency.json
```

这仅测 batch=1 **NumPy 输入→Torch Actor 推理→NumPy 输出**的 wall latency、p50/p95/p99、超 1 ms 次数；不包含状态估计、线程唤醒、消息总线、CTBR编码、控制器跟踪和操作系统实时调度。**实际 1 kHz 可靠发送需要额外实现并验证硬件接口的确定性 1 ms 截止时间、丢包/过期指令处理、超时安全行为和 HIL 测试**。仿真成功不构成真实硬件实时安全认证。

## 9. 修改文件及回归

`config/defaults.yaml`、`config/schema.py`、`config/loader.py`、`simulation/targets.py`（Singer Z OU 阻尼按实际秒数调整）、`simulation/environment.py`、`simulation/events.py`、`evaluation/scenarios.py`（修复已注册场景时间步被重置）、`evaluation/rollout.py`（按设定毫秒更新 Actor）、`training/trainer.py`（阻止假 1 kHz 训练）、`artifacts/trace.py`（降采样但保留精确终端事件）、`evaluation/metrics/{mission,maneuver,envelope}.py`、`evaluation/benchmark.py`、新增配置、脚本及测试。旧默认配置仍维持 dt=0.05、substeps=5 以兼容 V6.4 基线；使用 1kHz 必须明确传入新版 YAML。

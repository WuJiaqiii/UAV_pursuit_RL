# 训练、迁移与课程

[返回 README](../README.md) · [架构](ARCHITECTURE.md) · [评估](EVALUATION.md) · [诊断](DIAGNOSTICS.md)

## 基础任务从头训练

```bash
DEVICE=cuda PPO_STEPS=20000000 bash scripts/run_rescue.sh
```

默认使用 `configs/rescue/velocity_50hz_2m.json`，输出 `runs/v680_rescue/velocity_50hz_2m/seed_42/`。先初始化，再恢复 latest 做 PPO，最后用相同测试初态比较 `bc/best/latest`，写入 `comparison.json`。

默认初始化：4 万 BC 状态/1200 次更新，一轮 DAgger 的 2 万状态/500 次更新，再用当前随机策略完整回合 MC 目标拟合 critic 600 次。随后 PPO 示教辅助系数为零，网络执行不混合参考动作。

环境变量：`PYTHON`、`DEVICE`、`PROFILE`、`RUN_ROOT`、`SEEDS`、`PPO_STEPS`、`EVAL_EPISODES`、`TEST_SEED`、`MIN_VALID_SUCCESS`。`PPO_STEPS` 是累计预算，不是每次调用追加的预算；脚本默认 200 万步只适合初期检查。

```bash
DEVICE=cuda SEEDS="42 43 44" PPO_STEPS=20000000 \
  RUN_ROOT=runs/rescue_multiseed bash scripts/run_rescue.sh
```

这是三个从头初始化运行。复用一个训练好 actor、只改变微调种子，不能视为三个独立完整训练链。

## 预算与连续运行

PPO 预算为所有环境的物理步之和；BC/DAgger、MC 预训练/校准和评估交互另计。

```text
每环境名义仿真秒数 = PPO物理步 × physics.dt / num_envs
每轮决策样本数     = rollout_decisions × num_envs
每轮物理步         = rollout_decisions × num_envs × decision_interval_steps
```

64 环境、1 ms 积分、20 个物理步一决策时，256 决策 rollout 覆盖每环境 5.12 秒。2000640 聚合物理步只对应每环境 31.26 秒，小于课程 40/60 秒截止；短阶段“完成回合全成功”不能当作完整时域成功率。

从重置开始，至少约 256 万/384 万聚合物理步，才有机会让未捕获 PPO 回合走到 40/60 秒超时。这不是收敛的充分预算；GAE 可以从部分轨迹 bootstrap。完整 MC 预训练和评估不受该 PPO 预算限制，仍能观察超时。

恢复会重置未完成环境。长回合研究应连续运行足够长时间，避免频繁退出/恢复将困难回合反复截断。计数向完整决策批次取整。

## 恢复与任务迁移

| 操作 | 行为 |
|---|---|
| `--resume` | 相同任务/配置恢复模型、优化器、随机状态和已完成更新；允许延长总预算，未完成回合重开；须在原输出目录恢复 |
| `--init-from` | 新任务只复制 actor（含动作记忆）；重建 critic/优化器，按新 profile 重置探索标准差并做 MC 拟合；须满足控制/观测兼容性 |

已有基础任务目录的预算延长：

```bash
python scripts/rescue.py train \
  --profile configs/rescue/velocity_50hz_2m.json --device cuda --seed 42 \
  --resume runs/v680_rescue/velocity_50hz_2m/seed_42/checkpoints/latest.pt \
  --physics-steps 20000000 \
  --out runs/v680_rescue/velocity_50hz_2m/seed_42
```

新任务示例，替换 checkpoint 为真实路径：

```bash
python scripts/rescue.py train \
  --profile configs/curriculum_v681/01_singer_05.json \
  --init-from /path/to/baseline/best.pt \
  --device cuda --seed 42 --physics-steps 10000000 \
  --out runs/singer05_long
```

课程 profile 要求 `--init-from` 或 `--resume`。当前 `bc_min_success` 的初始化停止检查仅在 `bc_updates > 0` 时执行；零 BC 迁移没有这项训练前自动门槛，应先冻结评估迁移起点。模型和参数必须使用匹配入口，不能把 rescue checkpoint 交给旧 `main.py`。

## 九阶段课程

基线布局为 `/path/to/baselines/velocity_50hz_2m/seed_42/checkpoints/best.pt`，Git 不包含权重。短程流程检查示例：

```bash
bash scripts/run_curriculum.sh --device cuda \
  --baseline-root /path/to/baselines/velocity_50hz_2m \
  --phases 01_singer_05 02_singer_1 --physics-steps 2000000 \
  --out runs/v681_curriculum_check
```

完整计划省略 `--phases`。默认基线根为 `baseline_models/velocity_50hz_2m`，默认输出根为 `runs/v681_curriculum`。多种子用 `--seeds 42 43 44`，每个种子须提供各自基线。

| 阶段 | 变化 |
|---|---|
| 01–03 | Singer 加速度上限 0.5、1、2 m/s² |
| 04–05 | Dubins、Hybrid switchback，2 m/s² |
| 06 | 直线/Dubins/Singer/Hybrid 混合 |
| 07–08 | 混合任务初始距离上限增至 60、100 米 |
| 09 | 初始目标速度上限增至 8 m/s，截止由 40 改为 60 秒 |

均保持 2 米首次捕获、50 Hz 策略/1 kHz 内环。`--phases` 按计划顺序串联所选阶段，从基线起步；继续已有链时保持相同阶段序列或运行完整计划。延长上游预算会改变下游父模型，请为改变后的整条链使用新输出根目录。

当前门槛是：当前任务**最新一次验证**成功率 ≥95%；所选 `best` 的基础任务回归成功率 ≥95%；基础回归惩罚后均值耗时 ≤基线的 1.25 倍。下一阶段继承 `best`。父 checkpoint SHA 不符时停止，整条链有 `curriculum_summary.json`。

当前任务 seed78000、回归 seed79000 参与选择，不是最终独立测试集。门槛不要求新增 PPO 胜过初始化，因此“阶段通过”可能仅来自迁移能力。应另用[初始化/latest 对照](EVALUATION.md)检查训练收益。

## 输出和可选路线

| 文件 | 含义 |
|---|---|
| `checkpoints/bc.pt` | 零轮本阶段 PPO 初始化；迁移阶段可能未做 BC，须看 origin |
| `checkpoints/best.pt` | 按验证成功率优先、成功捕获时间次之选择，可能保留初始化 |
| `checkpoints/latest.pt` | 最近完成更新后的模型，恢复使用它 |
| `base_config.json`、`rescue_profile*.json`、`control_contract.json` | 配置与动作/任务契约 |
| `metrics.jsonl`、`training_summary.json` | 更新、来源、计数和校准；来源累计计数不等于本轮总计算预算 |
| `initial_validation.json`、`validation_*.json` | 当前任务验证 |
| `baseline_regression/` | 课程 best 的基础回归 |
| `first_anomaly.pt` | 更新异常证据，不是普通恢复模型 |

约每 200 万物理步验证并做随机策略 MC 校准；RMSE >1.5 时额外拟合 critic 300 次。阈值单位为任务奖励乘 0.1；训练集 loss 和高 explained variance 不代替独立控制评估。

其他 profile：`velocity_50hz.json` 是 5 米任务；`ctbr_1khz_bc.json` 为 1 kHz CTBR；`ctbr_20hz_bc.json` 为 20 Hz CTBR；`velocity_50hz_from_scratch.json` 为无 BC 消融。它们不是已验证等效的替换方案。

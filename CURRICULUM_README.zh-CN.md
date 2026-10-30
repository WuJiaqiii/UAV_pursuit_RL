# V6.8.1：逐级增加机动、速度与初始距离

V6.8.0 已在服务器上恢复基础追踪。本版本保留原策略、奖励尺度和物理时间参数，增加冻结模型的复杂场景评估、显式任务迁移以及带回归检查的课程执行。原 V6.8.0 工程和下载结果没有被修改。

## 先做预评估

在新项目根目录运行：

```bash
python scripts/challenge.py \
  --checkpoint baseline_models/velocity_50hz_2m/seed_42/checkpoints/best.pt \
  --device cuda --episodes 256 --seeds 1000000 1010000 1020000 \
  --scenarios baseline singer_a05 singer_a1 singer_a2 dubins_a2 hybrid_a2 \
              distance_30_60 distance_60_100 speed_5_8 speed_8_12 speed_8_12_open combined_moderate \
  --out runs/v681_challenge
```

每个场景都对比同初始条件的冻结网络和显式参考控制器，完整记录训练任务与评估任务的差异；不会修改模型。初筛可先用 `--episodes 32 --seeds 1000000`。`--stochastic` 可检查当前随机采样策略的 MC 回报；确定性执行的价值误差只作参考。

`baseline` 始终恢复原来的 2 米、10–30 米距离、0–5 m/s 初始速度、直线目标任务；`training_task` 才表示 checkpoint 当前训练的任务。`actuator_lag_150` 和 `drag_150` 修改实际随机机体参数，内环仍用原名义参数，避免把正确物理参数提前提供给控制器。

每回合记录目标实际最大速度/加速度、目标边界回拉区域激活步数，以及终止时机体速度/倾角。目标速度范围指初始化速度；机动过程中的实际速度另行记录。

`speed_8_12_open` 扩大目标活动边界，保留追踪器原工作空间和约束，用来区分真实追击与目标边界回拉对捕获的帮助。参考控制器失败不能直接证明任务物理不可达。

## 从已训练模型进入课程

随包 `baseline_models/velocity_50hz_2m/seed_42/checkpoints/best.pt` 是你提供的服务器最终模型。第一轮建议只执行前两个阶段：

```bash
bash scripts/run_curriculum.sh --device cuda \
  --phases 01_singer_05 02_singer_1 --physics-steps 2000000
```

前两阶段通过后，使用相同输出目录运行完整计划；已完成阶段保持原预算，会检查后继续：

```bash
bash scripts/run_curriculum.sh --device cuda --physics-steps 2000000
```

若使用原服务器实验路径或其他匹配训练种子的模型：

```bash
bash scripts/run_curriculum.sh --device cuda \
  --baseline-root /path/to/runs/v680_rescue/velocity_50hz_2m \
  --seeds 42 43 44 --physics-steps 2000000
```

每个训练种子必须有自己的基线 `seed_N/checkpoints/best.pt`。从同一个 seed42 模型分别用 43/44 微调，不能视为三条独立的完整训练链。代码不会自动用 seed42 替代缺少的基线。

### 默认课程

所有阶段保留 2 米首次捕获任务、50 Hz 策略、1 kHz 内环、原机体极限和控制尺度。

| 阶段 | 变化 |
|---|---|
| 01 | Singer，目标加速度上限 0.5 m/s²，原距离/速度 |
| 02 | Singer，上限 1 m/s² |
| 03 | Singer，上限 2 m/s² |
| 04 | Dubins 转弯，上限 2 m/s² |
| 05 | Hybrid switchback，上限 2 m/s² |
| 06 | 直线/Dubins/Singer/Hybrid 混合，保留基础目标类型 |
| 07 | 混合任务距离范围扩大到 10–60 米 |
| 08 | 混合任务距离范围扩大到 10–100 米 |
| 09 | 初始目标速度范围扩大到 0–8 m/s，任务时长改为 60 秒 |

前八阶段截止时间保留 40 秒；各阶段默认增加约 200 万 PPO 物理步，向完整决策批次取整。MC 初始化、评估等交互额外计数。2 米是首次进入捕获球，不是持续保持误差。

单独选择 `--phases` 会按计划顺序串联所选阶段，从原始基线起步。继续已有链时保持相同阶段序列，或运行完整计划。延长前面阶段预算会改变下游父模型；要改变整条链的预算，请使用新的 `--out`，避免混用旧下游结果。

## 新任务迁移的含义

`--resume` 仍只用于相同任务的中断恢复或预算延长。新参数 `--init-from` 用于新任务：

```bash
python scripts/rescue.py train \
  --profile configs/curriculum_v681/01_singer_05.json \
  --init-from baseline_models/velocity_50hz_2m/seed_42/checkpoints/best.pt \
  --device cuda --seed 42 --physics-steps 2000000 \
  --out runs/v681_manual_singer05
```

- 复制 actor，包括已学到的动作记忆权重。
- critic 和两个优化器重新初始化，使用新任务的完整回合 MC 回报拟合 critic。
- 探索标准差使用新 profile 的设置。默认课程不重复 BC，不把 actor 拉回较慢的参考策略。
- 新阶段 PPO 步数从零计数，另外保留父模型与累计训练步数、来源文件 SHA256。
- 检查动作单位、速度尺度、观测归一化、决策周期、内环和名义机体参数；不同动作含义不能仅凭网络形状相同就迁移。
- 课程 profile 要求 `--init-from`；漏传时直接报错，避免意外从随机策略开始投入预算。

初始化验证低于 80% 时保存初始化结果并停止 PPO。这个下限可在 profile 中调整；默认不自动用更多训练掩盖缺乏可用起点的问题。

## 每阶段的通过条件

1. 当前任务固定 64 回合验证成功率至少 95%。
2. 所选最佳模型在原基础任务的独立回归验证成功率至少 95%。
3. 基础回归的惩罚后平均捕获时间，不超过初始基线的 1.25 倍。失败回合按任务截止时间计，不会通过提前坠毁“变快”。

回归种子默认 79000；当前任务验证种子沿用 78000。它们用于课程门槛，不能充当最终独立测试集。训练结束后另用新的测试种子和足够回合数评估。参考控制器结果是对照，不混入网络动作。

每个阶段输出模型、`rescue_profile_latest.json`、训练计数和回归报告；整条链有 `curriculum_summary.json`。如果父模型文件已改变，已有下游阶段会停止并要求使用新输出目录，防止悄悄改写模型来源。

## 当前发现与后续代码优化

CPU 小样本预评估说明当前模型可以进入轻度机动任务，但高速泛化有明显缺口。原边界内的 8–12 m/s 测试有四个慢回合进入目标回拉区域；扩大目标边界后，同四个回合失败。详细初筛在 `NEXT_EXPERIMENTS.md` 和 `challenge_screen/`。

优先顺序：先验证课程覆盖和旧任务保留，再比较加速度前馈示教或预测信息。当前每个水平轴速度指令限幅 15 m/s，水平向量上限约 21.21 m/s；25 m/s 逃逸目标的某些工况超出此命令空间。更高速度阶段需要单独升级并验证动作接口，重新做合适的初始化。不要在加载旧网络时直接把速度尺度翻倍。

当前未更改 PPO 核心、奖励权重、gamma/lambda 和基础内环参数；没有声称完整高机动课程已收敛。短程 CPU 运行只验证迁移与课程链路。新代码已做单元测试和旧实现兼容检查；完整九阶段 CUDA 训练仍需执行。

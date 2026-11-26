# 静态虚拟航点与 PD 教师消融

在仿真中从定点悬停起飞，首次进入一个静止虚拟航点的球形区域时结束回合。
每个回合开始时采样一次航点，回合内保持固定。默认初始水平距离为 2–5 m，
球半径为 0.5 m，回合时限为 6 s。PD 的期望地速上限为 8 m/s，所有实验组共用
10 m/s 三维实际地速超限阈值。超过阈值判定为失败，即使同一步已到达航点。
实验使用 1 kHz 仿真和 50 Hz 控制决策，观测包含时间余量。

| 实验组 | 采样或执行 | 用途 |
| --- | --- | --- |
| `zero` | 零归一化 CTBR 命令 | 确认悬停不会被计为到达 |
| `pd` | 静态航点 PD | 验证仿真里单机可以到达航点 |
| `ppo` | PPO 自己采样 | 测量从零开始的表现 |
| `bc_ppo` | 先采集 PD 示范并预训练 Actor，再由 PPO 自己采样 | 分开测量行为克隆的收益和 PPO 更新后的变化 |
| `aux_ppo` | PPO 自己采样，PD 对相同状态提供动作标签 | 检查训练期间的辅助监督是否有效 |

所有学习组使用相同的 Actor 和 Critic 结构、同样的 PPO 决策步预算以及相同的
独立评估航点。`bc_ppo` 额外消耗示范样本，`aux_ppo` 额外消耗标签，结果应同时
报告这两项成本。评估使用确定性策略。并行环境按各回合到达或失败重置，
训练状态独立于评估环境。先跑测试和 PD 对照，确认基线结果再启动 GPU 消融。

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p test_virtual_waypoint.py -v

python -m scripts.run_waypoint_experiments \
  --out runs/waypoint_baselines --arms zero pd --device cpu

python -m scripts.run_waypoint_experiments \
  --out runs/waypoint_smoke --device cpu --envs 2 --episodes 2 \
  --rollout-steps 8 --updates 1 --batch-size 16 --bc-steps 8 --bc-epochs 1

python -m scripts.run_waypoint_experiments \
  --out runs/waypoint_ablation --device cuda --seeds 42 43 44
```

正式消融的 PPO 默认预算是每组每种子 `64 × 128 × 120 = 983,040` 个控制决策。
示范预训练默认额外采集 16,384 个 PD 控制决策。每组评估 64 个相同的独立回合。
各训练种子的成功率单独报告。PPO 更新阶段前后使用的评估航点一致，评估结果不参与
checkpoint 选择。较近的单机航点可通过 `--distance 0.75 1.5` 单独诊断，输出使用新目录。

每次启动创建一个新的输出目录，其中包含 `plan.json`、`summary.csv` 和
`progress.json`。进度文件分别记录已完成的实验组和 PPO 更新数。每组还保存逐回合
评估、更新指标及最终模型，行为克隆组另存 PPO 更新前的评估和模型。统计时同时看
首次到达率、最小距离、超速和姿态失败。航点任务的 6 s 时限是任务本身的一部分，
在时限处终止回合。采样批次结束时仍在飞行的回合保留价值估计。

- PD 未能在独立航点上稳定到达时，先检查仿真环境和基线。
- PD 能到达而 PPO 从零仍无法到达时，比较 `bc_ppo` 的 `after_bc` 结果，判断示范
  是否让 Actor 在更新前就有可用行为。
- `after_bc` 明显好于同组 `final` 时，先对照 `metrics.jsonl` 的 KL、裁剪比例和回合
  失败原因，定位 PPO 更新后表现变差的阶段。
- `aux_ppo` 和 `ppo` 的差异用于衡量同预算下辅助标签的收益。单种子和 CPU 冒烟
  只验证流程，不能据此判断学习是否收敛。

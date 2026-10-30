# V6.4 E1–E5 复杂追逐实验操作与研究协议

## 一、设计目标与边界

在保留 V6.3 的 31D/43D 非对称 PPO、CTBR 控制指令、原动力学模型、首次进入交会半径终止规则和 checkpoint 合约的基础上，新版增加五组实验。**本更新不是重新训练得到的新模型，也不是 AirSim 或真机动力学替代。**

- E1：Singer、Rapid Hybrid、高加速度、交替横向机动、Pulse 与同向极限追逐。
- E2：三档速度 × 五档机动加速度共 15 个独立任务；固定首入球半径 2 m，初距 300–400 m，150 s 上限。
- E3：独立验证随机种子上参数标定 PN/APN；保存冻结后的控制器参数；与 PPO 共享测试场景、目标轨迹、动力学和执行器。
- E4：独立评估扰动，包括执行器响应常数、位置/速度测量噪声、观测延迟和 CTBR 指令传输延迟。
- E5：自动生成每场景、每控制算法的轨迹比较、CPA/负载/角速度曲线、E2 成功率图、Pareto 非支配点和 CSV/LaTeX。

## 二、环境与启动

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q

# 历史 V5 权重仅用来验证兼容性和运行路径，不代表 V6.4 策略已重新训练
python main.py experiments \
  --checkpoint examples/v5_legacy_best.pt \
  --suites E1 E2 E3 E4 E5 --episodes 2 --calibration-episodes 2 \
  --quick --out runs/smoke_e1e5

# 正式实验（计算量较大），验证种子与测试种子分开
python main.py experiments \
  --checkpoint /path/to/V6/best.pt \
  --suites E1 E2 E3 E4 E5 --episodes 128 --calibration-episodes 32 \
  --seed 930000 --out runs/e1e5_v6_4_full

# 仅E1、E3和E5，优先完成经过标定的算法对照
python main.py experiments \
  --checkpoint /path/to/best.pt --suites E1 E3 E5 \
  --episodes 256 --calibration-episodes 32 --out runs/e1e3_256

# 不依赖 PPO checkpoint，仅验证 PN/APN
python main.py experiments --algorithms PN APN --quick --episodes 2 \
  --out runs/classic_only

# 在已有 E1-E4 结果上仅重新生成 E5（不修改已有轨迹）
python main.py experiments --algorithms PN APN --suites E5 \
  --out runs/classic_only
```

每次新 E1–E4 任务 **必须使用空输出目录**。E5 可以单独从现有完整结果重建图表。

## 三、对照公平性与基线含义

同组比较使用固定相同的场景配置、`seed`、初始状态与目标目标运动随机数。`paired_traces` 附带目标真实轨迹前缀一致性断言，若发现不同算法所面对的目标轨迹不相同，就抛错而不是生成误导性叠加图。

`GuidanceParameters`（PN/APN）含纵向速度管理、加速度限幅、姿态反馈环；论文中应称“PN/APN guidance with common CTBR attitude inner loop”，不是纯理想加速度比例制导。

E3 先跑默认基线，再在固定的 `validation_seed = test_seed + 10000000` 上扫描控制参数。PN 默认 16 组候选；APN 默认 24 组候选，按验证成功率优先、截尾时长小权重打破平局。固定最终参数后在原测试种子独立评价。`--quick` 仅 2 组候选，**不能用于正式论文标定**。

标定在固定分布 `singer_fast` 和 `rapid_hybrid` 上开展；正式结论需注明该验证分布，OOD 的 `ood_switchback_2m` 和 `overtaking_edge` 为独立测试子集。需要跨新目标分布泛化时，可更改验证分布，但不允许用测试回合调参。

## 四、测量与控制扰动（E4）

`sensor` 在评估中维护独立随机状态缓存；目标与自机位置、速度、目标加速度采用**同样的延迟与测量噪声**形成相对观测。PPO Actor 中的相对运动、目标加速度、CPA 特征由测量值替换，姿态和机体系角速度仍来自理想本机 IMU。PN/APN 制导几何、纵向速度管理和 APN 目标加速度补偿读取同一测量通道；底层姿态环仍读取真实机身姿态、角速度（简化 IMU 模型）。

`CommandChannel` 在 `env.step()` 之前缓冲归一化 CTBR。延迟期间输出悬停中性指令 `[0,0,0,0]`；传输延迟必须是 `physics.dt=0.05 s` 的整数倍。轨迹 `commanded_action` 与 `action` 分别保存发出的指令与环境真正执行的指令，`command_tracking_l2_mean` 可以评价传输延迟带来的跟踪差异。

**关键限制**：目前不模拟雷达/VIO 各自异步时间戳或 IMM 滤波器，也不模拟图像检测失效；E4 是简化的状态测量鲁棒性评估，不是完整多传感器自主飞行栈。

## 五、结果结构

```text
runs/e1e5_v6_4_full/
├── experiment_protocol.json
├── E1/benchmark/                    # 六种高机动场景
├── E2/benchmark/                    # 15格性能边界
├── E3/default/                      # 未标定PN/APN
├── E3/calibrated/                   # 标定后的PN/APN
│   └── ...
├── E3/PN_calibration.json
├── E3/APN_calibration.json
├── E3/locked_parameters.json
├── E4/nominal/ ...                  # 观测/动力学/执行器扰动
├── figures/                         # 轨迹、控制时序、边界热力图、Pareto
├── combined_summary.csv
├── combined_summary.json
├── combined_table.tex
├── E5_visualizations.json
└── experiment_complete.json
```

每组比较原样继承 V6 的 `protocol.json`、`summary.csv`、`summary_table.tex`、各算法目录 `metrics.json` 与 `episode_0000.npz` 等文件。实验恢复请用新输出目录或 E5 重绘入口；不允许覆盖已有结果。

## 六、指标解释及统计注意事项

`terminal_success` 与 `terminal_reach_time` 由环境首次入球事件产生，不使用离散最短距离代替。特别是 1m/2m 精度对比应在正确半径下重新运行，而不是对 5m 已提前终止的轨迹二次判定。

旧字段 `high_envelope_ratio` 继续仅指高推力载荷；新增 `high_envelope_any_ratio` 是速度、倾角、载荷、角速度中任一分量超过 90% 上限的时间比例，同时记录各分量最大、95 分位和高使用率比例。`lateral_accel_cost` 为横向加速度平方时间积分，`control_variation_per_s` 为按回合时长归一化的动作变化代价；它们不等同于实际电功耗。

Pareto 比较严格在**同一场景和同一 benchmark** 中进行：横轴默认所有回合的横向加速度代价均值，纵轴为成功率。不得把高难度和简单任务直接作为同一个 Pareto 前沿的数据点。成功回合的成本均值在结果表中单独提供；正式论文需一并给出失败原因、Wilson 区间及多训练 seed 方差。

对于不足 16 回合的 quick 结果，尤其 0/1 的成功率波动，仅作软件通路验证，不形成论文性能结论。建议 E1/E3 至少 128–256 个配对回合，各 PPO 模型至少三个独立训练 seed，E2 先网格小样本筛选，再在关键边界格增加回合数。

## 七、模块增量清单

- `pursuit_rl/simulation/disturbances.py`：传感器状态缓存、观测特征替换、CTBR 命令缓冲。
- `pursuit_rl/evaluation/scenario_grid.py`：E2 固定网格、E4 鲁棒性实验生成器。
- `pursuit_rl/evaluation/tuning.py`：验证集标定。
- `pursuit_rl/evaluation/experiment_suite.py`：E1–E5 一键编排与复合 CSV/报告。
- `pursuit_rl/evaluation/metrics/pareto.py`：同任务非支配解。
- `pursuit_rl/visualization/experiment.py`：配对 3D、控制时序、热力图和 Pareto。
- `main.py` 新子命令 `experiments`；适配扩展 `evaluation/benchmark.py` 与 `rollout.py`。
- 补充指标：`envelope.py`、`maneuver.py`、`prediction.py`、`report.py`；增强轨迹字段：`artifacts/trace.py`。

不修改 PPO 网络结构、Trainer、奖励函数和动力学积分算法，不破坏原 V6 的 `train`、`benchmark`、`evaluate` 等命令。**E1–E5 扩展只增加评估能力，训练增强课程学习仍应作为单独有明确实验对照的研究更改。**

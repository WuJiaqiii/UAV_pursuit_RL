# v7.0 实验操作

## 建议顺序

先执行测试与 smoke，再用 seed 42 跑课程；查看基础课门槛、critic_correction 与 forecast 指标后，再扩展到 43、44。

~~~bash
pytest -q
python main.py train --config configs/ctbr/smoke.json --device cpu --out runs/v700_smoke
SEEDS=42 DEVICE=cuda bash scripts/run_ctbr_curriculum.sh runs/ctbr_v700
~~~

新实验必须使用空目录；相同命令和输出目录可继续课程。不要复用 ctbr_v691 目录，也不要在同一 run 里切换代码、控制频率或预测模式。旧检查点与 v3 不兼容。

## 课程

| 阶段 | 变化 |
|---|---|
| 01–03 | 悬停、小扰动恢复、移动定点保持 |
| 04–09 | 近距低速跟随，加速度 0→0.25→0.5→1→1.5→2 |
| 10–11 | 逐步加入转弯和 Singer 机动 |
| 12 | 单独取消初速匹配 |
| 13–15 | 距离上限 5→10→30 m |
| 16–20 | 速度上限 4→6 m/s，距离上限 50 m，再转到 10–50 m、2–6 m/s |
| 21–24 | 随机模式、切换间隔、jerk 预算、加速度 3 m/s² |
| 25–30 | 速度 8 m/s、距离 70 m、OU 时间、快速切换、响应时间 |
| 31–34 | 加速度 5 m/s²、速度 10 m/s、距离 100 m、持续 60 s |

每级 2000 万物理步是最大预算。程序先对父模型做两组种子的完整时长筛查，已掌握的级别跳过。训练通过多组当前任务和旧技能/精度门槛后提前升级；否则预算结束后仍失败就停止，并保留结果。

01 级保留 BC/DAgger 初始化，02–04 级可进行带接受检查的教师迁移拟合；05 级起 transfer_dagger_updates=0。PPO 动作从不混入教师动作。初始化通过不是 PPO 改善。

训练混合比例按并行环境数四舍五入，因此小规模检查未必恰好 60/20/20。日志 training_distribution 给出实际比例。保留锚点轮换参与采样，所有锚点参与升级验证。

## 单项运行、继续和迁移

~~~bash
python main.py train --config configs/ctbr/01_hover_small.json \
  --device cuda --out runs/v700_hover42 --seed 42

python main.py train --config configs/ctbr/01_hover_small.json \
  --device cuda --out runs/v700_hover42 --seed 42 \
  --resume runs/v700_hover42/checkpoints/latest.pt
~~~

单项 resume 只允许变更总预算和设备，必须保留相同源代码。在课程中运行的模型还需相同混合任务，应重新执行课程命令恢复。

单独在兼容 v3 模型间转移任务：

~~~bash
python main.py train --config configs/ctbr/05_near_a025.json --device cuda \
  --out runs/v700_a025 --init-from PATH_TO_V3_BEST_CHECKPOINT
~~~

改变时钟、预测模块结构、奖励或 ValueNorm 接口需要新实验，不允许静默加载不匹配权重。

## 独立评估

~~~bash
python main.py evaluate --checkpoint PATH_TO_V3_CHECKPOINT \
  --config configs/ctbr/20_maneuver_pursuit.json --device cuda \
  --seed 110042 --episodes 256 --out runs/test_det.json

python main.py evaluate --checkpoint PATH_TO_V3_CHECKPOINT \
  --config configs/ctbr/20_maneuver_pursuit.json --device cuda --stochastic \
  --seed 110042 --episodes 256 --out runs/test_stoch.json
~~~

--reference 运行相同决策频率的几何控制器；--trace-dir 可以保存轨迹。critic_matches_policy 仅在随机策略评估时为 true。确定性追踪回报不能直接当作随机策略价值真值。

最终脚本根据 curriculum_summary.json 的真实选定模型和任务 profile 评估，支持最后一级被跳过的情况。

## 关键输出

- config.json、curriculum_manifest.json：完整配置及课程/源代码约束。
- metrics.jsonl：PPO、预测器拟合、持续纠偏、实际采样分配。
- initial_validation.json、validation_*.json：完整时长确定性表现与冻结预测误差。
- calibration_*.json：随机策略初始价值校准测量。
- critic_correction_*.json：持续 MC 纠偏的训练/校准验证种子、拟合前后误差、接受或回滚。
- gate/summary.json、promotion_*/summary.json：两组当前任务及旧技能精度门槛。
- phase_skip.json：已掌握任务复用的真实父模型，不代表该级训练过。
- curriculum_summary.json：通过状态、模型来源、选定路径和 SHA256。
- first_anomaly.pt：首个 PPO/采样数值异常的状态和 rollout。
- checkpoints/initial.pt、best.pt、latest.pt、update_*.pt：保存 Actor、critic、预测器、ValueNorm、三个优化器与随机状态。

预测报告中的 learned、constant_velocity、constant_acceleration 为相同有效窗口上的每坐标位置 RMSE，按预测时长分别报告。预测训练误差必须与独立冻结评估分开看。样本跨终止或未来不足时不计入，样本数同时输出。

prediction_fit.accepted 表示预测更新是否通过拟合保护检查；updates 只统计接受的优化步，attempted_updates 包含被回滚的尝试。PPO 与课程日志同样区分“更新后通过”和“相对初始化确有改善”。

## 对照开关

创建新的 profile 和输出目录，其他参数保持一致：

- prediction.enabled=false：关闭历史预测特征。
- prediction.mode=constant_velocity / constant_acceleration / lstm：比较预测模型。
- critic.value_norm=false：关闭价值目标归一化。
- training.mc_calibration_every_updates=0：关闭持续纠偏，保留初始化预热和校准测量。
- policy.decision_steps=1 或 10：比较 1 kHz 和 100 Hz 决策。

频率对照还应保持相同物理 rollout 时长与 BC 数据采样间隔：例如 410×10 ms 对应约 4100×1 ms；bc_stride=2 对应 20。探索相关性、gamma 和 lambda 自动按秒换算；历史窗口覆盖的物理时长也需匹配。分别改变 ValueNorm、MC、预测器与采样分布才能判断各自贡献。

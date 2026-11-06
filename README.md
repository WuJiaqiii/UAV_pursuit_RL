# UAV pursuit RL · v7.0.1

直接 CTBR（总推力与机体系角速度）持续追踪；100 Hz 网络决策、1 kHz 物理积分。保留 v7.0 的因果预测、ValueNorm、完整 MC 校准及模型来源记录，本版重点解决大距离输入、课程精度和模型选择。

## 首轮 A/B/C 实验

| 组别 | Actor 位置误差编码 | PPO 预算 | 初始探索标准差 |
|---|---|---:|---:|
| A | 幅值限制在 5 m，保留方向 | 0，冻结迁移后的策略 | 0.07 |
| B | 同上 | 最多 400 万物理步 | 0.07 |
| C | 同上 | 最多 400 万物理步 | 0.03 |

三组使用同一个已审计 v700 第 15 级 best 初始化，预测器全程冻结。B/C 先练习 45 秒近距混合机动，目标成功率至少 95%、末段 RMSE 不超过 0.5 m，并保护悬停及 0–30 m 接近能力。任务原有 2 m 成功半径保持原定义，精度是额外门槛。

最终分别评估近距机动、0–30 m 接近和悬停，包含确定性与随机策略。额外冻结评估 C 的初始化随机策略，用于区分降低噪声与 PPO 学习的贡献。

## 运行

Python 3.10+、PyTorch 2.1+；按服务器 CUDA 环境安装 PyTorch 后：

~~~bash
pip install -e '.[dev]'
pytest -q
SEEDS=42 DEVICE=cuda bash scripts/run_ctbr_v701_experiments.sh runs/ctbr_v701
~~~

脚本默认读取：

~~~text
runs/ctbr_v700/seed_42/15_distance_30/seed_42/checkpoints/best.pt
~~~

如果检查点放在其他位置，设置 SOURCE_CHECKPOINT。默认计划核对其 SHA256，防止误用 latest 或其他阶段。程序自动迁移权重、重设各组探索标准差、重新预热 B/C 的 critic、训练并评估。

先运行 seed 42；确认后可使用 SEEDS="42 43 44"。完全相同的命令和目录会复用已完成结果并继续未完成组。每个训练种子共用同一个 v700 预训练初始化，比较的是微调随机性。

## 模型与门槛

- Actor 使用有界位置误差，critic 保留真实位置状态。
- 当前课程同时检查成功率和末段 RMSE，失败时优先返回，减少无效旧技能评估。
- best_current.pt 保存任务评分最好模型；best_feasible.pt 仅在满足全部配置门槛时存在。晋级不会覆盖更好的初始化。
- best.pt 是 best_current.pt 的兼容别名，不能据此判断已通过门槛。
- 未达到精度目标也是实验结果，B 不达标仍会执行 C 并生成对照报告。

v701 使用 v4 检查点契约。v700/v3 可以只读评估，保持原有未限幅策略；训练需显式迁移到新的实验目录。v690/v691 不支持直接迁移。

[实验操作与输出](docs/EXPERIMENTS.md) · [架构与兼容性](docs/ARCHITECTURE.md) · [验证记录](docs/VALIDATION.md)

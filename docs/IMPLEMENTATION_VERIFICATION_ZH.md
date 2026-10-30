# V6.4 E1–E5 实施与验证记录

更新模块：场景参数扫描、独立验证集参数标定、传感器观测噪声与延迟、CTBR 延迟、飞行包线补全、CPA/控制时序分析、多算法配对 3D 和 Pareto。

**代码行为验证**：在构建环境执行 `python -m pytest -q tests`，验证结果为 102 passed。完整 E1–E5 `--quick --episodes 1 --calibration-episodes 1` 流程使用包内历史 PPO checkpoint 运行成功，导出 30 行汇总和 31 张实验图。仅 1 回合的数据不能作为任何算法性能优劣的证据。

**保留不变**：原 PPO/Actor-Critic 维数、PPO 训练器、奖励计算、目标模型与动力学积分、首次捕获事件语义、checkpoint 结构、旧的 `benchmark`/`evaluate`/`train` CLI。

**边界**：本版没有实施新的 PPO 课程训练算法、IMM、图像检测、真实 AirSim、风场、真实硬件执行模型。扩展在评估侧完成，相关实验中的噪声与延迟不自动注入训练；如后续需要鲁棒训练，必须另建对照实验与新策略权重。

**发布策略**：原始 V6 附带约 25MB 的旧运行归档为历史结果，源码包去除旧运行缓存并只保留未改动的 2.6MB 历史 `best.pt` 作为载入样例；不将旧结果冒充新训练结果。`SOURCE_MANIFEST.json` 重新计算以匹配当前源码。

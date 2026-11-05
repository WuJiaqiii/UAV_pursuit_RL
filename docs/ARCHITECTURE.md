# v7.0 架构

## 数据路线

main.py → ctbr/curriculum.py 或 ctbr/training.py → CTBRBatch / TaskMixtureBatch → AsymmetricPPO。

- simulation/rigid_body.py：NED 刚体运动与 SO(3) 积分，一阶推力和角速度响应。
- ctbr/environment.py：目标生成、观测、安全判定、奖励、持续追踪指标，以及决策到物理子步的展开。
- learning/prediction.py：因果历史编码、监督标签窗口、独立预测器训练和冻结评估指标。
- learning/ppo.py、buffers.py：条件动作概率、PPO、以奖励单位计算 GAE，以及数值诊断。
- learning/valuenorm.py、ctbr/value_learning.py：价值归一化、输出头重标定和完整 MC 回报纠偏。
- ctbr/mixture.py：当前任务和已掌握任务的新鲜在策略数据采集。
- ctbr/checkpoint.py：模型、预测器、归一化统计、优化器、随机状态和训练来源。

## 控制与时钟

physics.dt=0.001，policy.decision_steps=10，决策周期为 0.01 秒。CTBR 是期望推力及机体系角速度，每次决策保持到下一次。当前模型使用一阶闭环响应近似执行器，默认推力/角速度时间常数为 0.1/0.13 秒；没有显式旋翼、ESC 或惯量/力矩 PID 模型。

每个物理子步都检查终止条件。中途终止的环境立即冻结，剩余子步不再推进其状态、奖励或计数。获得目标半径仅记为一次事件；成功必须安全持续到完整截止时间，满足完整末段窗口的距离占比和相对速度条件。

决策奖励是各物理子步奖励的折扣和，决策 gamma=exp(-decision_dt/discount_time_s)，gamma×lambda=exp(-decision_dt/advantage_time_s)。动作变化惩罚按决策间隔计算一次；动作历史每次决策写入一次。训练预算统计实际活跃物理子步，decision_transitions 单独计数。最后一个批次可能使预算超过目标少量子步。

## 因果历史与预测

默认当前观测为 66 维，保留索引 32 开始的 8 个 raw-action 历史。额外保存 40×10 维目标历史：已观测位置、速度、加速度和有效标记。默认原始存储维度 Actor=466、critic=510。

网络不直接使用绝对目标历史坐标。预测器先将历史位置重基准到最新目标位置，缩放运动状态，用单层 64 维 LSTM 预测未来目标位移。恒加速度预测作为基线，输出层零初始化，网络学习残差。3 个预测点共 9 维拼接到当前观测；因此实际 Actor MLP 输入 75 维，critic MLP 输入 119 维。critic 另外保留已有 44 维特权状态，Actor 不接收这些内部状态。

历史在 reset 时单独清空，缺失帧使用有效标记。未来标签只在 rollout 结束后生成；任何跨终止、跨重置或未完整观察的预测窗口均丢弃。训练顺序是：

1. 固定预测器，采集当前策略数据。
2. 用同一个预测器重算概率，完成 PPO。
3. 用刚才轨迹的已实现未来位置单独训练预测器。
4. 下一轮用新预测器重新采样。

每次监督拟合留出 20% 窗口作为更新保护集，若候选预测误差退化则恢复权重和优化器。该集合来自同一 rollout，可能包含相关窗口，只用于防止明显的坏更新；泛化能力仍由独立种子的冻结轨迹评估。

预测器不接受 PPO 反向传播。检查点包含预测器权重与优化器。可通过 prediction.mode 对照 constant_velocity、constant_acceleration、lstm；或将 enabled 设为 false。

思路参考 [OPEN](https://arxiv.org/abs/2409.15866)。本实现为单追踪器重新实现，没有复制其多智能体 MAPPO 和注意力架构。

## 价值学习

critic 内部输出归一化价值；sample、values_tensor、GAE、诊断和 MC 全部使用还原后的奖励单位。奖励仍采用原有物理时间积分和固定 reward.scale，不因加入 ValueNorm 而改变任务奖励。

ValueNorm 用 beta=0.995 的指数滑动均值和二阶矩，进行初期偏置校正；方差下限 0.01。更新统计后对最后一层进行仿射重标定，保持原奖励单位下输出不变，仅存在浮点舍入误差。这是额外的 PopArt 式输出保真处理；被重标定输出头的 Adam 状态会清空，隐藏层状态保留。

PPO 每轮只更新一次统计，在全部 minibatch/epoch 中固定。旧价值先按同一套新统计归一化，再进行 value clipping；损失为归一化目标上的 Huber loss。优势仍单独标准化。

持续 MC 默认每 16 次 PPO 更新和预算结束时触发：

- 当前 Actor、探索分布和预测器冻结，采集完整随机策略回合。
- 用完整回报拟合 critic，额外采样少见的回合初始状态。
- 另一组种子提前生成校准验证状态和回报。拟合前后使用完全相同的数据比较。
- 全状态与初始状态误差均在容许退化预算内才接受；否则恢复 critic、归一化统计及其优化器。
- Actor、预测器不参与这次纠偏；跨策略的旧 MC 数据不会混用。

参考 [SimpleFlight 的 ValueNorm](https://github.com/thu-uav/SimpleFlight/blob/f5ae8fc3689edc7c982e07cbfd587eeeab72f279/omni_drones/learning/utils/valuenorm.py)。SimpleFlight 使用 GAE 加 ValueNorm；持续 MC 纠偏与输出头保真是本版额外设计。

## 课程与保留

34 个 profile 的参数显式保存在 configs/ctbr；计划记录每次改变的难度轴。第 04 级起采用 2 m 半径、10 秒末段窗口和至少 90% 半径内占比；第 04–33 级时长 45 秒，最后一级为 60 秒。每个验证种子单独达到最低成功率才通过。

机动任务分布依次引入 switchback、turn、Singer，然后独立改变初速匹配、距离与速度，最后增加随机模式、切换速度、加速度及远距范围。response_time_s 显式配置，验证 2×max_accel/response_time_s <= jerk_limit，避免幅度变化隐式改变响应时间。

旧技能锚点使用相同种子建立精度基线；之后既检查成功率，也要求末段 RMSE <= 基线×1.2+0.05 m。两个门槛种子组是验证数据。持续 MC 的独立数据也用于校准选择，均不能代替最终测试集。

课程跳级不复制或伪造训练记录，summary 指向真正复用的 checkpoint。训练和初始化来源单独记录；passed_after_ppo 表示更新后通过但未超过初始化基准，只有实际改善才标记 improved_by_training。不能将通过门槛直接计为 PPO 学习改善。

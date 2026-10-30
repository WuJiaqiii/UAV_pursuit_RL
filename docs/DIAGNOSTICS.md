# 时间尺度、奖励与 critic 诊断

[返回 README](../README.md) · [训练](TRAINING.md) · [评估](EVALUATION.md)

## 先确定接口与单位

物理积分频率、网络决策频率和执行器响应是三个量。当前主线物理步 `h=0.001 s`，每 `K=20` 步决策一次，即 50 Hz；执行器仍有约 0.10/0.13 秒推力/角速度滞后。1 kHz 仿真不等于执行器瞬时响应，也不等于真机满足 1 ms 截止。

rescue 默认按物理时间设置折扣与 GAE：

```text
gamma_tick     = exp(-h / discount_time_s)             # 默认 20 秒
gamma_decision = exp(-K*h / discount_time_s)
lambda_decision= exp(-K*h / advantage_time_s) / gamma_decision
                                                     # 默认 GAE 衰减 3 秒
reward_decision= 0.1 × Σ gamma_tick^i × reward_tick[i]
```

有限任务截止视为真正终止，尾部不 bootstrap；rescue 另外施加默认 25 的未缩放超时惩罚。首事件后奖励与终止观测冻结，reset 不进入上一回合目标。旧 `main.py` 的超时/bootstrap 语义应独立检查，不能沿用这项结论。

## 20 Hz → 1 kHz 的换算原则

| 对象 | 处理 |
|---|---|
| 速度/角速度/推力/倾角上限、执行器时间常数 | 保持物理单位，不乘除 50 |
| 折扣 gamma | 保持物理衰减时，`gamma_new = gamma_old^(dt_new/dt_old)` |
| GAE | 保持 gamma×lambda 对应的物理衰减，而非只看 lambda 数字 |
| rollout、总预算与评估间隔 | 换算为每环境物理秒，另报样本量/更新次数 |
| 学习率、KL、clip、熵系数 | 无统一频率缩放公式，分别对照 |

旧 20 Hz 的 256 步覆盖 12.8 秒；1 kHz 同窗口需要 12800 步。直接 1 kHz CTBR 的 2048 步只有 2.048 秒，而当前 50 Hz rescue 的 256 决策为 5.12 秒。不同入口的步数必须先转换单位。

## 奖励

连续时间/距离/闭合速度和风险项已有 `dt`，进度奖励为距离差，成功失败为一次性事件；不能再把它们统一乘除 50 或 1000。rescue 的整体 `reward_scale=0.1` 是一致数值尺度，价值/回报校准也在同一尺度报告。

动作变化率平方的时间积分为 `||Δu||²/Δt`；旧 `dt*||Δu||²` 表示另一种量，频率迁移时不能混用。当前基础配置动作变化项为零。修改奖励是新实验，应重新标记配置，不能将数值缩放当作控制问题已经解决。

## 探索与内环

执行器会滤除高频独立命令噪声。在线性一阶近似中，`a=exp(-dt/tau)`，输出/输入噪声标准差比为 `sqrt((1-a)/(1+a))`；tau=0.13 秒时，20 Hz 约 0.436，1 kHz 约 0.062。这只是近似，不包含 tanh 与闭环反馈。

检查策略残差、实际推力/角速度和状态变化，而不只看 `log_std`。时间相关探索必须使用匹配的条件概率，不能在动作后加滤波却继续用旧 log probability。rescue 已实现条件高斯支持，旧主线仍为每步独立采样约定。

归一化动作没有接近 ±1，也不证明内环没有限幅；还应检查加速度范数、目标倾角和角速度命令的限幅，以及期望/实际速度误差。

## critic 的检查顺序

1. 对比 V、next_V、TD 残差、原始优势、bootstrap targets 与完整回合 MC 回报，区分真实回报尺度与自举漂移。
2. 核对终止/截断、最终观测、reset、折扣聚合与条件概率重算。
3. 固定策略，用随机执行的独立完整回报检查平均偏差和 RMSE。确定性动作的 V 误差只作参考。
4. 区分训练拟合集与独立集，并按初始/中段/近捕获/长回合分层。当前评估 JSON 的校准只报告初始状态 V(s₀) 对 G₀，不能直接推断整条轨迹的 GAE 误差。
5. 在目标可靠后再测试 critic 数据覆盖、迁移/重建、学习率、拟合预算或其他稳定化；不把高 explained variance 或小 loss 当成控制成功。

随机 MC RMSE 包含回报随机性；阈值应结合系统偏差和重复采样判断。只把每 batch return 标准化，却未同步变换 V、bootstrap、保存恢复，会改变价值含义。

## 当前 rescue 的证据文件

`metrics.jsonl` 的 `ppo_update` 含数值分布、终止事件、梯度、更新前概率误差与实际更新计数；`critic_pretrain`、`validation` 事件记录 MC 拟合和校准。首次异常保存 `first_anomaly.pt`，包含更新前模型/优化器、随机状态及 rollout。该包用于重放诊断，不是 `--resume` checkpoint。

普通阈值越界会保存证据，非有限输入/损失/梯度会中断。阈值不是价值/回报裁剪边界。统计分位数采用有限采样，均值/极值使用有限有效数据；查看日志里的实际样本数。

## 旧 Trainer 诊断

旧路线的独立 JSON 配置由 `PURSUIT_DIAGNOSTICS_CONFIG` 指定，不改变旧训练配置哈希：

```bash
PURSUIT_DIAGNOSTICS_CONFIG=configs/diagnostics/numerics.json \
python main.py train --config configs/basic_v67/s0_standard_pure.yaml \
  --device cuda --simulation-backend torch --seed 42 \
  --total-steps 20000000 --out runs/legacy_diagnostics
```

上述是长训练，输出目录须为空。旧输出为 `diagnostics.jsonl`、`train.jsonl`、`critic_calibration.jsonl` 和 `diagnostics/first_anomaly_*.pt`。完整 MC 校准默认关闭，示例配置会启用额外开销；API 可传 `Trainer(..., diagnostics={...})`。

`configs/diagnostics/` 保留 rollout8192、3秒 GAE、低 critic LR 及组合消融。它们属于旧路线，不是已证明有效的修复，也不控制 rescue。增大 rollout 会同时改变每轮样本数和优化 minibatch 数，解释时须报告这些变化。

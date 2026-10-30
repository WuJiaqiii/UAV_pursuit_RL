# V6.8.0：恢复可学习的追踪策略

本版本新增独立的恢复训练入口。推荐先运行 **2 米捕获、50 Hz 速度策略 + 1 kHz 几何内环**。原 V6.7.2 的 `main.py` 和训练方式保留，新 checkpoint 请用 `scripts/rescue.py`。

## 先直接验证随包模型

在解压后的项目根目录、原训练服务器的 Python 环境中运行：

```bash
python scripts/rescue.py evaluate \
  --checkpoint pretrained/velocity_50hz_2m_ppo20m_seed42.pt \
  --profile configs/rescue/velocity_50hz_2m.json \
  --device cuda --episodes 256 \
  --out runs/v680_pretrained_2m_eval.json
```

CPU 环境改为 `--device cpu`。模型自带完整配置，checkpoint 中记录的原机器基础配置路径不需要存在。

本地结果、原始评估和训练曲线见 `LOCAL_VALIDATION.md` 与 `local_validation/`。本地测试使用 CPU；尚未在你的 CUDA 服务器上复测。成功含义沿用原任务：在失败或截止时间前，首次进入指定捕获球；不等同于持续保持误差。

## 从头执行恢复训练

```bash
# 默认 2 米任务，先做 BC/DAGGER 和 critic 初始化，再训练约 200 万物理步。
DEVICE=cuda bash scripts/run_rescue.sh

# 对相同目录续训至累计 2000 万物理步；不会重复 BC。
DEVICE=cuda PPO_STEPS=20000000 bash scripts/run_rescue.sh

# 初步有效后，扩展至三个训练种子。
DEVICE=cuda SEEDS="42 43 44" PPO_STEPS=20000000 bash scripts/run_rescue.sh
```

可用环境变量：`PYTHON`、`DEVICE`、`PROFILE`、`RUN_ROOT`、`SEEDS`、`PPO_STEPS`、`EVAL_EPISODES`、`TEST_SEED`、`MIN_VALID_SUCCESS`。预算指累计 PPO 物理步；不包括示教、MC 和评估轨迹。64 环境、20 个物理步一决策时，预算向完整批次取整。

脚本依次完成初始化、PPO、`bc/best/latest` 的同种子独立测试，输出 `comparison.json`。默认要求最新阶段验证成功率至少 95%；BC 初始化单独有 80% 下限，低于下限即保存初始化模型并停止投入 PPO 预算。测试集不用于这个自动停止条件。若中断发生在首次 `latest.pt` 生成之前，保留日志并换一个 `RUN_ROOT` 重启初始化。

断点恢复保留模型、优化器、随机状态及已完成更新计数；未完成回合重开，不承诺逐步完全一致的轨迹。只允许保持配置并延长预算；改变任务、动作单位、频率或其他训练参数请创建新实验。

## 可用方案

| 配置 | 神经网络决策频率 | 执行器/内环更新 | 初始化 | 用途 |
|---|---:|---:|---|---|
| `configs/rescue/velocity_50hz_2m.json` | 50 Hz | 1 kHz | BC + 一轮 DAgger | 推荐，直接优化 2 米捕获 |
| `configs/rescue/velocity_50hz.json` | 50 Hz | 1 kHz | BC + 一轮 DAgger | 5 米任务，与旧诊断任务对照 |
| `configs/rescue/ctbr_1khz_bc.json` | 1 kHz | 1 kHz | BC + 两轮 DAgger | 保留直接 CTBR 的路线 |
| `configs/rescue/ctbr_20hz_bc.json` | 20 Hz | 1 kHz | BC + 两轮 DAgger | 低频直接 CTBR 对照，尚未长程实测 |
| `configs/rescue/velocity_50hz_from_scratch.json` | 50 Hz | 1 kHz | 无 BC | 纯 PPO 消融，尚未验证收敛 |

例如：

```bash
PROFILE=configs/rescue/ctbr_1khz_bc.json DEVICE=cuda \
  PPO_STEPS=2000000 bash scripts/run_rescue.sh
```

主路线的三个输出为 NED 坐标系期望速度，比例分别为 15、15、5 m/s。内环只使用自身速度和姿态，把速度误差转成限幅加速度，再生成 CTBR。它不读取目标位置来替神经网络追踪。原机体推力、角速度、执行器时间常数及域随机化保留；新增内环加速度限幅 8 m/s²、目标倾角上限 35°，留在原失败边界以内。

直接 CTBR 路线每毫秒重新调用网络。动作采用可观测上一 raw action 的条件高斯过程，相关时间 20 ms；不是采样后偷偷混入滤波噪声。采样与 PPO 更新使用同一个条件均值、方差和 tanh Jacobian。

## 初始化与 PPO 的边界

先用参考控制器收集 4 万个状态标签、做 1200 次 BC 更新，再用网络自己执行时访问的状态补充 2 万个标签、做 500 次更新。参考控制器只提供训练标签和显式基线。随后 PPO 的示教辅助系数为零，评估也不混合参考动作。

`bc.pt` 是零轮 PPO 的初始化；`best.pt` 可能仍是初始化；`latest.pt` 是当前模型。所有评估记录 `checkpoint_origin` 与实际 `ppo_updates`。请同时查看三者，不能把 BC 已成功当作 PPO 从零学会。

## critic 与时间尺度

新观测补充实际推力、角速度、边界余量、剩余时间和上一 raw action。速度策略 actor 为 35 维，直接 CTBR actor 为 36 维；critic 另加 8 维随机物理参数/目标模式信息。

| actor 索引 | 内容 |
|---|---|
| 0–2 | 相对位置 / 30 m |
| 3–5 | 相对速度 / 15 m/s |
| 6–8 | 当前目标加速度 / 基础配置加速度尺度 |
| 9–11 | 自身速度 / 15 m/s |
| 12–20 | 完整旋转矩阵 |
| 21–23 | 实际角速度 / 最大角速度尺度 |
| 24 | 实际推力 / 悬停推力 |
| 25–30 | 六个边界余量 |
| 31 | 剩余回合时间比例 |
| 32 及以后 | 上一次 raw action |

真实部署需要提供/估计这些观测，尤其是推力和目标加速度；critic 的随机参数不用于部署。BC 阶段把 raw memory 输入与对应权重设零，避免只复制上一动作。

先用当前已初始化策略按随机动作采样执行的完整回合 MC 回报训练 critic 600 次，再开始 PPO。每约 200 万物理步重新检查 MC 校准，RMSE 超过 1.5（学习奖励单位）时额外拟合 critic 300 次。PPO 的 KL 提前停止只停 actor，critic 继续完成本轮更新。Huber 回归、梯度裁剪与奖励常数缩放用于数值调节；校准始终报告真实 MC RMSE，不把较小 loss 当成控制成功。

基础 YAML 保留物理步奖励所需的 gamma。新训练实际使用 `RescueBatch.gamma/gae_lambda`，诊断文件记录这些决策步参数；旧 YAML 里的 `n_steps/total_steps/gae_lambda` 不控制此入口。

给定物理步 `h=0.001` 秒、每 `K` 步决策一次：

- `gamma_tick = exp(-h/20)`，`gamma_decision = exp(-K*h/20)`。
- `lambda_decision = exp(-K*h/3) / gamma_decision`，使 GAE 物理衰减时间为 3 秒。
- 一次决策奖励为 `0.1 * sum_i(gamma_tick**i * r_tick[i])`。
- 40 秒任务截止视为真正终止，额外未缩放超时惩罚 25，不继续 bootstrap。潜势奖励的终止势也清零。
- 中途命中/失败时，当回合奖励和最终观测冻结于首次事件；重置不会串进上一回合的价值目标。

原环境大部分连续奖励已乘物理 `dt`，进度项本身是距离变化。它们不再额外乘 50 或 1000；成功/失败奖励保持一次性事件的含义，再整体乘同一个数值尺度 0.1。若以后恢复变化率惩罚，应先明确量纲：积分平方变化率对应 `-c*||Δu||²/Δt`，不同于当前旧式的 `-c*Δt*||Δu||²`。本次基础配置的动作变化项为零。

## 改动位置与验证

新增 `config/rescue.py`、`simulation/rescue_control.py`、`simulation/rescue_environment.py`、`learning/rescue_checkpoint.py`、`evaluation/rescue.py`、`training/rescue.py`，以及两个入口脚本和配置。`learning/ppo.py` / `learning/buffers.py` 增加可选动作维度与条件策略支持，默认四维独立高斯的旧更新已做逐参数一致性审计。`torch_backend.py` 增加可选的有限任务终止潜势处理与跳过无用观测计算。

本地完整测试 156 项通过、2 项跳过；另外补做外部旧实现更新一致性审计 1 项通过。唯一尚未补做的硬件验证为真实 CUDA 项。两个跳过项分别是默认不启用的外部源码审计及 CUDA 测试，外部审计已另外通过。

`scripts/run_rescue.sh` 已做语法检查和真实 CPU 小任务的首次执行、延长预算续训验证；新增测试覆盖控制器目标独立性、条件 log probability、AR 方差/相关性、有限截止、折扣奖励聚合、MC 不依赖错误 bootstrap、KL 不阻断 critic、异常更新前快照和完整训练/评估链路。

原始上传项目本地路径已不存在；本版本基于与服务器诊断实验源码指纹一致的保留副本创建，未覆盖原服务器工程。新旧 checkpoint 的观测和动作含义不同，不能交叉使用。

方法背景：[PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)、[DAgger](https://arxiv.org/abs/1011.0686)。这些文献支持所用方法；本项目的有效性证据是随包本地实验，不是文献对本任务的保证。

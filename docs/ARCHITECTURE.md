# 代码架构

[返回 README](../README.md) · [训练](TRAINING.md) · [评估](EVALUATION.md)

## 当前主线

```text
JSON profile + YAML/extends → config/rescue.py → training/rescue.py
  ├─ simulation/rescue_environment.py：RescueBatch、奖励聚合、有限截止
  ├─ learning/ppo.py：AsymmetricPPO、策略采样和更新
  ├─ learning/buffers.py：TensorRolloutBuffer、GAE 与诊断
  ├─ evaluation/rescue.py：完整回合评估与 MC 数据
  └─ learning/rescue_checkpoint.py：配置契约、保存与恢复

actor观测 → 条件高斯/tanh → 3D期望速度
  → simulation/rescue_control.py：自身速度/姿态反馈 → 4D CTBR
  → simulation/torch_backend.py：1 ms 动力学与目标运动
```

课程编排在 `scripts/curriculum.py`。底层动力学不承担 CLI 分发和实验目录管理。

## 网络与观测

默认速度策略为两个独立 ELU MLP：actor `35→128→128→3`，critic `43→128→128→1`。可训练高斯 `log_std` 与 actor 一起更新；两网络不共享主干，优化器独立。实际结构以 profile/checkpoint 为准。

| actor 索引 | 内容 |
|---|---|
| 0–2 | NED 相对位置 / `position_scale`，默认 30 m |
| 3–5 | 相对速度 / `velocity_obs_scale`，默认 15 m/s |
| 6–8 | 目标当前加速度 / `observation.target_accel_scale` |
| 9–11 | 自身速度 / `velocity_obs_scale` |
| 12–20 | 完整旋转矩阵 |
| 21–23 | 实际角速度 / 配置最大角速度尺度 |
| 24 | 实际推力 / 悬停推力 |
| 25–30 | 六个归一化边界余量 |
| 31 | 剩余任务时间比例 |
| 32–34 | 上一 raw action，重置时为零 |

critic 另加 8 维：阻力、角速度时间常数、推力时间常数、质量比例及目标模式 4 维 one-hot。当前未输入目标内部 OU 状态或 Hybrid 切换计时。部署需要提供或估计 actor 所需的量，包括目标加速度与实际执行器状态。

rescue 的直接 CTBR 路线为 4D 动作，因此 actor 36D / critic 44D；旧 `main.py` 的 31D/43D 编码是另一契约，见[旧接口](LEGACY.md)。

## 控制与探索

默认速度动作按 `[15,15,5] m/s` 缩放，自身速度误差乘增益 2 得到期望加速度，范数限幅 8 m/s²。几何内环目标倾角上限 35°，原机体推力/角速度极限、执行器滞后和域随机化继续生效。

学习策略动作解码器只读取自身状态，不使用目标位置替网络制导。`reference_action()` 使用目标状态，但仅用于示教标签与明确标记的参考控制器评估。

PPO 使用 tanh 高斯与 Jacobian 修正；默认速度策略为独立噪声，直接 CTBR profile 可使用上一 raw action 可观测的条件相关高斯。采样与概率重算使用同一条件分布。BC 屏蔽动作记忆输入；无 BC 的迁移完整复制已有 actor 权重。

## 模块分工

| 目录 | 职责 |
|---|---|
| [config](../pursuit_rl/config/) | 默认配置、继承/校验、只读 Config；`rescue.py` 主线 profile；`challenge.py` 场景与迁移兼容性 |
| [simulation](../pursuit_rl/simulation/) | 动力学、目标、事件、奖励、观测，NumPy/Torch 后端；`rescue_*` 当前接口包装与内环 |
| [learning](../pursuit_rl/learning/) | PPO、buffer/GAE、`rescue_checkpoint.py` 和旧 `checkpoints.py` 两套模型契约 |
| [training](../pursuit_rl/training/) | `rescue.py` 当前训练；`trainer.py` 旧训练 |
| [evaluation](../pursuit_rl/evaluation/) | rescue/challenge 当前评估；benchmark、PN/APN 标定、E1–E5 和离线指标 |
| [control](../pursuit_rl/control/) | 旧接口 PN/APN/PD 与几何制导；当前内环在 `simulation/rescue_control.py` |
| [artifacts](../pursuit_rl/artifacts/) | 日志、实验归属、轨迹、导出和源码指纹；完整 Experiment/session 体系主要由旧 `main.py` 使用 |
| [visualization](../pursuit_rl/visualization/) | 轨迹、动画、指标与实验绘图 |
| [scripts](../scripts/) / [tests](../tests/) | 运行编排与功能回归 |

读代码建议：`scripts/rescue.py` → `config/rescue.py` → `simulation/rescue_environment.py` → `rescue_control.py` → `learning/ppo.py` / `buffers.py` → `training/rescue.py` → 评估与 checkpoint。

## 配置和契约

YAML 基准在 `pursuit_rl/config/defaults.yaml`。支持相对 `extends`，`stages` 按名称覆盖，`curriculum` 整体替换。`Config` 深层只读，`with_overrides()` 生成并校验新对象；数值流程也使用解析后的字典。

主线先读取 JSON profile，再读取其 `base_config` YAML。JSON 中的控制周期、网络、PPO 设置和物理时间常数决定 rescue 的实际参数；旧 YAML 的 `n_steps/total_steps/gae_lambda` 不控制这个入口。

checkpoint 保存完整配置/profile、动作与观测契约、指纹、模型、优化器、随机状态和来源计数。改变动作单位/归一化不能只凭网络形状匹配而迁移。恢复和任务迁移规则见[训练](TRAINING.md)。

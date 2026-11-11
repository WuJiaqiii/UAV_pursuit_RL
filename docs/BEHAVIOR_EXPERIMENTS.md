# 稳态偏差：同状态定位与完整闭环修正

前提：diagnostics_v702 显示匀速误差约为速度乘以 0.6 s，且末段相对速度很小。这是速度相关稳态偏差的证据，不是 0.6 s 真实延迟的辨识。以下入口独立于 PPO；不改变原 BC/PPO、奖励或模型接口。

## 1. 同状态位置扫描

项目根目录，airsim 环境中执行。先测 1 m/s，随后将 --speed 改为 0.5 和 2，每次使用不同输出目录：

```bash
python -m pursuit_rl.ctbr.behavior probe \
  --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
  --out runs/behavior_probe_v703_speed1 --device cuda \
  --speed 1 --warmup 35 --num-envs 8 \
  --offsets -0.5 0 0.5 --horizon 0.3
```

先让策略飞行 35 s，再复制同一个环境状态。沿目标速度方向移动追踪机位置，正 offset 表示增加落后距离；目标位置/速度、动作记忆和执行器状态保持原值，观测重新计算。目标预测历史保持不变，因为目标没有被移动。

对每个 offset 分别从相同状态开始运行策略和 teacher，闭环推进 0.3 s。输出 probe.json 含动作、实际执行器状态、Actor 位置特征、raw 动作记忆、沿目标速度方向的速度增量、最终距离及终止状态。若分支终止，实际时长可能短于名义窗口。可用 --horizon 0.1/0.5 复核时间窗口敏感性。CPU/GPU、配置和初始模型记录在 manifest.json。

判断：

- offset=0 时，teacher 加速追近、策略维持落后：控制映射差距得到直接证据。
- 增加落后距离后策略修正几乎不变：位置反馈弱、限幅或其他特征抵消值得优先检查。
- 策略能修正但出现持续过冲：检查姿态/执行器与闭环阻尼。
- 一个时刻/一个航向不能确定唯一根因；改变 --train-seed、速度和预热时长复核。

当前不包含物理配平状态求解、输入 Jacobian、单独屏蔽预测特征或任意旋转反事实实验；避免将不一致输入扫描误当闭环证据。

## 2. 完整回合 DAgger 修正

先单轮、小学习率：

```bash
python -m pursuit_rl.ctbr.behavior distill \
  --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
  --out runs/behavior_dagger_v703 --device cuda \
  --episodes 32 --num-envs 8 --rounds 1 --updates 500 \
  --lr 0.00003 --stride 5 --validation-episodes 32 \
  --train-seed 710042 --validation-seeds 810042 910042
```

每轮在原机动任务上执行完整策略回合：安全失败仍正常终止，不强制每 0.5 s 重置。每 5 个决策采样当前完整观测和 teacher 动作；行为始终来自确定性策略，不混入 teacher 动作。无随机化匀速任务用于验证，本轮不用于拟合。

仅训练 actor，预测器、critic、ValueNorm 和 log_std 冻结。损失直接比较 tanh(rho*历史raw动作 + (1-rho)*actor输出) 与 teacher 动作，保留实际动作记忆路径。预测特征由冻结预测器计算并缓存。已接受轮次的数据与本轮数据聚合拟合；拒绝候选时也丢弃本轮数据。每轮新建模仿优化器，不沿用 PPO 动量。当前 loss 为标准化动作空间 MSE，不代表物理推力/角速度误差等权；查看 probe 中各动作维度差异。

验证包括原机动、0.5/1/2 m/s 匀速、悬停和 0–30 m 接近。每个任务/验证种子：原成功率与联合精度成功率不能降低；末段 RMSE 不超过前一已接受模型的 1.1 倍加 0.02 m；原机动平均 RMSE 需改善超过 0.01 m。只有全部通过才接受，否则恢复 actor。该门槛较保守，被拒绝不等于模仿完全无效，需查看 round_1.json 的逐任务原因。

保留 baseline_validation.json、每轮候选完整结果、summary.json 和 selected.pt。没有候选通过时 selected.pt 保留初始 actor，origin 为 unchanged_behavior_baseline；不能只看文件存在就认为修正成功。验证种子用于选模型，不作为最终独立测试。

## 3. 独立验收与后续 PPO

使用 diagnostics 入口，在相同新种子上比较 A 与修正模型：

```bash
python -m pursuit_rl.ctbr.diagnostics \
  --checkpoint A=runs/ctbr_v701/seed_42/initializations/A.pt \
  --checkpoint D=runs/behavior_dagger_v703/selected.pt \
  --out runs/behavior_test_v703 --episodes 64 \
  --seeds 1010042 1110042 --device cuda --stochastic
```

优先验收 1/2 m/s 的逐回合 0.5 m 联合精度、原机动误差和悬停安全。确定性改善不保证随机策略改善，因此两种策略均需测试。若通过，再以 selected.pt 作为新实验的 --init-from，现有训练器会重新校准 critic。不要 --resume 此文件：它是行为初始化，未保存可续训的环境进度或 DAgger 优化器状态。

若完整模仿有效但接 PPO 后退化，才进入奖励/优势消融；若模仿动作误差下降但闭环不改善，应先查状态覆盖、误差分布与闭环稳定性。暂不加入 0.6*v 的位置补偿。

所有输出目录要求为空。原模型、原 runs 目录和 main 分支不被覆盖。完整长时训练需在审阅后执行。

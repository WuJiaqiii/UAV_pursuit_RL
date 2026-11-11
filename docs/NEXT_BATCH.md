# 下一批实验：先定位水平滞后，再测训练预算

本分支以 v7.0.1 为基底，新增诊断和配置，不更改动力学、动作映射、PPO、奖励、原有成功率或模型输入输出契约。目录名 ctbr_v702 表示实验批次；CLI 和 checkpoint schema 不升版，现有 v701 模型直接可读。

## 批次 1：冻结模型配对诊断（先执行）

在项目根目录、airsim 环境执行：

```bash
conda activate airsim
python -m pursuit_rl.ctbr.diagnostics \
  --checkpoint A=runs/ctbr_v701/seed_42/initializations/A.pt \
  --checkpoint B_latest=runs/ctbr_v701/seed_42/B/checkpoints/latest.pt \
  --checkpoint C=runs/ctbr_v701/seed_42/C/checkpoints/best_current.pt \
  --out runs/diagnostics_v702 --episodes 64 --seeds 310042 410042 \
  --device cuda --stochastic
```

默认测试固定速度 0.5/1/2 m/s 匀速及原近距机动任务。匀速组匹配初速度，关闭初始姿态/速度扰动和动力学随机化；距离、时长、精度阈值继承配置。机动组保留原配置，因此两组不能直接归因为“只有加速度不同”。如需单因素对照，应另存任务配置，只改变加速度。

每个任务/种子下 teacher 和所有模型使用相同环境配置、环境数及初始状态摘要；模型同时测确定性/随机策略。不要在同一比较中改变 num_envs，它会改变场景序列。新输出目录必须为空；中断后使用新目录，当前入口不支持缓存续评。

输出包含完整配置、模型 SHA256、源码指纹、逐回合数据和 summary.json：

- tail_axis_rmse_m：逐回合 X/Y/Z 末段 RMSE，按物理 tick 精确累积；summary 为完整 tail 回合的均值。
- tail_axis_mean_error_m：target-pursuer 的有符号误差。
- tail_velocity_projection_s：误差沿速度投影除以速度，正值表示落后；只计速度≥0.2 m/s 的 tick，并记录样本数。它不是系统延迟辨识，静止目标为 null。
- joint_precision_success_rate：安全成功、完整 tail 且该回合位置 RMSE 达标的联合比例，失败回合计零。
- tail_position_rmse_p95_m：完整 tail 回合的 P95；同时查看 full_tail_episodes，避免幸存者偏差。

先检查：teacher 是否逐场景通过 0.5 m；策略匀速误差是否随速度增长；随机与确定性的差距是否主要在 Z 轴。不要只看原有 success_rate。

## 批次 2：预算对照（诊断后再执行）

保留原 A/B/C 配置、replay、优化器设置和预测器冻结；将 B/C 上限增至 4000 万物理步。该计划从已有冻结 A 重新初始化，不是接着 latest 续训，便于和原 400 万实验比较。保留晋级提前停止，4000 万是上限，需报告实际更新数。

```bash
python main.py experiment \
  --plan configs/ctbr_v702/budget_plan.json \
  --source-checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
  --out runs/budget_v702 --seeds 42 --device cuda
```

计划锁定本次审计 A.pt 的 SHA256；不匹配会拒绝执行，不能悄悄更换初始模型。最终测试种子偏移改为 610000；验证种子保持原设置，以比较学习曲线。原 400 万结果使用了不同最终测试种子，不能直接做逐回合配对；应通过批次 1 入口将旧/新模型在相同新种子上重评。

训练每约 200 万步有验证文件。比较 initial、各 validation、best_current 和 latest，防止“选回初始化”掩盖训练变化。若 1200 万步左右仍在约 1 m 平台，可先停止并分析，不必机械跑满。确认单种子趋势后再运行 --seeds 42 43 44；这些是共同预训练初始化的微调种子。

改动后的 source_fingerprint 不同，不能 main.py train --resume 旧源码的运行；使用新实验目录。不要覆盖原 runs/ctbr_v701。

## 后续分支条件

1. 匀速也明显落后、teacher 达标：优先比较完整闭环 teacher 模仿与原 BC。现有 transfer_dagger_updates 能做短片段模仿，但仍用旧的 0.5 s 采样/删减特征流程，不能充当完整闭环消融；本分支尚未实现此训练变体。
2. 匀速达标、机动不达标：在相同随机化/初始状态下逐级增加目标加速度，再分析动作响应和前馈。
3. 延长 PPO 有稳定收益：先保留算法，扩充预算及独立种子。
4. 延长仍停滞：另开近距奖励消融，并处理 reward interface 与 critic 重新校准；不能仅改 JSON 奖励再绕过 init-from 契约。

暂不同时修改推力映射、积分器、网络规模和奖励。teacher 在同环境约 0.20 m，首先定位策略与 teacher 的差距。

## 验证

```bash
python -m pytest -q
```

新增测试覆盖物理 tick 轴向统计、选择性 reset、失败回合的联合精度、固定速度配置和批次端到端输出。完整长训练未在提交前执行。

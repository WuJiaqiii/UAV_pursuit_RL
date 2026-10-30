# 实验操作与判读

## 正式实验

先运行 README 的 smoke，确认 CPU/GPU 依赖与保存流程。它只有 0.12 秒，不能证明悬停成功。再运行前两个基础阶段，查看完整 20/25 秒任务与独立种子结果，最后扩展到全部课程。

```bash
python main.py curriculum --device cuda --seed 42 --out runs/foundation42 \
  --phases 01_hover_small 02_hover_recovery

# 同目录继续全课程；已完成阶段保留原更新与优化器。
python main.py curriculum --device cuda --seed 42 --out runs/foundation42
```

课程默认以独立门槛种子要求当前任务成功率≥90%，并回测已经完成的悬停、较快平飞和直线追踪锚点。只有通过已存在的锚点才会继续。门槛数据参与了课程决策，属于验证数据；最终测试应使用新种子。32/64 个样本只能作为推进门槛，最终比较建议至少 3 个训练种子，各使用 256 个以上测试回合并给出区间估计。

每阶段预算至少应覆盖每环境两个完整截止时间，课程入口会拒绝低于该值的预算。默认 2000 万 / 32 × 0.001=625 秒/环境。BC、MC、验证额外计数，不从 PPO 预算中偷换。若训练结束仍有回合未完成，它们会作为 `unfinished_episodes` 报告，不进入成功率分母。

## 单阶段、迁移与恢复

```bash
python main.py train --config configs/ctbr/03_goto_hold.json --device cuda \
  --init-from runs/foundation42/02_hover_recovery/seed_42/checkpoints/best.pt \
  --out runs/goto42 --seed 42

python main.py train --config configs/ctbr/03_goto_hold.json --device cuda \
  --resume runs/goto42/checkpoints/latest.pt --out runs/goto42 --seed 42 \
  --physics-steps 30000000
```

迁移加载兼容 actor 与 critic 权重，优化器重新建立，并以新任务 MC 回报重新校准 critic。初始探索标准差按新配置建立。动作/观察、物理、奖励与探索时间常数必须一致；旧版 checkpoint 一律拒绝。需要改变这些定义时应新建模型与实验，不能默默重解释旧权重。

恢复保留权重、优化器、PPO 计数、Torch RNG 与最佳记录，允许改变设备和增加总预算。恢复要求同一 Python 源码指纹与其他配置，且必须在原输出目录。**仿真环境与环境 RNG 没有序列化**：恢复会重新开始未完成回合，因此不承诺逐位复现中断前的连续轨迹。正常保存不会重置环境，长回合可跨多个 rollout 持续运行。课程重新进入时还会核验上一阶段最佳 checkpoint 的文件哈希，防止串错来源。

## 模型与指标

每阶段输出：

| 文件 | 用途 |
|---|---|
| `config.json` | 合并默认值后的完整配置 |
| `metrics.jsonl` | BC/DAgger、PPO、数值诊断、验证事件 |
| `critic_warmup_heldout.json` | 预热后的独立随机策略 MC 检验 |
| `initial_validation.json` | PPO 前初始化的固定种子成绩 |
| `validation_*.json` / `calibration_*.json` | 更新后的确定性成绩与随机策略 critic 检验 |
| `checkpoints/initial.pt` | 初始化完成后的对照模型 |
| `checkpoints/best.pt` | 包含初始化在内的最佳模型 |
| `checkpoints/latest.pt` | 最后一次 PPO 更新后的模型 |
| `checkpoints/update_*.pt` | 按配置保存的中间更新 |
| `training_summary.json` | 独立交互计数、来源、已完成及未完成回合 |
| `first_anomaly.pt` | 首次数值异常的原始 rollout 与更新前状态，仅异常时出现 |
| `gate_current.json` / `regression/*.json` | 当前课程与历史技能保留门槛 |

选择模型时先比较持续成功率，再比较安全到达完整截止时间的比例，最后比较末尾位置误差；提前崩溃不能靠“末尾样本为零”获得优先。相同前两项下，误差至少改善 max(1 cm, 原误差的 1%) 才替换最佳模型。

`best_origin` 为 `bc_initial`、`transfer_initial`、`transfer_dagger_initial` 或 `ppo`。最佳模型仍来自初始化时，即使训练确实执行了很多更新，也只能说“PPO 尚无可观察收益”。课程总结同时保存来源、哈希与被选模型的本阶段 PPO 更新数；总训练更新数应查看 `training_summary.json`，不能使用最佳模型保存时的计数代替。

## 冻结评估

```bash
python main.py evaluate --device cuda --checkpoint runs/goto42/checkpoints/best.pt \
  --seed 110042 --episodes 256 --out runs/goto42/test_det.json \
  --trace-dir runs/goto42/traces

python main.py evaluate --device cuda --checkpoint runs/goto42/checkpoints/best.pt \
  --seed 110042 --episodes 256 --stochastic --out runs/goto42/test_stochastic.json
```

可以用 `--config` 显式评估另一种兼容任务，或用 `--reference` 单独评估几何参考控制器。参考控制器从不与学习策略混合。轨迹文件为首批前两个样本的 20 ms 采样 NPZ，包含位置、动作与有效掩码；完成后的 padding 必须按掩码忽略。事件判定及误差统计仍在每个 1 ms 物理步进行。

比较 initial/best/latest 与参考控制器时保持配置、种子、episodes、num-envs 相同，核对 `initial_state_digest` 一致。环境随机生成器独立于策略采样；评估还会保存和恢复 Torch RNG，避免验证改变随后的训练随机序列。改变批大小会改变随机序列，不应继续声称同一测试集。

## 如何判断下一步

- 几何参考控制器也失败：先检查控制可行性、飞行约束与任务难度，避免直接扩大 PPO。
- BC 在完整基础课程中失败：先增加教师数据覆盖、检查动作单位和状态缩放；短回合成功不能替代持续悬停。
- 初始策略合格而 PPO 退化：比较三份模型、条件 logprob 重算误差、KL、动作变化与实际执行器，核对首次异常快照。
- critic MC 偏差持续显著：使用独立随机策略回报判断；增加完整 MC 样本或 critic 预热更新，必要时减小 critic 学习率。高机动未来随机性会带来不可消除的回报方差，RMSE 无法保证降至零。
- 当前任务通过但锚点退化：课程停止；回到上一阶段权重，减小 actor 学习率或增加该阶段数据预算，再重新验证。

本分支移除了旧评估、ZIP/补丁分发脚本及旧接口专用测试，替换为新契约下的回归测试。历史结果不与新的持续追踪成功定义直接比较；不能把旧 95% 首次捕获率与新版完整末尾保持成功率等同。

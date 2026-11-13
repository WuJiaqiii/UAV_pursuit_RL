# v708 配平诊断与恢复样本对照

v707 的两组第 10 次更新候选通过了六任务验证。1 m/s 的达标数从 A 的 6/64 提高到 12/64 和 13/64，但同状态探针仍落后约 0.60 m。v708 先诊断零误差状态，再检验在配平附近补样本的效果。

## D0/D1/D2 配平、阻力与输入敏感性

```bash
conda activate airsim
python -m pursuit_rl.ctbr.trim \
  --checkpoint A=runs/ctbr_v701/seed_42/initializations/A.pt \
  --checkpoint near10=runs/staged_v707_near_zero/update_0010/candidate.pt \
  --checkpoint recovery10=runs/staged_v707_recovery/update_0010/candidate.pt \
  --device cuda --out runs/trim_v708_1mps \
  --speeds 1 --drag-factors 0 0.5 1 --scenarios 8
```

同种子生成匀速场景，姿态和推力按实际阻力配平。速度匹配，设置沿程误差 −0.2、0、0.2、0.5、0.8 m。正值表示落后。固定配平动作的 0.3 s 自检要求每个场景的速度和距离变化分别小于 0.005 m/s 和 0.005 m。通过自检后，同起点比较 teacher 与各策略的 0.1、0.3、1.0 s 响应。配置、校验值、起始姿态、逐轴指令、实际响应和终止原因随结果保存。

D2 保存位置误差、相对速度与自身速度沿程扰动的中心差分，并记录将配平动作历史清零后的动作变化。扰动仅用于离线敏感性分析。阻力档位同时改变配平姿态和推力，结果描述策略对重新配平后状态的敏感性。零误差配平状态与自然飞到末段的 A 探针分别报告。

通过 1 m/s 自检后可用新输出目录增加 `--speeds 0.5 1 2`。可选 `--full-episodes 32` 在各阻力档记录标准匀速完整回合，同时包含几何 teacher 基线。此处 teacher 使用匹配的名义阻力，参数失配对照留待后续。

## P0/P1 相同预算训练

```bash
for arm in recovery trim_recovery; do
  python -m pursuit_rl.ctbr.staged_behavior \
    --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
    --device cuda --out "runs/staged_v708_${arm}" \
    --data-mix "$arm" --samples 28800 \
    --episodes 32 --cruise-episodes 8 --recovery-episodes 12 \
    --anchor-weight 1 --milestones 10 25 50 --rank-by-precision || break
done
```

P0 复现 v707 的恢复样本。P1 从相同冻结 A 的末段场景构造配平状态，使用 −0.2、0、+0.2 m 落后量，并由 teacher 执行 0.8 s。等量替换 3,600 条训练样本。三种速度、三种误差和两个恢复阶段均分，每桶 200 条。两组均保留相同机动、匀速策略和近零误差样本，使用同一学习率、动作约束和更新预算。样本不足时增加对应回合数，重新使用空输出目录。

## 验收

共享 A 探针用于比较短时纠偏。六任务验收沿用原防退化门槛。用已有 diagnostics 入口在新种子上评估 A 与通过筛选的候选，含确定性和随机模式。匀速报告增加逐回合末段有符号沿程偏差，以及绝对值中位数与 P90，只使用完整末段。核对参与统计回合数与失败原因。1 m/s 的建议修复目标为中位数 ≤0.10 m、P90 ≤0.20 m，且逐种子联合精度提高、保留任务稳定。

本批实现输入敏感性诊断、配平样本对照和匹配阻力下的几何基线。梯度冲突、分任务约束、参数失配及有界学习残差作为后续对照。

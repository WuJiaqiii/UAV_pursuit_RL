# v707 匀速恢复样本对照

两组都从冻结 A 开始，固定 28,800 个样本、学习率 3e-5、动作约束权重 1 和累计更新点 10/25/50。两组复用相同的 14,400 个机动样本和 7,200 个匀速策略样本。恢复组将 3,600 个近零误差 teacher 样本等量替换为恢复样本。

恢复组在每个速度用独立种子让 A 飞到回合的 75% 和 85%。仅使用仍活跃且沿程相对速度绝对值不超过 0.1 m/s 的状态。复制状态并沿目标速度方向设置 0.3、0.5、0.8 m 的落后量，保留横向误差与历史。teacher 从复制状态控制 0.8 s。每速度、每落后量、每时间半段抽取 200 个样本，共 3,600 个。首轮统一使用受控偏移，原始落后量随样本记录，未另设自然状态样本桶。

```bash
conda activate airsim
for arm in near_zero recovery; do
  python -m pursuit_rl.ctbr.staged_behavior \
    --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
    --device cuda --out "runs/staged_v707_${arm}" \
    --data-mix "$arm" --samples 28800 \
    --episodes 32 --cruise-episodes 8 --recovery-episodes 12 \
    --anchor-weight 1 --milestones 10 25 50 --rank-by-precision || break
done
```

若某桶样本不足，增加 --recovery-episodes 并使用新的输出目录。collection.json 记录各来源、配额、抽样索引和恢复样本的初始状态、偏移、回合终止原因、0.8 s 速度变化与闭合距离。若提前终止，记录的是终止时刻的实际变化。

两组使用同一共享探针及闭环筛选。优先按 1 m/s 联合精度达标率提名，同分比较 RMSE，再比较更新次数。候选仍需通过六任务、两个种子的原验收门槛。对验收候选使用 diagnostics 在预留种子 1210042、1310042 上评估确定性和随机策略。重点比较零额外偏移时的沿程提速、逐种子 1 m/s 达标数以及悬停和远距接近。

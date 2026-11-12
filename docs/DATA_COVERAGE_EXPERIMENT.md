# v706 数据覆盖对照

从同一个 A 开始，三个方案使用相同的 28,800 个训练样本预算、Adam 学习率 0.00003、动作约束权重 1，以及第 10、25、50 次累计更新。采样、训练小批次、筛选及完整验证种子固定。

| 组 | 机动策略样本 | 匀速策略样本 | 匀速 teacher 近零误差样本 |
|---|---:|---:|---:|
| mixed | 28,800 | 0 | 0 |
| cruise | 14,400 | 14,400 | 0 |
| near_zero | 14,400 | 7,200 | 7,200 |

匀速部分在 0.5、1、2 m/s 间等分。near_zero 组沿用 cruise 组的一半匀速策略样本，并替换另一半。替换样本来自同速的完整 teacher 回合，采样时三维距离不超过 0.2 m。由于替换同时改变了访问状态和行为控制器，该组用于检验这种数据配方的效果。各桶无放回抽样，样本不足时会报出具体桶及缺口，可增加对应的 --cruise-episodes 或 --episodes 后重跑该组。

```bash
conda activate airsim
for arm in mixed cruise near_zero; do
  python -m pursuit_rl.ctbr.staged_behavior \
    --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
    --device cuda --out "runs/staged_v706_${arm}" \
    --data-mix "$arm" --samples 28800 \
    --episodes 32 --cruise-episodes 8 \
    --anchor-weight 1 --milestones 10 25 50 || break
done
```

每个更新点的 shared_probe.json 记录从冻结 A 在 1 m/s 匀速下飞行 35 s 得到的 8 个状态。候选与 teacher 从相同状态及相同偏移开始，推进 0.3 s。比较不同输出目录时先核对 initial_state_digest，再按 slot 和 added_along_error_m 配对。先看零偏移时沿程提速和末端距离，再看负偏移下是否减弱修正。collection.json 记录实际来源、配额、采样种子和回合统计。

筛选和完整验证沿用 v704 的规则。每组从快速筛选中提名 1 m/s 末段 RMSE 最低且成功率未下降的候选，随后完成六类任务、两个种子的验证。重点检查 1 m/s 逐回合达标数、倾角终止、悬停和接近能力。通过原门槛的候选可用 diagnostics 在新种子 1210042 和 1310042 上配对独立验收。各组的训练张量、候选权重、逐次损失和筛选轨迹均保留。

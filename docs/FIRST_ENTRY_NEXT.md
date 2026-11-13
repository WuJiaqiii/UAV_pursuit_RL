# 首次进入任务的第二批实验

静止任务的配对评估中，A 和 teacher 都为 64/64。平均首次进入时间分别约 1.283 s 和 1.285 s。400 万步静止训练的验证始终为 64/64，最终选中的仍是初始化。下一批先筛查任务难度，再决定是否投入训练。

| 配置 | 与 1 m/s 匀速基线相比 | 检验能力 |
|---|---|---|
| cruise_05 | 速度降至 0.5 m/s | 较低目标速度下的进入 |
| cruise_10 | 匹配初速，初距 1–2 m | 1 m/s 匀速进入 |
| cruise_10_unmatched | 追踪机从零速度开始 | 初速失配时加速追近 |
| cruise_10_far | 初距增至 3–5 m，匹配初速 | 更远距离接近 |
| maneuver_10 | 匹配初速，初始目标速度 1 m/s，加速度上限 0.75 m/s² | 温和机动时的首次进入 |

所有组保持 0.5 m 球体、12 s 截止时间和关闭动力学随机化。机动组的 1 m/s 是初始速度，后续速度受运动模型影响。不同任务组的状态摘要可不同，各组内部 A 与 teacher 使用相同种子和环境数。

```bash
conda activate airsim
# 绘图机若已安装 Matplotlib，可跳过这一步
pip install -e '.[plot]'

checkpoint=runs/ctbr_v701/seed_42/initializations/A.pt
for task in cruise_05 cruise_10 cruise_10_unmatched cruise_10_far maneuver_10; do
  for seed in 1610042 1710042; do
    dest="runs/entry_next/${task}/${seed}"
    python main.py evaluate --reference \
      --config "configs/first_entry/${task}.json" --device cuda \
      --episodes 64 --seed "$seed" --num-envs 32 --trace-count 8 \
      --trace-dir "$dest/teacher" --out "$dest/teacher.json" || exit 1
    python main.py evaluate --checkpoint "$checkpoint" \
      --config "configs/first_entry/${task}.json" --device cuda \
      --episodes 64 --seed "$seed" --num-envs 32 --trace-count 8 \
      --trace-dir "$dest/A" --out "$dest/A.json" || exit 1
  done
done
```

使用同一个 slot 比较 A 和 teacher。图包含世界平面轨迹、相对目标的平面轨迹、三维距离及倾角和机体角速度。平面圆为球体投影，是否进入以三维距离及成功终止原因判定。终点采用准确的终止帧，其余轨迹约每 20 ms 采样。日志只导出首批指定数量的场景。若失败在其他场景，保留失败 JSON 并单独安排对应场景重放。

```bash
dest=runs/entry_next/cruise_10_unmatched/1610042
for policy in A teacher; do
  python scripts/plot_entry_trace.py "$dest/$policy/trace_seed_1610042.npz" \
    --slot 0 --out "$dest/${policy}_slot0.png"
done
```

先按任务、种子记录进入数、超时和安全失败。耗时与相对速度只在成功回合中计算，比较耗时时同时报告成功分母。若双方进入率都饱和，继续增加一个难度变量或转向连续保持实验。若 teacher 稳定进入而 A 出现较多失败，优先训练该单项配置，保留冻结 A 和 teacher 对照。若 teacher 也普遍失败，先检查初始条件、截止时间和轨迹可达性。

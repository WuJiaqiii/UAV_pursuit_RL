# CTBR 悬停冷启动

检验从新模型开始时，零均值动作初始化和更敏感的近点奖励能否帮助 Actor 学会基本控制。使用 PPO、32 个环境、100 Hz 控制和 1000 Hz 物理积分，关闭 BC 与 DAgger。零均值组只清零 Actor 输出层，保留随机隐藏层、可训练的输出层和探索噪声。后续阶段传递权重。

| 组别 | 初始 Actor 均值 | 奖励 |
|---|---|---|
| random-original | 默认随机 | 原悬停奖励 |
| zero-original | 零 | 原悬停奖励 |
| random-local | 默认随机 | 近点奖励 |
| zero-local | 零 | 近点奖励 |

原奖励沿用上一轮基础课程的进度系数 0.125 和成功奖励 6.0。近点组将位置与相对速度指数项的衰减长度分别从 4 m、3 m/s 缩至 0.5 m、0.5 m/s，并关闭起始在球内即可触发的一次性进入奖励。近点组测量三项调整的合计效果。

| 阶段 | 时长 | 初始距参考点 | 初始倾角上限 | 速度噪声 | 末段位置 RMSE 门槛 | 物理步上限 |
|---|---:|---:|---:|---:|---:|---:|
| 名义悬停 | 3 s | 0 m | 0° | 0 m/s | 0.15 m | 200 万 |
| 小倾角恢复 | 5 s | 0 m | 1° | 0 m/s | 0.20 m | 400 万 |
| 位置与速度扰动 | 8 s | 0–0.15 m | 2° | 0.05 m/s | 0.25 m | 600 万 |
| 原悬停关 | 20 s | 0–0.30 m | 5° | 0.10 m/s | 0.10 m | 2,000 万 |

所有阶段使用 0.75 m 悬停半径、末段至少 90% 时间在球内和 1 m/s 的相对速度 RMSE 门槛。晋级取两个种子，每组至少成功 29/32，同时满足关卡的末段位置 RMSE。名义悬停检查控制平衡点，零均值组可能在第一次 PPO 更新前通过。后续比较请报告实际 PPO 步数和过关阶段。

在仓库根目录运行。先对两个零均值组跑到第 2 关可比较奖励，替换成 random-original 或 random-local 可比较初始化。通过后将 --through 2 改为 --through 4，沿用同一输出目录。

    conda activate airsim
    for cell in zero-original zero-local; do
      python scripts/run_hover_coldstart.py --cell "$cell" --seed 42 --through 2 \
        --out "runs/hover_coldstart_${cell}_s42"
    done

training/curriculum_summary.json 记录晋级及选中检查点。各关目录保留 config.json、metrics.jsonl、initial_validation.json、周期验证和两组晋级报告。比较成功率、末段位置 RMSE、球内时间占比和安全终止。门槛未过时，启动器返回 2 并保留数据。用 --prepare-only 可只生成配置。

在第 2 关实际训练过的组中，可对初始、最佳和最后三个检查点导出前八个场景的轨迹，比较前 2 秒的漂移和姿态。

    run=runs/hover_coldstart_zero-local_s42
    phase=01_hover_tilt
    for model in initial best_current latest; do
      python main.py evaluate --device cuda --seed 97042 --episodes 32 --num-envs 32 \
        --checkpoint "$run/training/$phase/seed_42/checkpoints/$model.pt" \
        --config "$run/profiles/$phase.json" \
        --trace-count 8 --trace-dir "$run/diagnostics/$phase/$model" \
        --out "$run/diagnostics/$phase/$model/report.json"
    done

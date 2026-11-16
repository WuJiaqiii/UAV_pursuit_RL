# 单机三维巡检课程

从现有单机导航检查点继续时，先复测原 CTBR 小扰动悬停、悬停恢复和定点保持。通过的关卡可以跳过训练，并保留为后续模型的能力检查。从随机权重开始时，仍可使用悬停冷启动课程。

三维导航阶段继承原 CTBR 课程逐项改变难度的设计，使用独立于飞行器状态的虚拟导航参考。各关使用相同的横向速度范围、转向率上限、初始扰动和末段通过门槛。

| 关卡 | 本关改动 |
|---|---|
| 01_planar_bridge | 20 s 平面随机参考，水平速度 0.4–1.0 m/s，指令间隔 1–2 s |
| 02_altitude_changes | 加入平滑升降，沿初始高度的偏移最多 ±0.45 m |
| 03_shorter_intervals | 横向指令间隔缩至 0.75–1.5 s |
| 04_longer_flight | 延长至 28 s，前 20 s 的路线与上一关配对一致 |

从仓库根目录运行。基础关和三维关都输出两个种子的逐回合评估。三维关用几何控制作为配对基准，每组 16 回合，至少 15 回合满足完整飞行及末段精度。加入 --stop-on-failure 可在第一处未通过的关卡停下。此入口评估冻结 Actor 并记录结果。

    model=runs/random_survey_s42/adapted/random_survey_bc.pt
    out=runs/survey_3d_curriculum_s42
    python scripts/check_survey_foundation.py --checkpoint "$model" --out "$out/foundation"
    python scripts/evaluate_random_survey.py --checkpoint "$model" --out "$out/validation" \
      --plan configs/survey/progression_3d.json --seeds 5890042 5990042

每关生成一个三维 GIF 和一个静态 PNG。三维图使用东、北、相对高度，垂直轴与水平轴的视觉比例不同。右侧是三维位置误差随时间的曲线。已有的采样轨迹可直接重绘：

    python scripts/render_survey_replay.py \
      --run runs/random_survey_s42/adapted/heldout/03_frequent_changes/4790042 \
      --out runs/survey_3d_curriculum_s42/replays/prior_frequent_changes_3d

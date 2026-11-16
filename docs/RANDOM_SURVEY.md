# 单机随机巡检航线

每个场景根据种子随机抽取速度与转向指令、指令间隔，平滑插值后生成水平导航参考。参考仅由时间和种子决定。几何控制器和冻结 Actor 使用相同初始状态和导航参考，脚本用摘要核对配对。各关均采用全程飞行和末段误差判据。

| 关卡 | 时长 | 指令速度 | 最大转向率 | 随机指令间隔 |
|---|---:|---:|---:|---:|
| 随机转向 | 12 s | 0.7 m/s | 0.35 rad/s | 1.8–3.2 s |
| 随机变速和转向 | 16 s | 0.4–1.0 m/s | 0.50 rad/s | 1.5–2.8 s |
| 高频变化 | 20 s | 0.4–1.0 m/s | 0.50 rad/s | 1.0–2.0 s |

每关每个评估种子包含 16 场景，组内至少 15 个场景通过。单场景须飞完全程，最后 4 秒至少 90% 时间的三维误差在 0.45 m 内，且相对参考速度 RMSE ≤0.6 m/s、位置 RMSE ≤0.32 m。横向和高度误差由约 10 Hz 的全程采样计算。每关生成一个代表场景的 GIF 和 PNG，并保留完整的采样轨迹 NPZ 和逐场景 JSON。若评估集合中存在失败，GIF 选择首个种子中全程位置 RMSE 最大的失败场景。退出码 2 表示有一个或多个 Actor 关卡未过门槛。

先按 [单机航线课程](WAYPOINT_CURRICULUM.md) 生成混合示范检查点与配置，再从仓库根目录运行：

    python scripts/evaluate_random_survey.py \
      --checkpoint runs/waypoint_curriculum_s42/mixed_route_v2/mixed_bc.pt \
      --out runs/random_survey_s42

示范适配继续使用行为克隆，混合旧定点、旧圆弧和新随机航线示范。适配脚本另测旧定点和圆弧。两个新种子用于评估适配后的随机航线：

    python scripts/adapt_random_survey.py \
      --checkpoint runs/waypoint_curriculum_s42/mixed_route_v2/mixed_bc.pt \
      --profiles runs/waypoint_curriculum_s42/profiles \
      --out runs/random_survey_s42/adapted

    python scripts/evaluate_random_survey.py \
      --checkpoint runs/random_survey_s42/adapted/random_survey_bc.pt \
      --out runs/random_survey_s42/adapted/heldout \
      --seeds 4790042 4890042

续跑时使用同一版本、配置和输出目录。用 --no-gif 可在新目录只运行评估及保存轨迹。GIF 需要已有的 Matplotlib 和 Pillow。

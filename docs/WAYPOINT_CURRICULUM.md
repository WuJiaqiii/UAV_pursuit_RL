# 单机预设航线课程

本实验使用固定航点和按时间推进的预设圆弧作为导航参考。任务使用持续跟踪判据，在完整飞行后衡量末段空间误差和速度误差。

| 关卡 | 速度 | 时长 | 起始扰动 |
|---|---:|---:|---|
| 定点飞行 | 0 m/s | 10 s | 距参考点 0.5–1.25 m |
| 巡检圆弧 | 0.7 m/s | 12 s | 同上 |
| 延长圆弧 | 0.7 m/s | 20 s | 同上 |
| 扰动起步 | 0.7 m/s | 20 s | 额外加入 3° 倾角上限和 0.05 m/s 速度噪声 |

每组用相同的仿真初始状态。晋级种子为 3290042 和 3390042，独立测试种子为 3490042 和 3590042，每组 32 回合。每组至少 29 回合同时满足持续跟踪成功和末段 3D 位置 RMSE ≤0.32 m。持续跟踪还要求最后 4 秒内至少 90% 的时间误差 ≤0.45 m，且相对参考速度 RMSE ≤0.6 m/s。

在仓库根目录使用现有 airsim 环境运行：

    python scripts/run_waypoint_curriculum.py --out runs/waypoint_curriculum_s42

定点关比较几何控制、行为克隆、行为克隆加 50 万步 PPO。后续关卡先评估直接迁移，未通过时尝试 PPO 并回测定点。程序用退出码 2 表示有课程关卡未通过，已有报告仍保留。

混合示范实验从定点行为克隆检查点开始，混合定点、圆弧和扰动起步的示范。训练后依次检验四关，通过后换种子复验。末两关用于检验时间延长和扰动起步，混合示范数据包含扰动起步场景。

    python scripts/adapt_survey_arc.py \
      --out runs/waypoint_curriculum_s42/mixed_route_v2 \
      --base runs/waypoint_curriculum_s42/01_waypoint/bc_train/checkpoints/initial.pt

curriculum_summary.json 保存晋级结果，heldout 保存独立评估、逐回合报告及第一批 8 个场景的轨迹。混合示范权重保存为 mixed_bc.pt。

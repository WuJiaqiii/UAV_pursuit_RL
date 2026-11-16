# 从随机权重训练单机三维巡检 Actor

课程由 configs/survey/scratch_bc_plan.json 定义。训练开始时创建新的 CTBR Actor，整个流程使用仿真几何控制器生成示范，由行为克隆拟合 Actor。评估时使用 Actor 自己的动作。Critic 与预测器不参与优化，当前脚本不执行 PPO。

| 阶段 | 场景 |
|---|---|
| 01_hover | 8 s 短时悬停，2° 初始倾角上限 |
| 02_hover_recovery | 12 s 悬停，5° 上限及更大初始速度扰动 |
| 03_waypoint | 从 0.5–1.25 m 外飞到定点并保持 |
| 04_constant_speed_turns | 0.7 m/s 恒速随机转向 |
| 05_random_speed | 水平速度扩展至 0.4–1.0 m/s |
| 06_altitude_changes | 增加平滑升降，相对起始高度偏移最多 ±0.45 m |
| 07_shorter_intervals | 指令变化间隔由 1–2 s 缩至 0.75–1.5 s |
| 08_longer_flight | 回合从 20 s 延至 28 s |

每关从上一关通过的 Actor 权重继续，混合本关示范、上一关示范以及配置标记的旧关示范。两组晋级种子各需至少 15/16 个回合达标，同时用另一组种子复测参与回放的旧关。悬停、定点和三维导航都使用完整飞行及末段精度判据。所有通过的关卡在结尾用两个独立种子重新评估。三维关会保留评估轨迹和 GIF。若关卡或最终评估没过门槛，脚本返回 2，并保留模型和各回合数据。

在仓库根目录运行：

    conda activate airsim
    python scripts/train_survey_from_scratch.py \
      --plan configs/survey/scratch_bc_plan.json \
      --seed 42 --out runs/survey_scratch_bc_s42

可追加 --through 03_waypoint 先验证三个基础关。之后去掉 --through 并沿用相同 --out，即可从已经通过的关卡继续。更换配置、种子或代码版本时使用不同输出目录。追加 --no-gif 可以跳过图像生成。

输出包括 random_initial_actor.pt、各关 actor.pt 和 demonstrations.npz、逐关 gate.json、curriculum_summary.json，以及 heldout 中的最终复测报告。该检查点保留标准 CTBR 模型接口，可用于之后独立的 PPO 对照。

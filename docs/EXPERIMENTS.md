# v701 下一步实验

## 第一轮：单种子 A/B/C

在仓库目录、原训练环境中执行：

~~~bash
SEEDS=42 DEVICE=cuda PYTHON_BIN=python \
bash scripts/run_ctbr_v701_experiments.sh runs/ctbr_v701
~~~

若原检查点位于其他位置：

~~~bash
SOURCE_CHECKPOINT="/absolute/path/to/v700/best.pt" SEEDS=42 DEVICE=cuda \
bash scripts/run_ctbr_v701_experiments.sh runs/ctbr_v701
~~~

默认 source 是已核查的 v700 第 15 级 best，SHA256 为：

~~~text
6725525b6b4b9d49275440789d8982335214315b6e477e334942b57d2322156e
~~~

使用其他模型需要复制计划并明确修改 source_sha256，然后通过 main.py experiment --plan 指定。原始检查点不会修改。

## 实验设置

| 内容 | 配置 |
|---|---|
| 决策/积分 | 100 Hz / 1 kHz |
| Actor 位置误差限幅 | 5 m；critic 保留真实距离 |
| A | 冻结迁移后的策略，标准差 0.07 |
| B | 最多 400 万物理步，初始标准差 0.07 |
| C | 最多 400 万物理步，初始标准差 0.03 |
| B/C 当前任务 | 初始距离 0–2 m、速度 0.5–2 m/s、混合机动 2 m/s² |
| 时长与原任务定义 | 完整 45 s，末段 10 s，2 m 半径及 90% 占比 |
| 额外精度门槛 | 成功率至少 95%，末段位置 RMSE≤0.5 m |
| 预测器 | 保留同一预测器权重，三组均冻结 |
| replay | 平飞与悬停，新鲜在策略采样 |
| ValueNorm/MC | B/C 重新 MC 预热，持续校准保持同一设置 |

本轮固定奖励、网络规模和推力映射。风险分层 MC、平滑推力映射作为后续独立对照。

400 万是每个 B/C 的 PPO 预算，不包含 MC、验证或测试模拟步。每组有合格候选可提前结束；无候选则记录未达标并完成评估，随后继续下一组。不要因为看到 precision failed 就当作程序异常。

## 看哪些结果

experiment_summary.json 汇总全部种子，seed_42/comparison.json 保存完整对照。每组均测试近距机动、0–30 m 接近和悬停。

- primary_test_passed：独立测试中的近距成功率和精度。
- precision_validation_passed：是否在验证中取得 best_feasible。
- local_ppo_updates：实际执行更新数。
- selected_ppo_updates：最终选中模型执行过多少次局部更新；可能为零。
- vs_frozen_A_deterministic：与冻结基线的配对比较。
- vs_own_initial_stochastic：B 与 A、C 与同噪声 C_initial 的随机策略比较。
- lost_successes 和 paired_complete_tail_episodes：防止只看幸存回合误差而忽略安全退化。

A 的 0–30 m 成功率高而精度约 1 m，是预期基线；本轮先改善近距精度，保留远距接近安全性。额外精度门槛不会改变原任务 success_rate 定义。

## 目录

| 文件/目录 | 用途 |
|---|---|
| experiment_manifest.json | 源模型、源码、配置和评估约束 |
| seed_42/initializations/ | A/B/C 显式迁移检查点与回执 |
| seed_42/references/ | 冻结 A 的保留任务验证基线 |
| seed_42/B、C/metrics.jsonl | PPO、ValueNorm、MC 和实际采样分配 |
| B、C/checkpoints/best_current.pt | 任务评分最优候选，包含初始化 |
| B、C/checkpoints/best_feasible.pt | 满足全部晋级条件时才存在 |
| B、C/selection_*/summary.json | 当前精度及保留门槛，含提前返回原因 |
| A、B、C/test/ | 独立确定性和随机策略测试 |
| seed_42/C_initial/ | 同噪声冻结随机基线 |

迁移回执标明标准差是否重设。A 的 critic 未在新策略下预热，critic_matches_policy=false；B/C 校准后再开始学习。

## 中断与多种子

重新执行完全相同的命令和目录，可继续未完成的 B/C，复用已完成迁移与评估。保持源码、设备和回合数一致。修改设置后使用新实验目录。

确认 seed 42 后：

~~~bash
SEEDS="42 43 44" DEVICE=cuda \
bash scripts/run_ctbr_v701_experiments.sh runs/ctbr_v701
~~~

这是共同预训练初始化上的三个微调随机种子，不代表三次独立从零训练。

## 显式迁移与后续课程

独立迁移示例：

~~~bash
python main.py migrate --checkpoint PATH_TO_V700_BEST \
  --config configs/ctbr_v701/B.json --out runs/v701_transfer/B.pt --device cuda
~~~

新版只读评估旧 v700 时自动保留未限幅行为；迁移后才使用 5 m Actor 编码。不能直接 resume v700。

后续可从同契约 v701 检查点运行精度课程：

~~~bash
python main.py curriculum --plan configs/ctbr_v701/precision_plan.json \
  --init-from PATH_TO_V701_CHECKPOINT --out runs/v701_precision \
  --device cuda --seed 42 --episodes 64 --minimum-success 0.95
~~~

该计划先检查悬停并建立保留基线，再依次训练平飞、混合机动、初速恢复及 15/20/30 m 距离，阶段预算 400 万。先审阅 A/B/C 结果再选择续接模型。旧的 34 级计划仍保留，当前推荐从较小的精度计划开始。

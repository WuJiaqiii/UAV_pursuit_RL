# V6.7.0 统一代码与 Stage 0 长周期 PPO 实验指南

## 版本与复现边界（先读）

本包以交付的 **V6.6.1 ZIP** 为唯一可检查源码基线，纳入了用户最近三组实验中被报告但**未上传实际修改源码**的 PD 辅助监督思想。因此本版 PD 辅助代码是**新实现/复刻，不能声称与服务器上的 `stage0_pd_hold` 实现逐字一致**。旧模型 checkpoint 只在网络 contract 匹配时才能初始化；本包不包含用户近期 `best.pt` 权重。历史的 32.81%/10.94%/1.56% 是用户上一批实验数据，不是 V6.7 实测。

重要：本项目的“基础目标跟踪”按**首次进入半径 5 m 的捕获球即成功**定义，**不等于持续伴飞保持距离**。物理仿真 1 ms、PPO 1 ms 更新 CTBR 指令，不代表 HIL 最坏时延符合 1 ms。

## 核心实现

- Actor 31D / Critic 43D、CTBR 4D，不改网络与动作 contract；PyTorch GPU 路径保留。
- 新 `training.aux_pd_coef`（默认 0）控制训练期 **Smooth L1[tanh(actor_mean), PD_teacher_action]** 的系数。教师只在**训练的额外损失**中使用，采样动作仍来自 PPO 分布，old/new log-prob ratio 不变；评估时只执行 PPO Actor，**绝不混入 PD 动作**。
- PD 教师采用位置/速度误差近似控制：`desired_acc = clip(0.2*(target_pos-pursuer_pos) + 0.75*(target_vel-pursuer_vel), -7,7)`，再映射为滚转/俯仰率和推力；属于启发式辅助监督，不是最优专家，也未证明稳定性。Torch/NumPy 双实现单元测试对齐原 PD。
- `aux_pd_schedule = constant | linear`；推荐 `constant`，`linear` 只是为了复现“不合适的按步数退火”作消融，不推荐初期使用。
- `training.challenge_eval_enabled=false` 时节省 Stage 0 一轮重复验证的开销，验证日志中 `challenge_independent=false` 且 challenge 值仅为 stage 值拷贝，**禁止解释为独立测试**。
- `reward.w_progress` 显式配置=8，对照 `0`；计算 1 ms 前后相对距离变化，旧默认 0 保持兼容。
- `configs/basic_v67/` 统一 YAML; `scripts/basic_experiment_matrix.py` 顺序启动长训练并生成 `*.launch.json`；`scripts/eval_basic_matrix.py` 使用独立固定种子复测 best_stage0。

## 安装与验证

在 RTX 3090 Ubuntu 服务器：

```bash
unzip pursuit_rl_v6_7_unified_basic_ppo.zip
cd pursuit_rl_v6_7_unified_basic_ppo
pip install -e '.[dev,plots]'
python -m pytest -q
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
# 无 GPU 也可执行的功能测试：0.02s 仿真，不能证明策略学会跟踪
python main.py train --config configs/basic_v67/smoke.yaml --device cpu --simulation-backend torch --out runs/v67_smoke
```

确认基础 config: `physics.dt=0.001`、`policy_interval_steps=1`、`simulation_backend=torch`、`device=cuda`、`reward.w_progress=8`、`training.aux_pd_coef=0.2`、`training.max_stage=0`；真正生效的解析后 config 会保存在每次运行的 `runs/.../config.yaml`，**以此为准**。

## 具体两阶段实验

**Stage E：容易目标**（只检查基础控制学习）: 距离 10–15 m、目标速度 0–2 m/s、Straight、20 s 时限、5 m 捕获。3 组：Pure PPO / 固定 PD=0.2 / 固定 PD=0.5。3 个随机种子 42、43、44；每组每个种子 **20M** 环境步。`64 env ×2048` 步 Rollout；评估每约 4.19M 环境步、32 回合；正式独立评估 256 回合。

**Stage S：原始标准目标**（与用户之前结果物理任务一致）: 距离 10–30 m、速度 0–5 m/s、Straight、40 s 时限、5 m 捕获。7 组：Pure PPO、PD=0.2、PD=0.5、PD=0.2且无进展奖励、PD=0.2且GAE λ=0.9995、PD=0.2且减半 LR、PD=0.2线性衰减到零。3 种子 × 每运行 **60M** 环境步。

**重要试验设计原则：** E/S 两个任务难度不同，不能将两个组的成功率直接比较成某个超参数的因果效果；S 组内才使用相同初始任务、网络和验证设置。`--init-from` 如使用，应让**所有 S 组共享同一个**来源 checkpoint，且将它们标记为“微调消融”，不可与从零训练直接混合。长期训练记录同一随机种子，不意味着 PPO 采样和 reset 过程严格成对；正式最好报告至少 3 个训练种子上的结果。

## 启动命令（每组命令单 GPU **顺序执行**，不并发争抢显存）

```bash
# 所有实验命令先预览，无实际文件写入
python scripts/basic_experiment_matrix.py --group easy --seeds 42 43 44 --dry-run

# 初始检查：两套主要策略小预算训练（示例 4M 每运行）
python scripts/basic_experiment_matrix.py --group easy \
  --only easy_pure easy_pd02 --seeds 42 --steps 4000000 \
  --out-root runs/v67_easy_pilot --device cuda --backend torch

# 正式容易任务：3种策略 × 3个随机种子 × 20M 环境步
python scripts/basic_experiment_matrix.py --group easy --seeds 42 43 44 \
  --out-root runs/v67_easy_formal --device cuda --backend torch

# 正式标准任务：7种消融 × 3个随机种子 × 60M 环境步
python scripts/basic_experiment_matrix.py --group standard --seeds 42 43 44 \
  --out-root runs/v67_standard_formal --device cuda --backend torch
```

如果只想优先测试 `standard_pd02` 与 `standard_pure`：

```bash
python scripts/basic_experiment_matrix.py --group standard \
  --only standard_pd02 standard_pure --seeds 42 43 44 \
  --out-root runs/v67_standard_priority --device cuda --backend torch
```

将相同已训练权重作为 S 组所有消融的公共起点（需 checkpoint 网络兼容）：

```bash
python scripts/basic_experiment_matrix.py --group standard --seeds 42 43 44 \
  --init-from /absolute/path/to/COMMON_START/checkpoints/best_stage0.pt \
  --out-root runs/v67_standard_common_init --device cuda --backend torch
```

长时间远程 SSH 建议放入 `tmux`，以免 SSH 断线中止训练。注意不能重复使用同一非空 `--out-root` 来更改步骤预算；每个实验以 `run.json`、实际 `config.yaml`、`*.launch.json` 为追溯依据。中断训练如需继续，用本项目原生 `--resume` 和**该运行目录保存的相同 `config.yaml`**，不能通过新实验矩阵覆盖旧运行。

```bash
tmux new -s ppo_v67
python scripts/basic_experiment_matrix.py --group standard \
  --only standard_pd02 standard_pure --seeds 42 43 44 \
  --out-root runs/v67_standard_priority --device cuda --backend torch \
  2>&1 | tee v67_training_console.log
# Ctrl+b, d 可退出 tmux 但不停止训练
```

## 独立评估与验收

```bash
python scripts/eval_basic_matrix.py --runs-root runs/v67_easy_formal \
  --report-dir runs/v67_easy_independent --episodes 256 --seed 990000 \
  --device cuda --backend torch
python scripts/eval_basic_matrix.py --runs-root runs/v67_standard_formal \
  --report-dir runs/v67_standard_independent --episodes 256 --seed 990000 \
  --device cuda --backend torch
```

每运行生成 JSON，汇总 `independent_summary.csv`；推荐检查：独立成功率、速度越界、倾角越界、空间越界、超时比例、有效捕获时间、PPO 的 `approx_kl/clip_fraction/explained_variance/aux_pd_loss/action_std`。最终报告给出三种训练 seed 的均值与区间，不能把 32 回合周期验证当作最后独立测试，也不能把未成功回合的 `mean_capped_time` 当实际飞行时长。

推荐阶段门槛：容易任务独立成功率 ≥80% 再推广至标准任务；标准任务独立成功率达到 ≥80% 且多种子稳定后，再考虑高速 Singer/Hybrid 和 E1～E5。**阈值是工程验收建议，不是已取得的实验结果。**

## 当前尚未验证的部分

- 当前交付环境没有 NVIDIA GPU；只运行了 CPU 上的 PyTorch 后端、测试和极短 PPO 训练。CUDA 长跑吞吐、显存、真实成功率及 1 ms 实时截止不作任何保证。
- 本包没有最近 PD Hold / Pure PPO 的真实改版源码，因此新写的 PD teacher 只与已知旧版启发式 PD 输出对齐，**不能断言复现用户之前的 PD 监督实现**。
- 若要进行“持续跟随目标、保持距离”而不是首次接近捕获，需要重写终止定义与位置/相对速度控制目标；本阶段未做。

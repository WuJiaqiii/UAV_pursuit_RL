# 预设航线 PPO 微调对照

在原有 `scripts.run_waypoint_curriculum` 入口下，用 `ppo-compare` 在单架仿真无人机的预设圆弧航线上检验 PPO 微调。原入口仍可通过 `--out`、`--device` 和 `--seed` 运行逐阶段课程。两个入口共享相同的静态航点和圆弧航线定义。

## 参数和分组

训练航线为 `02_survey_arc`，虚拟参考线速度为 0.7 m/s。训练及评估使用 10 m/s 的仿真地速超限阈值。验证比较 `02_survey_arc` 和 `01_waypoint`，最终测试另包含 `03_long_arc` 和 `04_offset_start`。

| 组别 | 设置 |
| --- | --- |
| `mixed_bc` | 对原有 `mixed_bc.pt` 进行评估 |
| `critic_only` | Actor 学习率为零，并逐更新核对 Actor、预测器和动作标准差 |
| `ppo` | 更新 Actor 和 Critic |
| `ppo_bc` | 标准 PPO 加固定 BC 的动作保持损失，权重 0.1 |

三组更新实验在同一个训练种子下共享同一份预热后的 Actor、Critic 和 Critic 优化器状态。PPO 更新使用原仿真和训练模块，行为采样始终来自当前策略。默认用 32 个环境、410 个决策步/rollout、2048 个样本/minibatch、3 个 epoch，Actor/Critic 学习率分别是 `3e-5` 和 `1e-4`。参数在 [`configs/experiments/route_ppo_compare.json`](../configs/experiments/route_ppo_compare.json)，可用 `--plan` 指定一个副本。

- `pilot` 用种子 42，每组 4 次更新，每次评估，合计 12 次更新。
- `confirm` 用种子 42、43、44，每组每种子 16 次，每 4 次评估，合计 144 次更新。
- 每条验证航线有 64 个配对回合。默认要求完成全部回合、至少 63 个达到联合精度条件，且末段位置 RMSE 相对原始 BC 的增幅不超过 0.5 cm。通过后按圆弧验证成绩选择 checkpoint，未超过原始 BC 时保留原始 checkpoint。

## 启动

在训练仓库根目录切到 `feat/route-ppo-ablation`。若另建工作树，`ckpt` 使用原实验目录的绝对路径。该分支基于 `temp` 的 `07c262e`，原 `mixed_bc.pt` 可以直接加载。运行前应在当前 Python 环境中安装原仓库的依赖。

```bash
ckpt="$PWD/runs/waypoint_curriculum_s42/mixed_route_v2/mixed_bc.pt"
out="$PWD/runs/route_ppo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"

python -m scripts.run_waypoint_curriculum ppo-compare --mode pilot --dry-run
python -m scripts.run_waypoint_curriculum ppo-compare --check-source

python -m scripts.run_waypoint_curriculum ppo-compare \
  --checkpoint "$ckpt" --out "$out/preflight" --preflight

nohup python -u -m scripts.run_waypoint_curriculum ppo-compare \
  --checkpoint "$ckpt" --out "$out/pilot" --mode pilot \
  > "$out/pilot.log" 2>&1 &
tail -f "$out/pilot.log"
```

试跑确认后，将 `--out "$out/pilot" --mode pilot` 改为 `--out "$out/confirm" --mode confirm --final-test`，并换一个日志文件。确认阶段从原始 BC 起点重新训练。独立测试种子为 `3690042` 和 `3790042`，测试结果不用于挑选 checkpoint。预检可以用 `--device cpu`，默认使用 CUDA。`--out` 使用尚未创建的目录。

输出中的 `progress.json` 记录整批进度，`summary.csv` 记录配对指标。各组的 `metrics.jsonl` 同时记录沿用采样 batch 和改用训练 batch 时的更新前 log probability 误差。`selection.json` 中的 `update: 0` 表示选择了原始 BC，`final.pt` 表示末次更新的结果。

## 加载 `mixed_bc.pt` 时提示 first-entry-only

`mixed_bc.pt` 使用 `pursuit-direct-ctbr-checkpoint-v4` 格式。出现 `this first-entry-only build does not load old versions` 时，运行目录加载了另一个版本的 `pursuit_rl/ctbr/checkpoint.py`。只复制新脚本到 `main`，或把新脚本提交单独拣选到 `main`，也会出现此错误。在原训练仓库根目录执行以下命令，另建一个完整工作树并沿用原 checkpoint：

```bash
ckpt="$PWD/runs/waypoint_curriculum_s42/mixed_route_v2/mixed_bc.pt"
worktree="$(dirname "$PWD")/route-ppo-worktree"
out="$PWD/runs/route_ppo_preflight_$(date +%Y%m%d_%H%M%S)"

git fetch origin feat/route-ppo-ablation
git -c core.autocrlf=false worktree add --detach "$worktree" FETCH_HEAD
cd "$worktree"

python -m scripts.run_waypoint_curriculum ppo-compare --check-source
python -m scripts.run_waypoint_curriculum ppo-compare \
  --checkpoint "$ckpt" --out "$out" --preflight
```

`--check-source` 会在加载 PyTorch 前验证源码指纹。若 `route-ppo-worktree` 已存在，可换一个新目录名。该步骤不需要迁移或重新训练 checkpoint。

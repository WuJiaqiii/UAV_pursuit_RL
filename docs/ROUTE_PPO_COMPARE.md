# 预设航线 PPO 微调实验

这批实验研究单架仿真无人机在固定圆弧航线上的 PPO 微调。问题是：在已有混合航线 BC 的基础上，PPO 能否改善圆弧跟踪，并保留静态航点的精度？训练统一从原实验的 `mixed_route_v2/mixed_bc.pt` 起步，沿用 v4 仿真、PPO 损失、奖励和 GAE 时间常数。

## 任务与对照

| 类型 | 航线 | 设置 |
| --- | --- | --- |
| 唯一训练航线 | `02_survey_arc` | 固定圆弧，参考速度 0.7 m/s，12 s |
| 基础能力回归 | `01_waypoint` | 静态虚拟航点，10 s |
| 持续能力测试 | `03_long_arc` | 相同参考速度的长圆弧，20 s |
| 初态扰动测试 | `04_offset_start` | 20 s 圆弧，初始倾角 3°，初始速度扰动 0.05 m/s |

各航线统计最后 4 s，仿真地速超限阈值为 10 m/s。航线和难度定义共用 `scripts/survey_route_profiles.py`。本实验各组使用同一条训练航线，四条航线用于评估能力及泛化。

| 实验组 | 更新的部分 | 用途 |
| --- | --- | --- |
| `mixed_bc` | 无，直接评估初始权重 | 给出精度和稳定性的参照 |
| `critic_only` | 只更新 Critic。逐更新核对 Actor、探索标准差和预测器权重 | 检查采样到评估的一致性，并分离价值学习的影响 |
| `ppo` | 原 PPO 更新 Actor 和 Critic | 测量标准微调的收益与退化 |
| `ppo_bc` | 相同的 PPO，再加权重 0.1 的冻结 BC 动作标签损失 | 检验保留示范监督能否减少微调退化 |

同一训练种子内，训练组共享 Actor、预热后的 Critic 和 Critic 优化器状态。Critic 预热使用 8 个随机策略回合做 96 次拟合，并在另一批 8 个回合上报告价值 RMSE 和偏差。执行动作始终来自当前学习策略。各组沿用相同的种子、预算、学习率和预设航线。

## 预算与评估

训练参数集中在 [route_ppo_compare.json](../configs/experiments/route_ppo_compare.json)。默认使用 32 个并行环境、410 决策步/rollout、minibatch 2048、3 个 epoch，Actor/Critic 学习率分别为 `3e-5`、`1e-4`。

| 阶段 | 训练种子 | 每组每种子更新数 | 整批更新数 | 验证频率 |
| --- | --- | ---: | ---: | --- |
| 预检 `--preflight` | — | 0 | 0 | 只复测现有 BC |
| 流程测试 `smoke` | 42 | 1 | 3 | 每次，采用 2 个环境、32 步 rollout、1 epoch 和每个评估种子 4 回合 |
| 短程 `pilot` | 42 | 4 | 12 | 每次 |
| 确认 `confirm` | 42、43、44 | 16 | 144 | 每 4 次 |

- **筛选。** 静态航点和短圆弧均用种子 `3290042`、`3390042`，每种子 32 回合。脚本核对每条航线的初始状态指纹。两条航线都需完成全部回合、至少 63/64 个回合精度达标、末段位置 RMSE 相对各自的 BC 基线增加不超过 0.5 cm。
- **选模。** 合格的模型按圆弧精度达标数排序，再按圆弧末段位置 RMSE 排序。只有严格改善时才替换原 BC。冻结 Actor 组还会检查同一批回合的确定性飞行指标是否保持一致。
- **最终测试。** `confirm --final-test` 只对验证选中的模型和原 BC 在 `3690042`、`3790042` 上测试四条路线。测试成绩不参与选模。报告每个训练种子的达标率，以及相对原 BC 的末段位置和高度 RMSE 变化。流程测试和单种子短程用于诊断，三种子结果用于判断效果能否重复。
- **诊断。** 每轮更新记录使用采样形状和优化 batch 形状重算的动作 log probability 误差 P50/P99/最大值、KL、clip fraction 和 Critic 指标。若冻结组指标变化，优先检查装载和评估链路。若标准 PPO 退化、BC 保持组稳定，再做跨种子复验。

## 运行

建议在原仓库旁建一个工作树，复用原实验数据。源码自检要求 v4 仿真和训练模块与保存原 checkpoint 的版本一致。不要把脚本单独复制到 first-entry-only 的分支运行。

```bash
stamp=$(date +%Y%m%d_%H%M%S)
ckpt="$PWD/runs/waypoint_curriculum_s42/mixed_route_v2/mixed_bc.pt"
out="$PWD/runs/route_ppo_suite_$stamp"
tree="$(dirname "$PWD")/route-ppo-suite-$stamp"

git fetch origin refactor/route-ppo-suite &&
git -c core.autocrlf=false worktree add --detach "$tree" FETCH_HEAD &&
cd "$tree" &&
python -m scripts.route_ppo_compare --check-source
```

激活已有的 `airsim` 环境。按顺序执行，上一项通过后再启动下一项：

```bash
python -m scripts.route_ppo_compare --mode pilot --dry-run
python -m scripts.route_ppo_compare --checkpoint "$ckpt" --out "$out/preflight" --preflight
python -u -m scripts.route_ppo_compare --checkpoint "$ckpt" --out "$out/smoke" --mode smoke
python -u -m scripts.route_ppo_compare --checkpoint "$ckpt" --out "$out/pilot" --mode pilot
```

短程结果符合预期后，可在同一个终端后台启动完整确认轮：

```bash
nohup python -u -m scripts.route_ppo_compare \
  --checkpoint "$ckpt" --out "$out/confirm" --mode confirm --final-test \
  > "$out/confirm.log" 2>&1 &
tail -f "$out/confirm.log"
```

输出目录必须是新目录。用 `--plan` 指向参数副本，或用 `--device cpu` 做不依赖 CUDA 的预检。原 `scripts.run_waypoint_curriculum ppo-compare` 调用会进入同一套代码。

| 输出 | 内容 |
| --- | --- |
| `plan.json` | 生效参数、源码版本及原始 checkpoint 哈希 |
| `progress.json` | 整批完成更新数、组别和阶段，失败时记录原因 |
| `summary.csv` | 各组各路线的精度达标数、位置和高度 RMSE |
| `critic_warmup_s*.json` | 价值拟合与独立回合上的误差 |
| 各组 `metrics.jsonl` | 紧凑的 PPO 与 logprob 诊断 |
| `selection.json` | 验证选中的 checkpoint，更新 0 表示保留原混合 BC |

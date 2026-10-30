# 冻结评估与对照

[返回 README](../README.md) · [训练](TRAINING.md) · [诊断](DIAGNOSTICS.md)

## 当前训练任务

准备真实存在的 rescue checkpoint。它自带配置，原机器 YAML 路径无需存在：

```bash
python scripts/rescue.py evaluate \
  --checkpoint /path/to/checkpoints/best.pt \
  --device cuda --episodes 256 --seed 1010000 \
  --out runs/frozen_eval/best.json
```

默认确定性执行取条件分布均值；默认无相关噪声的速度 profile 等价于 `tanh(actor(obs))`。加 `--stochastic` 才与训练 critic 的随机策略分布一致。CPU 用 `--device cpu`。

结果包括成功、失败、捕获时间、逐回合记录、初始 V 与完整折扣回报偏差/RMSE。`--capture-radius` 支持显式变更捕获半径；已经在大半径提前结束的轨迹不能反推小半径成功率。

## 复杂场景和参考控制器

```bash
python scripts/challenge.py \
  --checkpoint /path/to/checkpoints/best.pt --device cuda \
  --episodes 64 --num-envs 64 --seeds 1010000 \
  --scenarios baseline singer_a05 singer_a1 singer_a2 dubins_a2 hybrid_a2 \
              distance_30_60 distance_60_100 speed_5_8 speed_8_12 \
              speed_8_12_open combined_moderate \
  --out runs/challenge_new
```

默认评估冻结网络和参考控制器，配对初始条件；参考动作不混入网络策略。`--no-reference` 省略参考组；`--stochastic` 改为随机网络执行。多个评估种子和更多回合会增加完整仿真开销。

| 场景 | 含义 |
|---|---|
| `training_task` | checkpoint 当前训练任务 |
| `baseline` | 原基础直线目标、2 米球、距离 10–30 米、初速 0–5 m/s |
| `singer_a05/a1/a2`、`dubins_a2`、`hybrid_a2` | 改变机动类型与加速度上限 |
| `distance_30_60`、`distance_60_100` | 改变初始距离 |
| `speed_5_8`、`speed_8_12` | 改变初速，实际最大速度另记 |
| `speed_8_12_open` | 扩大目标活动边界，追踪器包线不变 |
| `actuator_lag_150`、`drag_150` | 实际机体参数增至 1.5 倍，内环仍使用名义参数 |
| `combined_moderate` | 多因素组合 |

报告包含训练/评估契约、模型 SHA、初态哈希、目标实际最大速度/加速度、边界修正区域步数和终止机体状态。边界计数不证明边界帮助捕获；参考控制器失败也不证明物理不可达。

## 初始化与 latest 收益核查

根目录脚本只评估，不训练：

```bash
DEVICE=cuda bash evaluate_v681_learning_gain.sh
```

默认读取 `runs/v681_curriculum/09_mixed_speed8/seed_42/checkpoints/` 的 `bc.pt` 与 `latest.pt`，在 seed1000000、64 回合、64 环境下运行初始化/latest × 确定性/随机四组。新结果位于 `runs/v681_gain_audit/<时间>/`，汇总为 `gain_comparison.json`。

```bash
PHASES="08_mixed_distance100 09_mixed_speed8" \
EPISODES=256 SEEDS="1010000 1020000 1030000" \
DEVICE=cuda bash evaluate_v681_learning_gain.sh
```

可指定 `PROJECT_ROOT`、`RESULT_ROOT`、`OUT_ROOT`、`TRAIN_SEED`、`NUM_ENVS`、`PYTHON`。配对比较保持相同环境数；设备/向量化批次改变会改变随机数流，同一个数字 seed 不保证相同初态。seed1000000 已用于诊断，后续最终验证应保留未参与调参的数据。

## 结果解释

- 同时报初始化、best、latest 与真实 PPO 更新来源，best 保留初始化不表示新增训练有效。
- 成功回合均值删去了失败；同时报告失败按截止计入的全回合均值、失败原因与 P90/P95，检查近失和长尾。
- `V(s₀)` 对 `G₀` 的随机 MC RMSE 含回报噪声，不只是函数误差；确定性执行的 critic 误差只作参考。当前 JSON 未保存各回合独立 G₀/V₀ 或逐时刻误差，不能恢复所有配对统计。
- 验证门槛、独立测试和训练种子分开；单 seed、小样本不能证明稳定增益/退化。
- 首次进入捕获球不等于持续伴飞。真机、传感器和硬件截止时延需单独验证。

旧 checkpoint 的 PN/APN、E1–E5 与轨迹可视化见[旧接口](LEGACY.md)，不自动兼容 rescue 契约。

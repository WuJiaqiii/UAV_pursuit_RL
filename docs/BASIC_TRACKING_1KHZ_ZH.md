# V6.6.1 基础目标追踪（1 kHz CTBR）训练与诊断手册

> 本分支保留原 V6.6 GPU/Torch 与 NumPy 双后端、PPO 31D Actor/43D Critic、4D CTBR 动作及 1 ms 决策周期。此阶段只研究 Stage 0 直线目标的**首次进入 5 m 捕获球**，不是长时间编队跟随（持续距离保持）。E1～E5 的原代码仍保留，但不参与本轮训练与评价。

## 1. 本次发现与改动

- **已证实**：原 `curriculum_1khz.yaml` 的 `eval_episodes=16`、最高训练阶段 5 会造成低成功率时的 checkpoint 选择噪声；基础追踪采用 `max_stage=0`、`eval_episodes=64`，并仅评价当前任务。
- **已实现**：新增可关闭的 `w_progress` 距离进展奖励：
  `r_progress = w_progress * (d_before - d_after) / progress_scale_m`。该项使用**实际物理步**两端的距离变化，不再额外乘 `dt`；在 1 kHz 和较低控制频率下代表同一物理位移。它是密集奖励，并不严格符合 potential-based policy-invariance 条件，因此作为单独可消融的任务设计，不能声称不改变最优策略。
- **已实现**：可配置 `training.log_std_min` 探索下限，防止低成功率阶段标准差一直下降。仍然采用原 PPO 每 1 ms 独立高斯动作采样，**尚未实现时间相关探索**。
- **已实现**：统计 `explained_variance`、训练失败原因、探索标准差；早期课程阶段不再以偶然的最高难度成功事件覆盖全局 `best.pt`。
- **保持不变**：PPO 网络拓扑、观测/CTBR 映射、GPU 仿真器、物理参数与 legacy E1～E5 配置的 `w_progress=0` 行为。

## 2. 最小功能测试（CPU）

在项目根目录：

```bash
pip install -e '.[dev]'
python -m pytest -q
python main.py train \
  --config configs/basic/basic_cpu_smoke.yaml \
  --out runs/basic_cpu_smoke
```

这一测试仅执行 16 环境步，检验导入、采样、GAE、更新和 checkpoint；不能用于评价收敛。

## 3. 先做旧策略兼容性诊断（推荐）

优先确认训练失败不是 1 ms 物理积分或动作解释引起。用之前已经有效的 **V6.4 Hard checkpoint** 与一个全新的评估种子测试 Stage 0：

```bash
python main.py evaluate \
  --config configs/basic/basic_stage0_1khz.yaml \
  --algorithm PPO \
  --checkpoint /path/to/v64_hard/checkpoints/best.pt \
  --stage 0 --episodes 32 --seed 440000 \
  --out runs/basic_old_policy_eval.json
```

此处仅验证旧策略跨频率、跨任务设置的可执行性，不用于主实验性能比较；低成功率不等价于旧策略训练失败。若旧 checkpoint 的 `policy_contract` 不兼容，程序会主动拒绝加载，避免悄悄修改观测语义。

## 4. 从零训练 Stage 0

在 CUDA 版 PyTorch 及显卡驱动均正常的 RTX 3090 服务器中运行：

```bash
python main.py train \
  --config configs/basic/basic_stage0_1khz.yaml \
  --device cuda \
  --simulation-backend torch \
  --seed 42 \
  --out runs/basic_stage0_seed42
```

默认预算 16,000,000 环境步、64 并行环境、每环境 4096 步 Rollout，相当于每段 4.096 秒模拟时间；最大真实任务时间 30 秒。`n_steps*64=262144` 个样本/次 Rollout，约 61 次 Rollout 批次，不保证收敛。

前 2～4 次评估检查 `stage_success`、`train_failure_reasons`、`action_std`、`approx_kl`、`clip_fraction`、`explained_variance`、`reward_components/progress`；如果连续评估均为 0%，应停止扩大计算预算，先排查闭环状态和奖励是否有效。

### 逐回合验收

```bash
python main.py evaluate \
  --config configs/basic/basic_stage0_1khz.yaml \
  --algorithm PPO \
  --checkpoint runs/basic_stage0_seed42/checkpoints/best.pt \
  --stage 0 --episodes 128 --seed 540000 \
  --out runs/basic_stage0_independent_eval.json
```

128 个独立随机种子生成的回合成功率达到约 80% 可作为阶段性目标；同时核查超速/倾角/越界的失败比例及真实完成时间。无独立验证通过，不升级难度。

## 5. 新模型直线移动目标测试与微调

仅在 Stage 0 基础成功率稳定后，尝试速度 2～6 m/s、距离 20～50 m 的直线移动目标：

```bash
python main.py train \
  --config configs/basic/basic_moving_1khz.yaml \
  --init-from runs/basic_stage0_seed42/checkpoints/best.pt \
  --device cuda --simulation-backend torch --seed 43 \
  --out runs/basic_moving_seed43

python main.py evaluate \
  --config configs/basic/basic_moving_1khz.yaml \
  --algorithm PPO \
  --checkpoint runs/basic_moving_seed43/checkpoints/best.pt \
  --stage 0 --episodes 128 --seed 540100 \
  --out runs/basic_moving_independent_eval.json
```

对于不同配置，**使用 `--init-from`，不要使用 `--resume`**。目前增加了配置字段，旧 V6.6 的 checkpoint 恢复应使用 `--init-from` 而不是强行严格恢复旧优化器。

## 6. Reward 消融（务必单独训练）

仅关闭本轮新加的进展奖励，其余基本训练条件保持相同：

```bash
python main.py train \
  --config configs/basic/basic_no_progress_ablation.yaml \
  --device cuda --simulation-backend torch --seed 42 \
  --out runs/basic_no_progress_seed42
```

若需要从头比较基础 PPO 是否有收益，应使用相同训练步数、多个 seed、相同独立评估种子。该配置主要用于分离 Reward 效果，不是另一个任务模型。

## 7. 轻量诊断报告

```bash
python scripts/basic_training_diagnostics.py \
  --run runs/basic_stage0_seed42 \
  --out runs/basic_stage0_seed42/diagnosis.json
```

将 `config.yaml`、`train.jsonl`、`validation.jsonl`、`diagnosis.json`、必要时的 `episodes.jsonl` 上传即可，不需要上传整套 NPZ / checkpoint。脚本不会覆盖训练文件。

## 8. 已验证与未验证范围

本次在 **Torch CPU** 环境执行了原有和新增测试、极短 PPO 训练与 checkpoint 保存。由于当前运行容器没有 NVIDIA 设备，**没有实测 CUDA 吞吐或做百万级完整收敛实验**；不能据此宣称已经解决 Stage 0 训练停滞。下一轮应先进行小规模真实 GPU 诊断，再确定是否需要动作时间相关探索或其他 PPO 结构调整。

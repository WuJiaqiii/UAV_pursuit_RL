# V6.6 GPU 原生训练与 E1–E5 评估使用说明

## 1. 修改内容与当前验证范围

本分支以 V6.5 1 kHz CTBR 工程为基础，提供可选的 `simulation_backend: torch`：

- PPO Actor/Critic、梯度更新、Rollout 张量存储、GAE 均可驻留 CUDA；训练过程中不再每个控制步把全部状态/动作转换成 NumPy。
- 新增 Torch 物理后端：6DoF 推力/角速度一阶滞后、SO(3)、目标运动、31D Actor/43D Critic 观测、奖励、连续段穿越捕获事件和飞行包线积分。
- PPO、PN、APN、PD 共用 Torch 物理后端；PN/APN 控制律也为 PyTorch 实现。E4 噪声、传感器延迟、CTBR 指令延迟可以在同一设备运行。
- 完整 Benchmark 的 NPZ、CSV、LaTeX 和图形接口不变。为减小 CPU I/O，只有按 `evaluation.trace_stride` 采样和首次结束事件时才需要拷贝轨迹到主机。
- 同时保留原 `simulation_backend: numpy` 路径作为回归基线。

**验证限制：本包构建环境仅安装了 CPU 版 PyTorch（CUDA 不可用）。** 已运行 Torch-on-CPU 的原生训练、PPO/PN/APN Benchmark、轨迹导出、噪声延迟 E4、动力学方程对照和自动化测试。**没有在 RTX 3090 上实测 CUDA 吞吐、显存、数值稳定性或真实 1 ms 截止时延**。CUDA 加速仍需在用户服务器上确认。新 Torch 后端产生的随机轨迹与 NumPy 后端不具有相同随机数序列，不能直接用同 seed 混合配对。正式 PPO/PN/APN 比较必须统一后台并重新生成数据。

## 2. 安装与 CUDA 检查

```bash
pip install -e '.[dev,plots,logging]'
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
```

如果输出 `False`，需要先按服务器系统与驱动适配的官方方法安装带 CUDA 的 PyTorch。只安装 NVIDIA 驱动并不能让 CPU 版 PyTorch 使用 GPU。

## 3. 1 kHz GPU 训练（保持原 31D/43D 网络合同）

首先跑短流程：

```bash
python main.py train \
  --config configs/high_frequency/smoke_torch_1khz.yaml \
  --device cuda \
  --out runs/gpu_smoke
```

从旧 V6.5 训练好的 20 Hz 或 1 kHz checkpoint 做新的 1 kHz 初始化训练：

```bash
python main.py train \
  --config configs/high_frequency/ppo_1khz.yaml \
  --device cuda \
  --simulation-backend torch \
  --num-envs 256 \
  --init-from /path/to/best.pt \
  --out runs/v66_gpu_1khz_seed42
```

完整新的从零课程训练：

```bash
python main.py train \
  --config configs/high_frequency/curriculum_1khz.yaml \
  --device cuda \
  --simulation-backend torch \
  --num-envs 256 \
  --out runs/v66_gpu_curriculum_seed42
```

`--init-from` 会从旧权重启动新实验，不能将 NumPy 训练配置的优化器及 RNG 状态当作 Torch 后端同一实验 `--resume`。Tensor checkpoint 支持在同配置下继续训练，但未完成回合将重新抽样，不能实现逐步完全相同的恢复。

## 4. GPU 评估及 E1～E5

```bash
python main.py evaluate \
  --algorithm PPO \
  --checkpoint runs/v66_gpu_1khz_seed42/checkpoints/best.pt \
  --config configs/high_frequency/ppo_1khz.yaml \
  --device cuda --simulation-backend torch \
  --stage 5 --episodes 64 \
  --out runs/v66_gpu_eval.json
```

小规模 PPO/PN/APN 同场景 Benchmark：

```bash
python main.py benchmark \
  --checkpoint runs/v66_gpu_1khz_seed42/checkpoints/best.pt \
  --config configs/high_frequency/ppo_1khz.yaml \
  --device cuda --simulation-backend torch \
  --algorithms PPO PN APN \
  --scenarios precision_2m ood_pulse_2m ood_switchback_2m \
  --episodes 8 --no-plots \
  --out runs/v66_gpu_benchmark_pilot
```

E1～E5 试运行：

```bash
python main.py experiments \
  --checkpoint runs/v66_gpu_1khz_seed42/checkpoints/best.pt \
  --config configs/high_frequency/ppo_1khz.yaml \
  --device cuda --simulation-backend torch \
  --suites E1 E2 E3 E4 E5 \
  --episodes 4 --calibration-episodes 2 --quick --no-plots \
  --out runs/v66_gpu_e1e5_pilot
```

正式实验应重新冻结随机种子与任务协议，并在同一 Torch 后端上评价 PPO 与传统制导。`--no-plots` 只跳过图像生成，不一定关闭 NPZ 导出。1 kHz 评估的轨迹仍然可能很大，优先使用配置中的 `evaluation.trace_stride: 20` 或更高的采样间隔；控制与奖励积分仍采用全频率。

## 5. 真实吞吐剖析和参数调优

```bash
python scripts/profile_backends.py \
  --config configs/high_frequency/ppo_1khz.yaml \
  --backend torch --device cuda \
  --num-envs 256 --steps 1000 --warmup 100 \
  --out runs/profile_gpu_256.json
```

对比 64 / 128 / 256 / 512 环境并行吞吐、PPO update 占比与显存峰值。推荐先试 256，再判断是否需要 512，不能假定越多越快。用 `nvidia-smi` 观察显存与利用率。

**硬件 1 kHz 截止时延与离线训练吞吐是不同要求**：此脚本测量全批次 Actor+环境吞吐，不包含真实状态估计、通信与飞控实时线程调度。真实硬件部署仍须在目标设备上进行端到端 P99/P99.9 抖动和 deadline miss 检查。

## 6. 性能与数值注意事项

1. Torch 后端默认使用 float32 物理状态，NumPy 原实现为 float64。飞行器模型和确定性短轨迹可对照，但长轨迹在捕获边界处可能出现不同的厘米级结果。正式结论必须在统一数值后端上重跑。
2. 新后端避免了每步整批 CPU/GPU 数组拷贝，但训练中的 reset 检查仍会同步一个布尔标量。CUDA Eager 操作可能发起大量小 kernel；对于仅 64 个环境，CPU NumPy 可能更快。
3. 新 Torch 环境中随机数使用 `torch.Generator`，与 NumPy 的 `default_rng` 不具备跨后端逐点复现性。PPO、PN、APN 在同 Torch 后端使用同场景 seed 可保持各自任务的目标轨迹一致。
4. NPZ 轨迹压缩及 matplotlib 绘图仍属 CPU I/O，GPU 评估不是“零 CPU”。不同硬件上的速度必须实测。
5. 不建议擅自把 SO(3) 计算与捕获事件改为混合精度；先用 float32 通过短轨迹回归，再做大规模验证。
6. 原环境控制方式为 CTBR 设定值直接进入简化一阶响应模型，GPU 后端**没有**新增真实飞控 1 kHz PID、电机/ESC 或 AirSim 的精细物理模型。

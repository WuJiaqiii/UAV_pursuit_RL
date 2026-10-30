# V6.7.1 CUDA Smoke 修复说明（2026-10-27）

## 出错原因

`torch.device("cuda") != torch.device("cuda:0")`：配置中的 `cuda` 是未指定编号的设备描述，实际 CUDA 张量却报告完整的 `cuda:0`。V6.7.0 错误地直接比较二者，导致 GPU 模式在首个 PPO 动作采样时抛出 `sample_tensor requires observations on policy device`。

## 修改内容

- 新增 `pursuit_rl/simulation/torch_device.py`，把设备规范化为明确的 `cuda:N`；未安装 CUDA PyTorch 时提前报清楚错误。
- 修复 `pursuit_rl/learning/ppo.py` 的模型设备及张量采样检查。
- 修复 GPU 仿真后端、Rollout Buffer 和 GPU Evaluation 中相同的设备问题。
- 修复 `main.py` 和 Benchmark 重新加载 checkpoint 时的设备属性设置。
- 不改变 PPO、PD 监督、Reward 或 1kHz 物理积分参数。

## 安装热修复

将 `v6_7_1_cuda_device_hotfix.zip` 下载到服务器，运行：

```bash
unzip -o v6_7_1_cuda_device_hotfix.zip -d /path/to/pursuit_rl_v6_7_unified_basic_ppo
cd /path/to/pursuit_rl_v6_7_unified_basic_ppo
python -m pytest -q tests/test_v671_cuda_device.py
python main.py train --config configs/basic_v67/smoke.yaml \
  --device cuda --simulation-backend torch --out runs/v671_cuda_smoke
```

请使用**新的** `--out`：之前的失败运行目录可能存在元数据，训练器会拒绝覆盖。

## 完整源码包

若使用完整 V6.7.1 源码包，解压后进入 `pursuit_rl_v6_7_1_unified_basic_ppo/`，直接运行上述命令即可；不需叠加热修复。

## 检查设备

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
```

当前执行环境没有 NVIDIA GPU，验证了 125 项 CPU 回归测试，另有 1 项真实 CUDA 集成测试在无 GPU 情况下自动跳过；应在服务器上补跑真实 GPU smoke。

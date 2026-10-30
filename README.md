# V6.8.1：复杂场景评估与逐级迁移训练

本轮新增功能请先阅读 [CURRICULUM_README.zh-CN.md](CURRICULUM_README.zh-CN.md)。随包包含你的服务器 seed42 基线模型，可以直接执行复杂场景预评估和课程训练。下方保留 V6.8.0 及更早接口说明。

---

# V6.8.0 追踪恢复实验版

本轮新增的 **50 Hz 高层策略 + 1 kHz 内环、BC/DAGGER 初始化、MC critic 校准**，请先阅读 [RESCUE_README.zh-CN.md](RESCUE_README.zh-CN.md)，使用 `scripts/rescue.py` 或 `scripts/run_rescue.sh`。

下文和其他历史文档保留原 V6.7.2 接口说明；旧的 `main.py` 不用于加载本次新动作接口的 checkpoint。

---

# Pursuit RL V6.7.0 — unified 1 kHz basic PPO tracking

**Start here:** [Chinese Stage0 experiment guide](docs/V67_BASIC_PPO_EXPERIMENTS_ZH.md).
This release is based on the packaged **V6.6.1** source. It reconstructs a
training-only PD auxiliary objective because the user's *actual modified PD
source* was not supplied; it is **not** a byte-for-byte copy of that private
PD implementation. E1–E5 remain in the repository for compatibility but are
out of scope for the V6.7 basic-tracking protocol.

```bash
pip install -e '.[dev]'
python -m pytest -q
python main.py train --config configs/basic_v67/smoke.yaml --device cpu --simulation-backend torch --out runs/v67_smoke
python scripts/basic_experiment_matrix.py --group easy --seeds 42 43 44 --dry-run
```

## V6.6.1 基础目标追踪（1 kHz PPO）

本次推荐从 `configs/basic/basic_stage0_1khz.yaml` 开始，先验证基础直线目标的 5 m 捕获任务，不运行 E1～E5。详情见 [中文基础训练手册](docs/BASIC_TRACKING_1KHZ_ZH.md)。

---

## V6.6 GPU Torch 后端（可选）

增加 GPU 设备驻留的批量仿真、PPO Rollout/GAE、训练和 PPO/PN/APN 评估。详见 [中文 GPU 使用与性能测试手册](docs/GPU_TORCH_BACKEND_ZH.md)。

GPU 服务器示例：

```bash
python main.py train --config configs/high_frequency/ppo_1khz.yaml \
  --device cuda --simulation-backend torch --num-envs 256 \
  --init-from /path/to/best.pt --out runs/v66_gpu
```

**验证边界：发布环境无 NVIDIA GPU，仅完成 Torch CPU 模式与数值/集成测试；CUDA 性能需要在服务器验证。**

---

# Pursuit RL 6.5 · 真正 1 kHz PPO→CTBR / E1–E5：按功能分包的无人机强化学习仿真

向量化动力学、非对称 PPO 和 CTBR 控制的无人机首次交会仿真。目标是运动学机动体；世界系 NED，机体系 FRD。成功按控制周期相对线段首次进入捕获球判定，成功优先于同一步物理失败。

## 代码解读

完整的逐文件说明、Actor/Critic 结构、观测动作、训练评估流程和依赖索引见 [当前代码解读](docs/CODE_GUIDE_ZH.md)。初次阅读建议先看第 1～4 节，再按第 8 节查文件。

## 安装与统一入口

```bash
python -m pip install -e ".[dev,plots]"
python -m pytest -q
python main.py --help
```

Python>=3.10。所有顶层参数、配置选择、数据输入输出与分发由 main.py 负责；安装后等价使用 `pursuit <子命令>`。子命令：train、evaluate、benchmark、**experiments**、trace-metrics、visualize、report、compare、export、feasibility。图表为可选依赖，TensorBoard 使用 `.[logging]`。

## 代码目录与依赖

```text
main.py                       # 参数与顶层任务编排
pursuit_rl/
├── __init__.py               # 包说明与版本
├── config/                   # 配置对象、YAML/CLI 解析、校验、默认资源
├── simulation/               # 批仿真、动力学、目标、观测、奖励、事件
├── control/                  # 控制器协议与 PD/PN/APN 解析制导
├── learning/                 # PPO、采样缓冲区、模型契约、checkpoint
├── training/                 # 训练循环、验证调度与课程晋级
├── evaluation/               # 在线评估、场景、配对比较、可行性、metrics/
├── artifacts/                # 实验归属、日志、轨迹、JSON、导出与源码指纹
└── visualization/            # 静态轨迹、动画、基准图表
configs/                      # 用户与实验 YAML 覆盖
tests/                        # 单元、流程与架构约束测试
scripts/                      # 批量实验和检查脚本
docs/                         # 设计、配置和重构说明
runs/                         # 实验数据，与源码包分离
```

训练流程调用评估和学习模块；评估调用控制器、学习模块的模型读取接口与仿真；仿真和配置不导入训练、日志或图表。实验记录不依赖训练实现。包内禁止循环模块依赖，tests/test_architecture.py 持续检查边界。配置、仿真和 PN/APN 的导入不要求加载 PyTorch 或 Matplotlib。

Python API 使用新的功能路径，例如 `pursuit_rl.simulation.environment.PursuitBatch`、`pursuit_rl.learning.ppo.AsymmetricPPO`、`pursuit_rl.training.trainer.Trainer`。Config 入口仍是 `from pursuit_rl.config import Config`。旧根模块已删除，不保留转发空壳；已有命令参数与模型文件格式保持。完整文件迁移表见 [模块组织说明](docs/MODULE_LAYOUT_ZH.md)。

## 统一实验目录

新实验默认只有一个根目录 runs，每个直接子目录表示一次实验，例如：

```text
runs/20261016_143000_train_precision_s43_a1b2c3d4/
├── experiment.json                 # 实验身份与来源
├── config.yaml                     # 完整训练配置
├── run.json                        # 训练身份，恢复不覆盖
├── checkpoints/
├── logs/<session>/
├── tensorboard/                    # 可选
├── train.jsonl
├── episodes.jsonl
├── validation.jsonl
├── validation_details.jsonl
├── events.jsonl
└── results/
    ├── evaluate/<日期_类型_ID>/     # report.json、可选 traces/、日志
    ├── benchmark/<日期_类型_ID>/    # 协议、轨迹、指标、汇总与图表
    ├── visualize/<日期_类型_ID>/
    ├── report/<日期_类型_ID>/
    ├── trace-metrics/<日期_类型_ID>/
    ├── compare/<日期_类型_ID>/
    ├── export/<日期_类型_ID>/
    └── feasibility/<日期_类型_ID>/
```

名称由上海时间、类型、可选 --name、seed 和唯一 ID 组成。每次评估/分析使用独立操作目录并保存 operation.json，重复执行不会混入旧轨迹。旧实验目录保留名称，避免破坏 checkpoint 路径。

## 训练与恢复

```bash
python main.py train --config configs/smoke.yaml --name smoke
python main.py train --name precision --seed 43 --set training.actor_lr=0.0001
python main.py train --resume runs/<实验名>/checkpoints/latest.pt
python main.py train --config configs/experiments/precision_fixed_2m.yaml --init-from runs/<实验名>/checkpoints/best.pt --name precision_finetune
```

不指定 --out 时自动创建新的 runs 子目录；--runs-dir 可改实验根目录。恢复默认留在 checkpoint 所属实验，并读取其 config.yaml，仍要求相同训练配置与当前 checkpoint/reward 版本。新训练/微调要求空目录。显式 --out 保留自定义路径能力。

## 结果自动归属

```bash
python main.py evaluate --checkpoint runs/<实验名>/checkpoints/best.pt --stage 0 --episodes 8 --save-traces
python main.py benchmark --checkpoint runs/<实验名>/checkpoints/best.pt --algorithms PPO PN APN --scenarios standard precision_3m precision_2m precision_1m --episodes 128
python main.py report --input runs/<实验名>/results/benchmark/<比较目录>/summary.json
python main.py visualize --trace runs/<实验名>/results/evaluate/<评估目录>/traces/episode_0000.npz --gif
python main.py export --run runs/<实验名>
```

checkpoint、轨迹、报告都可反向找到所属实验；也可显式 --run 指定归属。没有所属实验的独立 PN/APN 评估会创建自己的 runs 子目录。跨两个实验的比较默认创建独立 compare 实验，保留双方来源，避免随意归到某一方。--out/--log-dir 等显式路径仍按用户指定执行，并记录所属实验。默认不再生成顶层 results 或旁置 .run 日志目录。

--stage 按训练配置评估；--scenarios 使用独立固定测试分布，二者互斥。OOD 场景包括 ood_random_2m、ood_switchback_2m、ood_pulse_2m。在线 evaluate 默认保存完整回合报告，--save-traces 另外保存逐步轨迹；benchmark 始终保存其比较所需轨迹。--no-plots 关闭比较图表。

## 配置对象

Config 统一默认值、YAML/extends/stages、checkpoint 回退与 CLI 覆盖。优先级为：

```text
默认值 / checkpoint 回退 → 显式 YAML → --set → 显式训练参数
```

```python
from pursuit_rl.config import Config

config = Config.load('configs/smoke.yaml')
print(config.training.num_envs)
config = config.with_overrides({'training': {'actor_lr': 0.0001}})
config.save('resolved_config.yaml')
```

根、嵌套字典和列表都只读；修改使用 with_overrides 返回验证后的新对象。to_dict 返回独立普通数据，供现有计算模块使用。保存使用 safe_dump，不写 torch.device 的 Python 专用标签。device 支持 cpu/cuda/auto，只有显式 auto 才选择设备；保存实际解析的字符串值。

运行参数和 Experiment 路径独立于仿真配置，不进入模型兼容性哈希。evaluate/benchmark 的 --seed 是评估随机种子，不重写模型原训练配置中的 seed。--set 支持 `training.actor_lr=0.0001`、`stages.stage0.time=20` 等 YAML 值，未知字段和非法值直接报错。

## 记录与兼容

每次执行和恢复使用独立 session，记录配置、输入 SHA256、参数、软件/设备环境、指标、checkpoint 和完成/失败/中断状态。phase=collected 的回合表示完成采样；含 session 的 episode_id 区分恢复后的重新采集。JSONL 是复核依据，TensorBoard 为可选展示。

旧 V5 权重仍支持推理或兼容契约下的 init-from；旧版 resume 退役。当前版本恢复会重采样未完成回合，不保证逐随机数一致。旧 runs/results 迁移按 checkpoint 哈希核实，原文件字节保留，路径映射见 runs/asset_migration.json。

配置设计与迁移细节见 [实验目录与配置对象](docs/EXPERIMENT_CONFIG_ZH.md)，日志字段见 [统一入口与日志](docs/ENTRY_LOGGING_ZH.md)。短训练只验证软件流程，不代表收敛性能或真实飞行能力。


## E1–E5 深入实验（V6.4 新增）

参阅 [E1-E5 实验完整操作手册](docs/E1_E5_EXPERIMENT_GUIDE_ZH.md)。新入口支持高机动场景、速度×机动幅值网格、独立验证集标定 PN/APN、扰动鲁棒性与自动可视化。

```bash
# 小规模软件验证（不是论文性能数据）
python main.py experiments --checkpoint runs/v5_seed42/checkpoints/best.pt \
  --quick --episodes 2 --calibration-episodes 2 --out runs/e1e5_smoke

# E1+E3 正式对照；全部任务使用一致的真实动力学和独立种子
python main.py experiments --checkpoint runs/v5_seed42/checkpoints/best.pt \
  --suites E1 E3 E5 --episodes 128 --calibration-episodes 32 \
  --seed 930000 --out runs/e1_e3_paired

# 可先不跑PPO，只做经典制导的软件验证
python main.py experiments --algorithms PN APN --quick --episodes 2 \
  --out runs/baseline_smoke
```

**说明**：默认噪声模型只作用于评估观测；CTBR 指令延迟作用于执行，不修改 PPO 权重或训练协议。E3 标定集使用独立随机种子，测试集参数冻结。E5 对相同任务分组计算 Pareto 前沿，不混合不同环境难度的成本指标。旧 checkpoint 可以读取，但不是新训练的 V6.4 模型。


## V6.5 新增：外部 PPO 每 1 ms 发新 CTBR

使用 [`docs/V65_1KHZ_EXPERIMENT_GUIDE_ZH.md`](docs/V65_1KHZ_EXPERIMENT_GUIDE_ZH.md) 中的 1 kHz 专用训练/评估配置与分频实验。默认 20 Hz 配置仍保留，以支持旧结果复现；升级不会将旧 20 Hz checkpoint 自动标记成经过 1 kHz 训练的策略。

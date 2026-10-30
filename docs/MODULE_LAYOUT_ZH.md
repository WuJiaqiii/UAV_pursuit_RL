# 6.3 模块划分与迁移说明

## 划分依据

按“谁负责什么、谁依赖谁”组织，而不是按类、函数或文件长度机械分类。根目录 main.py 继续承担参数解析与顶层任务编排；pursuit_rl 根目录只保留包说明和版本，其余进入功能包。

参考了 [Stable-Baselines3 的算法与公共模块组织](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3)，以及 [Gymnasium 的 envs、vector、wrappers 等功能包划分](https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium)。这里只借鉴边界设计，没有复制其实现或新增框架依赖。当前项目只有一种 PPO 实现，不额外引入多算法注册器、插件系统或通用基类体系。

## 当前目录

```text
pursuit_rl/
├── __init__.py
├── config/
│   ├── __init__.py            # Config 等公共配置接口
│   ├── defaults.yaml         # 唯一完整默认配置，随包安装
│   ├── loader.py             # YAML、extends、CLI 覆盖、旧元数据读取、Config
│   ├── schema.py             # 字段结构与物理/训练参数校验
│   └── values.py             # 深层只读值、属性访问、字典副本
├── simulation/
│   ├── environment.py        # PursuitBatch，协调一个控制周期
│   ├── dynamics.py           # QuadrotorBatch，动力学积分
│   ├── targets.py            # TargetBatch，目标机动与采样
│   ├── geometry.py           # 坐标变换、SO(3)、CPA、首次相交
│   ├── observations.py       # actor/critic 特征、尺度、维度
│   ├── rewards.py            # 奖励分项与奖励版本
│   └── events.py             # 成功、失败、超时、回合记录
├── control/
│   └── guidance.py           # Controller 协议、PD/PN/APN 与动作映射
├── learning/
│   ├── ppo.py                # AsymmetricPPO 网络与更新
│   ├── buffers.py            # RolloutBuffer 与 GAE
│   ├── contracts.py          # 模型观测/动作/网络兼容契约
│   └── checkpoints.py        # 推理加载、权重初始化、保存、恢复状态
├── training/
│   └── trainer.py            # Trainer、采样/更新、验证调度、课程晋级
├── evaluation/
│   ├── rollout.py            # 固定种子在线评估、任务哈希、在线汇总
│   ├── benchmark.py          # 配对基准任务、协议与结果汇总
│   ├── scenarios.py          # 命名测试分布与场景覆盖
│   ├── comparison.py         # 回合配对检查与比较
│   ├── feasibility.py        # 理想拦截计算与 CSV 报告
│   └── metrics/
│       ├── __init__.py        # evaluate_trace 组合入口
│       ├── mission.py        # 捕获、首达、最小距离
│       ├── maneuver.py       # 机动成本、控制变化
│       ├── envelope.py       # 飞行包线
│       ├── prediction.py     # CPA 预测指标
│       ├── statistics.py     # Wilson 区间与公共任务统计
│       └── report.py         # 离线聚合、CSV、LaTeX
├── artifacts/
│   ├── layout.py             # Experiment/Operation、归属与路径规划
│   ├── logging.py            # ExpLogger，会话生命周期和结构化记录
│   ├── io.py                 # 严格 JSON/JSONL 序列化
│   ├── trace.py              # 轨迹格式规范化、读取、录制和保存
│   ├── export.py             # 轻量实验归档
│   └── provenance.py         # 整个包及 main.py 的源码指纹
└── visualization/
    ├── trajectory.py         # 三维轨迹与控制量静态图
    ├── animation.py          # 时间一致的 GIF/MP4
    └── summary.py            # 基准比较图
```

每个目录都有 __init__.py。config 和 evaluation/metrics 的 __init__.py 分别是实际配置接口和指标组合入口；其他初始化文件只描述包职责，避免导入一个包就加载整套训练或绘图库。它们不承担旧路径转发。

## 依赖方向

| 功能包 | 可以依赖的其他项目包 |
| --- | --- |
| config | 无 |
| simulation | 无 |
| control | 无 |
| artifacts | config |
| learning | config、simulation、artifacts |
| visualization | artifacts |
| evaluation | config、simulation、control、artifacts、learning、visualization |
| training | config、simulation、artifacts、learning、evaluation |

表中不列标准库和第三方库，同一功能包内部可按模块职责依赖。`from pursuit_rl import __version__` 是公共元数据读取。

训练是最上层的库内流程，所以从 learning 单独划出 training，避免评估为了读取模型而反向引入 Trainer。模型契约和 checkpoint 依赖学习模型及仿真特征，不依赖训练循环。checkpoint 的保存/恢复函数接受 Trainer 状态，但不导入 Trainer 类。

配置通过参数取得值，仿真只计算状态；两者不调用 main、不写实验日志、不绘图。artifacts 保存数据但不决定学习或仿真规则。可视化读取已保存轨迹，不修改仿真状态。benchmark 需要出图时调用 visualization；普通基线任务导入不加载 Matplotlib，配置和仿真导入不加载 PyTorch。

tests/test_architecture.py 检查所有静态项目导入可解析、符合上表且无模块循环，也检查无 PyTorch/Matplotlib 时的基线导入和源码指纹覆盖范围。

## 合并和拆分

1. controllers.py 的单一 Controller 协议并入 control/guidance.py，与解析控制器共享维护。
2. curriculum.py 的 promotion_state 并入 training/trainer.py，保持纯函数和原行为。
3. feasibility.py 与 feasibility_report.py 合并为 evaluation/feasibility.py，同一诊断功能的计算与文件报告放在一起。
4. policy.py 拆为 learning/ppo.py 和 learning/buffers.py，PPO 优化与采样存储分别维护。
5. 原 config.py 拆为读取/覆盖、校验、不可变值三部分；原 configuration/defaults.yaml 移入同一 config 包，避免代码与默认资源分散。
6. metrics 整体归入 evaluation，plotting 改名为 visualization；统计模块归入 metrics，日志和文件操作归入 artifacts。

simulation 中的 dynamics、targets、observations、rewards、events 保持独立。这些模块对应不同的仿真规则和修改原因，合并成单个 environment 文件会增加耦合。三个可视化文件也保持分离，动画和静态图的依赖与处理流程不同。

## 旧文件到新文件

以下路径均相对于 pursuit_rl：

| 原路径 | 当前路径 |
| --- | --- |
| config.py | config/loader.py、config/schema.py、config/values.py；公共接口 config/__init__.py |
| configuration/defaults.yaml | config/defaults.yaml |
| environment.py、dynamics.py、targets.py、geometry.py | simulation/ 下同名文件 |
| observations.py、rewards.py、events.py | simulation/ 下同名文件 |
| baselines.py、controllers.py | control/guidance.py |
| policy.py | learning/ppo.py、learning/buffers.py |
| contracts.py、checkpoints.py | learning/ 下同名文件 |
| training.py、curriculum.py | training/trainer.py |
| evaluation.py | evaluation/rollout.py |
| benchmarking.py | evaluation/benchmark.py |
| scenarios.py、comparison.py | evaluation/ 下同名文件 |
| feasibility.py、feasibility_report.py | evaluation/feasibility.py |
| metrics/ | evaluation/metrics/ |
| statistics.py | evaluation/metrics/statistics.py |
| experiments.py | artifacts/layout.py |
| logger.py | artifacts/logging.py |
| io.py、trace.py、provenance.py | artifacts/ 下同名文件 |
| exporting.py | artifacts/export.py |
| plotting/ | visualization/ |

## 使用与兼容

main.py 的九个子命令及参数保持，scripts 中的 Python 导入、测试和文档已同步。Python 代码应使用新路径：

```python
from pursuit_rl.config import Config
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.learning.buffers import RolloutBuffer
from pursuit_rl.learning.checkpoints import load_model
from pursuit_rl.training.trainer import Trainer
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.artifacts.layout import Experiment
```

旧根模块文件、旧 metrics/plotting/configuration 目录删除，不创建空壳兼容文件。若项目外有自己的 Python 脚本，需要按上表更新导入。

默认 YAML 的数值、模型 state_dict 键、checkpoint/reward/评估协议版本保持；包版本更新为 6.3.0。既有模型使用张量字典保存，不依赖 Python 类旧路径。现有 runs 的文件和位置保持。provenance 移动后调整包根定位，指纹仍覆盖所有功能包、默认 YAML 和 main.py。

验证包括原有 92 项测试及新增 3 项架构检查。固定种子对照中，120 步、4 个并行环境的 36,960 个观测/奖励/结束标记值，PN 评估报告、历史 checkpoint 预测及配置哈希均与重组前完全一致。另检查构建后的安装包包含新配置资源，且可在源码目录之外运行。短流程测试用于确认软件行为，不能替代完整训练收敛实验。

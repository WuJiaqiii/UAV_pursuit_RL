# UAV Pursuit RL

无人机追逐仿真与非对称 PPO。当前主线是 **50 Hz 期望速度策略 + 1 kHz 几何内环**，先 BC/DAgger 初始化，再用 PPO 优化，并逐步增加目标机动、初始距离与初速。

成功定义为截止或失败前**首次进入捕获球**，默认半径 2 米；不代表持续保持 2 米追踪误差，也不代表真机或硬件在环验证。

## 文档导航

| 需求 | 文档 |
|---|---|
| 网络、观测、动作与模块关系 | [代码架构](docs/ARCHITECTURE.md) |
| 初始化、训练、迁移、恢复与课程 | [训练指南](docs/TRAINING.md) |
| 冻结评估、参考控制器与收益对照 | [评估指南](docs/EVALUATION.md) |
| 时间尺度、奖励、critic 与异常证据 | [诊断指南](docs/DIAGNOSTICS.md) |
| 旧 `main.py`、直接 CTBR、E1–E5 与轨迹分析 | [旧接口参考](docs/LEGACY.md) |
| 已验证结果、版本演变和当前研究缺口 | [实验与版本历史](docs/HISTORY.md) |

日常使用先读本页和训练指南。命令均在仓库根目录执行。

## 安装与检查

项目声明 Python ≥3.10。先准备与服务器驱动匹配的 PyTorch，再安装项目：

```bash
git clone https://github.com/WuJiaqiii/UAV_pursuit_RL.git
cd UAV_pursuit_RL
python -m pip install -e '.[dev,plots]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
```

只查看入口、不启动实验：

```bash
python scripts/rescue.py --help
python scripts/challenge.py --help
python scripts/curriculum.py --help
```

## 模型与实验数据

Git 仓库不包含 `.pt` 权重和 `runs/` 结果。历史说明中的 `pretrained/`、`local_validation/`、`challenge_screen/` 也不在当前仓库中。评估已有模型时，准备自己的 checkpoint 并指定真实路径。

课程默认查找 `baseline_models/velocity_50hz_2m/seed_42/checkpoints/best.pt`；也可通过 `--baseline-root` 指向已有服务器实验根目录。没有模型时，可从头训练：

```bash
DEVICE=cuda PPO_STEPS=20000000 bash scripts/run_rescue.sh
```

这是需要实际算力的基础任务训练，不是快速功能检查。默认输出为 `runs/v680_rescue/velocity_50hz_2m/seed_42/`。预算、模型来源与恢复规则见[训练指南](docs/TRAINING.md)。

## 两套接口

| 路线 | 入口 | 默认观测/动作 |
|---|---|---|
| 当前恢复与课程路线 | `scripts/rescue.py`、`scripts/challenge.py`、`scripts/curriculum.py` | 速度 actor 35D / critic 43D，3D 期望速度 |
| 旧实验与分析路线 | `main.py`，安装后也可用 `pursuit` | actor 31D / critic 43D，4D 推力和机体系角速度（CTBR） |

同名 `best.pt` 不表示相同模型契约。两条路线的 checkpoint 不应交叉传入；`main.py experiments` 不能直接用于 rescue checkpoint。

基础任务已获得有效策略；困难任务新增 PPO 尚未获得可靠收益，课程通过可能只来自原模型迁移能力。证据与范围见[历史](docs/HISTORY.md)。

## 维护

通过 Git 提交和 Pull Request 审阅源码变化。实验保存配置、来源、真实更新计数，并将模型选择数据与独立测试分开。

`SOURCE_MANIFEST.json`、`docs/HISTORICAL_SOURCE_MANIFEST_*.json` 和 `VERIFICATION_V681.json` 是历史交付清单/验证快照，不是当前工作树的完整性证明，也不是本次提交重跑的测试结果。

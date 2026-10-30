# 旧接口参考：main.py、CTBR 与 E1–E5

[返回 README](../README.md) · [当前架构](ARCHITECTURE.md) · [诊断](DIAGNOSTICS.md)

本页用于维护 V6.3–V6.7 的已有实验。入口 `main.py` / `pursuit` 默认 actor31D / critic43D，输出4D CTBR；当前 `scripts/rescue.py` 是不同模型契约，不可直接替换。

## 训练和频率对照

CPU 小功能测试不证明收敛：

```bash
python main.py train --config configs/high_frequency/smoke_1khz.yaml \
  --device cpu --out runs/legacy_smoke
```

Torch/CUDA 可使用 `configs/high_frequency/smoke_torch_1khz.yaml`。`--simulation-backend torch` 使动力学、rollout、GAE 和网络驻留同一设备；NumPy 后端保留作回归。GPU 可用性、吞吐和数值结果应在实际主机记录。

旧 CTBR 新实验初始化示例：

```bash
python main.py train --config configs/high_frequency/ppo_1khz.yaml \
  --init-from /path/to/legacy/best.pt --device cuda \
  --simulation-backend torch --seed 42 --out runs/legacy_1khz
```

旧 `--init-from` 加载兼容 actor/critic，不等于 rescue 的 actor-only 迁移；旧 `--resume` 要求严格配置兼容，不能套用 rescue 的预算延长规则。配置/步数含义以该入口解析结果为准。

旧训练要求 `physics.policy_interval_steps=1`；评估可保持同一权重、1 ms 积分，仅改变决策频率：

```bash
python scripts/frequency_study_1khz.py \
  --checkpoint /path/to/legacy/best.pt \
  --config configs/high_frequency/ppo_1khz.yaml \
  --hz 20 50 100 200 500 1000 --algorithms PPO PN APN \
  --scenarios precision_2m ood_pulse_2m ood_switchback_2m \
  --episodes 16 --seed 990000 --out runs/legacy_frequency
```

这是固定权重的部署频率消融，不是各频率分别训练的比较。比较训练频率应匹配每环境物理时间，见[诊断指南](DIAGNOSTICS.md)。`measure_actor_1khz.py` 测网络时延，`profile_backends.py` 测后端；它们不能代替硬件最坏截止时延验证。

`configs/basic/` 保留基础追踪/进度消融；`configs/basic_v67/` 保留容易/标准任务、纯 PPO 和训练期 PD 辅助。先预览实验矩阵：

```bash
python scripts/basic_experiment_matrix.py --group easy --seeds 42 43 44 --dry-run
```

PD 辅助是额外训练监督，不混入执行动作；不同难度组不能作为单超参数因果对照。具体参数以 YAML 和脚本 `--help` 为准。

## E1–E5

| 套件 | 内容 |
|---|---|
| E1 | Singer、Rapid Hybrid、高加速度、交替横向、Pulse 与同向极限追逐 |
| E2 | 三档速度×五档加速度网格；固定2米首入球，初距300–400米，150秒上限 |
| E3 | 在独立验证种子上标定并冻结 PN/APN，再与 PPO 比较 |
| E4 | 执行器参数、位置/速度噪声、观测延迟与指令延迟扰动 |
| E5 | 轨迹、CPA/控制时序、性能边界、同任务 Pareto、CSV/LaTeX |

不依赖 checkpoint 的经典控制软件检查：

```bash
python main.py experiments --algorithms PN APN --quick --episodes 2 \
  --out runs/classic_only
```

已有兼容旧模型的研究对照：

```bash
python main.py experiments --checkpoint /path/to/legacy/best.pt \
  --suites E1 E3 E5 --episodes 256 --calibration-episodes 32 \
  --seed 930000 --out runs/legacy_e1e3
```

`--quick` 只验证软件通路，标定候选也更少。E1–E4 使用空输出目录；已有完整结果可单独生成 E5：

```bash
python main.py experiments --algorithms PN APN --suites E5 --out runs/classic_only
```

### 公平性和边界

同组共享任务、初态和目标随机过程，配对轨迹检查真实目标轨迹前缀。PN/APN 含纵向速度管理、加速度限幅与公共 CTBR 姿态内环，应按这一完整控制器描述。

E3 默认在 `test_seed+10000000` 上标定：PN16组、APN24组；quick仅2组。默认标定分布为 `singer_fast` 与 `rapid_hybrid`；参数冻结后再测独立场景，不能用测试结果反复调参。

E4 使用共享的状态测量噪声/延迟通道，本机姿态/角速度仍为简化理想 IMU；不包含完整检测、雷达/VIO异步估计系统。指令延迟按当前物理 dt 的整数倍实现，轨迹区分发出与实际执行动作。

首次捕获由环境事件产生，在线采样最小距离与离线线段最小距离有不同含义。`high_envelope_ratio` 旧字段只指高推力，`high_envelope_any_ratio` 覆盖多个包线分量。横向加速度平方积分/动作变化代价不是电功耗。Pareto 只在同场景、同 benchmark 中比较，并同时报告失败与多训练 seed 不确定性。

## 日志、轨迹和实验归属

`main.py` 子命令有 train、evaluate、benchmark、experiments、trace-metrics、visualize、report、compare、export、feasibility。安装后的 `pursuit` 进入同一个 `main.main`。

新训练默认创建带时间/名称/seed的 `runs/` 实验；checkpoint/输入已归属某实验时，新结果放入其 `results/<命令>/<操作ID>`。`--run` 指定归属，`--out` 保留显式自定义路径。每次操作记录 `operation.json`、会话状态和来源。

配置优先级为默认→checkpoint回退→YAML/extends→`--set`→显式训练CLI参数。实际解析配置、输入SHA、源码指纹、原始argv和session用于追溯，不保证轨迹完全确定。日志时间采用 Asia/Shanghai。

`episodes.jsonl` 中 collected 仅表示采样完成，不保证成功做过优化更新；中断未完成回合不算完整样本。恢复保留原实验身份，新建会话，未完成回合重采样。`validation_details.jsonl` 是可复核明细，TensorBoard 是标量展示。

常用入口：

```bash
python main.py benchmark --checkpoint /path/to/legacy/best.pt \
  --config configs/high_frequency/ppo_1khz.yaml --stage 0 \
  --algorithms PPO PN APN --episodes 8 --no-plots --out runs/legacy_benchmark
python main.py trace-metrics /path/to/episode_0000.npz --out runs/trace_metrics.json
python main.py visualize --trace /path/to/episode_0000.npz --out runs/trajectory.png
python main.py report --input /path/to/summary.json --out runs/report.csv
```

轨迹降采样只影响记录，不改变实际闭环频率。引用历史模型和协议时必须带来源，不能把旧模型软件运行成功当作当前任务已重新训练成功。

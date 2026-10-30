# 核心仿真重构说明

## 行为修复

- CPA potential 奖励只加入一次；成功和物理失败的势函数归零，超时保留 bootstrap。
- 命名 holdout 固定物理/目标/任务条件；按 checkpoint 配置评估需显式使用 --stage。
- 新实验拒绝非空目录；当前版本 resume 核对配置、checkpoint 和奖励版本，记录恢复事件并保留原 run.json。
- 配对比较核对任务哈希、批种子、回合 ID 和初态，修复旧 get 默认参数提前求值问题。
- 推理加载不再创建优化器；目标加速度尺度写入模型契约，OOD 任务上限与输入编码分离。

## 模块拆分与整合

simulation/environment.py 只协调状态更新、奖励、事件和观测。观测特征有命名切片；奖励分项可通过 env.reward_components 查看；事件规则集中在 simulation/events.py。模型读写位于 learning/checkpoints.py；课程晋级函数与训练流程一起维护在 training/trainer.py。artifacts/trace.py 的 TraceRecorder 统一回合录制；离线指标只规范化一次。artifacts/io.py 严格处理 NumPy，拒绝未知类型和非有限 JSON 数值。静态图、捕获放大图和动画都在 visualization，通过一个 visualize 命令使用。当前完整结构见 MODULE_LAYOUT_ZH.md。

## 配置

完整基准位于 pursuit_rl/config/defaults.yaml。configs/train.yaml 表示不覆盖基准；实验仅声明差异。支持 `extends: 相对父配置路径`，循环继承会报错；`stages` 按名称覆盖字段，`curriculum` 则整体替换。

```yaml
extends: ../train.yaml
training:
  initial_stage: 5
  max_stage: 5
stages:
  precision_2m:
    time: 150.0
```

运行保存完整解析配置。target.min_speed、training.test_seed、reward.success_radius 删除；实际半径仅由阶段字段决定。旧 checkpoint 读取时显式移除这些废弃元数据并补充原归一化尺度，这一小型读取适配不再保留旧实验系统。

## 清理与兼容

删除依赖缺失版本的 generate_reference_report、旧 campaign/V3/V4 测试与 showcase 筛选/重放/OOD 脚本、无调用且字段不匹配的 Pareto 工具。旧测试中可复用的特征、边界、限速、精度、warm-start、迁移检查已迁移到当前职责测试。动画/绘图/报告能力整合后删除整个 tools_showcase；过时展示 CSV 同时移除。三份相同完整配置及复制配置改为包资源与小覆盖。default/stage0/stage3 历史配置路径退役，使用当前 experiments 或默认基准。缓存清理，runs/results 保留。

命令统一为 main.py 的 train、evaluate、benchmark、trace-metrics、visualize、report、compare、export、feasibility 子命令；程序接口使用 pursuit_rl.learning.checkpoints.load_model。旧 V5 模型可推理/新实验初始化，当前版本模型可恢复；修改总步数或奖励视为新实验。6.2 已按 checkpoint 哈希将旧 results 移入所属 runs；6.3 只重组源码包，保持已有实验文件内容与位置。

依赖按本次实测兼容范围声明（NumPy>=1.25、Torch>=2.1、Matplotlib>=3.8），绘图与 TensorBoard 为可选 extra；这不是依赖版本锁。核心 PPO 更新、GAE、动力学积分和批 RNG 顺序保持，CPA potential 的旧数值错误不保持。命名评估协议与 reward/checkpoint schema 升级为 V6，新的结果不能直接冒充旧协议结果。

## 仍有的研究边界

首次相交为控制周期的线段近似；动作控制成本按保存的完整采样区间累计，未静默改成精确首达截断成本。在线 minimum_distance 是采样最小距离，离线 min_distance 是线段最小值。resume 重采样未完成回合，非逐随机数一致。真实硬件适用性和长期收敛性能需另行验证。


## 6.1 入口与日志更新

根目录包装与 cli/ 已由 main.py 子命令取代，详细说明见 README.md 与 ENTRY_LOGGING_ZH.md。旧模型读取和当前版本恢复规则保持。


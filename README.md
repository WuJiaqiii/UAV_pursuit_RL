# UAV pursuit RL

当前默认任务是首次安全进入目标周围 **0.5 m 三维球体**。每个回合从 1–2 m 开始，最多 12 s。以 1 ms 物理步判断进入，进入当步安全即成功并终止。倾角、速度、边界或目标合法性失败优先于成功。到期未进入记为 timeout。任务奖励使用接近进展、成功奖励和失败惩罚，并保留原动作与姿态正则项。

## 首次进入实验

三个配置分别为静止目标、0.5 m/s 和 1 m/s 匀速目标。匀速阶段匹配初速度，各阶段关闭动力学随机化。

```bash
conda activate airsim

# 配对评估 teacher 和冻结 A
python main.py evaluate --reference --out runs/entry_static_teacher.json --device cuda
python main.py evaluate \
  --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
  --config configs/first_entry/static.json \
  --out runs/entry_static_A.json --device cuda

# 从 A 初始化训练。默认配置为静止目标
python main.py train \
  --init-from runs/ctbr_v701/seed_42/initializations/A.pt \
  --out runs/entry_static_train --device cuda
```

分别将配置换成 `configs/first_entry/cruise_05.json` 或 `cruise_10.json`，并更换输出路径。用同一评估种子和环境数配对。首先比较进入率、成功回合进入时间、进入时相对速度和失败原因。训练按进入率选模，其次比较安全结束比例、成功进入时间。默认可行门槛为进入率至少 95%。进入时相对速度是诊断指标。

## 旧实验与检查点

持续跟踪训练与评估配置仍支持原回合语义。旧检查点通过显式 first_entry 配置可参与新任务评估或 `--init-from`。任务切换会改变回报含义，训练需在新目录重新校准 critic。旧运行的 `--resume` 需要原始源码和配置。

专用 A/B/C 入口、预算组配置、行为拟合、分段样本和配平诊断等 v701–v708 实验工具已从当前入口和工作树移除。复现这些实验可在独立 checkout 使用 [v708 最后提交](https://github.com/WuJiaqiii/UAV_pursuit_RL/tree/320807fd8a43fabb5f81fb3b9408f33b694eb5c8)，其 Git 历史包含此前各版本。仓库的持续跟踪训练、课程和检查点迁移仍可使用。

建议保留冻结 A、v707 两个通过验证的第 10 次检查点、v708 各组被选中的权重，以及各轮正式汇报使用的关键对照。同步保存运行清单、配置、验证报告、Git 提交和校验值。仅对仍需续训的实验保留 latest 及其原运行目录。重复的检查点和其他中间权重可以在确认无引用并备份后再清理。

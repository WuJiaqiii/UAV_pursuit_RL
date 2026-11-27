# V18-A Controlled Diagnostics — frozen value study

独立非飞行强化学习标准基准。使用 V16 `linear16_custom_{42..46}/last.pt` 的冻结随机策略，比较有限时域价值预测。不会更新 Actor、PPO 或原应用系统，不输出训练后的策略或价值网络权重。

## 实验问题

- E01 Frozen：原始冻结 Critic 在新采样的状态上的误差。
- E02 Critic-only probe：复制原 Critic，仅在独立 MC 数据上做监督拟合；额外输入恒为 0。
- E03 Time-aware probe：与 E02 使用相同网络容量、初始权重、数据、训练批次、优化器和训练轮数，唯一差异是额外输入为归一化剩余时间。

所有标签为固定 100 步有限回合的折扣回报（gamma=0.99），终点零尾值，不使用被测 Critic 自举。各状态独立重复采样随机策略延续；训练、测试来自不同初始轨迹种子，测试状态还重新独立采样一批标签。四个 anchor 时刻 0/25/50/75 同属一个初始轨迹的状态不视为独立样本。每个 checkpoint 单独建模，不跨种子混合。复制的探针不作为原策略的训练组件，也不保存探针权重。

**注意**：V16 的 Critic 采用时间截断自举，与有限时域标签定义不完全相同；误差降低仅说明对本实验价值定义的拟合改善，不能证明原 Critic 错误或 PPO 性能提高。E02、E03 是离线监督价值探针，并非在线 PPO Critic-only 更新。增加时间输入会改变值函数信息集。预测误差含 MC 标签噪声，报告独立标签重复误差。训练预算固定，不使用测试集调参。

## 运行

在本仓库 benchmarks/ppo_control 目录下执行：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -u -m v18.study --archive /path/to/v16_all_full.zip --preset quick --out runs/v18_quick
python -u -m v18.study --archive /path/to/v16_all_full.zip --preset full --out runs/v18_full
python -m v18.study --out runs/v18_full --status
# 中断后保留原参数及目录，添加 --resume
python -u -m v18.study --archive /path/to/v16_all_full.zip --preset full --out runs/v18_full --resume
```

CPU only; 不需要 Gymnasium / SB3 / MuJoCo。`--seeds 44` 可单独复评种子。输出 `summary.csv`、`summary.json`、`seed_*.json`、`anchors_*.csv`、`manifest.json`、`progress.json`。原 V17 两个只读入口仍保留。

| preset | 每 split 初始轨迹 | 每 split anchor 状态 | 每状态 MC 延续数 | 每 probe epochs |
|---|---:|---:|---:|---:|
| smoke | 4 | 16 | 4 | 40 |
| quick | 24 | 96 | 12 | 400 |
| full | 64 | 256 | 32 | 800 |

每个 seed 包含训练、测试和同一测试状态的独立标签重复三批 MC 延续。Full 五个种子共 122880 条延续（不等同完整 100 步回合），另有生成 anchor 的初始轨迹模拟。两个价值探针使用同一批标签。Quick 与 Full 使用重叠初始种子范围，不是十个独立训练重复。

## 验证与版本范围

原 V18-A 交付包包含实际 CPU Quick 和 Full 结果；两组各 5 个 seed 均完成，35 项测试通过、1 项官方环境依赖缺失跳过。此 Git 分支只包含源码、测试和依赖文件，不包含结果、模型或日志。原应用代码保持不变，本目录不接入该应用。未实现在线 PPO 更新或 V18-B。请在独立目录运行，只使用可信的 V16 标准基准归档。

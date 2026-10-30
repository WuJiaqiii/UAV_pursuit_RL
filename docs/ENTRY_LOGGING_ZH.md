# 统一入口与实验记录

当前实现位置：`pursuit_rl/artifacts/logging.py`；实验归属与目录规则位于 `pursuit_rl/artifacts/layout.py`。完整模块结构见 [MODULE_LAYOUT_ZH.md](MODULE_LAYOUT_ZH.md)。

## 提供的 logger.py 如何借鉴

保留按配置分组显示、控制台 INFO、文件 DEBUG 和独立实验日志的思路。去掉 YOLO、数据集、类别划分等当前项目没有的字段，使用配置字典遍历 RL 参数，新增机器可读的会话、指标与回合记录。

原参考文件缺少 logging/datetime 的 import；ColorFormatter 未使用且直接修改 LogRecord；重复创建同名 logger 会增加重复 handler；logging.setLoggerClass 会影响其他库；控制台与文件时间格式未统一。因此采用独立 ExpLogger，而不直接复制该文件。每个 session 独立 logger，格式化复制 record，UTC+8 时间统一，异常/中断均写入状态并关闭 handler。

## 单一入口

只保留 main.py 作为 CLI，实现 build_parser、配置和模型选择、数据读取、输出写入与分发。子命令：train、evaluate、benchmark、trace-metrics、visualize、report、compare、export、feasibility。安装后使用 pursuit 同样进入 main.main。移除旧 root 包装和 pursuit_rl/cli 包；更新脚本与集成测试。

evaluate 是在线仿真，trace-metrics 是离线数据分析；二者能力不同，但不再需要两个名字近似的 Python 入口。顶层流程集中，PPO、动力学、指标、绘图等实现留在库中。

## 日志边界

- 参数语法或配置在实验创建前就不合法时，main 返回错误，不创建假实验。
- 新训练/比较输出目录保护仍然生效，拒绝写入已有实验后不会添加日志。
- 已创建实验的初始化、执行、保存或分析异常会记录 failed 和堆栈；KeyboardInterrupt 记录 interrupted。
- 每次恢复写入独立 session，run.json 身份保持，回合编号从 checkpoint 继续。
- episodes.jsonl 记录完成回合；中断时尚未完成的回合不冒充完整样本。
- phase=collected 表示已经完成采样，不保证该回合已用于一次成功的优化器更新；失败更新前的采样记录仍保留。恢复时进度计数从 checkpoint 继续，含会话 ID 的 episode_id 区分重新采集的回合。
- validation_details.jsonl 保存完整验证结果，TensorBoard 只保存可视化的标量；JSONL 是可复核的数据源。
- logger 模块不使用全局 setLoggerClass，不修改全局 root handler。

输入 SHA256、配置哈希、源码指纹、版本和原始 argv 用于追溯；日志与 checkpoint 本身不保证训练完全确定性。会话时间采用 Asia/Shanghai，elapsed_s 使用单调时钟。输入路径保留在记录中，便于定位原始数据。


## 6.2 目录与配置

新任务默认记录到 runs/<实验>/results/<命令>/<操作ID>；显式 --out 保持自定义路径及旁置日志的兼容用法。目录与配置规则以 README 和 EXPERIMENT_CONFIG_ZH.md 为准。

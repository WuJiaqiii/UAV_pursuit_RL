# 实验目录与配置对象

当前配置实现集中在 `pursuit_rl/config/`：loader.py 负责读取、覆盖、迁移及 Config；schema.py 负责默认资源与校验；values.py 负责深层只读值。外部仍从 `pursuit_rl.config` 导入。实验目录实现位于 `pursuit_rl/artifacts/layout.py`。

## 参考 config.py 的取舍

采用统一配置入口、派生目录属性、配置保存、带时区的命名和冻结的设计思路。参考文件本身没有 YAML 读入，只有 yaml.dump 保存；其 torch.device 会产生 Python 专用 YAML 表示，自动设备选择覆盖 CLI，freeze 只保护已有顶层属性，不能阻止新增属性及嵌套修改。分类项目的 classes 与本仿真无关，因此不移植。

当前 Config 是 Mapping，支持 cfg.training.num_envs 和 cfg['training']['num_envs']，解析完成后深层只读。嵌套列表变为 tuple，嵌套映射同样只读。with_overrides 创建并校验新对象，to_dict 返回独立副本；保留 load_config 的字典接口，避免改动数值计算流程。

默认 → checkpoint 回退 → YAML/extends → --set → 显式训练 CLI 参数。显式 YAML 仍按完整默认契约解析，不是任意混合外部对象。--set 的 stages 按名称合并，curriculum 整体替换。Config.from_args 在一个入口完成这些步骤，保存实际设备字符串。auto 仅在显式请求时探测 PyTorch CUDA。

构造 Config 和规划 Experiment 都没有文件系统副作用。Config 只含仿真、PPO、奖励和课程等已验证字段；args/out/日期/日志路径属于运行上下文，不混入 checkpoint 的配置哈希。Experiment 提供 directory、checkpoints、logs、config_file、results 等路径属性，注册与执行时才创建目录。

## 目录规则

- 新训练 runs/YYYYMMDD_HHMMSS_train_名称_sSEED_ID；默认名字不会碰撞。
- 有 checkpoint 或输入数据所属实验时，结果归入该实验 results/命令/操作ID。
- --run 显式指定已有实验；--runs-dir 指定新实验的根目录。
- 无归属的独立任务创建自己的实验；跨实验比较同样独立，记录所有来源。
- 每次操作的 operation.json 记录归属、来源、输出路径与 completed/failed/interrupted。
- 显式 --out 保留原来的自定义路径语义；外置结果的 operation.json 能反向定位所属实验。显式 --log-dir 也保留。
- 训练恢复沿用原实验。旧实验不强制改名，以免破坏模型路径。
- 导出实验时排除正在生成的本次 export 操作目录，避免把自身未完成的日志打入压缩包。

## 已有资产迁移

当前旧 results/v5_smoke/protocol.json 的 checkpoint_sha256 唯一匹配 runs/v5_smoke/checkpoints/best.pt。因此整个评估目录迁移到 runs/v5_smoke/results/benchmark/legacy_v5_smoke，内部相对结构与文件内容不变。runs/v5_smoke 与 runs/v5_seed42 保留名字并补充实验身份。

逐文件核对原有资产的 SHA256，保存旧路径、新路径和哈希到 runs/asset_migration.json。顶层 results 仅在空目录时删除。旧 run.json、config.yaml、checkpoint、轨迹和统计结果不重新序列化；历史记录中出现的旧路径可通过迁移映射查询。原创建时间不明的旧实验不伪造 created_at，另记 registered_at。

## 示例

```bash
python main.py train --config configs/smoke.yaml --name smoke
python main.py evaluate --run runs/<实验名> --checkpoint runs/<实验名>/checkpoints/best.pt --stage 0 --episodes 8 --save-traces
python main.py benchmark --run runs/<实验名> --checkpoint runs/<实验名>/checkpoints/best.pt --algorithms PPO PN APN --stage 0 --episodes 8
python main.py report --input runs/<实验名>/results/benchmark/<操作目录>/summary.json
```

没有 --out 时输出位置由实验和操作决定；不要用“最新目录”猜测模型或结果归属，应使用明确路径或 --run。

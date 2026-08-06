# G1 视觉导航工程文档索引

本文档集以 Git 提交 `0c191e2` 为 MuJoCo 冻结基线，只做架构整理和迁移准备，不包含 Isaac Lab 实现。

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 模块边界、在线数据流、训练与评估流程 |
| [IMITATION_LEARNING_FAILURES.md](IMITATION_LEARNING_FAILURES.md) | 教师/DAgger数据和三类模型失败分析 |
| [MUJOCO_DWA_BASELINE.md](MUJOCO_DWA_BASELINE.md) | DWA配置、11场景、Monte Carlo与公平对比协议 |
| [FUTURE_RL_INTERFACE.md](FUTURE_RL_INTERFACE.md) | 未来483维观测、3维动作、奖励、终止与成功字段 |
| [MIGRATION_READINESS.md](MIGRATION_READINESS.md) | 当前解耦程度、迁移缺口与训练机验收顺序 |

DWA继续作为默认规划器、baseline和未来RL fallback；Unitree官方Walking Policy与`SafetySystem`不可修改。缺少独立迁移分支时不得创建新的`isaaclab_port/`。

# Priority 6 学习型导航接入基线

更新时间：2026-08-06 CST

## 已完成

- 定义框架无关的 `LocalNavigator` 合同，现有DWA无需改算法即可满足合同。
- `NavigationRunConfig` 支持注入任意合同实现，Walking Policy推理与PD控制链未修改。
- 新增 `LearnedNavigator`，固定Costmap/Goal输入和三维物理速度输出。
- 对学习策略输出执行有限值检查、物理限幅和独立轨迹碰撞否决。
- Debug Viewer可继续读取候选轨迹和有效性标志。

## 验证

- 31项单元测试通过，包括DWA合同、观测编码、速度限幅、碰撞否决和非法输出测试。
- 0.3秒MuJoCo端到端注入烟雾测试通过：机器人存活、碰撞0、生成15行结构化日志。
- Python编译检查与Git空白检查通过。

## 边界

本阶段完成的是学习策略接入合同和可验证基线，没有伪造或发布未经训练的模型权重。
下一步是构建训练数据/环境和候选模型，然后与DWA在相同Benchmark上对比成功率、碰撞、
规划延迟和泛化能力。任何候选模型都不得替换或修改Unitree官方Walking Policy。

# G1 Navigation Pipeline 开发路线

更新时间：2026-08-05 17:04 CST

## 不可变更的基线

- 不重新训练 Walking Policy。
- 不修改 Unitree 官方 ONNX。
- 保持 `Depth → Costmap → Planner → Velocity → Official ONNX → MuJoCo`。
- 当前阶段不接 DDS、`LowCmd` 或真实机器人。

## 模块边界

| 目录 | 职责 | 状态 |
|---|---|---|
| `navigation/` | 导航运行时、场景定义、单次日志 | 已建立 |
| `camera/` | 深度相机与可迁移地面分割 | Priority 3 已完成 |
| `costmap/` | 局部代价地图稳定入口 | 已建立 |
| `planner/` | DWA 局部规划稳定入口 | 已建立 |
| `benchmark/` | 批量统计、报告和可恢复 Monte Carlo | Priority 4 已完成 |
| `visualization/` | Pipeline Debug Viewer | Priority 2 已完成 |
| `logging/` | 结构化逐帧调试日志与字段说明 | Priority 5 已完成 |

`g1_nav/` 继续保留已验证的相机、代价地图、DWA、安全层和策略合同实现，新目录先提供稳定入口，避免一次性大规模搬迁导致回归。

## Priority 1：Navigation Benchmark

### 已完成

- 11 个场景：单障碍、双障碍、三障碍、随机箱体、窄通道、宽墙、L墙、U墙、死胡同、迷宫、动态障碍。
- 批量运行入口 `benchmark_navigation.py`。
- 逐帧 CSV、单次汇总 `summary.csv`、场景汇总 `success_rate.csv`和中文 `navigation_report.md`。
- 统计成功率、碰撞事件、时间、路径长度、最小净空、平均速度、CPU、FPS和规划耗时。
- 保留 `navigation_sim.py` 单障碍兼容入口。

### 首次基线

- 11 场景运行 1 次：8 成功、3 失败，总成功率 72.7%。
- 总碰撞事件为 0。
- U墙、死胡同和迷宫超时，机器人未跌倒，显示局部 DWA 缺少全局引导和恢复行为。
- 无界面物理 FPS 约 172～283，单次感知+规划约 167～279 ms，尚未满足真机实时要求。

## 后续顺序

1. **Priority 2：Visualization（已完成）**：显示 Depth Image、Point Cloud、Occupancy Costmap、DWA 候选路径、选中路径、Robot Pose 和 Goal。
2. **Priority 3：Ground Segmentation（已完成）**：使用 RANSAC/Least Square/Gravity Constraint 替换 MuJoCo geom ID，并用相同 Benchmark 做回归。
3. **Priority 4：Monte Carlo（已完成）**：参数化障碍宽度、间距、高度、目标点和随机地图，运行 100 次统计。
4. **Priority 5：Debug Logging（已完成）**：补齐 Depth FPS、Planning FPS/Time、Control Latency、Velocity、Pose、Goal Error 和 Collision Event 日志。
5. **Priority 6：Learning-based Navigation**：只替换 Costmap/DWA 到速度的导航层，不替换官方 Walking Policy。
6. **Priority 7：Isaac Lab**：研究 Height Scan、Terrain Curriculum 和 Navigation Policy，不训练 Walking。
7. **Priority 8：真机设计**：最后再引入 RealSense、标定、延时测量、Safety Layer、急停和速度限制。

## Priority 2 验收结果

- [x] Debug Viewer 能在同一时间轴观察深度图、点云、代价地图和规划路径。
- [x] 候选路径区分有效/碰撞轨迹，选中路径、机器人和目标独立标记。
- [x] 状态栏显示世界位姿、速度指令、规划耗时和有效候选数。
- [x] 界面关闭会正常结束仿真，不影响无界面 Benchmark。
- [x] Debug 数据仅在注册回调时收集，单元测试证明开关不改变 DWA 选择结果。
- [x] 已通过 1904×1148 PNG 无界面渲染验证，四个面板均有有效像素内容。

## Priority 3 验收结果

- [x] 新增重力约束 RANSAC 地面平面估计和最小二乘精修。
- [x] Costmap 使用相对拟合平面的有符号高度，不再假设固定水平 `z`。
- [x] 稀疏/空深度帧使用最近有效平面降级，不中断导航。
- [x] 导航感知运行时不读取 `floor_id` 或 `point_geom_ids`。
- [x] 22 项测试通过，11 场景仍为 8/11 成功、零碰撞。
- [x] 10 个场景逐帧完全一致；窄通道最大位置差 8.6 毫米，结果不变。
- [x] Debug Viewer 增加障碍点红色叠加和地面内点率/缓存状态。

## Priority 4 实施状态

- [x] 建立障碍宽度、障碍间距、障碍高度、目标点偏移和随机地图五个等量测试族。
- [x] 默认 100 次，每组 20 次；`run_id` 和种子唯一且可复现。
- [x] 每次运行后增量写入 checkpoint，`--resume` 可跳过已完成样本。
- [x] 输出分组/总体成功率、95% Wilson 置信区间、碰撞和运动指标。
- [x] 断点续跑烟雾测试通过：先跑2次，恢复后完成剩余3次，无重复。
- [x] 五组各1次完整试验均成功、零碰撞，总墙钟约2.9分钟。
- [x] 正式100次低优先级批次已完成。
- 最终结果：87/100成功，95% Wilson CI 79.0%～92.2%；前四组成功率85%～100%，随机地图成功率60%并产生32个碰撞事件。
- 精简报告见 `reports/monte_carlo_report.md`；逐帧数据和服务日志仅保留在本地。

## Priority 5 验收结果

- [x] 单一CSV时间轴记录世界位姿、目标误差、规划速度、安全层指令和实际机体速度。
- [x] 记录运行时Depth FPS、Planning FPS、单帧规划耗时和规划到控制应用延迟。
- [x] 区分单日志周期新增碰撞事件和整次运行累计碰撞数。
- [x] 字段名包含单位，`logging/README.md` 明确指标语义和数据保留边界。
- [x] 27项单元测试、Python编译检查和0.3秒端到端CSV烟雾测试通过。

## Priority 6 实施状态

- [x] 阶段A：定义可替换DWA的`LocalNavigator`合同并让运行时支持依赖注入。
- [x] 阶段A：建立学习策略Costmap/Goal输入与三维物理速度输出合同。
- [x] 阶段A：增加有限值、速度限幅和独立轨迹碰撞否决，不绕过现有Safety System。
- [x] 阶段A：31项测试及MuJoCo端到端假策略基线通过，Walking Policy链路未修改。
- [x] 阶段B首次迭代：构建教师数据、训练轻量候选模型并与DWA做同口径Benchmark。

接口和验收说明见`planner/README.md`及`reports/learning_navigation_contract.md`。

### Priority 6 阶段B首次候选结果

- [x] 固定种子生成800个DWA教师样本，600训练、200独立验证；原始数据仅本地保存。
- [x] 纯NumPy岭回归候选模型完成训练，三维命令验证MAE为0.0458。
- [x] 模型JSON、训练脚本、断点数据复用入口和DWA同场景对比脚本已固化。
- [x] 单障碍、双障碍、窄通道闭环对比完成：DWA 3/3成功，岭回归0/3成功，双方碰撞均为0。
- [ ] 岭回归只作为失败基线，不进入部署候选；下一轮需使用闭环数据聚合和非线性模型解决分布偏移。

详细数据见`reports/navigation_ridge_training.md`和`reports/learning_navigation_benchmark.md`。

### Priority 6 阶段B第二次迭代

- [x] 在失败岭回归策略的实际闭环状态上完成第一轮DAgger采集，共500个DWA重标注样本。
- [x] 实现确定性随机ReLU特征模型，并增加占据图距离场特征；无需新增机器学习依赖。
- [x] 训练仅使用单/双障碍325个聚合样本，窄通道175个聚合样本不参与训练。
- [x] 独立随机验证MAE 0.0457，含安全适配平均约2.6 ms，安全否决率0.5%。
- [x] 三场景闭环仍为0/3成功、零碰撞；DWA为3/3成功。
- [ ] 随机特征模型同样只保留为失败基线；后续应转向具备时序状态和明确闭环训练目标的模型。

报告见`reports/navigation_dagger_round1.md`、`reports/navigation_random_relu_clearance_training.md`和`reports/learning_navigation_random_relu_clearance_benchmark.md`。

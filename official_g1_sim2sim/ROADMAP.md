# G1 Navigation Pipeline 开发路线

更新时间：2026-08-05 15:55 CST

## 不可变更的基线

- 不重新训练 Walking Policy。
- 不修改 Unitree 官方 ONNX。
- 保持 `Depth → Costmap → Planner → Velocity → Official ONNX → MuJoCo`。
- 当前阶段不接 DDS、`LowCmd` 或真实机器人。

## 模块边界

| 目录 | 职责 | 状态 |
|---|---|---|
| `navigation/` | 导航运行时、场景定义、单次日志 | 已建立 |
| `camera/` | 深度相机稳定入口 | 已建立 |
| `costmap/` | 局部代价地图稳定入口 | 已建立 |
| `planner/` | DWA 局部规划稳定入口 | 已建立 |
| `benchmark/` | 批量统计和报告 | Priority 1 已建立 |
| `visualization/` | Pipeline Debug Viewer | Priority 2 已完成 |
| `logging/` | Priority 5 日志边界说明 | 待扩展 |

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
2. **Priority 3：Ground Segmentation（下一阶段）**：使用 RANSAC/Least Square/Gravity Constraint 替换 MuJoCo geom ID，并用相同 Benchmark 做回归。
3. **Priority 4：Monte Carlo**：参数化障碍宽度、间距、高度、目标点和随机地图，运行 100 次统计。
4. **Priority 5：Debug Logging**：补齐 Depth FPS、Planning FPS/Time、Control Latency、Velocity、Pose、Goal Error 和 Collision Event 日志。
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

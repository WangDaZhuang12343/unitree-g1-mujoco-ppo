# MuJoCo 视觉自主导航架构

基线提交：`0c191e2`。

## 模块边界

| 层 | 模块 | 输入 → 输出 | 属性 |
|---|---|---|---|
| 场景/仿真 | `navigation/scenarios.py`、`simulate.py` | 场景 → MuJoCo模型/状态 | 后端相关 |
| 深度感知 | `g1_nav/l4_camera.py` | 几何 → 64×40深度/机器人坐标点云 | 相机后端需替换 |
| 地面分割 | `camera/ground_segmentation.py` | 点云+重力 → 地面平面/障碍点 | NumPy算法 |
| 局部地图 | `g1_nav/l2_costmap.py` | 障碍点+平面 → 占据图/距离场 | CPU NumPy/SciPy |
| 上层导航 | `planner/` | Costmap+局部目标 → 3维速度 | `LocalNavigator`可替换 |
| 安全层 | `g1_nav/l6_safety.py` | 期望速度+姿态 → 限幅/斜坡/急停速度 | 必须保留 |
| 底层步态 | `policy_contract.py`、`OrtRunner` | 官方480维观测 → 29维关节目标 | 官方ONNX，不可修改 |
| 执行/日志 | `navigation/runtime.py`、`run_log.py` | 关节目标 → PD/物理/日志 | 后端相关 |
| 评估 | `benchmark/`、`benchmark_*.py` | 运行结果 → CSV/成功率/Wilson区间 | 判据应跨后端复用 |

## 在线数据流

```text
MuJoCo mj_multiRay (10 Hz)
 → Depth 64×40 / PointCloud(robot frame)
 → gravity-constrained RANSAC + least-squares plane
 → plane-relative obstacle points
 → LocalCostMap (occupancy + distance field)
 → LocalNavigator [DWA default / learned experimental]
 → desired (vx, vy, omega)
 → SafetySystem
 → applied velocity command
 → official Walking Policy observation history (480)
 → immutable Unitree ONNX action (29)
 → action scale/offset + joint map + PD torque
 → MuJoCo physics
```

感知/规划周期0.10秒；步态策略周期由官方配置`step_dt`决定。上层导航器只产生速度，不接触关节、官方观测拼接或PD控制。

## 关键合同

- `PolicyContract`强制官方观测480维、动作29维、关节映射为0～28完整排列。
- `LocalNavigator.plan(costmap, goal_body)`统一返回`PlanResult(vx, vy, omega, score, trajectory)`。
- `NavigationRunConfig.navigator`支持注入；未指定时始终使用`DWANavigator`。
- 所有导航器输出经过同一`SafetySystem`，学习策略不能绕过安全层。
- 成功定义：目标误差不超过0.30米、机器人存活、障碍碰撞事件为0。

## 模仿学习流程

```text
procedural maps + goals → DWA labels
 → 800 samples (600 train / 200 validation)
 → compact occupancy/clearance features
 → ridge / random-ReLU / temporal random-ReLU
 → LearnedNavigator safety adapter
 → same MuJoCo closed-loop benchmark
```

DAgger第一轮在失败策略闭环状态上用DWA重标注500条；训练使用单/双障碍325条，窄通道175条保留为未见场景检查。

## 评估流程

1. 单元测试验证相机、分割、Costmap、规划合同和官方策略元数据。
2. 11个确定性场景输出逐帧日志和汇总。
3. 100次参数化Monte Carlo按5个测试族运行，支持checkpoint恢复。
4. DWA与学习策略使用相同场景、Walking Policy、SafetySystem、成功判据闭环比较。
5. Debug Viewer同步显示Depth、PointCloud、Costmap、候选/选中轨迹、位姿和目标。

未来RL只替换`Costmap/features → LocalNavigator → velocity`，底层Walking链保持独立。

# 迁移就绪度审查

范围：提交`0c191e2`的MuJoCo代码。只评估，不实现Isaac Lab。

## 已具备的解耦

| 能力 | 证据 | 结论 |
|---|---|---|
| 上层动作独立 | `LocalNavigator`只输出3维速度 | 可复用边界 |
| 导航器可注入 | `NavigationRunConfig.navigator`，DWA默认 | 可保留DWA baseline/fallback |
| 底层合同明确 | `PolicyContract`验证480→29、映射和缩放 | 可跨后端对照 |
| 安全层独立 | `SafetySystem.update()`不依赖MuJoCo对象 | 语义可迁移 |
| 场景数据化 | `NavigationScenario`/`ObstacleBox` | 可建立后端场景映射 |
| 评估结构化 | `NavigationRunResult`、CSV、Wilson统计 | 可复用判据 |
| 感知无geom决策 | RANSAC/Costmap不读geom ID | 有利于迁移 |
| 可复现 | 固定种子、checkpoint、模型与报告 | 基线可追踪 |

## 尚未就绪

1. `run_navigation()`把模型、感知、规划、Walking、PD、碰撞和日志放在单一MuJoCo循环中，不是Gym/VecEnv环境。
2. `SimulatedDepthCamera`直接调用`mj_multiRay`，需批量深度/射线后端。
3. RANSAC、Costmap、DWA是CPU NumPy/SciPy，不能直接进入GPU并行step热路径。
4. `OrtRunner`是本机C++ ORT桥；训练机需验证批量推理、device、历史和关节顺序。
5. `SafetySystem`保存单机器人NumPy状态，未来需每环境独立状态并保持相同语义。
6. 碰撞和动态障碍仍使用MuJoCo geom handle，需映射到PhysX contact与actor状态。
7. 当前上层学习观测不是正式483维张量合同，需按接口文档固化shape测试。
8. 当前Benchmark串行；VecEnv final info、自动reset和episode聚合需验证。

## 推荐适配边界

```text
Backend adapter
 ├─ robot state batch
 ├─ depth/pointcloud batch
 ├─ contact/fall batch
 └─ apply joint targets batch

Navigation environment contract
 ├─ build upper observation (483)
 ├─ map action to desired velocity (3)
 ├─ apply Safety semantics per env
 ├─ build official walking observation (480, separate)
 ├─ run immutable Walking Policy (29)
 └─ reward/done/info
```

不要直接移植`run_navigation()`主循环；保留合同和判据，把仿真器调用放入后端适配层。

## 训练机验收顺序

### 静态合同

- 确认Isaac Lab/Sim、PyTorch、训练器版本和API。
- 断言上层483、动作3、底层480、Walking输出29。
- 对照29关节名称/顺序、默认位置、缩放、PD和周期。

### 32环境冒烟

- G1资产加载、初始姿态、接触和reset隔离。
- Walking批量输出有限，零命令稳定站立。
- 固定速度命令响应方向与MuJoCo一致。
- reward分项、terminated/truncated、`info['success']`和final info正确。
- fallback开关不改变Walking/Safety链。

### 扩展64/128与策略验收

- 记录显存、物理FPS、策略FPS、step延迟和reset稳定性。
- 固定种子复现短训练，无NaN、历史串环境或统计丢失。
- 训练/验证/测试种子隔离；先过11场景，再跑相同100次Monte Carlo并报告Wilson区间。

## 结论

项目在分层合同、DWA fallback和评估定义上具备迁移基础；批量环境、GPU张量感知、Walking批量推理和VecEnv episode语义尚未实现。独立迁移分支缺失时应等待同步，不创建新的`isaaclab_port/`。

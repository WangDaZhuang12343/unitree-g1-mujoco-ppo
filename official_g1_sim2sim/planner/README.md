# 可替换导航策略合同

`LocalNavigator` 是 Costmap 到速度计划的唯一替换边界。DWA 和学习策略都实现：

```text
LocalCostMap + goal_body → PlanResult(vx, vy, omega, score, trajectory)
```

该边界不接触关节、力矩、官方 ONNX Walking Policy 或 MuJoCo 控制器。输出仍需经过
现有 `SafetySystem` 的速度斜坡、姿态和急停检查。

## 学习策略输入输出

`LearnedNavigator` 接收一个框架无关的 `predictor(observation)` 可调用对象：

- 输入前2维为归一化机器人坐标目标 `(goal_x/front_range, goal_y/side_range)`；
- 后续维度为按行展开的二值局部占据图；默认Costmap下共7202维；
- 输出为物理单位 `[vx(m/s), vy(m/s), omega(rad/s)]`，必须是3个有限数值。

适配器对输出执行限幅，并用独立前向轨迹检查否决碰撞命令。被否决的命令转换为
零速度，原候选轨迹仍进入Debug Viewer，便于定位策略问题。

当前仓库只固化接口、安全适配器和可复现的假策略基线，不声称已有训练完成的学习
模型。后续模型可使用ONNX、PyTorch或其他运行时，只需包装成上述可调用对象，并用
现有11场景Benchmark和100次Monte Carlo做同口径对比。

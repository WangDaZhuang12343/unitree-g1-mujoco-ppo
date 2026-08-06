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
适配器还根据已测得的官方步态低速死区，将小于激活阈值的前进输出置零，将已激活但
低于0.25 m/s的输出整形到最低连续行走速度。

当前仓库固化接口、安全适配器和明确标注验收结果的实验模型，不把失败基线描述为
可用策略。后续模型可使用ONNX、PyTorch或其他运行时，只需包装成上述可调用对象，
并用现有11场景Benchmark和100次Monte Carlo做同口径对比。

## 当前训练基线

`train_navigation_policy.py`支持岭回归与确定性随机ReLU特征两种纯NumPy模型，也可
复用本地教师NPZ并追加`collect_navigation_dagger.py`生成的闭环重标注数据。已发布的
前两个无时序模型均在三场景闭环测试中0/3成功，只用于复现实验和后续改进，不是部署候选。

时序候选在紧凑地图和距离场之外维护帧间特征变化与上一输出，并通过`reset()`确保场景
之间状态隔离。它改善了单/双障碍的最终目标距离，但闭环仍为0/3成功，同样不属于部署候选。

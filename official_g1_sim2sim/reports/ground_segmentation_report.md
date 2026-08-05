# Ground Segmentation 回归报告

测试日期：2026-08-05

## 实现

```text
Depth → 重力约束 RANSAC 平面 → 最小二乘精修
      → Ground / Obstacle Cloud → Plane-relative Costmap
```

- 平面模型：`normal · point + offset = 0`
- RANSAC 距离阈值：0.018 米
- 障碍物离地阈值：0.025 米
- 最大地面倾角：25 度
- 稀疏帧：使用最近有效平面，不中断导航

## 测试结果

- 22 项单元测试全部通过。
- 合成带噪斜地法向量一致性大于 0.995，地面分类率大于 98%，障碍点检出率大于 95%。
- 仿真单障碍首帧与 geom 真值对照：地面分类大于 99%，障碍点检出大于 95%。
- 单障碍 741 帧与旧基线完全一致：位置和 `(vx, vy, omega)` 最大差异均为 0。
- 11 场景仍为 8 成功、3 失败，总碰撞事件为 0。
- 10 个场景的逐帧位置和速度指令完全一致。
- 窄通道提前 0.02 秒到达，最大位置差 8.6 毫米，最大指令差 0.05，成功和零碰撞判定不变。

## 性能

| 环节 | 耗时 |
|---|---:|
| `mj_multiRay` 相机 | 约 2 ms |
| Ground Segmentation | 约 5～9 ms |
| Costmap | 约 2～4 ms |
| Python DWA | 约 410～714 ms |

当前性能主要瓶颈是 DWA 的 Python 候选轨迹循环，不是 Ground Segmentation。本阶段不修改 DWA 算法，后续在 Priority 5 做等价性能优化。

## 依赖边界

- 导航点云分类和 Costmap 输入不读取 `floor_id`、`point_geom_ids` 或 `geom_ids`。
- MuJoCo 动态障碍位置更新和 Benchmark 碰撞真值仍使用 geom handle。
- 上述 geom handle 不参与感知、Costmap 或 DWA 决策。

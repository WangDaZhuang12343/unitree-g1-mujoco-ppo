# DWA 与轻量学习导航闭环对比

| 场景 | 规划器 | 成功 | 碰撞 | 仿真时间(s) | 路径(m) | 最小净空(m) | 最终距离(m) | 规划(ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single_obstacle | dwa | 1 | 0 | 14.80 | 5.20 | 0.241 | 0.298 | 173.93 |
| single_obstacle | random_relu_clearance_v2 | 0 | 0 | 30.00 | 10.40 | 0.044 | 4.493 | 3.64 |
| double_obstacle | dwa | 1 | 0 | 17.70 | 6.08 | 0.329 | 0.300 | 170.34 |
| double_obstacle | random_relu_clearance_v2 | 0 | 0 | 35.00 | 0.78 | 1.030 | 4.900 | 5.73 |
| narrow_corridor | dwa | 1 | 0 | 12.96 | 5.26 | 0.294 | 0.297 | 169.92 |
| narrow_corridor | random_relu_clearance_v2 | 0 | 0 | 35.00 | 0.04 | 0.732 | 5.513 | 5.80 |

学习策略与DWA使用相同场景、深度感知、Costmap、Safety System、官方Walking Policy和成功判据。
逐帧CSV保留在本地，不纳入Git。

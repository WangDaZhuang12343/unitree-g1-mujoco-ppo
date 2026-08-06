# DWA 与轻量学习导航闭环对比

| 场景 | 规划器 | 成功 | 碰撞 | 仿真时间(s) | 路径(m) | 最小净空(m) | 最终距离(m) | 规划(ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single_obstacle | dwa | 1 | 0 | 14.80 | 5.20 | 0.241 | 0.298 | 153.09 |
| single_obstacle | random_relu_v1 | 0 | 0 | 30.00 | 10.18 | -0.055 | 4.419 | 2.79 |
| double_obstacle | dwa | 1 | 0 | 17.70 | 6.08 | 0.329 | 0.300 | 164.55 |
| double_obstacle | random_relu_v1 | 0 | 0 | 35.00 | 0.74 | 1.043 | 4.891 | 4.81 |
| narrow_corridor | dwa | 1 | 0 | 12.96 | 5.26 | 0.294 | 0.297 | 171.98 |
| narrow_corridor | random_relu_v1 | 0 | 0 | 35.00 | 0.03 | 0.733 | 5.508 | 4.83 |

学习策略与DWA使用相同场景、深度感知、Costmap、Safety System、官方Walking Policy和成功判据。
逐帧CSV保留在本地，不纳入Git。

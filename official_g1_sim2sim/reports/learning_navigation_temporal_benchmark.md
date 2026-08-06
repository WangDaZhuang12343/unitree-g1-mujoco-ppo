# DWA 与轻量学习导航闭环对比

| 场景 | 规划器 | 成功 | 碰撞 | 仿真时间(s) | 路径(m) | 最小净空(m) | 最终距离(m) | 规划(ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single_obstacle | dwa | 1 | 0 | 14.80 | 5.20 | 0.241 | 0.298 | 186.71 |
| single_obstacle | temporal_random_relu_v1 | 0 | 0 | 30.00 | 9.90 | 0.057 | 3.699 | 2.59 |
| double_obstacle | dwa | 1 | 0 | 17.70 | 6.08 | 0.329 | 0.300 | 167.72 |
| double_obstacle | temporal_random_relu_v1 | 0 | 0 | 35.00 | 11.48 | 0.343 | 3.534 | 3.56 |
| narrow_corridor | dwa | 1 | 0 | 12.96 | 5.26 | 0.294 | 0.297 | 179.19 |
| narrow_corridor | temporal_random_relu_v1 | 0 | 0 | 35.00 | 0.17 | 0.727 | 5.502 | 4.71 |

学习策略与DWA使用相同场景、深度感知、Costmap、Safety System、官方Walking Policy和成功判据。
逐帧CSV保留在本地，不纳入Git。

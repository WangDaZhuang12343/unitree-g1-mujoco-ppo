# DWA 与轻量学习导航闭环对比

| 场景 | 规划器 | 成功 | 碰撞 | 仿真时间(s) | 路径(m) | 最小净空(m) | 最终距离(m) | 规划(ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single_obstacle | dwa | 1 | 0 | 14.80 | 5.20 | 0.241 | 0.298 | 158.99 |
| single_obstacle | learned_ridge | 0 | 0 | 30.00 | 2.02 | 1.358 | 4.146 | 4.57 |
| double_obstacle | dwa | 1 | 0 | 17.70 | 6.08 | 0.329 | 0.300 | 162.22 |
| double_obstacle | learned_ridge | 0 | 0 | 35.00 | 1.52 | 1.632 | 5.450 | 4.87 |
| narrow_corridor | dwa | 1 | 0 | 12.96 | 5.26 | 0.294 | 0.297 | 162.06 |
| narrow_corridor | learned_ridge | 0 | 0 | 35.00 | 0.04 | 0.733 | 5.512 | 4.96 |

学习策略与DWA使用相同场景、深度感知、Costmap、Safety System、官方Walking Policy和成功判据。
逐帧CSV保留在本地，不纳入Git。

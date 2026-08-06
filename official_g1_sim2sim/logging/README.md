# Logging 模块

结构化调试日志的运行时实现在 `navigation/run_log.py`，避免顶层包名
`logging` 覆盖 Python 标准库。每次导航运行的 CSV 使用同一时间轴记录：

- 仿真时间、世界坐标位姿和目标距离；
- 安全层输出指令、DWA 规划速度和实际机体速度；
- Depth FPS、Planning FPS、单帧规划耗时和规划到控制应用延迟；
- 当前净空、单周期碰撞事件数和累计碰撞事件数。

`control_latency_ms` 从最新规划结果生成后开始计时，到安全层在策略周期应用该
结果时结束；两个感知周期之间沿用最近一次测量值。`collision_event` 是上一个日志
周期以来的新接触事件数，`collision_count` 是整次运行的累计值。

日志只写入用户指定的 `--output` 路径，不在无输出配置下积累逐帧数据文件。

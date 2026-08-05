"""导航运行过程 CSV 输出。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


NAVIGATION_LOG_FIELDS = (
    "time_s", "x_m", "y_m", "height_m", "yaw_rad",
    "cmd_vx", "cmd_vy", "cmd_omega", "plan_vx", "plan_vy", "plan_omega",
    "plan_score", "goal_distance_m", "clearance_m", "planning_ms", "collision_count",
)


def write_navigation_log(path: Path, rows: Iterable[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(NAVIGATION_LOG_FIELDS)
        writer.writerows(rows)

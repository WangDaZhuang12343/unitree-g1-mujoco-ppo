#!/usr/bin/env python3
"""测量官方 G1 策略对侧向速度和偏航角速度指令的真实响应。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs/command_benchmark"
DURATION = 8.0

CASES = [
    ("forward", 0.35, 0.0, 0.0),
    ("lateral_left_010", 0.0, 0.10, 0.0),
    ("lateral_right_010", 0.0, -0.10, 0.0),
    ("mixed_left_010", 0.35, 0.10, 0.0),
    ("mixed_right_010", 0.35, -0.10, 0.0),
    ("turn_left_010", 0.35, 0.0, 0.10),
    ("turn_right_010", 0.35, 0.0, -0.10),
    ("turn_left_020", 0.35, 0.0, 0.20),
    ("turn_right_020", 0.35, 0.0, -0.20),
]


def metrics(path: Path, command: tuple[float, float, float]) -> dict:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    time_s = np.asarray([float(row["time_s"]) for row in rows])
    x = np.asarray([float(row["x_m"]) for row in rows])
    y = np.asarray([float(row["y_m"]) for row in rows])
    yaw = np.unwrap(np.asarray([float(row["yaw_rad"]) for row in rows]))
    elapsed = float(time_s[-1])
    return {
        "cmd_vx": command[0],
        "cmd_vy": command[1],
        "cmd_omega": command[2],
        "elapsed": elapsed,
        "mean_world_vx": float((x[-1] - x[0]) / elapsed),
        "mean_world_vy": float((y[-1] - y[0]) / elapsed),
        "mean_omega": float((yaw[-1] - yaw[0]) / elapsed),
        "final_yaw_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "survived": elapsed >= DURATION - 0.1,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, vx, vy, omega in CASES:
        output = OUTPUT_DIR / f"{name}.csv"
        print(f"\n测试 {name}: ({vx:+.2f}, {vy:+.2f}, {omega:+.2f})", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "simulate.py"),
                "--duration",
                str(DURATION),
                "--vx",
                str(vx),
                "--vy",
                str(vy),
                "--yaw",
                str(omega),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        result = metrics(output, (vx, vy, omega))
        result["name"] = name
        results.append(result)

    fields = [
        "name", "cmd_vx", "cmd_vy", "cmd_omega", "elapsed", "mean_world_vx",
        "mean_world_vy", "mean_omega", "final_yaw_deg", "survived",
    ]
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# G1 速度指令响应测试",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "| 测试 | 指令 vx | 指令 vy | 指令 omega | 实测世界 vx | 实测世界 vy | 实测 omega | 最终转角 | 存活 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['cmd_vx']:+.2f} | {row['cmd_vy']:+.2f} | {row['cmd_omega']:+.2f} | "
            f"{row['mean_world_vx']:+.3f} | {row['mean_world_vy']:+.3f} | {row['mean_omega']:+.3f} | "
            f"{row['final_yaw_deg']:+.1f}度 | {'是' if row['survived'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "说明：位移速度在世界坐标系统计；转弯时不能直接等同于机身坐标速度。",
        ]
    )
    report = OUTPUT_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告：{report}", flush=True)


if __name__ == "__main__":
    main()

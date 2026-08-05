#!/usr/bin/env python3
"""分级测试官方 G1 策略跨越低矮横向障碍的能力。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heights", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--vx", type=float, default=0.45, help="前向速度指令（米/秒）")
    parser.add_argument("--duration", type=float, default=8.0, help="单次测试时长（秒）")
    parser.add_argument("--obstacle-x", type=float, default=1.5, help="障碍中心位置 x（米）")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/obstacle_benchmark")
    return parser.parse_args()


def read_result(path: Path, duration: float, obstacle_x: float) -> dict[str, float | bool]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"elapsed": 0.0, "x": 0.0, "y": 0.0, "yaw_deg": 0.0, "passed": False}
    last = rows[-1]
    elapsed = float(last["time_s"])
    x = float(last["x_m"])
    passed = elapsed >= duration - 0.1 and x >= obstacle_x + 0.5
    return {
        "elapsed": elapsed,
        "x": x,
        "y": float(last["y_m"]),
        "yaw_deg": float(last["yaw_rad"]) * 57.295779513,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for height in args.heights:
        output = args.output_dir / f"bar_{height:.3f}m.csv"
        command = [
            sys.executable,
            str(ROOT / "simulate.py"),
            "--terrain",
            "bar",
            "--obstacle-height",
            str(height),
            "--obstacle-x",
            str(args.obstacle_x),
            "--vx",
            str(args.vx),
            "--duration",
            str(args.duration),
            "--output",
            str(output),
        ]
        print(f"\n测试障碍高度 {height:.3f} 米", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        result = read_result(output, args.duration, args.obstacle_x)
        result["height"] = height
        results.append(result)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["height", "elapsed", "x", "y", "yaw_deg", "passed"])
        writer.writeheader()
        writer.writerows(results)

    report_path = args.output_dir / "report.md"
    lines = [
        "# G1 横向障碍分级测试",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"速度指令：{args.vx:.2f} 米/秒；单次时长：{args.duration:.1f} 秒；障碍位置：x={args.obstacle_x:.2f} 米。",
        "",
        "| 障碍高度（米） | 存活时间（秒） | 最终 x（米） | 横向偏移（米） | 偏航（度） | 结果 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['height']:.3f} | {result['elapsed']:.2f} | {result['x']:.3f} | "
            f"{result['y']:.3f} | {result['yaw_deg']:.1f} | {'通过' if result['passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "判定标准：机器人运行完整测试时长，且机身 x 坐标超过障碍物后方 0.5 米。",
            "",
            "当前策略没有地形高度观测，本测试衡量的是盲走鲁棒性，不代表具备视觉越障规划能力。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    passed = [float(result["height"]) for result in results if result["passed"]]
    print(f"\n报告：{report_path}", flush=True)
    print(f"最高通过高度：{max(passed):.3f} 米" if passed else "没有通过任何测试高度", flush=True)


if __name__ == "__main__":
    main()

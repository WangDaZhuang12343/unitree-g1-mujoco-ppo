#!/usr/bin/env python3
"""评估官方 G1 策略在斜坡、台阶、楼梯和随机起伏地形上的表现。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs/terrain_benchmark"
VX = 0.45


CASES = [
    {"name": "ramp_5deg", "label": "斜坡 5 度", "terrain": "ramp", "args": ["--slope-angle", "5", "--obstacle-x", "1.8"], "duration": 12.0, "pass_x": 4.1},
    {"name": "ramp_8deg", "label": "斜坡 8 度", "terrain": "ramp", "args": ["--slope-angle", "8", "--obstacle-x", "1.8"], "duration": 12.0, "pass_x": 4.1},
    {"name": "ramp_10deg", "label": "斜坡 10 度", "terrain": "ramp", "args": ["--slope-angle", "10", "--obstacle-x", "1.8"], "duration": 12.0, "pass_x": 4.1},
    {"name": "step_2cm", "label": "单台阶 2 厘米", "terrain": "step", "args": ["--obstacle-height", "0.02"], "duration": 10.0, "pass_x": 2.4},
    {"name": "step_4cm", "label": "单台阶 4 厘米", "terrain": "step", "args": ["--obstacle-height", "0.04"], "duration": 10.0, "pass_x": 2.4},
    {"name": "step_6cm", "label": "单台阶 6 厘米", "terrain": "step", "args": ["--obstacle-height", "0.06"], "duration": 10.0, "pass_x": 2.4},
    {"name": "stairs_1cm", "label": "楼梯级高 1 厘米", "terrain": "stairs", "args": ["--obstacle-height", "0.01", "--obstacle-x", "1.0"], "duration": 10.0, "pass_x": 3.3},
    {"name": "stairs_2cm", "label": "楼梯级高 2 厘米", "terrain": "stairs", "args": ["--obstacle-height", "0.02", "--obstacle-x", "1.0"], "duration": 10.0, "pass_x": 3.3},
    {"name": "stairs_3cm", "label": "楼梯级高 3 厘米", "terrain": "stairs", "args": ["--obstacle-height", "0.03", "--obstacle-x", "1.0"], "duration": 10.0, "pass_x": 3.3},
    {"name": "rough_1cm", "label": "随机起伏 1 厘米", "terrain": "rough", "args": ["--roughness", "0.01", "--obstacle-x", "0.8"], "duration": 12.0, "pass_x": 4.3},
    {"name": "rough_2cm", "label": "随机起伏 2 厘米", "terrain": "rough", "args": ["--roughness", "0.02", "--obstacle-x", "0.8"], "duration": 12.0, "pass_x": 4.3},
    {"name": "rough_4cm", "label": "随机起伏 4 厘米", "terrain": "rough", "args": ["--roughness", "0.04", "--obstacle-x", "0.8"], "duration": 12.0, "pass_x": 4.3},
]


def read_metrics(path: Path, case: dict) -> dict:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    last = rows[-1]
    elapsed = float(last["time_s"])
    return {
        "name": case["name"],
        "label": case["label"],
        "elapsed": elapsed,
        "x": float(last["x_m"]),
        "y": float(last["y_m"]),
        "yaw_deg": float(last["yaw_rad"]) * 57.295779513,
        "passed": elapsed >= case["duration"] - 0.1 and float(last["x_m"]) >= case["pass_x"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        output = OUTPUT_DIR / f"{case['name']}.csv"
        command = [
            sys.executable,
            str(ROOT / "simulate.py"),
            "--terrain",
            case["terrain"],
            "--vx",
            str(VX),
            "--duration",
            str(case["duration"]),
            "--output",
            str(output),
            *case["args"],
        ]
        print(f"\n测试：{case['label']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        results.append(read_metrics(output, case))

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "label", "elapsed", "x", "y", "yaw_deg", "passed"])
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# G1 综合地形测试",
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"前向速度指令：{VX:.2f} 米/秒。所有地形使用固定参数和随机种子，可重复测试。",
        "",
        "| 地形 | 存活时间（秒） | 最终 x（米） | 横向偏移（米） | 偏航（度） | 结果 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['label']} | {result['elapsed']:.2f} | {result['x']:.3f} | "
            f"{result['y']:.3f} | {result['yaw_deg']:.1f} | {'通过' if result['passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "判定标准：运行完整测试时长，并越过地形终点后至少继续前进约 0.5 米。",
            "",
            "本报告测试无地形感知的官方预训练策略，只代表盲走鲁棒性。",
        ]
    )
    report = OUTPUT_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    passed_count = sum(bool(result["passed"]) for result in results)
    print(f"\n完成：{passed_count}/{len(results)} 项通过，报告：{report}", flush=True)


if __name__ == "__main__":
    main()

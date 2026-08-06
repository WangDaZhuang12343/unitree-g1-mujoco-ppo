#!/usr/bin/env python3
"""在相同MuJoCo场景中对比DWA与轻量学习导航策略。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from navigation import NavigationRunConfig, get_scenario, run_navigation
from planner import DWANavigator, LearnedNavigator, load_navigation_policy
from simulate import MODEL_PATH, ROOT, OrtRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="+",
        default=["single_obstacle", "double_obstacle", "narrow_corridor"],
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--model", type=Path, default=ROOT / "models/navigation_ridge_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/learning_benchmark")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/learning_navigation_benchmark.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_navigation_policy(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    runner = OrtRunner(MODEL_PATH)
    try:
        for scene_name in args.scenes:
            scenario = get_scenario(scene_name)
            learned_name = args.model.stem.removeprefix("navigation_")
            for planner_name, navigator in (
                ("dwa", DWANavigator()),
                (learned_name, LearnedNavigator(policy)),
            ):
                output = args.output / f"{scene_name}_{planner_name}.csv"
                result = run_navigation(
                    NavigationRunConfig(
                        scenario=scenario,
                        duration=args.duration,
                        output=output,
                        run_id=f"{scene_name}_{planner_name}",
                        navigator=navigator,
                    ),
                    policy_runner=runner,
                )
                records.append({
                    "scene": scene_name,
                    "planner": planner_name,
                    "success": result.success,
                    "reached": result.reached,
                    "survived": result.survived,
                    "collision_count": result.collision_count,
                    "elapsed_sim_s": result.elapsed_sim_s,
                    "path_length_m": result.path_length_m,
                    "min_clearance_m": result.min_clearance_m,
                    "final_distance_m": result.final_distance_m,
                    "average_planning_ms": result.average_planning_ms,
                })
                print(
                    f"{scene_name}/{planner_name}: success={result.success} "
                    f"collision={result.collision_count} distance={result.final_distance_m:.3f}m"
                )
    finally:
        runner.close()

    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# DWA 与轻量学习导航闭环对比", "",
        "| 场景 | 规划器 | 成功 | 碰撞 | 仿真时间(s) | 路径(m) | 最小净空(m) | 最终距离(m) | 规划(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['scene']} | {row['planner']} | {int(row['success'])} | "
            f"{row['collision_count']} | {row['elapsed_sim_s']:.2f} | {row['path_length_m']:.2f} | "
            f"{row['min_clearance_m']:.3f} | {row['final_distance_m']:.3f} | "
            f"{row['average_planning_ms']:.2f} |"
        )
    lines.extend([
        "", "学习策略与DWA使用相同场景、深度感知、Costmap、Safety System、官方Walking Policy和成功判据。",
        "逐帧CSV保留在本地，不纳入Git。", "",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

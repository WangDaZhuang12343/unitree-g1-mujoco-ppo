#!/usr/bin/env python3
"""打开 G1 Navigation Pipeline 实时四联调试界面。"""

from __future__ import annotations

import argparse
from pathlib import Path

from navigation import NavigationRunConfig, get_scenario, run_navigation, scenario_names
from simulate import ROOT
from visualization import NavigationDebugViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=scenario_names(), default="single_obstacle")
    parser.add_argument("--duration", type=float, help="覆盖场景默认时长")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/navigation_debug/latest.csv")
    parser.add_argument("--snapshot", type=Path, help="结束时保存最后一帧 PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = get_scenario(args.scene, seed=args.seed)
    viewer = NavigationDebugViewer(f"G1 Navigation Debug - {scenario.name}")
    try:
        result = run_navigation(NavigationRunConfig(
            scenario=scenario,
            duration=args.duration,
            output=args.output,
            realtime=True,
            debug_callback=viewer.update,
        ))
        if args.snapshot and viewer.is_open:
            viewer.save(args.snapshot)
    finally:
        viewer.close()
    print(
        f"结果：scenario={result.scenario} success={result.success} "
        f"collision={result.collision_count} final_distance={result.final_distance_m:.3f}m "
        f"output={result.output}"
    )


if __name__ == "__main__":
    main()

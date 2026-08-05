#!/usr/bin/env python3
"""单障碍视觉导航兼容入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from navigation import NavigationRunConfig, NavigationScenario, ObstacleBox, run_navigation
from simulate import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--obstacle-x", type=float, default=2.5)
    parser.add_argument("--obstacle-height", type=float, default=0.15)
    parser.add_argument("--obstacle-width", type=float, default=0.8)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/navigation/latest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = NavigationScenario(
        name="single_obstacle",
        description="单障碍兼容场景",
        goal=(args.goal_x, args.goal_y),
        obstacles=(
            ObstacleBox(
                name="single",
                x=args.obstacle_x,
                y=0.0,
                size_x=0.12,
                size_y=args.obstacle_width,
                height=args.obstacle_height,
            ),
        ),
        duration=args.duration,
    )
    result = run_navigation(
        NavigationRunConfig(
            scenario=scenario,
            duration=args.duration,
            output=args.output,
            viewer=args.viewer,
            realtime=args.viewer,
        )
    )
    print(
        f"结果：reached={result.reached} collision_count={result.collision_count} "
        f"final_distance={result.final_distance_m:.3f}m survived={result.survived} "
        f"path={result.path_length_m:.2f}m min_clearance={result.min_clearance_m:.3f}m "
        f"fps={result.fps:.1f} output={args.output}"
    )


if __name__ == "__main__":
    main()

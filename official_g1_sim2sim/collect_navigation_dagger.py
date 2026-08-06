#!/usr/bin/env python3
"""在学习策略闭环状态上查询DWA教师动作，生成DAgger聚合数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from costmap import LocalCostMap
from navigation import NavigationDebugFrame, NavigationRunConfig, get_scenario, run_navigation
from planner import DWANavigator, LearnedNavigator, load_navigation_policy
from simulate import MODEL_PATH, ROOT, OrtRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="+",
        default=["single_obstacle", "double_obstacle", "narrow_corridor"],
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stride", type=int, default=2, help="每N个感知帧保留一个教师样本")
    parser.add_argument("--model", type=Path, default=ROOT / "models/navigation_ridge_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/learning/dagger_round1.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/learning/dagger_round1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride 必须大于等于1")
    policy = load_navigation_policy(args.model)
    observations: list[np.ndarray] = []
    targets: list[list[float]] = []
    scene_counts: dict[str, int] = {}
    runner = OrtRunner(MODEL_PATH)
    try:
        for scene_name in args.scenes:
            scenario = get_scenario(scene_name)
            teacher = DWANavigator()
            frame_index = 0
            start_count = len(observations)

            def collect(frame: NavigationDebugFrame) -> bool:
                nonlocal frame_index
                keep = frame_index % args.stride == 0
                frame_index += 1
                if not keep:
                    return True
                costmap = LocalCostMap()
                costmap.obstacle_map[:] = frame.obstacle_map
                costmap.distance_field[:] = frame.distance_field
                goal = frame.goal_body
                observation = LearnedNavigator.encode_observation(costmap, goal)
                label = teacher.plan(costmap, goal)
                observations.append(observation)
                targets.append([label.vx, label.vy, label.omega])
                return True

            result = run_navigation(
                NavigationRunConfig(
                    scenario=scenario,
                    duration=args.duration,
                    output=None,
                    run_id=f"dagger_{scene_name}",
                    navigator=LearnedNavigator(policy),
                    debug_callback=collect,
                ),
                policy_runner=runner,
            )
            scene_counts[scene_name] = len(observations) - start_count
            print(
                f"{scene_name}: samples={scene_counts[scene_name]} reached={result.reached} "
                f"collision={result.collision_count} final_distance={result.final_distance_m:.3f}m"
            )
    finally:
        runner.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.stack(observations),
        targets=np.asarray(targets, dtype=np.float32),
    )
    summary = {
        "source_model": str(args.model),
        "scenes": args.scenes,
        "stride": args.stride,
        "scene_samples": scene_counts,
        "total_samples": len(observations),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""批量运行 G1 Navigation Benchmark 并生成 CSV/Markdown 报告。"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import write_benchmark_reports
from navigation import NavigationRunConfig, get_scenario, run_navigation, scenario_names
from simulate import MODEL_PATH, ROOT, OrtRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", default="all",
        help="all 或逗号分隔场景名：" + ",".join(scenario_names()),
    )
    parser.add_argument("--repetitions", type=int, default=1, help="每个场景重复次数")
    parser.add_argument("--duration", type=float, help="覆盖场景默认时长，主要用于快速回归")
    parser.add_argument("--seed", type=int, default=7, help="随机场景基础种子")
    parser.add_argument("--exclude-dynamic", action="store_true", help="暂不运行动态障碍")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/navigation_benchmark")
    return parser.parse_args()


def selected_scenes(value: str, include_dynamic: bool) -> tuple[str, ...]:
    available = scenario_names(include_dynamic=include_dynamic)
    if value == "all":
        return available
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"未知场景：{', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("至少选择一个场景")
    return selected


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions 必须大于等于1")
    scenes = selected_scenes(args.scenes, include_dynamic=not args.exclude_dynamic)
    results = []
    runner = OrtRunner(MODEL_PATH)
    try:
        for scene_name in scenes:
            for repetition in range(args.repetitions):
                seed = args.seed + repetition
                scenario = get_scenario(scene_name, seed=seed)
                run_id = f"run_{repetition + 1:03d}"
                output = args.output / "runs" / scene_name / f"{run_id}.csv"
                print(f"[{scene_name}] {run_id} seed={seed}")
                result = run_navigation(
                    NavigationRunConfig(
                        scenario=scenario, duration=args.duration, output=output, run_id=run_id
                    ),
                    policy_runner=runner,
                )
                results.append(result)
                print(
                    f"  success={result.success} collision={result.collision_count} "
                    f"time={result.elapsed_sim_s:.2f}s path={result.path_length_m:.2f}m "
                    f"clearance={result.min_clearance_m:.3f}m fps={result.fps:.1f}"
                )
    finally:
        runner.close()
    write_benchmark_reports(args.output, results)
    print(f"报告：{args.output / 'navigation_report.md'}")


if __name__ == "__main__":
    main()

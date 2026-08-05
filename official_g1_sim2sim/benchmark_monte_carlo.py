#!/usr/bin/env python3
"""可恢复的 G1 Navigation 100 次 Monte Carlo 批处理。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from benchmark.monte_carlo import (
    MonteCarloRecord,
    append_checkpoint,
    generate_samples,
    load_checkpoint,
    write_monte_carlo_report,
)
from navigation import NavigationRunConfig, run_navigation
from simulate import MODEL_PATH, ROOT, OrtRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100, help="总样本数，必须是5的倍数")
    parser.add_argument("--base-seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/navigation_monte_carlo")
    parser.add_argument("--resume", action="store_true", help="跳过 checkpoint 中已完成的 run_id")
    parser.add_argument("--max-new-runs", type=int, help="本次最多新跑几次，用于分批或烟雾测试")
    parser.add_argument("--duration", type=float, help="覆盖场景时长，仅用于快速回归")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = generate_samples(args.runs, args.base_seed)
    checkpoint = args.output / "checkpoint.csv"
    records = load_checkpoint(checkpoint) if args.resume else []
    if checkpoint.exists() and records and not args.resume:
        raise RuntimeError(f"{checkpoint} 已存在；使用 --resume 继续，或更换 --output")
    completed = {record.run_id for record in records}
    pending = [sample for sample in samples if sample.run_id not in completed]
    if args.max_new_runs is not None:
        if args.max_new_runs < 1:
            raise ValueError("max-new-runs 必须大于等于1")
        pending = pending[: args.max_new_runs]
    print(f"Monte Carlo: completed={len(records)} pending_this_run={len(pending)} total={len(samples)}")
    runner = OrtRunner(MODEL_PATH)
    batch_start = time.perf_counter()
    try:
        for sample_index, sample in enumerate(pending, start=1):
            output = args.output / "runs" / sample.profile / f"{sample.run_id}.csv"
            print(
                f"[{len(records) + 1}/{len(samples)}] {sample.run_id} "
                f"params={sample.parameters}"
            )
            result = run_navigation(
                NavigationRunConfig(
                    scenario=sample.scenario,
                    duration=args.duration,
                    output=output,
                    run_id=sample.run_id,
                ),
                policy_runner=runner,
            )
            record = MonteCarloRecord.from_result(sample, result)
            append_checkpoint(checkpoint, record)
            records.append(record)
            write_monte_carlo_report(args.output, records, expected_runs=len(samples))
            average_wall = (time.perf_counter() - batch_start) / sample_index
            remaining = len(samples) - len(records)
            print(
                f"  success={result.success} collision={result.collision_count} "
                f"sim={result.elapsed_sim_s:.1f}s wall={result.elapsed_wall_s:.1f}s "
                f"progress={len(records)}/{len(samples)} eta={average_wall * remaining / 60.0:.1f}min"
            )
    except KeyboardInterrupt:
        print(f"\n已中断，完成 {len(records)}/{len(samples)}；使用 --resume 续跑。")
        if records:
            write_monte_carlo_report(args.output, records, expected_runs=len(samples))
        raise SystemExit(130)
    finally:
        runner.close()
    if records:
        write_monte_carlo_report(args.output, records, expected_runs=len(samples))
    print(f"报告：{args.output / 'navigation_report.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the frozen checkpoint gate to an Isaac benchmark summary CSV."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab_nav.benchmark_gate import BenchmarkGateConfig, evaluate_benchmark, load_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--minimum_success_rate", type=float, default=0.20)
    parser.add_argument("--maximum_collision_rate", type=float, default=0.20)
    parser.add_argument("--maximum_fall_rate", type=float, default=0.20)
    parser.add_argument("--minimum_completed_scenarios", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decision = evaluate_benchmark(
        load_summary(args.summary),
        BenchmarkGateConfig(
            minimum_completed_scenarios=args.minimum_completed_scenarios,
            minimum_success_rate=args.minimum_success_rate,
            maximum_collision_rate=args.maximum_collision_rate,
            maximum_fall_rate=args.maximum_fall_rate,
        ),
    )
    payload = {
        "accepted": decision.accepted,
        "reasons": decision.reasons,
        "metrics": asdict(decision.metrics),
        "summary": str(args.summary.resolve()),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if decision.accepted else 2)


if __name__ == "__main__":
    main()

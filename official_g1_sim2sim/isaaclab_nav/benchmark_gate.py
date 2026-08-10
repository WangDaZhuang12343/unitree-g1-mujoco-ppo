"""Pure-Python benchmark metrics and deployment-candidate gate."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkGateConfig:
    """Minimum evidence required before spending more training budget."""

    minimum_completed_scenarios: int = 10
    minimum_success_rate: float = 0.20
    maximum_collision_rate: float = 0.20
    maximum_fall_rate: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_completed_scenarios <= 0:
            raise ValueError("minimum_completed_scenarios must be positive")
        for name, value in (
            ("minimum_success_rate", self.minimum_success_rate),
            ("maximum_collision_rate", self.maximum_collision_rate),
            ("maximum_fall_rate", self.maximum_fall_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class BenchmarkMetrics:
    completed: int
    successes: int
    collision_scenarios: int
    fall_scenarios: int
    success_rate: float
    collision_rate: float
    fall_rate: float
    wilson_low: float
    wilson_high: float


@dataclass(frozen=True)
class BenchmarkGateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: BenchmarkMetrics


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - margin, center + margin


def summarize_rows(rows: list[dict[str, object]]) -> BenchmarkMetrics:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        raise ValueError("benchmark contains no completed static scenarios")
    successes = sum(_as_bool(row.get("success", False)) for row in completed)
    collisions = sum(int(float(row.get("collision_count", 0))) > 0 for row in completed)
    falls = sum(row.get("termination_reason") == "fall" for row in completed)
    total = len(completed)
    low, high = wilson_interval(successes, total)
    return BenchmarkMetrics(
        completed=total,
        successes=successes,
        collision_scenarios=collisions,
        fall_scenarios=falls,
        success_rate=successes / total,
        collision_rate=collisions / total,
        fall_rate=falls / total,
        wilson_low=low,
        wilson_high=high,
    )


def load_summary(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def evaluate_benchmark(
    rows: list[dict[str, object]], config: BenchmarkGateConfig | None = None
) -> BenchmarkGateDecision:
    cfg = config or BenchmarkGateConfig()
    metrics = summarize_rows(rows)
    reasons: list[str] = []
    if metrics.completed < cfg.minimum_completed_scenarios:
        reasons.append(
            f"completed {metrics.completed} < required {cfg.minimum_completed_scenarios}"
        )
    if metrics.success_rate < cfg.minimum_success_rate:
        reasons.append(
            f"success rate {metrics.success_rate:.3f} < {cfg.minimum_success_rate:.3f}"
        )
    if metrics.collision_rate > cfg.maximum_collision_rate:
        reasons.append(
            f"collision rate {metrics.collision_rate:.3f} > {cfg.maximum_collision_rate:.3f}"
        )
    if metrics.fall_rate > cfg.maximum_fall_rate:
        reasons.append(f"fall rate {metrics.fall_rate:.3f} > {cfg.maximum_fall_rate:.3f}")
    return BenchmarkGateDecision(not reasons, tuple(reasons), metrics)


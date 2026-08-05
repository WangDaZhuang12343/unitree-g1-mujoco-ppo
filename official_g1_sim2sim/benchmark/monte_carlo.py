"""Priority 4 参数化导航场景与 Monte Carlo 报告。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from navigation.runtime import NavigationRunResult
from navigation.scenarios import NavigationScenario, ObstacleBox


PROFILES = ("obstacle_width", "obstacle_spacing", "obstacle_height", "goal_offset", "random_map")


@dataclass(frozen=True)
class MonteCarloSample:
    run_id: str
    profile: str
    seed: int
    scenario: NavigationScenario
    parameters: dict[str, float | int]


@dataclass(frozen=True)
class MonteCarloRecord:
    run_id: str
    profile: str
    seed: int
    parameters_json: str
    scenario_json: str
    success: bool
    reached: bool
    survived: bool
    collision_count: int
    elapsed_sim_s: float
    elapsed_wall_s: float
    path_length_m: float
    min_clearance_m: float
    average_velocity_mps: float
    cpu_usage_percent: float
    fps: float
    depth_fps: float
    planning_fps: float
    average_planning_ms: float
    final_distance_m: float
    output: str

    @classmethod
    def from_result(cls, sample: MonteCarloSample, result: NavigationRunResult) -> "MonteCarloRecord":
        scenario_data = {
            "goal": sample.scenario.goal,
            "duration": sample.scenario.duration,
            "obstacles": [asdict(obstacle) for obstacle in sample.scenario.obstacles],
        }
        return cls(
            run_id=sample.run_id,
            profile=sample.profile,
            seed=sample.seed,
            parameters_json=json.dumps(sample.parameters, ensure_ascii=False, sort_keys=True),
            scenario_json=json.dumps(scenario_data, ensure_ascii=False, sort_keys=True),
            success=result.success,
            reached=result.reached,
            survived=result.survived,
            collision_count=result.collision_count,
            elapsed_sim_s=result.elapsed_sim_s,
            elapsed_wall_s=result.elapsed_wall_s,
            path_length_m=result.path_length_m,
            min_clearance_m=result.min_clearance_m,
            average_velocity_mps=result.average_velocity_mps,
            cpu_usage_percent=result.cpu_usage_percent,
            fps=result.fps,
            depth_fps=result.depth_fps,
            planning_fps=result.planning_fps,
            average_planning_ms=result.average_planning_ms,
            final_distance_m=result.final_distance_m,
            output=str(result.output or ""),
        )


def _box(name: str, x: float, y: float, size_x: float, size_y: float, height: float) -> ObstacleBox:
    return ObstacleBox(name, x, y, size_x, size_y, height)


def _random_boxes(rng: np.random.Generator, count: int, goal: tuple[float, float]) -> tuple[ObstacleBox, ...]:
    obstacles: list[ObstacleBox] = []
    attempts = 0
    while len(obstacles) < count and attempts < 500:
        attempts += 1
        candidate = _box(
            f"random_{len(obstacles)}",
            float(rng.uniform(1.5, goal[0] - 0.7)),
            float(rng.uniform(-1.25, 1.25)),
            float(rng.uniform(0.14, 0.38)),
            float(rng.uniform(0.28, 0.75)),
            float(rng.uniform(0.08, 0.35)),
        )
        if all(
            abs(candidate.x - other.x) > (candidate.size_x + other.size_x) / 2.0 + 0.30
            or abs(candidate.y - other.y) > (candidate.size_y + other.size_y) / 2.0 + 0.30
            for other in obstacles
        ):
            obstacles.append(candidate)
    if len(obstacles) != count:
        raise RuntimeError("无法在约束内生成随机地图")
    return tuple(obstacles)


def make_sample(profile: str, index: int, seed: int) -> MonteCarloSample:
    if profile not in PROFILES:
        raise ValueError(f"未知 Monte Carlo 测试族：{profile}")
    rng = np.random.default_rng(seed)
    run_id = f"{profile}_{index + 1:03d}_seed_{seed}"
    if profile == "obstacle_width":
        width = float(rng.uniform(0.35, 2.40))
        height = float(rng.uniform(0.08, 0.35))
        goal = (float(rng.uniform(5.0, 6.0)), float(rng.uniform(-0.35, 0.35)))
        obstacles = (_box("width", float(rng.uniform(2.2, 3.0)), 0.0, 0.14, width, height),)
        parameters = {"width_m": width, "height_m": height, "goal_x_m": goal[0], "goal_y_m": goal[1]}
        duration = 45.0
    elif profile == "obstacle_spacing":
        spacing = float(rng.uniform(0.75, 2.10))
        lateral = float(rng.uniform(0.18, 0.55))
        width = float(rng.uniform(0.55, 1.05))
        goal = (6.0, float(rng.uniform(-0.25, 0.25)))
        first_x = float(rng.uniform(1.8, 2.4))
        obstacles = (
            _box("spacing_a", first_x, lateral, 0.16, width, 0.25),
            _box("spacing_b", first_x + spacing, -lateral, 0.16, width, 0.25),
        )
        parameters = {"spacing_m": spacing, "lateral_m": lateral, "width_m": width, "goal_y_m": goal[1]}
        duration = 45.0
    elif profile == "obstacle_height":
        height = float(rng.uniform(0.04, 0.40))
        width = float(rng.uniform(0.55, 1.40))
        goal = (5.2, float(rng.uniform(-0.25, 0.25)))
        obstacles = (_box("height", float(rng.uniform(2.2, 2.9)), 0.0, 0.16, width, height),)
        parameters = {"height_m": height, "width_m": width, "goal_y_m": goal[1]}
        duration = 40.0
    elif profile == "goal_offset":
        goal = (float(rng.uniform(4.8, 6.5)), float(rng.uniform(-1.10, 1.10)))
        obstacle_y = float(rng.uniform(-0.35, 0.35))
        obstacles = (
            _box("goal_a", 2.2, obstacle_y, 0.16, float(rng.uniform(0.55, 1.0)), 0.25),
            _box("goal_b", 3.7, -obstacle_y, 0.16, float(rng.uniform(0.55, 1.0)), 0.25),
        )
        parameters = {"goal_x_m": goal[0], "goal_y_m": goal[1], "obstacle_y_m": obstacle_y}
        duration = 50.0
    else:
        goal = (float(rng.uniform(5.5, 7.0)), float(rng.uniform(-0.8, 0.8)))
        count = int(rng.integers(4, 9))
        obstacles = _random_boxes(rng, count, goal)
        parameters = {"box_count": count, "goal_x_m": goal[0], "goal_y_m": goal[1]}
        duration = 55.0
    scenario = NavigationScenario(run_id, f"Monte Carlo: {profile}", goal, obstacles, duration)
    return MonteCarloSample(run_id, profile, seed, scenario, parameters)


def generate_samples(total_runs: int = 100, base_seed: int = 20260805) -> list[MonteCarloSample]:
    if total_runs < len(PROFILES) or total_runs % len(PROFILES) != 0:
        raise ValueError(f"总次数必须是 {len(PROFILES)} 的倍数且至少为 {len(PROFILES)}")
    per_profile = total_runs // len(PROFILES)
    samples = []
    for profile_index, profile in enumerate(PROFILES):
        for index in range(per_profile):
            seed = base_seed + profile_index * 100_000 + index
            samples.append(make_sample(profile, index, seed))
    return samples


def append_checkpoint(path: Path, record: MonteCarloRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(record)), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(record))
        stream.flush()


def _parse_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def load_checkpoint(path: Path) -> list[MonteCarloRecord]:
    if not path.exists():
        return []
    records = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records.append(MonteCarloRecord(
                run_id=row["run_id"], profile=row["profile"], seed=int(row["seed"]),
                parameters_json=row["parameters_json"], scenario_json=row["scenario_json"],
                success=_parse_bool(row["success"]), reached=_parse_bool(row["reached"]),
                survived=_parse_bool(row["survived"]), collision_count=int(row["collision_count"]),
                elapsed_sim_s=float(row["elapsed_sim_s"]), elapsed_wall_s=float(row["elapsed_wall_s"]),
                path_length_m=float(row["path_length_m"]), min_clearance_m=float(row["min_clearance_m"]),
                average_velocity_mps=float(row["average_velocity_mps"]),
                cpu_usage_percent=float(row["cpu_usage_percent"]), fps=float(row["fps"]),
                depth_fps=float(row["depth_fps"]), planning_fps=float(row["planning_fps"]),
                average_planning_ms=float(row["average_planning_ms"]),
                final_distance_m=float(row["final_distance_m"]), output=row["output"],
            ))
    return records


def _wilson(successes: int, runs: int) -> tuple[float, float]:
    if runs == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / runs
    denominator = 1.0 + z * z / runs
    center = (p + z * z / (2.0 * runs)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / runs + z * z / (4.0 * runs * runs)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def write_monte_carlo_report(output_dir: Path, records: Iterable[MonteCarloRecord], expected_runs: int) -> None:
    records = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    groups = list(PROFILES) + ["total"]
    for profile in groups:
        selected = records if profile == "total" else [record for record in records if record.profile == profile]
        if not selected:
            continue
        successes = sum(record.success for record in selected)
        low, high = _wilson(successes, len(selected))
        rows.append({
            "profile": profile, "runs": len(selected), "successes": successes,
            "success_rate": successes / len(selected), "ci95_low": low, "ci95_high": high,
            "collision_count": sum(record.collision_count for record in selected),
            "average_time_s": float(np.mean([record.elapsed_sim_s for record in selected])),
            "average_path_length_m": float(np.mean([record.path_length_m for record in selected])),
            "min_clearance_m": min(record.min_clearance_m for record in selected),
            "average_velocity_mps": float(np.mean([record.average_velocity_mps for record in selected])),
            "average_fps": float(np.mean([record.fps for record in selected])),
        })
    with (output_dir / "success_rate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    total = rows[-1]
    lines = [
        "# G1 Navigation Monte Carlo 报告", "", "## 进度", "",
        f"- 已完成：{len(records)} / {expected_runs}",
        f"- 总成功率：{total['success_rate']:.1%}",
        f"- 95% Wilson 置信区间：{total['ci95_low']:.1%} ～ {total['ci95_high']:.1%}",
        f"- 总碰撞事件：{total['collision_count']}", "",
        "## 分组结果", "",
        "| 测试族 | 次数 | 成功 | 成功率 | 95% CI | 碰撞 | 平均时间(s) | 平均路径(m) | 最小净空(m) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['runs']} | {row['successes']} | {row['success_rate']:.1%} | "
            f"{row['ci95_low']:.1%}～{row['ci95_high']:.1%} | {row['collision_count']} | "
            f"{row['average_time_s']:.2f} | {row['average_path_length_m']:.2f} | "
            f"{row['min_clearance_m']:.3f} | {row['average_fps']:.1f} |"
        )
    lines.extend([
        "", "## 说明", "",
        "- 五个测试族等量采样：障碍宽度、障碍间距、障碍高度、目标点偏移、随机地图。",
        "- 每个 `run_id` 由测试族、序号和种子唯一确定，相同命令可复现。",
        "- `checkpoint.csv` 每完成一次即刷新，中断后使用 `--resume` 继续。", "",
    ])
    (output_dir / "navigation_report.md").write_text("\n".join(lines), encoding="utf-8")

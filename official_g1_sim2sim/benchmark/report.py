"""导航 Benchmark CSV 与中文 Markdown 报告。"""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path

import numpy as np

from navigation.runtime import NavigationRunResult


def _aggregate(results: list[NavigationRunResult]) -> list[dict[str, float | int | str]]:
    rows = []
    for scenario in dict.fromkeys(result.scenario for result in results):
        selected = [result for result in results if result.scenario == scenario]
        rows.append({
            "scenario": scenario,
            "runs": len(selected),
            "success_rate": sum(result.success for result in selected) / len(selected),
            "collision_count": sum(result.collision_count for result in selected),
            "average_time_s": float(np.mean([result.elapsed_sim_s for result in selected])),
            "average_path_length_m": float(np.mean([result.path_length_m for result in selected])),
            "min_clearance_m": min(result.min_clearance_m for result in selected),
            "average_velocity_mps": float(np.mean([result.average_velocity_mps for result in selected])),
            "average_cpu_percent": float(np.mean([result.cpu_usage_percent for result in selected])),
            "average_fps": float(np.mean([result.fps for result in selected])),
            "average_planning_ms": float(np.mean([result.average_planning_ms for result in selected])),
        })
    return rows


def write_benchmark_reports(output_dir: Path, results: list[NavigationRunResult]) -> None:
    if not results:
        raise ValueError("Benchmark 结果不能为空")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_fields = [field.name for field in fields(NavigationRunResult)]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row = {name: getattr(result, name) for name in result_fields}
            row["output"] = str(result.output) if result.output else ""
            writer.writerow(row)

    aggregate = _aggregate(results)
    aggregate_fields = list(aggregate[0])
    with (output_dir / "success_rate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate)

    total_success = sum(result.success for result in results)
    lines = [
        "# G1 Navigation Benchmark 报告",
        "",
        "## 总览",
        "",
        f"- 总运行数：{len(results)}",
        f"- 成功数：{total_success}",
        f"- 总成功率：{total_success / len(results):.1%}",
        f"- 总碰撞事件：{sum(result.collision_count for result in results)}",
        "- CPU Usage 为当前进程的单核等效占用，100% 表示持续占用一个 CPU 核。",
        "- FPS 为 MuJoCo 物理步数除以墙钟时间，无界面批量测试不限制为实时。",
        "",
        "## 场景统计",
        "",
        "| 场景 | 次数 | 成功率 | 碰撞 | 平均时间(s) | 平均路径(m) | 最小净空(m) | 平均速度(m/s) | CPU(%) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['scenario']} | {row['runs']} | {row['success_rate']:.1%} | "
            f"{row['collision_count']} | {row['average_time_s']:.2f} | "
            f"{row['average_path_length_m']:.2f} | {row['min_clearance_m']:.3f} | "
            f"{row['average_velocity_mps']:.3f} | {row['average_cpu_percent']:.1f} | "
            f"{row['average_fps']:.1f} |"
        )
    lines.extend([
        "",
        "## 单次运行",
        "",
        "| 场景 | Run | 成功 | 到达 | 存活 | 碰撞 | 时间(s) | 路径(m) | 最小净空(m) | 目标误差(m) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        lines.append(
            f"| {result.scenario} | {result.run_id} | {'是' if result.success else '否'} | "
            f"{'是' if result.reached else '否'} | {'是' if result.survived else '否'} | "
            f"{result.collision_count} | {result.elapsed_sim_s:.2f} | {result.path_length_m:.2f} | "
            f"{result.min_clearance_m:.3f} | {result.final_distance_m:.3f} |"
        )
    lines.extend([
        "",
        "## 判定规则",
        "",
        "成功必须同时满足：进入目标点 0.30 米范围、机器人未跌倒、全程零障碍物碰撞。",
        "",
    ])
    (output_dir / "navigation_report.md").write_text("\n".join(lines), encoding="utf-8")

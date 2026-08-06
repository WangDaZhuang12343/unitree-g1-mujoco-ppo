#!/usr/bin/env python3
"""生成可复现DWA教师数据并训练轻量岭回归导航策略。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from costmap import LocalCostMap
from planner import DWANavigator, LearnedNavigator, RidgeNavigationPolicy
from simulate import ROOT


def sample_case(seed: int) -> tuple[LocalCostMap, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    costmap = LocalCostMap()
    points: list[np.ndarray] = []
    obstacle_count = int(rng.integers(0, 4))
    for _ in range(obstacle_count):
        center_x = float(rng.uniform(0.7, 3.6))
        center_y = float(rng.uniform(-1.45, 1.45))
        size_x = float(rng.uniform(0.10, 0.45))
        size_y = float(rng.uniform(0.25, 1.40))
        xs = np.arange(center_x - size_x / 2.0, center_x + size_x / 2.0 + 0.025, 0.05)
        ys = np.arange(center_y - size_y / 2.0, center_y + size_y / 2.0 + 0.025, 0.05)
        xx, yy = np.meshgrid(xs, ys)
        points.append(np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 0.18))))
    cloud = np.concatenate(points) if points else np.empty((0, 3), dtype=np.float32)
    costmap.update(cloud, ground_z_body=0.0)
    goal = (float(rng.uniform(2.5, 4.0)), float(rng.uniform(-1.0, 1.0)))
    return costmap, goal


def build_dataset(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    observations: list[np.ndarray] = []
    targets: list[list[float]] = []
    seeds = [seed + index for index in range(count)]
    for sample_seed in seeds:
        costmap, goal = sample_case(sample_seed)
        result = DWANavigator().plan(costmap, goal)
        observations.append(LearnedNavigator.encode_observation(costmap, goal))
        targets.append([result.vx, result.vy, result.omega])
    return np.stack(observations), np.asarray(targets, dtype=np.float32), seeds


def evaluate(
    policy: RidgeNavigationPolicy,
    observations: np.ndarray,
    targets: np.ndarray,
    seeds: list[int],
) -> dict[str, object]:
    start = time.perf_counter()
    predictions = np.stack([policy(row) for row in observations])
    inference_ms = 1000.0 * (time.perf_counter() - start) / max(len(observations), 1)
    absolute_error = np.abs(predictions - targets)
    safe_outputs: list[np.ndarray] = []
    veto_count = 0
    start = time.perf_counter()
    for sample_seed in seeds:
        costmap, goal = sample_case(sample_seed)
        navigator = LearnedNavigator(policy)
        safe = navigator.plan(costmap, goal)
        safe_outputs.append([safe.vx, safe.vy, safe.omega])
        veto_count += int(navigator.last_command_vetoed)
    safe_adapter_ms = 1000.0 * (time.perf_counter() - start) / max(len(seeds), 1)
    safe_outputs_array = np.asarray(safe_outputs, dtype=np.float32)
    return {
        "samples": len(seeds),
        "mae_vx": float(absolute_error[:, 0].mean()),
        "mae_vy": float(absolute_error[:, 1].mean()),
        "mae_omega": float(absolute_error[:, 2].mean()),
        "command_mae": float(absolute_error.mean()),
        "raw_inference_ms": inference_ms,
        "safe_adapter_ms": safe_adapter_ms,
        "safety_veto_count": veto_count,
        "safety_veto_rate": veto_count / max(len(seeds), 1),
        "safe_stop_rate": float(np.mean(np.all(np.isclose(safe_outputs_array, 0.0), axis=1))),
    }


def write_report(path: Path, settings: dict[str, object], metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 轻量学习导航训练报告", "",
        f"- 随机种子：{settings['seed']}",
        f"- 训练样本：{settings['train_samples']}",
        f"- 验证样本：{metrics['samples']}",
        f"- 正则系数：{settings['regularization']}", "",
        "## 验证集指标", "",
        f"- 三维命令平均绝对误差：{metrics['command_mae']:.4f}",
        f"- vx / vy / omega MAE：{metrics['mae_vx']:.4f} / {metrics['mae_vy']:.4f} / {metrics['mae_omega']:.4f}",
        f"- 原始模型平均推理：{metrics['raw_inference_ms']:.3f} ms",
        f"- 含轨迹安全适配平均耗时：{metrics['safe_adapter_ms']:.3f} ms",
        f"- 安全否决：{metrics['safety_veto_count']} / {metrics['samples']}（{metrics['safety_veto_rate']:.1%}）", "",
        "该结果是DWA模仿学习的离线验证，不等同于闭环导航成功率；闭环结果需使用相同MuJoCo Benchmark单独验收。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--model", type=Path, default=ROOT / "models/navigation_ridge_v1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/navigation_ridge_training.md")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/navigation_ridge_metrics.json")
    parser.add_argument("--dataset", type=Path, default=None, help="可选本地NPZ；默认不保存")
    parser.add_argument("--dataset-in", type=Path, default=None, help="复用已生成的本地NPZ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= args.validation or args.validation <= 0:
        raise SystemExit("--samples 必须大于 --validation，且验证样本数必须为正")
    if args.dataset_in is not None:
        dataset = np.load(args.dataset_in)
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        targets = np.asarray(dataset["targets"], dtype=np.float32)
        seeds = [int(value) for value in dataset["seeds"]]
        if len(observations) != args.samples:
            raise SystemExit("--samples 必须与 --dataset-in 中的样本数一致")
    else:
        observations, targets, seeds = build_dataset(args.samples, args.seed)
    split = args.samples - args.validation
    policy = RidgeNavigationPolicy.fit(
        observations[:split], targets[:split], regularization=args.regularization
    )
    policy.save(args.model)
    metrics = evaluate(policy, observations[split:], targets[split:], seeds[split:])
    settings = {
        "seed": args.seed,
        "samples": args.samples,
        "train_samples": split,
        "validation_samples": args.validation,
        "regularization": args.regularization,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps({"settings": settings, "metrics": metrics}, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.report, settings, metrics)
    if args.dataset is not None:
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dataset, observations=observations, targets=targets, seeds=seeds)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

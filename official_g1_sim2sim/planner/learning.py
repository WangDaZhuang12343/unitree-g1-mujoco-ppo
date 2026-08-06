"""无需外部机器学习依赖的轻量导航模仿学习基线。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CompactFeatureConfig:
    source_rows: int = 80
    source_cols: int = 90
    pooled_rows: int = 10
    pooled_cols: int = 15


def compact_features(
    observation: np.ndarray, config: CompactFeatureConfig | None = None
) -> np.ndarray:
    """将完整占据图压缩为最大池化特征，并保留目标/目标交互项。"""

    config = config or CompactFeatureConfig()
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    expected = 2 + config.source_rows * config.source_cols
    if observation.size != expected:
        raise ValueError(f"学习策略输入维度应为 {expected}，实际为 {observation.size}")
    goal = observation[:2]
    grid = observation[2:].reshape(config.source_rows, config.source_cols)
    row_groups = np.array_split(np.arange(config.source_rows), config.pooled_rows)
    col_groups = np.array_split(np.arange(config.source_cols), config.pooled_cols)
    pooled = np.asarray(
        [[np.max(grid[np.ix_(rows, cols)]) for cols in col_groups] for rows in row_groups],
        dtype=np.float32,
    ).reshape(-1)
    return np.concatenate((goal, pooled, pooled * goal[0], pooled * goal[1])).astype(np.float32)


class RidgeNavigationPolicy:
    """将紧凑Costmap特征映射到三维速度的岭回归策略。"""

    def __init__(
        self,
        weights: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        config: CompactFeatureConfig | None = None,
    ) -> None:
        self.config = config or CompactFeatureConfig()
        self.weights = np.asarray(weights, dtype=np.float32)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        feature_count = 2 + 3 * self.config.pooled_rows * self.config.pooled_cols
        if self.weights.shape != (feature_count + 1, 3):
            raise ValueError("weights形状与特征合同不一致")
        if self.feature_mean.shape != (feature_count,) or self.feature_scale.shape != (feature_count,):
            raise ValueError("特征归一化参数形状不一致")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        features = compact_features(observation, self.config)
        normalized = (features - self.feature_mean) / self.feature_scale
        design = np.concatenate((normalized, np.ones(1, dtype=np.float32)))
        return (design @ self.weights).astype(np.float32)

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        targets: np.ndarray,
        regularization: float = 1e-2,
        config: CompactFeatureConfig | None = None,
    ) -> "RidgeNavigationPolicy":
        config = config or CompactFeatureConfig()
        features = np.stack([compact_features(row, config) for row in observations]).astype(np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (features.shape[0], 3):
            raise ValueError("targets必须是[N, 3]")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-6] = 1.0
        design = np.column_stack(((features - mean) / scale, np.ones(features.shape[0])))
        penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return cls(weights, mean, scale, config)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "g1_navigation_ridge_v1",
            "config": self.config.__dict__,
            "weights": self.weights.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RidgeNavigationPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "g1_navigation_ridge_v1":
            raise ValueError("不支持的学习导航模型格式")
        config = CompactFeatureConfig(**payload["config"])
        return cls(payload["weights"], payload["feature_mean"], payload["feature_scale"], config)

"""无需外部机器学习依赖的轻量导航模仿学习基线。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class CompactFeatureConfig:
    source_rows: int = 80
    source_cols: int = 90
    pooled_rows: int = 10
    pooled_cols: int = 15
    include_clearance: bool = False


def _feature_count(config: CompactFeatureConfig) -> int:
    cells = config.pooled_rows * config.pooled_cols
    return 2 + cells * (4 if config.include_clearance else 3)


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
    parts = [goal, pooled, pooled * goal[0], pooled * goal[1]]
    if config.include_clearance:
        clearance = distance_transform_edt(grid < 0.5).astype(np.float32)
        clearance /= max(config.source_rows, config.source_cols)
        pooled_clearance = np.asarray(
            [[np.min(clearance[np.ix_(rows, cols)]) for cols in col_groups] for rows in row_groups],
            dtype=np.float32,
        ).reshape(-1)
        parts.append(pooled_clearance)
    return np.concatenate(parts).astype(np.float32)


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
        feature_count = _feature_count(self.config)
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


class RandomFeatureNavigationPolicy:
    """确定性随机ReLU特征加岭回归的轻量非线性策略。"""

    def __init__(
        self,
        output_weights: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        hidden_features: int = 64,
        projection_seed: int = 20260806,
        config: CompactFeatureConfig | None = None,
    ) -> None:
        self.config = config or CompactFeatureConfig()
        self.hidden_features = int(hidden_features)
        self.projection_seed = int(projection_seed)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        feature_count = _feature_count(self.config)
        self.output_weights = np.asarray(output_weights, dtype=np.float32)
        if self.output_weights.shape != (feature_count + self.hidden_features + 1, 3):
            raise ValueError("非线性策略输出权重形状不一致")
        if self.feature_mean.shape != (feature_count,) or self.feature_scale.shape != (feature_count,):
            raise ValueError("特征归一化参数形状不一致")
        rng = np.random.default_rng(self.projection_seed)
        self.projection = (
            rng.standard_normal((feature_count, self.hidden_features)) / np.sqrt(feature_count)
        ).astype(np.float32)
        self.hidden_bias = rng.standard_normal(self.hidden_features).astype(np.float32) * 0.1

    def _design(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = np.maximum(normalized @ self.projection + self.hidden_bias, 0.0)
        return np.concatenate((normalized, hidden, np.ones(1, dtype=np.float32)))

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return (self._design(compact_features(observation, self.config)) @ self.output_weights).astype(
            np.float32
        )

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        targets: np.ndarray,
        regularization: float = 10.0,
        hidden_features: int = 64,
        projection_seed: int = 20260806,
        config: CompactFeatureConfig | None = None,
    ) -> "RandomFeatureNavigationPolicy":
        config = config or CompactFeatureConfig()
        features = np.stack([compact_features(row, config) for row in observations]).astype(np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-6] = 1.0
        rng = np.random.default_rng(projection_seed)
        projection = rng.standard_normal((features.shape[1], hidden_features)) / np.sqrt(features.shape[1])
        hidden_bias = rng.standard_normal(hidden_features) * 0.1
        normalized = (features - mean) / scale
        hidden = np.maximum(normalized @ projection + hidden_bias, 0.0)
        design = np.column_stack((normalized, hidden, np.ones(features.shape[0])))
        penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return cls(weights, mean, scale, hidden_features, projection_seed, config)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "g1_navigation_random_relu_v1",
            "config": self.config.__dict__,
            "hidden_features": self.hidden_features,
            "projection_seed": self.projection_seed,
            "output_weights": self.output_weights.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RandomFeatureNavigationPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "g1_navigation_random_relu_v1":
            raise ValueError("不支持的非线性导航模型格式")
        return cls(
            payload["output_weights"], payload["feature_mean"], payload["feature_scale"],
            payload["hidden_features"], payload["projection_seed"],
            CompactFeatureConfig(**payload["config"]),
        )


def load_navigation_policy(
    path: Path,
) -> RidgeNavigationPolicy | RandomFeatureNavigationPolicy | TemporalRandomFeatureNavigationPolicy:
    """按模型格式加载已固化的轻量导航策略。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    model_format = payload.get("format")
    if model_format == "g1_navigation_ridge_v1":
        return RidgeNavigationPolicy.load(path)
    if model_format == "g1_navigation_random_relu_v1":
        return RandomFeatureNavigationPolicy.load(path)
    if model_format == "g1_navigation_temporal_random_relu_v1":
        return TemporalRandomFeatureNavigationPolicy.load(path)
    raise ValueError(f"不支持的学习导航模型格式: {model_format}")


class TemporalRandomFeatureNavigationPolicy:
    """使用地图变化和上一命令的轻量有状态随机ReLU策略。"""

    def __init__(
        self,
        output_weights: np.ndarray,
        temporal_mean: np.ndarray,
        temporal_scale: np.ndarray,
        hidden_features: int = 64,
        projection_seed: int = 20260806,
        config: CompactFeatureConfig | None = None,
    ) -> None:
        self.config = config or CompactFeatureConfig(pooled_rows=6, pooled_cols=9, include_clearance=True)
        self.hidden_features = int(hidden_features)
        self.projection_seed = int(projection_seed)
        self.temporal_mean = np.asarray(temporal_mean, dtype=np.float32)
        self.temporal_scale = np.asarray(temporal_scale, dtype=np.float32)
        self.base_count = _feature_count(self.config)
        self.temporal_count = 2 * self.base_count + 3
        self.output_weights = np.asarray(output_weights, dtype=np.float32)
        if self.temporal_mean.shape != (self.temporal_count,) or self.temporal_scale.shape != (
            self.temporal_count,
        ):
            raise ValueError("时序归一化参数形状不一致")
        if self.output_weights.shape != (self.temporal_count + self.hidden_features + 1, 3):
            raise ValueError("时序策略输出权重形状不一致")
        rng = np.random.default_rng(self.projection_seed)
        self.projection = (
            rng.standard_normal((self.temporal_count, self.hidden_features))
            / np.sqrt(self.temporal_count)
        ).astype(np.float32)
        self.hidden_bias = rng.standard_normal(self.hidden_features).astype(np.float32) * 0.1
        self.reset()

    def reset(self) -> None:
        self.previous_features: np.ndarray | None = None
        self.previous_command = np.zeros(3, dtype=np.float32)

    def _temporal_features(self, current: np.ndarray) -> np.ndarray:
        previous = current if self.previous_features is None else self.previous_features
        return np.concatenate((current, current - previous, self.previous_command)).astype(np.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        current = compact_features(observation, self.config)
        temporal = self._temporal_features(current)
        normalized = (temporal - self.temporal_mean) / self.temporal_scale
        hidden = np.maximum(normalized @ self.projection + self.hidden_bias, 0.0)
        design = np.concatenate((normalized, hidden, np.ones(1, dtype=np.float32)))
        command = (design @ self.output_weights).astype(np.float32)
        self.previous_features = current
        self.previous_command = command
        return command

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        targets: np.ndarray,
        sequence_starts: np.ndarray,
        regularization: float = 300.0,
        hidden_features: int = 64,
        projection_seed: int = 20260806,
        config: CompactFeatureConfig | None = None,
    ) -> "TemporalRandomFeatureNavigationPolicy":
        config = config or CompactFeatureConfig(pooled_rows=6, pooled_cols=9, include_clearance=True)
        current_features = np.stack([compact_features(row, config) for row in observations])
        targets = np.asarray(targets, dtype=np.float32)
        sequence_starts = np.asarray(sequence_starts, dtype=bool)
        if sequence_starts.shape != (len(observations),) or not sequence_starts[0]:
            raise ValueError("sequence_starts必须与样本等长且首样本为True")
        temporal_rows: list[np.ndarray] = []
        previous_features = current_features[0]
        previous_command = np.zeros(3, dtype=np.float32)
        for index, current in enumerate(current_features):
            if sequence_starts[index]:
                previous_features = current
                previous_command = np.zeros(3, dtype=np.float32)
            temporal_rows.append(np.concatenate((current, current - previous_features, previous_command)))
            previous_features = current
            previous_command = targets[index]
        temporal = np.asarray(temporal_rows, dtype=np.float64)
        mean = temporal.mean(axis=0)
        scale = temporal.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (temporal - mean) / scale
        rng = np.random.default_rng(projection_seed)
        projection = rng.standard_normal((temporal.shape[1], hidden_features)) / np.sqrt(temporal.shape[1])
        hidden_bias = rng.standard_normal(hidden_features) * 0.1
        hidden = np.maximum(normalized @ projection + hidden_bias, 0.0)
        design = np.column_stack((normalized, hidden, np.ones(len(temporal))))
        penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return cls(weights, mean, scale, hidden_features, projection_seed, config)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "g1_navigation_temporal_random_relu_v1",
            "config": self.config.__dict__,
            "hidden_features": self.hidden_features,
            "projection_seed": self.projection_seed,
            "output_weights": self.output_weights.tolist(),
            "temporal_mean": self.temporal_mean.tolist(),
            "temporal_scale": self.temporal_scale.tolist(),
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TemporalRandomFeatureNavigationPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "g1_navigation_temporal_random_relu_v1":
            raise ValueError("不支持的时序导航模型格式")
        return cls(
            payload["output_weights"], payload["temporal_mean"], payload["temporal_scale"],
            payload["hidden_features"], payload["projection_seed"],
            CompactFeatureConfig(**payload["config"]),
        )

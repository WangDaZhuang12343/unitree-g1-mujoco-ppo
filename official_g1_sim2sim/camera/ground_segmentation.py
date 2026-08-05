"""不依赖仿真 geom ID 的重力约束 RANSAC 地面分割。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundSegmentationConfig:
    distance_threshold: float = 0.018
    obstacle_clearance: float = 0.025
    maximum_tilt_deg: float = 25.0
    ransac_iterations: int = 80
    minimum_inliers: int = 80
    minimum_inlier_ratio: float = 0.25
    seed: int = 7


@dataclass(frozen=True)
class GroundPlane:
    normal: np.ndarray
    offset: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.normal + self.offset


@dataclass(frozen=True)
class GroundSegmentationResult:
    plane: GroundPlane
    ground_points: np.ndarray
    obstacle_points: np.ndarray
    ground_mask: np.ndarray
    obstacle_mask: np.ndarray
    inlier_ratio: float
    used_cached_plane: bool


class GroundSegmenter:
    def __init__(self, config: GroundSegmentationConfig | None = None) -> None:
        self.config = config or GroundSegmentationConfig()
        self.last_plane: GroundPlane | None = None

    @staticmethod
    def _orient(normal: np.ndarray, expected_up: np.ndarray) -> np.ndarray:
        return normal if float(np.dot(normal, expected_up)) >= 0.0 else -normal

    @staticmethod
    def _fit_least_squares(points: np.ndarray, expected_up: np.ndarray) -> GroundPlane:
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
        normal = GroundSegmenter._orient(vh[-1], expected_up)
        normal = normal / np.linalg.norm(normal)
        return GroundPlane(normal.astype(np.float64), -float(np.dot(normal, centroid)))

    def _fallback(self, points: np.ndarray, expected_up: np.ndarray) -> GroundPlane:
        projections = points @ expected_up
        offset = -float(np.quantile(projections, 0.20))
        return GroundPlane(expected_up.copy(), offset)

    def segment(self, points_body: np.ndarray, gravity_body: np.ndarray) -> GroundSegmentationResult:
        points = np.asarray(points_body, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("点云必须为 (N, 3)")
        gravity = np.asarray(gravity_body, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("重力向量必须是有限三维向量")
        gravity_norm = float(np.linalg.norm(gravity))
        if gravity_norm < 1e-6:
            raise ValueError("重力向量长度不能为0")
        expected_up = -gravity / gravity_norm
        if len(points) < 3:
            if self.last_plane is None:
                raise ValueError("首帧地面分割至少需要3个点")
            signed = self.last_plane.signed_distance(points)
            ground_mask = np.abs(signed) <= self.config.distance_threshold
            obstacle_mask = signed >= self.config.obstacle_clearance
            return GroundSegmentationResult(
                plane=self.last_plane,
                ground_points=points[ground_mask].astype(np.float32),
                obstacle_points=points[obstacle_mask].astype(np.float32),
                ground_mask=ground_mask,
                obstacle_mask=obstacle_mask,
                inlier_ratio=float(np.count_nonzero(ground_mask) / max(len(points), 1)),
                used_cached_plane=True,
            )
        cosine_limit = float(np.cos(np.radians(self.config.maximum_tilt_deg)))
        rng = np.random.default_rng(self.config.seed)
        best_mask: np.ndarray | None = None
        best_score = -1

        for indices in rng.integers(0, len(points), size=(self.config.ransac_iterations, 3)):
            first, second, third = points[indices]
            normal = np.cross(second - first, third - first)
            length = float(np.linalg.norm(normal))
            if length < 1e-8:
                continue
            normal = self._orient(normal / length, expected_up)
            if float(np.dot(normal, expected_up)) < cosine_limit:
                continue
            offset = -float(np.dot(normal, first))
            distances = np.abs(points @ normal + offset)
            mask = distances <= self.config.distance_threshold
            score = int(np.count_nonzero(mask))
            if score > best_score:
                best_score = score
                best_mask = mask

        minimum = max(
            self.config.minimum_inliers,
            int(np.ceil(len(points) * self.config.minimum_inlier_ratio)),
        )
        if best_mask is None or best_score < minimum:
            plane = self._fallback(points, expected_up)
        else:
            plane = self._fit_least_squares(points[best_mask], expected_up)
            if float(np.dot(plane.normal, expected_up)) < cosine_limit:
                plane = self._fallback(points, expected_up)

        signed = plane.signed_distance(points)
        self.last_plane = plane
        ground_mask = np.abs(signed) <= self.config.distance_threshold
        obstacle_mask = signed >= self.config.obstacle_clearance
        return GroundSegmentationResult(
            plane=plane,
            ground_points=points[ground_mask].astype(np.float32),
            obstacle_points=points[obstacle_mask].astype(np.float32),
            ground_mask=ground_mask,
            obstacle_mask=obstacle_mask,
            inlier_ratio=float(np.count_nonzero(ground_mask) / len(points)),
            used_cached_plane=False,
        )

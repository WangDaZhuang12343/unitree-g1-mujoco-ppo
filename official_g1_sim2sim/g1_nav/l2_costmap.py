"""机器人局部坐标系下的瞬时代价地图。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass(frozen=True)
class CostMapConfig:
    resolution: float = 0.05
    front_range: float = 4.0
    rear_range: float = 0.5
    side_range: float = 2.0
    obstacle_height: float = 0.035
    maximum_height: float = 1.5
    inflation_radius: float = 0.28


class LocalCostMap:
    """每帧从本体坐标点云重建，避免机器人运动后遗留幽灵障碍。"""

    def __init__(self, config: CostMapConfig | None = None) -> None:
        self.config = config or CostMapConfig()
        self.nx = int(round((self.config.front_range + self.config.rear_range) / self.config.resolution))
        self.ny = int(round(2.0 * self.config.side_range / self.config.resolution))
        self.height_map = np.full((self.ny, self.nx), -np.inf, dtype=np.float32)
        self.obstacle_map = np.zeros((self.ny, self.nx), dtype=bool)
        self.distance_field = np.full((self.ny, self.nx), np.inf, dtype=np.float32)

    def local_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(np.floor((x + self.config.rear_range) / self.config.resolution))
        gy = int(np.floor((y + self.config.side_range) / self.config.resolution))
        return gx, gy

    def grid_to_local(self, gx: int, gy: int) -> tuple[float, float]:
        x = (gx + 0.5) * self.config.resolution - self.config.rear_range
        y = (gy + 0.5) * self.config.resolution - self.config.side_range
        return x, y

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.nx and 0 <= gy < self.ny

    def update(self, points_body: np.ndarray, ground_z_body: float) -> None:
        self.height_map.fill(-np.inf)
        self.obstacle_map.fill(False)
        points = np.asarray(points_body, dtype=np.float32)
        if points.size == 0:
            self.distance_field.fill(max(self.config.front_range, 2.0 * self.config.side_range))
            return
        heights = points[:, 2] - ground_z_body
        valid = (
            (points[:, 0] >= -self.config.rear_range)
            & (points[:, 0] < self.config.front_range)
            & (np.abs(points[:, 1]) < self.config.side_range)
            & (heights >= -0.03)
            & (heights <= self.config.maximum_height)
        )
        points = points[valid]
        heights = heights[valid]
        if points.size == 0:
            self.distance_field.fill(max(self.config.front_range, 2.0 * self.config.side_range))
            return
        gx = np.floor((points[:, 0] + self.config.rear_range) / self.config.resolution).astype(int)
        gy = np.floor((points[:, 1] + self.config.side_range) / self.config.resolution).astype(int)
        np.maximum.at(self.height_map, (gy, gx), heights)
        raw_obstacles = self.height_map >= self.config.obstacle_height
        inflate = max(1, int(np.ceil(self.config.inflation_radius / self.config.resolution)))
        yy, xx = np.ogrid[-inflate : inflate + 1, -inflate : inflate + 1]
        footprint = xx * xx + yy * yy <= inflate * inflate
        self.obstacle_map[:] = binary_dilation(raw_obstacles, structure=footprint)
        self.distance_field[:] = distance_transform_edt(~raw_obstacles).astype(np.float32) * self.config.resolution

    def clearance(self, x: float, y: float) -> float:
        gx, gy = self.local_to_grid(x, y)
        if not self.in_bounds(gx, gy):
            return 0.0
        return float(self.distance_field[gy, gx])

    def occupied(self, x: float, y: float) -> bool:
        gx, gy = self.local_to_grid(x, y)
        return not self.in_bounds(gx, gy) or bool(self.obstacle_map[gy, gx])

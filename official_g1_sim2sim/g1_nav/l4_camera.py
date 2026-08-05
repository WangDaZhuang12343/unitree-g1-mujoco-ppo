"""基于 MuJoCo mj_multiRay 的CPU仿真深度相机。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    width: int = 64
    height: int = 40
    horizontal_fov_deg: float = 80.0
    vertical_fov_deg: float = 50.0
    pitch_down_deg: float = 20.0
    max_range: float = 3.5
    mount_offset: tuple[float, float, float] = (0.12, 0.0, 0.32)


@dataclass(frozen=True)
class DepthFrame:
    depth: np.ndarray
    points_body: np.ndarray
    points_world: np.ndarray
    geom_ids: np.ndarray
    point_geom_ids: np.ndarray


class SimulatedDepthCamera:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._directions_body = self._make_directions()
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[0] = 1

    def _make_directions(self) -> np.ndarray:
        cfg = self.config
        horizontal = np.linspace(
            np.radians(cfg.horizontal_fov_deg) / 2.0,
            -np.radians(cfg.horizontal_fov_deg) / 2.0,
            cfg.width,
        )
        vertical = np.linspace(
            np.radians(cfg.vertical_fov_deg) / 2.0 - np.radians(cfg.pitch_down_deg),
            -np.radians(cfg.vertical_fov_deg) / 2.0 - np.radians(cfg.pitch_down_deg),
            cfg.height,
        )
        yaw, pitch = np.meshgrid(horizontal, vertical)
        directions = np.stack(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            axis=-1,
        ).reshape(-1, 3)
        return np.ascontiguousarray(directions, dtype=np.float64)

    @staticmethod
    def isolate_world_geoms(model: mujoco.MjModel) -> None:
        """将机器人几何放到可见组1，射线只检测世界组0。"""
        robot_geoms = model.geom_bodyid != 0
        model.geom_group[robot_geoms] = 1

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> DepthFrame:
        cfg = self.config
        rotation = data.xmat[body_id].reshape(3, 3)
        body_position = data.xpos[body_id]
        camera_position = body_position + rotation @ np.asarray(cfg.mount_offset)
        directions_world = np.ascontiguousarray(self._directions_body @ rotation.T)
        ray_count = directions_world.shape[0]
        geom_ids = np.full(ray_count, -1, dtype=np.int32)
        distances = np.full(ray_count, cfg.max_range, dtype=np.float64)
        mujoco.mj_multiRay(
            model,
            data,
            np.ascontiguousarray(camera_position, dtype=np.float64),
            directions_world.reshape(-1),
            self._geomgroup,
            1,
            -1,
            geom_ids,
            distances,
            ray_count,
            cfg.max_range,
        )
        valid = (geom_ids >= 0) & (distances >= 0.0) & (distances <= cfg.max_range)
        depth = np.where(valid, distances, cfg.max_range).reshape(cfg.height, cfg.width).astype(np.float32)
        points_world = camera_position + directions_world[valid] * distances[valid, None]
        points_body = (points_world - body_position) @ rotation
        return DepthFrame(
            depth=depth,
            points_body=points_body.astype(np.float32),
            points_world=points_world.astype(np.float32),
            geom_ids=geom_ids.reshape(cfg.height, cfg.width),
            point_geom_ids=geom_ids[valid].copy(),
        )

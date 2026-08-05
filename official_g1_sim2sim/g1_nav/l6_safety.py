"""导航速度指令的限幅、斜坡和姿态安全检查。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyConfig:
    max_vx: float = 0.45
    max_vy: float = 0.10
    max_omega: float = 0.20
    max_vx_acceleration: float = 0.60
    max_vy_acceleration: float = 0.30
    max_omega_acceleration: float = 0.60
    maximum_tilt_rad: float = 0.55
    minimum_height: float = 0.50


class SafetySystem:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.command = np.zeros(3, dtype=np.float32)
        self.emergency_stopped = False

    def reset(self) -> None:
        self.command.fill(0.0)
        self.emergency_stopped = False

    def stop(self) -> np.ndarray:
        self.emergency_stopped = True
        self.command.fill(0.0)
        return self.command.copy()

    def update(
        self,
        desired_vx: float,
        desired_vy: float,
        desired_omega: float,
        dt: float,
        base_height: float,
        roll: float,
        pitch: float,
    ) -> np.ndarray:
        if (
            self.emergency_stopped
            or base_height < self.config.minimum_height
            or max(abs(roll), abs(pitch)) > self.config.maximum_tilt_rad
        ):
            return self.stop()
        target = np.array(
            [
                np.clip(desired_vx, 0.0, self.config.max_vx),
                np.clip(desired_vy, -self.config.max_vy, self.config.max_vy),
                np.clip(desired_omega, -self.config.max_omega, self.config.max_omega),
            ],
            dtype=np.float32,
        )
        limits = np.array(
            [
                self.config.max_vx_acceleration * dt,
                self.config.max_vy_acceleration * dt,
                self.config.max_omega_acceleration * dt,
            ],
            dtype=np.float32,
        )
        delta = np.clip(target - self.command, -limits, limits)
        self.command += delta
        return self.command.copy()

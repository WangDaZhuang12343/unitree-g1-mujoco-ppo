"""与已验证G1速度范围一致的非完整DWA局部规划器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .l2_costmap import LocalCostMap


@dataclass(frozen=True)
class DWAConfig:
    min_walk_speed: float = 0.25
    max_walk_speed: float = 0.45
    max_lateral_speed: float = 0.10
    max_omega: float = 0.20
    vx_samples: int = 6
    vy_samples: int = 5
    omega_samples: int = 11
    predict_time: float = 10.0
    predict_dt: float = 0.10
    robot_radius: float = 0.22
    goal_weight: float = 1.2
    clearance_weight: float = 1.8
    speed_weight: float = 0.35
    turn_weight: float = 0.15
    turn_change_weight: float = 0.25
    lateral_weight: float = 0.10
    vx_response_gain: float = 0.80
    vy_response_gain: float = 0.65
    omega_response_gain: float = 0.70


@dataclass(frozen=True)
class PlanResult:
    vx: float
    vy: float
    omega: float
    score: float
    trajectory: np.ndarray


@dataclass(frozen=True)
class PlanningDebug:
    trajectories: np.ndarray
    valid: np.ndarray


class DWANavigator:
    def __init__(self, config: DWAConfig | None = None) -> None:
        self.config = config or DWAConfig()
        self.previous_omega = 0.0
        self.last_debug: PlanningDebug | None = None

    def _simulate(self, vx: float, vy: float, omega: float) -> np.ndarray:
        steps = int(round(self.config.predict_time / self.config.predict_dt))
        trajectory = np.zeros((steps + 1, 3), dtype=np.float32)
        for index in range(steps):
            x, y, yaw = trajectory[index]
            body_vx = vx * self.config.vx_response_gain
            body_vy = vy * self.config.vy_response_gain
            trajectory[index + 1, 0] = x + (body_vx * np.cos(yaw) - body_vy * np.sin(yaw)) * self.config.predict_dt
            trajectory[index + 1, 1] = y + (body_vx * np.sin(yaw) + body_vy * np.cos(yaw)) * self.config.predict_dt
            trajectory[index + 1, 2] = yaw + omega * self.config.omega_response_gain * self.config.predict_dt
        return trajectory

    def plan(
        self, costmap: LocalCostMap, goal_body: tuple[float, float], collect_debug: bool = False
    ) -> PlanResult:
        goal = np.asarray(goal_body, dtype=np.float32)
        candidates = [(0.0, 0.0, 0.0)]
        candidates.extend(
            (float(vx), float(vy), float(omega))
            for vx in np.linspace(self.config.min_walk_speed, self.config.max_walk_speed, self.config.vx_samples)
            for vy in np.linspace(-self.config.max_lateral_speed, self.config.max_lateral_speed, self.config.vy_samples)
            for omega in np.linspace(-self.config.max_omega, self.config.max_omega, self.config.omega_samples)
        )
        best = PlanResult(0.0, 0.0, 0.0, -np.inf, np.zeros((1, 3), dtype=np.float32))
        debug_trajectories: list[np.ndarray] = []
        debug_valid: list[bool] = []
        for vx, vy, omega in candidates:
            trajectory = self._simulate(vx, vy, omega)
            clearances = np.asarray([costmap.clearance(float(x), float(y)) for x, y in trajectory[:, :2]])
            valid = not np.any(clearances < self.config.robot_radius)
            if collect_debug:
                debug_trajectories.append(trajectory)
                debug_valid.append(bool(valid))
            if not valid:
                continue
            initial_distance = max(float(np.linalg.norm(goal)), 1e-6)
            trajectory_distances = np.linalg.norm(trajectory[:, :2] - goal, axis=1)
            progress = (initial_distance - float(np.min(trajectory_distances))) / initial_distance
            clearance_score = min(float(np.min(clearances)) / 1.0, 1.0)
            speed_score = vx / self.config.max_walk_speed
            turn_penalty = abs(omega) / self.config.max_omega
            turn_change_penalty = abs(omega - self.previous_omega) / (2.0 * self.config.max_omega)
            lateral_penalty = abs(vy) / self.config.max_lateral_speed
            score = (
                self.config.goal_weight * progress
                + self.config.clearance_weight * clearance_score
                + self.config.speed_weight * speed_score
                - self.config.turn_weight * turn_penalty
                - self.config.turn_change_weight * turn_change_penalty
                - self.config.lateral_weight * lateral_penalty
            )
            if score > best.score:
                best = PlanResult(vx, vy, omega, score, trajectory)
        self.previous_omega = best.omega
        if collect_debug:
            self.last_debug = PlanningDebug(
                trajectories=np.stack(debug_trajectories),
                valid=np.asarray(debug_valid, dtype=bool),
            )
        else:
            self.last_debug = None
        return best

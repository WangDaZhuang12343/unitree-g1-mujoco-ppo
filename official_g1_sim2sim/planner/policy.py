"""可替换 DWA 的局部导航策略合同与学习策略安全适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from costmap import LocalCostMap
from g1_nav.l3_dwa import PlanResult, PlanningDebug


@runtime_checkable
class LocalNavigator(Protocol):
    """导航层稳定合同；实现不得直接控制关节或调用 Walking Policy。"""

    last_debug: PlanningDebug | None

    def plan(
        self,
        costmap: LocalCostMap,
        goal_body: tuple[float, float],
        collect_debug: bool = False,
    ) -> PlanResult:
        """根据机器人坐标系代价地图和目标生成速度计划。"""


PolicyPredictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LearnedNavigatorConfig:
    """学习策略输出和独立安全校验的固定物理合同。"""

    max_forward_speed: float = 0.45
    minimum_walk_speed: float = 0.25
    walk_activation_threshold: float = 0.20
    max_lateral_speed: float = 0.10
    max_omega: float = 0.20
    predict_time: float = 2.0
    predict_dt: float = 0.10
    robot_radius: float = 0.22


class LearnedNavigator:
    """把任意推理函数适配为局部导航器，并对输出做限幅和碰撞否决。

    输入向量由归一化目标 ``[goal_x, goal_y]`` 和按行展开的二值占据图组成。
    推理函数输出物理单位 ``[vx(m/s), vy(m/s), omega(rad/s)]``。适配器不加载、
    不修改也不替换 Unitree Walking Policy。
    """

    def __init__(
        self,
        predictor: PolicyPredictor,
        config: LearnedNavigatorConfig | None = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or LearnedNavigatorConfig()
        self.last_debug: PlanningDebug | None = None
        self.last_command_vetoed = False

    def reset(self) -> None:
        """重置有状态预测器，防止不同导航运行共享历史。"""

        reset = getattr(self.predictor, "reset", None)
        if callable(reset):
            reset()
        self.last_debug = None
        self.last_command_vetoed = False

    @staticmethod
    def encode_observation(
        costmap: LocalCostMap, goal_body: tuple[float, float]
    ) -> np.ndarray:
        goal = np.asarray(goal_body, dtype=np.float32)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_body 必须是两个有限数值")
        goal_scale = np.asarray(
            [costmap.config.front_range, costmap.config.side_range], dtype=np.float32
        )
        normalized_goal = np.clip(goal / goal_scale, -1.0, 1.0)
        occupancy = costmap.obstacle_map.astype(np.float32, copy=False).reshape(-1)
        return np.concatenate((normalized_goal, occupancy)).astype(np.float32, copy=False)

    def _trajectory(self, vx: float, vy: float, omega: float) -> np.ndarray:
        steps = int(round(self.config.predict_time / self.config.predict_dt))
        trajectory = np.zeros((steps + 1, 3), dtype=np.float32)
        for index in range(steps):
            x, y, yaw = trajectory[index]
            trajectory[index + 1, 0] = x + (
                vx * np.cos(yaw) - vy * np.sin(yaw)
            ) * self.config.predict_dt
            trajectory[index + 1, 1] = y + (
                vx * np.sin(yaw) + vy * np.cos(yaw)
            ) * self.config.predict_dt
            trajectory[index + 1, 2] = yaw + omega * self.config.predict_dt
        return trajectory

    def plan(
        self,
        costmap: LocalCostMap,
        goal_body: tuple[float, float],
        collect_debug: bool = False,
    ) -> PlanResult:
        observation = self.encode_observation(costmap, goal_body)
        raw = np.asarray(self.predictor(observation), dtype=np.float32).reshape(-1)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            raise ValueError("学习导航策略必须输出三个有限数值 [vx, vy, omega]")
        raw_vx = float(np.clip(raw[0], 0.0, self.config.max_forward_speed))
        vx = (
            max(raw_vx, self.config.minimum_walk_speed)
            if raw_vx >= self.config.walk_activation_threshold
            else 0.0
        )
        vy = float(np.clip(raw[1], -self.config.max_lateral_speed, self.config.max_lateral_speed))
        omega = float(np.clip(raw[2], -self.config.max_omega, self.config.max_omega))
        candidate_trajectory = self._trajectory(vx, vy, omega)
        clearances = np.asarray(
            [costmap.clearance(float(x), float(y)) for x, y in candidate_trajectory[:, :2]],
            dtype=np.float32,
        )
        valid = bool(np.all(clearances >= self.config.robot_radius))
        self.last_command_vetoed = not valid
        trajectory = candidate_trajectory
        if not valid:
            vx = vy = omega = 0.0
            trajectory = np.zeros((1, 3), dtype=np.float32)
        if collect_debug:
            self.last_debug = PlanningDebug(
                trajectories=candidate_trajectory[None, ...],
                valid=np.asarray([valid], dtype=bool),
            )
        else:
            self.last_debug = None
        score = float(np.min(clearances)) if clearances.size else 0.0
        return PlanResult(vx, vy, omega, score, trajectory)

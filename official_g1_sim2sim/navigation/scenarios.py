"""Navigation Benchmark 的可重现场景库。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ObstacleBox:
    name: str
    x: float
    y: float
    size_x: float
    size_y: float
    height: float = 0.20
    velocity_y: float = 0.0
    motion_limit: float = 0.0
    rgba: tuple[float, float, float, float] = (0.85, 0.22, 0.08, 1.0)

    def center_at(self, time_s: float) -> tuple[float, float]:
        if self.velocity_y == 0.0 or self.motion_limit <= 0.0:
            return self.x, self.y
        phase = self.y + self.velocity_y * time_s
        span = 2.0 * self.motion_limit
        folded = (phase + self.motion_limit) % (2.0 * span)
        y = -self.motion_limit + (folded if folded <= span else 2.0 * span - folded)
        return self.x, y


@dataclass(frozen=True)
class NavigationScenario:
    name: str
    description: str
    goal: tuple[float, float]
    obstacles: tuple[ObstacleBox, ...]
    duration: float = 35.0

    @property
    def dynamic(self) -> bool:
        return any(obstacle.velocity_y != 0.0 for obstacle in self.obstacles)

    def with_seed(self, seed: int) -> "NavigationScenario":
        if self.name != "random_boxes":
            return self
        rng = np.random.default_rng(seed)
        obstacles = tuple(
            ObstacleBox(
                name=f"random_{index}",
                x=float(rng.uniform(1.7, 4.5)),
                y=float(rng.uniform(-1.1, 1.1)),
                size_x=float(rng.uniform(0.18, 0.35)),
                size_y=float(rng.uniform(0.30, 0.65)),
                height=float(rng.uniform(0.12, 0.30)),
            )
            for index in range(6)
        )
        return replace(self, obstacles=obstacles)


def _box(name: str, x: float, y: float, sx: float, sy: float, height: float = 0.20) -> ObstacleBox:
    return ObstacleBox(name, x, y, sx, sy, height)


_SCENARIOS = {
    "single_obstacle": NavigationScenario(
        "single_obstacle", "单个中置障碍", (5.0, 0.0), (_box("single", 2.5, 0.0, 0.12, 0.80, 0.15),), 30.0
    ),
    "double_obstacle": NavigationScenario(
        "double_obstacle", "两个交错障碍", (5.5, 0.0),
        (_box("double_a", 2.0, 0.25, 0.18, 0.85), _box("double_b", 3.5, -0.30, 0.18, 0.85)), 35.0
    ),
    "triple_obstacle": NavigationScenario(
        "triple_obstacle", "三个连续交错障碍", (6.0, 0.0),
        (_box("triple_a", 1.8, 0.30, 0.16, 0.80), _box("triple_b", 3.1, -0.35, 0.16, 0.80),
         _box("triple_c", 4.4, 0.30, 0.16, 0.80)), 40.0
    ),
    "random_boxes": NavigationScenario(
        "random_boxes", "可重现随机箱体", (6.0, 0.0), (), 45.0
    ),
    "narrow_corridor": NavigationScenario(
        "narrow_corridor", "1.2米宽狭长通道", (5.5, 0.0),
        (_box("corridor_left", 3.0, 0.66, 4.5, 0.12, 0.40),
         _box("corridor_right", 3.0, -0.66, 4.5, 0.12, 0.40)), 35.0
    ),
    "wide_wall": NavigationScenario(
        "wide_wall", "需要大幅横向绕行的宽墙", (5.5, 0.0),
        (_box("wide_wall", 2.7, 0.0, 0.16, 2.20, 0.35),), 45.0
    ),
    "l_wall": NavigationScenario(
        "l_wall", "L形墙体", (5.5, -0.8),
        (_box("l_cross", 2.7, 0.20, 0.16, 1.80, 0.35),
         _box("l_side", 3.4, 1.04, 1.55, 0.16, 0.35)), 45.0
    ),
    "u_wall": NavigationScenario(
        "u_wall", "U形凹槽与局部极小值", (5.5, 0.0),
        (_box("u_back", 3.4, 0.0, 0.16, 2.00, 0.35),
         _box("u_left", 2.8, 0.94, 1.35, 0.16, 0.35),
         _box("u_right", 2.8, -0.94, 1.35, 0.16, 0.35)), 50.0
    ),
    "dead_end": NavigationScenario(
        "dead_end", "封闭死胡同，验证失败检测", (5.5, 0.0),
        (_box("dead_back", 3.5, 0.0, 0.16, 1.80, 0.35),
         _box("dead_left", 2.5, 0.84, 2.15, 0.16, 0.35),
         _box("dead_right", 2.5, -0.84, 2.15, 0.16, 0.35)), 35.0
    ),
    "maze": NavigationScenario(
        "maze", "交错隔墙构成的局部迷宫", (6.5, 0.0),
        (_box("maze_a", 2.0, 0.55, 0.16, 1.70, 0.35),
         _box("maze_b", 3.3, -0.55, 0.16, 1.70, 0.35),
         _box("maze_c", 4.6, 0.55, 0.16, 1.70, 0.35),
         _box("maze_d", 5.7, -0.55, 0.16, 1.70, 0.35)), 55.0
    ),
    "dynamic_obstacle": NavigationScenario(
        "dynamic_obstacle", "横向往复移动障碍（实验项）", (5.5, 0.0),
        (ObstacleBox("dynamic", 2.7, -0.9, 0.22, 0.35, 0.35, velocity_y=0.28, motion_limit=0.9),), 40.0
    ),
}


def scenario_names(include_dynamic: bool = True) -> tuple[str, ...]:
    return tuple(name for name, scenario in _SCENARIOS.items() if include_dynamic or not scenario.dynamic)


def get_scenario(name: str, seed: int = 7) -> NavigationScenario:
    try:
        return _SCENARIOS[name].with_seed(seed)
    except KeyError as error:
        raise ValueError(f"未知导航场景 {name}，可选：{', '.join(scenario_names())}") from error

"""Deterministic Isaac terrain adapter for the frozen MuJoCo benchmark scenarios."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from navigation.scenarios import NavigationScenario, get_scenario


@configclass
class NavigationScenarioTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    """One exact benchmark layout, expressed in the terrain-local frame."""

    function = None
    scenario_name: str = MISSING
    seed: int = 7


def navigation_scenario_terrain(
    difficulty: float, cfg: NavigationScenarioTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a plane and boxes while keeping the robot spawn at local (0, 0)."""

    del difficulty
    scenario = get_scenario(cfg.scenario_name, seed=cfg.seed)
    origin = np.asarray((cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0))
    meshes = [make_plane(cfg.size, 0.0, center_zero=False)]
    for obstacle in scenario.obstacles:
        # Dynamic motion is deliberately not approximated as static during evaluation.
        # The t=0 mesh exists for layout inspection only; callers mark it unsupported.
        x, y = obstacle.center_at(0.0)
        center = origin + np.asarray((x, y, obstacle.height / 2.0))
        meshes.append(
            trimesh.creation.box(
                (obstacle.size_x, obstacle.size_y, obstacle.height),
                trimesh.transformations.translation_matrix(center),
            )
        )
    return meshes, origin


def make_benchmark_terrain_cfg(
    scenario_names: tuple[str, ...], seed: int = 7
) -> TerrainGeneratorCfg:
    """Map each scenario deterministically to one terrain column."""

    if not scenario_names:
        raise ValueError("at least one benchmark scenario is required")
    proportion = 1.0 / len(scenario_names)
    sub_terrains = {
        name: NavigationScenarioTerrainCfg(
            function=navigation_scenario_terrain,
            proportion=proportion,
            scenario_name=name,
            seed=seed,
        )
        for name in scenario_names
    }
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=(16.0, 8.0),
        border_width=2.0,
        num_rows=1,
        num_cols=len(scenario_names),
        color_scheme="none",
        difficulty_range=(0.0, 0.0),
        sub_terrains=sub_terrains,
    )


def scenario_for_env(
    scenarios: tuple[NavigationScenario, ...], terrain_type: int
) -> NavigationScenario:
    """Small pure helper used by both the environment and contract tests."""

    if terrain_type < 0 or terrain_type >= len(scenarios):
        raise IndexError(f"terrain type {terrain_type} has no benchmark scenario")
    return scenarios[terrain_type]

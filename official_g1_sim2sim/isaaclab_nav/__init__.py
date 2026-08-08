"""Isaac Lab adapters and task registration for G1 navigation."""

from __future__ import annotations

import gymnasium as gym

from .contracts import (
    UPPER_FRAME_DIM,
    UPPER_HISTORY_LENGTH,
    UPPER_OBSERVATION_DIM,
    BatchSafety,
    UpperObservationHistory,
)
from .walking import FrozenWalkingPolicy

TASK_ID = "Unitree-G1-29dof-Visual-Navigation"


def register_task() -> None:
    """Register the task once, without modifying Unitree RL Lab's tasks."""

    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab_nav.env:G1VisualNavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_nav.env_cfg:G1VisualNavigationEnvCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_nav.ppo_cfg:G1NavigationPPORunnerCfg",
        },
    )


register_task()

__all__ = [
    "TASK_ID",
    "UPPER_FRAME_DIM",
    "UPPER_HISTORY_LENGTH",
    "UPPER_OBSERVATION_DIM",
    "BatchSafety",
    "FrozenWalkingPolicy",
    "UpperObservationHistory",
    "register_task",
]

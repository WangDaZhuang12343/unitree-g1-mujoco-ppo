"""CPU adapter that preserves the frozen MuJoCo DWA as an exceptional fallback."""

from __future__ import annotations

import numpy as np
import torch

from g1_nav.l2_costmap import LocalCostMap
from g1_nav.l3_dwa import DWANavigator


class BatchedDwaFallback:
    """Run existing DWA only for selected environments, outside the PPO hot path."""

    def __init__(self, num_envs: int) -> None:
        self._costmaps = [LocalCostMap() for _ in range(num_envs)]
        self._planners = [DWANavigator() for _ in range(num_envs)]

    def reset(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.detach().cpu().tolist():
            self._planners[env_id].previous_omega = 0.0

    def plan(
        self,
        points_body: torch.Tensor,
        ground_z_body: torch.Tensor,
        goal_body: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if points_body.ndim != 3 or points_body.shape[-1] != 3:
            raise ValueError("points_body must have shape (N, rays, 3)")
        command = torch.zeros((points_body.shape[0], 3), device=points_body.device)
        for env_id in torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist():
            points = points_body[env_id].detach().cpu().numpy()
            points = points[np.isfinite(points).all(axis=1)]
            costmap = self._costmaps[env_id]
            costmap.update(points, ground_z_body=float(ground_z_body[env_id]))
            goal = goal_body[env_id, :2].detach().cpu().numpy()
            result = self._planners[env_id].plan(costmap, (float(goal[0]), float(goal[1])))
            command[env_id] = command.new_tensor((result.vx, result.vy, result.omega))
        return command

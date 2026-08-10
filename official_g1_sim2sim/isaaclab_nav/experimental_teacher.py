"""Experimental teacher contracts that leave the frozen DWA implementation untouched."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class BatchedNavigationTeacher(Protocol):
    """Minimal interface shared by frozen and future privileged teachers."""

    def plan(
        self,
        points_body: torch.Tensor,
        ground_z_body: torch.Tensor,
        goal_body: torch.Tensor,
    ) -> torch.Tensor: ...

    def reset(self, env_ids: torch.Tensor) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RecoveryTeacherConfig:
    """Conservative stuck detector; disabled unless explicitly selected."""

    minimum_progress_m: float = 0.02
    stagnant_steps: int = 20
    recovery_vx: float = 0.08
    recovery_vy: float = 0.04
    recovery_omega: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_progress_m < 0.0 or self.stagnant_steps <= 0:
            raise ValueError("invalid recovery stuck detector")
        if not 0.0 <= self.recovery_vx <= 0.45:
            raise ValueError("recovery_vx is outside the walking command contract")
        if not 0.0 <= self.recovery_vy <= 0.10:
            raise ValueError("recovery_vy is outside the walking command contract")
        if not 0.0 <= self.recovery_omega <= 0.20:
            raise ValueError("recovery_omega is outside the walking command contract")


class RecoveryAugmentedTeacher:
    """Wrap a teacher with a bounded turn-and-creep recovery state.

    This is a separate experimental teacher.  It neither edits nor subclasses
    ``DWANavigator`` and is never used by the online fallback path.
    """

    def __init__(
        self,
        base: BatchedNavigationTeacher,
        num_envs: int,
        device: torch.device | str,
        config: RecoveryTeacherConfig | None = None,
    ) -> None:
        self.base = base
        self.config = config or RecoveryTeacherConfig()
        self._last_distance = torch.full((num_envs,), float("nan"), device=device)
        self._stagnant = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.recovery_used = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def plan(
        self,
        points_body: torch.Tensor,
        ground_z_body: torch.Tensor,
        goal_body: torch.Tensor,
    ) -> torch.Tensor:
        command = self.base.plan(points_body, ground_z_body, goal_body)
        distance = torch.linalg.norm(goal_body[:, :2], dim=1)
        progress = self._last_distance - distance
        initialized = torch.isfinite(self._last_distance)
        stagnant = initialized & (progress < self.config.minimum_progress_m)
        self._stagnant = torch.where(stagnant, self._stagnant + 1, torch.zeros_like(self._stagnant))
        self.recovery_used = self._stagnant >= self.config.stagnant_steps
        if self.recovery_used.any():
            lateral_sign = torch.where(goal_body[:, 1] >= 0.0, 1.0, -1.0)
            recovery = torch.stack(
                (
                    torch.full_like(distance, self.config.recovery_vx),
                    lateral_sign * self.config.recovery_vy,
                    lateral_sign * self.config.recovery_omega,
                ),
                dim=1,
            )
            command = torch.where(self.recovery_used[:, None], recovery, command)
        self._last_distance = distance
        return command

    def reset(self, env_ids: torch.Tensor) -> None:
        self.base.reset(env_ids)
        self._last_distance[env_ids] = float("nan")
        self._stagnant[env_ids] = 0
        self.recovery_used[env_ids] = False

    def close(self) -> None:
        self.base.close()


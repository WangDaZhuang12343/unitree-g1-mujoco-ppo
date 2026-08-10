"""GPU-friendly navigation curriculum state independent of Isaac Sim imports."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NavigationCurriculumConfig:
    """Promote sustained success and demote repeated unsafe/timeout episodes."""

    promotion_successes: int = 3
    demotion_failures: int = 2

    def __post_init__(self) -> None:
        if self.promotion_successes <= 0 or self.demotion_failures <= 0:
            raise ValueError("curriculum streak thresholds must be positive")


class NavigationCurriculumState:
    """Per-environment streak tracker kept on the simulation device.

    Terrain geometry remains owned by Isaac Lab.  This class only produces the
    ``move_up``/``move_down`` masks expected by ``TerrainImporter`` and is
    therefore unit-testable without launching Kit.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        config: NavigationCurriculumConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.config = config or NavigationCurriculumConfig()
        self.success_streak = torch.zeros(num_envs, dtype=torch.int16, device=device)
        self.failure_streak = torch.zeros_like(self.success_streak)

    def update(
        self,
        env_ids: torch.Tensor,
        completed: torch.Tensor,
        success: torch.Tensor,
        unsafe_or_timeout: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return promotion/demotion masks aligned with ``env_ids``."""

        size = len(env_ids)
        for name, value in (
            ("completed", completed),
            ("success", success),
            ("unsafe_or_timeout", unsafe_or_timeout),
        ):
            if value.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
        success = completed & success
        failure = completed & unsafe_or_timeout & ~success
        neutral = completed & ~success & ~failure

        self.success_streak[env_ids] = torch.where(
            success,
            self.success_streak[env_ids] + 1,
            torch.zeros_like(self.success_streak[env_ids]),
        )
        self.failure_streak[env_ids] = torch.where(
            failure,
            self.failure_streak[env_ids] + 1,
            torch.zeros_like(self.failure_streak[env_ids]),
        )
        # A neutral externally-reset episode must not carry stale evidence.
        self.success_streak[env_ids[neutral]] = 0
        self.failure_streak[env_ids[neutral]] = 0

        move_up = completed & (
            self.success_streak[env_ids] >= self.config.promotion_successes
        )
        move_down = completed & ~move_up & (
            self.failure_streak[env_ids] >= self.config.demotion_failures
        )
        decided = env_ids[move_up | move_down]
        self.success_streak[decided] = 0
        self.failure_streak[decided] = 0
        return move_up, move_down

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.success_streak.zero_()
            self.failure_streak.zero_()
        else:
            self.success_streak[env_ids] = 0
            self.failure_streak[env_ids] = 0


"""Batched contracts shared by the Isaac Lab navigation environment.

This module deliberately has no Isaac Sim imports.  It keeps the upper policy
contract testable without launching Kit and mirrors the frozen scalar safety
semantics in :mod:`g1_nav.l6_safety`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


UPPER_FRAME_DIM = 120
UPPER_HISTORY_LENGTH = 4
UPPER_GOAL_DIM = 3
UPPER_OBSERVATION_DIM = UPPER_FRAME_DIM * UPPER_HISTORY_LENGTH + UPPER_GOAL_DIM


@dataclass(frozen=True)
class BatchSafetyConfig:
    """Tensor equivalent of the immutable MuJoCo ``SafetyConfig``."""

    max_vx: float = 0.45
    max_vy: float = 0.10
    max_omega: float = 0.20
    max_vx_acceleration: float = 0.60
    max_vy_acceleration: float = 0.30
    max_omega_acceleration: float = 0.60
    maximum_tilt_rad: float = 0.55
    minimum_height: float = 0.50


class BatchSafety:
    """Per-environment safety state that stays on the simulation device."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        config: BatchSafetyConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.config = config or BatchSafetyConfig()
        self.command = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.emergency_stopped = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def num_envs(self) -> int:
        return self.command.shape[0]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.command.zero_()
            self.emergency_stopped.zero_()
            return
        self.command[env_ids] = 0.0
        self.emergency_stopped[env_ids] = False

    def stop(self, mask: torch.Tensor) -> None:
        if mask.shape != (self.num_envs,):
            raise ValueError(f"stop mask must have shape ({self.num_envs},)")
        self.emergency_stopped |= mask
        self.command[mask] = 0.0

    def update(
        self,
        desired_command: torch.Tensor,
        dt: float,
        base_height: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
    ) -> torch.Tensor:
        if desired_command.shape != (self.num_envs, 3):
            raise ValueError(f"desired_command must have shape ({self.num_envs}, 3)")
        for name, value in (("base_height", base_height), ("roll", roll), ("pitch", pitch)):
            if value.shape != (self.num_envs,):
                raise ValueError(f"{name} must have shape ({self.num_envs},)")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        cfg = self.config
        unsafe = (
            self.emergency_stopped
            | (base_height < cfg.minimum_height)
            | (torch.maximum(roll.abs(), pitch.abs()) > cfg.maximum_tilt_rad)
            | ~torch.isfinite(desired_command).all(dim=1)
        )
        self.stop(unsafe)

        lower = desired_command.new_tensor((0.0, -cfg.max_vy, -cfg.max_omega))
        upper = desired_command.new_tensor((cfg.max_vx, cfg.max_vy, cfg.max_omega))
        target = torch.minimum(torch.maximum(desired_command, lower), upper)
        limits = desired_command.new_tensor(
            (cfg.max_vx_acceleration, cfg.max_vy_acceleration, cfg.max_omega_acceleration)
        ) * dt
        delta = torch.minimum(torch.maximum(target - self.command, -limits), limits)
        safe = ~self.emergency_stopped
        self.command[safe] += delta[safe]
        self.command[self.emergency_stopped] = 0.0
        return self.command.clone()


class UpperObservationHistory:
    """Four-frame 483-D upper navigation observation history."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.buffer = torch.zeros(
            (num_envs, UPPER_HISTORY_LENGTH, UPPER_FRAME_DIM),
            dtype=torch.float32,
            device=device,
        )
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def num_envs(self) -> int:
        return self.buffer.shape[0]

    @staticmethod
    def make_frame(
        distance_field: torch.Tensor,
        base_linear_velocity: torch.Tensor,
        applied_command: torch.Tensor,
        base_angular_velocity: torch.Tensor,
        projected_gravity: torch.Tensor,
        safety_state: torch.Tensor,
        perception_health: torch.Tensor,
        progress_state: torch.Tensor,
    ) -> torch.Tensor:
        if distance_field.ndim == 3 and distance_field.shape[1:] == (10, 10):
            distance_field = distance_field.flatten(1)
        parts = (
            ("distance_field", distance_field, 100),
            ("base_linear_velocity", base_linear_velocity, 3),
            ("applied_command", applied_command, 3),
            ("base_angular_velocity", base_angular_velocity, 3),
            ("projected_gravity", projected_gravity, 3),
            ("safety_state", safety_state, 3),
            ("perception_health", perception_health, 3),
            ("progress_state", progress_state, 2),
        )
        num_envs = distance_field.shape[0]
        for name, value, width in parts:
            if value.shape != (num_envs, width):
                raise ValueError(f"{name} must have shape ({num_envs}, {width})")
        frame = torch.cat([value for _, value, _ in parts], dim=1).to(dtype=torch.float32)
        if frame.shape[1] != UPPER_FRAME_DIM:
            raise AssertionError(f"upper frame width changed to {frame.shape[1]}")
        return frame

    def reset(self, frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        if frame.shape != (self.num_envs, UPPER_FRAME_DIM):
            raise ValueError(f"frame must have shape ({self.num_envs}, {UPPER_FRAME_DIM})")
        if env_ids is None:
            self.buffer[:] = frame[:, None, :]
            self.initialized[:] = True
            return
        self.buffer[env_ids] = frame[env_ids, None, :]
        self.initialized[env_ids] = True

    def append(self, frame: torch.Tensor) -> None:
        if frame.shape != (self.num_envs, UPPER_FRAME_DIM):
            raise ValueError(f"frame must have shape ({self.num_envs}, {UPPER_FRAME_DIM})")
        new_ids = torch.nonzero(~self.initialized, as_tuple=False).flatten()
        if new_ids.numel():
            self.reset(frame, new_ids)
        existing = self.initialized.clone()
        existing[new_ids] = False
        self.buffer[existing, :-1] = self.buffer[existing, 1:].clone()
        self.buffer[existing, -1] = frame[existing]

    def observation(self, goal: torch.Tensor) -> torch.Tensor:
        if goal.shape != (self.num_envs, UPPER_GOAL_DIM):
            raise ValueError(f"goal must have shape ({self.num_envs}, {UPPER_GOAL_DIM})")
        observation = torch.cat((self.buffer.flatten(1), goal.to(dtype=torch.float32)), dim=1)
        if observation.shape != (self.num_envs, UPPER_OBSERVATION_DIM):
            raise AssertionError(f"upper observation shape changed to {observation.shape}")
        return observation

"""Pure-PyTorch contracts for DWA behavior cloning and PPO warm starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


ACTOR_INPUT_DIM = 483
ACTOR_OUTPUT_DIM = 3
ACTOR_HIDDEN_DIMS = (512, 256, 128)
ACTOR_CHECKPOINT_FORMAT = "g1_isaaclab_navigation_actor_v1"
RSL_RL_EXPORT_FORMAT = "g1_isaaclab_navigation_rsl_rl_export_v1"
NORMALIZER_EPS = 1.0e-2


def physical_command_to_normalized_action(command: torch.Tensor) -> torch.Tensor:
    """Convert ``[vx, vy, omega]`` to the environment's normalized action."""

    if command.shape[-1] != ACTOR_OUTPUT_DIM:
        raise ValueError("command must end in three [vx, vy, omega] values")
    action = torch.empty_like(command)
    action[..., 0] = command[..., 0] / 0.225 - 1.0
    action[..., 1] = command[..., 1] / 0.10
    action[..., 2] = command[..., 2] / 0.20
    return action.clamp(-1.0, 1.0)


def normalized_action_to_physical_command(action: torch.Tensor) -> torch.Tensor:
    """Apply the exact action transform used by ``G1VisualNavigationEnv``."""

    if action.shape[-1] != ACTOR_OUTPUT_DIM:
        raise ValueError("action must end in three normalized values")
    action = action.clamp(-1.0, 1.0)
    command = torch.empty_like(action)
    command[..., 0] = 0.225 * (action[..., 0] + 1.0)
    command[..., 1] = 0.10 * action[..., 1]
    command[..., 2] = 0.20 * action[..., 2]
    return command


def observation_moments(observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable per-feature moments in the RSL-RL normalizer convention."""

    if observation.ndim != 2 or observation.shape[1] != ACTOR_INPUT_DIM:
        raise ValueError(f"observation must have shape (N, {ACTOR_INPUT_DIM})")
    mean = observation.mean(dim=0, keepdim=True)
    std = observation.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-4)
    return mean, std


class NavigationActor(nn.Sequential):
    """MLP whose parameter names exactly match RSL-RL's actor module."""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(ACTOR_INPUT_DIM, ACTOR_HIDDEN_DIMS[0]),
            nn.ELU(),
            nn.Linear(ACTOR_HIDDEN_DIMS[0], ACTOR_HIDDEN_DIMS[1]),
            nn.ELU(),
            nn.Linear(ACTOR_HIDDEN_DIMS[1], ACTOR_HIDDEN_DIMS[2]),
            nn.ELU(),
            nn.Linear(ACTOR_HIDDEN_DIMS[2], ACTOR_OUTPUT_DIM),
        )


class NormalizedNavigationActor(nn.Module):
    """Deployment wrapper matching RSL-RL actor observation normalization."""

    def __init__(self, actor: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor((observation - self.mean) / (self.std + NORMALIZER_EPS))


def make_actor_checkpoint(
    actor: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    sample_count: int,
    validation_mae: float,
    action_std: float = 0.20,
) -> dict[str, Any]:
    """Build an actor-only warm-start checkpoint; critic and optimizer are excluded."""

    if mean.shape != (1, ACTOR_INPUT_DIM) or std.shape != (1, ACTOR_INPUT_DIM):
        raise ValueError("normalizer mean/std must have shape (1, 483)")
    if sample_count <= 0 or action_std <= 0.0:
        raise ValueError("sample_count and action_std must be positive")
    variance = torch.square(std)
    return {
        "format": ACTOR_CHECKPOINT_FORMAT,
        "actor_state_dict": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
        "actor_obs_normalizer_state_dict": {
            "_mean": mean.detach().cpu(),
            "_var": variance.detach().cpu(),
            "_std": std.detach().cpu(),
            "count": torch.tensor(sample_count, dtype=torch.long),
        },
        "action_std": float(action_std),
        "metadata": {
            "observation_dim": ACTOR_INPUT_DIM,
            "action_dim": ACTOR_OUTPUT_DIM,
            "hidden_dims": list(ACTOR_HIDDEN_DIMS),
            "sample_count": int(sample_count),
            "validation_mae": float(validation_mae),
            "normalizer_eps": NORMALIZER_EPS,
            "action_order": ["vx", "vy", "omega"],
        },
    }


def load_actor_initialization(policy: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load only actor, actor normalizer, and exploration std into an RSL-RL policy."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != ACTOR_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported actor checkpoint format: {payload.get('format')!r}")
    metadata = payload.get("metadata", {})
    expected = (ACTOR_INPUT_DIM, ACTOR_OUTPUT_DIM, list(ACTOR_HIDDEN_DIMS))
    actual = (
        metadata.get("observation_dim"),
        metadata.get("action_dim"),
        metadata.get("hidden_dims"),
    )
    if actual != expected:
        raise ValueError(f"actor checkpoint contract mismatch: expected {expected}, got {actual}")
    policy.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    if normalizer is None or not hasattr(normalizer, "_mean"):
        raise ValueError("PPO actor observation normalization must be explicitly enabled")
    normalizer.load_state_dict(payload["actor_obs_normalizer_state_dict"], strict=True)
    with torch.no_grad():
        if hasattr(policy, "std"):
            policy.std.fill_(float(payload["action_std"]))
        elif hasattr(policy, "log_std"):
            policy.log_std.fill_(float(torch.log(torch.tensor(payload["action_std"]))))
        else:
            raise ValueError("unsupported RSL-RL action noise parameter")
    return metadata


def actor_from_rsl_rl_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[NormalizedNavigationActor, dict[str, Any]]:
    """Extract the deterministic actor and its normalizer from an RSL-RL checkpoint."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("RSL-RL checkpoint is missing model_state_dict")
    actor = NavigationActor()
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(actor_state, strict=True)
    try:
        mean = state["actor_obs_normalizer._mean"]
        std = state["actor_obs_normalizer._std"]
    except KeyError as error:
        raise ValueError("RSL-RL checkpoint is missing actor observation normalization") from error
    if mean.shape != (1, ACTOR_INPUT_DIM) or std.shape != (1, ACTOR_INPUT_DIM):
        raise ValueError("RSL-RL actor normalizer must have shape (1, 483)")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0.0):
        raise ValueError("RSL-RL actor normalizer contains invalid values")
    metadata = {
        "format": RSL_RL_EXPORT_FORMAT,
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "iteration": int(payload.get("iter", -1)),
        "observation_dim": ACTOR_INPUT_DIM,
        "action_dim": ACTOR_OUTPUT_DIM,
        "hidden_dims": list(ACTOR_HIDDEN_DIMS),
        "normalizer_eps": NORMALIZER_EPS,
        "action_order": ["vx", "vy", "omega"],
    }
    return NormalizedNavigationActor(actor.eval(), mean.detach().clone(), std.detach().clone()).eval(), metadata

"""GPU-local perception helpers for the Isaac Lab navigation task."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalDistanceFieldConfig:
    """Geometry of the compact body-frame distance field."""

    rows: int = 10
    cols: int = 10
    rear_range: float = 0.5
    front_range: float = 4.0
    side_range: float = 2.0
    inflation_radius: float = 0.28
    max_clearance: float = 5.0


def local_distance_field(
    points_body: torch.Tensor,
    obstacle_mask: torch.Tensor,
    cfg: LocalDistanceFieldConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a Cartesian clearance field and robot-centric minimum clearance.

    The returned field is normalized to ``[0, 1]``. Zero denotes an occupied
    or inflated cell and one denotes clearance at or beyond the truncation
    distance. All computation stays on the input tensor's device.
    """

    if points_body.ndim != 3 or points_body.shape[-1] != 3:
        raise ValueError("points_body must have shape (N, rays, 3)")
    if obstacle_mask.shape != points_body.shape[:2]:
        raise ValueError("obstacle_mask must match the first two point dimensions")
    if cfg.rows <= 0 or cfg.cols <= 0 or cfg.max_clearance <= 0.0:
        raise ValueError("distance-field dimensions and max_clearance must be positive")

    dtype = points_body.dtype
    device = points_body.device
    x_step = (cfg.front_range + cfg.rear_range) / cfg.cols
    y_step = 2.0 * cfg.side_range / cfg.rows
    x = torch.linspace(
        -cfg.rear_range + 0.5 * x_step,
        cfg.front_range - 0.5 * x_step,
        cfg.cols,
        dtype=dtype,
        device=device,
    )
    y = torch.linspace(
        -cfg.side_range + 0.5 * y_step,
        cfg.side_range - 0.5 * y_step,
        cfg.rows,
        dtype=dtype,
        device=device,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    cells = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)

    points_xy = points_body[..., :2]
    distances = torch.linalg.norm(
        cells[None, :, None, :] - points_xy[:, None, :, :], dim=-1
    )
    distances = distances.masked_fill(~obstacle_mask[:, None, :], torch.inf)
    clearance = distances.amin(dim=2) - cfg.inflation_radius
    field = (clearance / cfg.max_clearance).clamp(0.0, 1.0)
    field = field.reshape(points_body.shape[0], cfg.rows, cfg.cols)

    robot_distance = torch.linalg.norm(points_xy, dim=-1)
    robot_distance = robot_distance.masked_fill(~obstacle_mask, torch.inf)
    robot_clearance = (robot_distance.amin(dim=1) - cfg.inflation_radius).clamp(
        0.0, cfg.max_clearance
    )
    return field, robot_clearance

"""Process-parallel data adapter around the unchanged frozen DWA planner."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch

from g1_nav.l2_costmap import LocalCostMap
from g1_nav.l3_dwa import DWANavigator


def _dwa_worker(connection) -> None:
    """Own one stateful planner in a CPU-only spawned process."""

    costmap = LocalCostMap()
    planner = DWANavigator()
    while True:
        message = connection.recv()
        operation = message[0]
        if operation == "close":
            connection.close()
            return
        if operation == "reset":
            planner.previous_omega = 0.0
            continue
        if operation != "plan":
            raise ValueError(f"unknown DWA worker operation: {operation}")
        _, points, ground_z, goal = message
        points = points[np.isfinite(points).all(axis=1)]
        costmap.update(points, ground_z_body=float(ground_z))
        result = planner.plan(costmap, (float(goal[0]), float(goal[1])))
        connection.send((result.vx, result.vy, result.omega))


class ParallelDwaTeacher:
    """Label independent Isaac environments concurrently on CPU."""

    def __init__(self, num_envs: int, workers: int) -> None:
        if num_envs <= 0 or workers <= 0:
            raise ValueError("num_envs and workers must be positive")
        if workers < num_envs:
            raise ValueError("process DWA labeling requires at least one worker per environment")
        context = mp.get_context("spawn")
        self._connections = []
        self._processes = []
        for _ in range(num_envs):
            parent, child = context.Pipe()
            process = context.Process(target=_dwa_worker, args=(child,), daemon=True)
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def reset(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.detach().cpu().tolist():
            self._connections[env_id].send(("reset",))

    def plan(
        self,
        points_body: torch.Tensor,
        ground_z_body: torch.Tensor,
        goal_body: torch.Tensor,
    ) -> torch.Tensor:
        points = points_body.detach().cpu().numpy()
        ground = ground_z_body.detach().cpu().numpy()
        goals = goal_body[:, :2].detach().cpu().numpy()
        for env_id, connection in enumerate(self._connections):
            connection.send(("plan", points[env_id], float(ground[env_id]), goals[env_id]))
        command = torch.zeros((points.shape[0], 3), device=points_body.device)
        for env_id, connection in enumerate(self._connections):
            value = connection.recv()
            command[env_id] = command.new_tensor(value)
        return command

    def close(self) -> None:
        for connection in self._connections:
            connection.send(("close",))
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

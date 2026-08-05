"""深度、点云、代价地图和 DWA 轨迹的实时四联调试界面。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np

from navigation.runtime import NavigationDebugFrame


class NavigationDebugViewer:
    def __init__(self, title: str = "G1 Navigation Pipeline Debug Viewer") -> None:
        plt.ion()
        self.figure, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), constrained_layout=True)
        self._interactive_canvas = self.figure.canvas.__class__.__module__ != "matplotlib.backends.backend_agg"
        self.figure.canvas.manager.set_window_title(title)
        self.depth_axis, self.cloud_axis, self.costmap_axis, self.planner_axis = axes.flat
        self._closed = False
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self.depth_image = self.depth_axis.imshow(
            np.zeros((40, 64), dtype=np.float32), vmin=0.0, vmax=3.5,
            cmap="viridis_r", interpolation="nearest", aspect="auto",
        )
        self.figure.colorbar(self.depth_image, ax=self.depth_axis, label="Depth (m)")
        self.depth_axis.set(title="Depth Image", xlabel="Pixel X", ylabel="Pixel Y")

        self.cloud_scatter = self.cloud_axis.scatter([], [], s=5, c=[], cmap="turbo", vmin=-0.8, vmax=0.8)
        self.obstacle_scatter = self.cloud_axis.scatter(
            [], [], s=16, facecolors="none", edgecolors="#dc2626", linewidths=0.8, label="Obstacle"
        )
        self.figure.colorbar(self.cloud_scatter, ax=self.cloud_axis, label="Height in body frame (m)")
        self.cloud_axis.set(
            title="Point Cloud (body frame)", xlabel="Forward X (m)", ylabel="Left Y (m)",
            xlim=(-0.5, 4.0), ylim=(-2.0, 2.0), aspect="equal",
        )
        self.cloud_axis.grid(True, alpha=0.25)
        self.cloud_axis.legend(loc="upper right")

        self.costmap_image = self.costmap_axis.imshow(
            np.zeros((80, 90), dtype=np.float32), origin="lower", interpolation="nearest",
            cmap="Greys", vmin=0.0, vmax=1.0, extent=(-0.5, 4.0, -2.0, 2.0), aspect="equal",
        )
        self.costmap_robot = Circle((0.0, 0.0), 0.22, facecolor="#1677ff", edgecolor="white", linewidth=1.2)
        self.costmap_axis.add_patch(self.costmap_robot)
        (self.costmap_goal,) = self.costmap_axis.plot([], [], marker="*", color="#16a34a", markersize=13)
        self.costmap_axis.set(title="Occupancy Costmap", xlabel="Forward X (m)", ylabel="Left Y (m)")

        self.invalid_paths = LineCollection([], colors="#dc2626", linewidths=0.35, alpha=0.12)
        self.valid_paths = LineCollection([], colors="#64748b", linewidths=0.45, alpha=0.18)
        self.planner_axis.add_collection(self.invalid_paths)
        self.planner_axis.add_collection(self.valid_paths)
        (self.selected_path,) = self.planner_axis.plot([], [], color="#16a34a", linewidth=3.0, label="Selected")
        self.planner_robot = Circle((0.0, 0.0), 0.22, facecolor="#1677ff", edgecolor="black", linewidth=1.0)
        self.planner_axis.add_patch(self.planner_robot)
        (self.planner_goal,) = self.planner_axis.plot([], [], marker="*", color="#f59e0b", markersize=14, label="Goal")
        self.planner_axis.set(
            title="DWA Candidate Paths", xlabel="Forward X (m)", ylabel="Left Y (m)",
            xlim=(-0.5, 4.0), ylim=(-2.0, 2.0), aspect="equal",
        )
        self.planner_axis.grid(True, alpha=0.25)
        self.planner_axis.legend(loc="upper right")
        self.status = self.figure.suptitle(title, fontsize=12)
        self.figure.canvas.draw_idle()
        if self._interactive_canvas:
            plt.show(block=False)

    def _on_close(self, _event) -> None:
        self._closed = True

    @property
    def is_open(self) -> bool:
        return not self._closed and plt.fignum_exists(self.figure.number)

    def update(self, frame: NavigationDebugFrame) -> bool:
        if not self.is_open:
            return False
        self.depth_image.set_data(frame.depth)
        finite_depth = frame.depth[np.isfinite(frame.depth)]
        if finite_depth.size:
            self.depth_image.set_clim(0.0, max(1.0, float(np.percentile(finite_depth, 99))))

        points = frame.points_body
        if points.size:
            self.cloud_scatter.set_offsets(points[:, :2])
            self.cloud_scatter.set_array(points[:, 2])
        else:
            self.cloud_scatter.set_offsets(np.empty((0, 2)))
            self.cloud_scatter.set_array(np.empty(0))
        obstacle_points = frame.obstacle_points_body
        self.obstacle_scatter.set_offsets(
            obstacle_points[:, :2] if obstacle_points.size else np.empty((0, 2))
        )

        self.costmap_image.set_data(frame.obstacle_map.astype(np.float32))
        self.costmap_image.set_extent(frame.map_extent)
        goal_x, goal_y = frame.goal_body
        self.costmap_goal.set_data([goal_x], [goal_y])

        trajectories = frame.candidate_trajectories[:, :, :2]
        self.valid_paths.set_segments(trajectories[frame.candidate_valid])
        self.invalid_paths.set_segments(trajectories[~frame.candidate_valid])
        selected = frame.selected_trajectory
        self.selected_path.set_data(selected[:, 0], selected[:, 1])
        self.planner_goal.set_data([goal_x], [goal_y])

        min_x, max_x, min_y, max_y = frame.map_extent
        display_max_x = max(max_x, min(max(goal_x, 0.0), 6.0) + 0.25)
        for axis in (self.costmap_axis, self.planner_axis):
            axis.set_xlim(min_x, display_max_x)
            axis.set_ylim(min_y, max_y)
        pose_x, pose_y, pose_yaw = frame.robot_world_pose
        vx, vy, omega = frame.command
        valid_count = int(np.count_nonzero(frame.candidate_valid))
        ground_state = "cached" if frame.ground_plane_cached else f"{frame.ground_inlier_ratio:.0%}"
        self.status.set_text(
            f"t={frame.time_s:5.1f}s | pose=({pose_x:+.2f}, {pose_y:+.2f}, {np.degrees(pose_yaw):+.1f} deg) | "
            f"cmd=({vx:.2f}, {vy:+.2f}, {omega:+.2f}) | "
            f"ground={ground_state} | DWA={frame.planning_ms:.1f} ms | "
            f"valid={valid_count}/{len(frame.candidate_valid)}"
        )
        if self._interactive_canvas:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)
        else:
            self.figure.canvas.draw()
        return self.is_open

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=140)

    def close(self) -> None:
        self._closed = True
        plt.close(self.figure)

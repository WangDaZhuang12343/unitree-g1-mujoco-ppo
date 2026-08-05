import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from navigation.runtime import NavigationDebugFrame
from visualization import NavigationDebugViewer


class DebugViewerTest(unittest.TestCase):
    def test_render_and_snapshot(self):
        trajectories = np.zeros((3, 8, 3), dtype=np.float32)
        trajectories[0, :, 0] = np.linspace(0.0, 1.0, 8)
        trajectories[1, :, 0] = np.linspace(0.0, 1.0, 8)
        trajectories[1, :, 1] = np.linspace(0.0, 0.5, 8)
        trajectories[2, :, 0] = np.linspace(0.0, 1.0, 8)
        trajectories[2, :, 1] = np.linspace(0.0, -0.5, 8)
        frame = NavigationDebugFrame(
            time_s=1.0,
            depth=np.linspace(0.5, 3.5, 40 * 64, dtype=np.float32).reshape(40, 64),
            points_body=np.array([[1.0, 0.2, 0.1], [1.5, -0.3, 0.2]], dtype=np.float32),
            obstacle_points_body=np.array([[1.5, -0.3, 0.2]], dtype=np.float32),
            ground_normal=(0.0, 0.0, 1.0),
            ground_offset=0.8,
            ground_inlier_ratio=0.82,
            ground_plane_cached=False,
            obstacle_map=np.zeros((80, 90), dtype=bool),
            distance_field=np.ones((80, 90), dtype=np.float32),
            map_extent=(-0.5, 4.0, -2.0, 2.0),
            candidate_trajectories=trajectories,
            candidate_valid=np.array([True, True, False]),
            selected_trajectory=trajectories[1],
            goal_body=(3.0, 0.0),
            robot_world_pose=(0.2, -0.1, 0.05),
            command=(0.3, 0.0, 0.1),
            planning_ms=12.5,
        )
        viewer = NavigationDebugViewer()
        try:
            self.assertTrue(viewer.update(frame))
            with TemporaryDirectory() as directory:
                output = Path(directory) / "debug.png"
                viewer.save(output)
                self.assertGreater(output.stat().st_size, 10_000)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()

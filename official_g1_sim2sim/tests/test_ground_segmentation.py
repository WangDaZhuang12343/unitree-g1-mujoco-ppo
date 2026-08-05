from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import yaml

from camera import GroundSegmenter
from g1_nav.l2_costmap import LocalCostMap
from g1_nav.l4_camera import SimulatedDepthCamera
from g1_nav.policy_contract import PolicyContract
from navigation.scenarios import get_scenario
from simulate import CONFIG_PATH, build_model, projected_gravity


class GroundSegmentationTest(unittest.TestCase):
    def test_inclined_noisy_plane_and_obstacle(self):
        rng = np.random.default_rng(21)
        x = rng.uniform(0.5, 3.5, 1200)
        y = rng.uniform(-1.5, 1.5, 1200)
        ground_z = 0.04 * x - 0.03 * y - 0.80 + rng.normal(0.0, 0.003, len(x))
        ground = np.column_stack([x, y, ground_z])
        obstacle = np.column_stack([
            rng.uniform(1.8, 2.2, 120),
            rng.uniform(-0.35, 0.35, 120),
            rng.uniform(-0.68, -0.55, 120),
        ])
        points = np.vstack([ground, obstacle]).astype(np.float32)
        expected_normal = np.array([-0.04, 0.03, 1.0])
        expected_normal /= np.linalg.norm(expected_normal)
        gravity = -expected_normal
        result = GroundSegmenter().segment(points, gravity)
        self.assertGreater(float(np.dot(result.plane.normal, expected_normal)), 0.995)
        self.assertGreater(np.mean(result.ground_mask[: len(ground)]), 0.98)
        self.assertGreater(np.mean(result.obstacle_mask[len(ground) :]), 0.95)

    def test_simulated_camera_matches_geom_oracle(self):
        scenario = get_scenario("single_obstacle")
        args = SimpleNamespace(
            terrain="flat", obstacle_height=0.15, obstacle_x=2.5, obstacle_width=0.8,
            slope_angle=5.0, roughness=0.02, terrain_seed=7,
            scene_obstacles=scenario.obstacles,
        )
        model = build_model(args)
        data = mujoco.MjData(model)
        with CONFIG_PATH.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        contract = PolicyContract.from_config(config)
        data.qpos[7 + contract.joint_map] = np.asarray(config["default_joint_pos"])
        mujoco.mj_forward(model, data)
        camera = SimulatedDepthCamera()
        camera.isolate_world_geoms(model)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        frame = camera.capture(model, data, body_id)
        result = GroundSegmenter().segment(frame.points_body, projected_gravity(data.qpos[3:7]))
        oracle_obstacle = frame.point_geom_ids != floor_id
        self.assertGreater(np.mean(result.ground_mask[~oracle_obstacle]), 0.99)
        self.assertGreater(np.mean(result.obstacle_mask[oracle_obstacle]), 0.95)
        self.assertGreater(result.inlier_ratio, 0.85)

    def test_costmap_uses_plane_relative_height(self):
        plane_normal = np.array([-0.05, 0.0, 1.0], dtype=np.float32)
        plane_normal /= np.linalg.norm(plane_normal)
        plane_offset = 0.8 / np.linalg.norm(np.array([-0.05, 0.0, 1.0]))
        x = np.linspace(1.0, 1.4, 20)
        ground_z = 0.05 * x - 0.8
        obstacle = np.column_stack([x, np.zeros_like(x), ground_z + 0.12])
        costmap = LocalCostMap()
        costmap.update(obstacle, ground_plane=(plane_normal, plane_offset))
        self.assertTrue(costmap.occupied(1.2, 0.0))

    def test_sparse_frame_uses_last_plane(self):
        ground = np.array([
            [1.0, -0.5, -0.8], [1.0, 0.5, -0.8], [2.0, -0.5, -0.8],
            [2.0, 0.5, -0.8], [3.0, 0.0, -0.8],
        ], dtype=np.float32)
        segmenter = GroundSegmenter()
        first = segmenter.segment(ground, np.array([0.0, 0.0, -1.0]))
        self.assertFalse(first.used_cached_plane)
        sparse = segmenter.segment(np.empty((0, 3)), np.array([0.0, 0.0, -1.0]))
        self.assertTrue(sparse.used_cached_plane)
        self.assertEqual(sparse.obstacle_points.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()

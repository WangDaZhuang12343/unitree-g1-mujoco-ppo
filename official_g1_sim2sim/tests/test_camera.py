from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import yaml

from g1_nav.l2_costmap import LocalCostMap
from g1_nav.l4_camera import SimulatedDepthCamera
from g1_nav.policy_contract import PolicyContract
from simulate import CONFIG_PATH, build_model


class CameraTest(unittest.TestCase):
    def test_camera_detects_course_bar(self):
        args = SimpleNamespace(
            terrain="bar",
            obstacle_height=0.15,
            obstacle_x=2.0,
            obstacle_width=1.8,
            slope_angle=5.0,
            roughness=0.02,
            terrain_seed=7,
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
        robot_geoms = model.geom_bodyid != 0
        self.assertTrue(np.all(model.geom_group[robot_geoms] == 1))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "course_bar")
        frame = camera.capture(model, data, body_id)
        self.assertEqual(frame.depth.shape, (40, 64))
        self.assertGreater(np.count_nonzero(frame.geom_ids == bar_id), 20)

        costmap = LocalCostMap()
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        non_ground = frame.point_geom_ids != floor_id
        costmap.update(frame.points_body[non_ground], ground_z_body=-float(data.xpos[body_id, 2]))
        self.assertTrue(any(costmap.occupied(x, 0.0) for x in np.linspace(1.5, 2.2, 15)))


if __name__ == "__main__":
    unittest.main()

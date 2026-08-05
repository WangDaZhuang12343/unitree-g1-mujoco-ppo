from pathlib import Path
import unittest

import numpy as np
import yaml

from g1_nav.policy_contract import ObservationHistory, PolicyContract
from simulate import CONFIG_PATH


class PolicyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Path(CONFIG_PATH).open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.contract = PolicyContract.from_config(cls.config)

    def test_observation_layout(self):
        expected = {
            "base_ang_vel": (0, 15),
            "projected_gravity": (15, 30),
            "velocity_commands": (30, 45),
            "joint_pos_rel": (45, 190),
            "joint_vel_rel": (190, 335),
            "last_action": (335, 480),
        }
        actual = {name: (value.start, value.stop) for name, value in self.contract.observation_slices.items()}
        self.assertEqual(actual, expected)
        self.assertEqual(
            self.contract.command_component_indices,
            {
                "vx": (30, 33, 36, 39, 42),
                "vy": (31, 34, 37, 40, 43),
                "omega": (32, 35, 38, 41, 44),
            },
        )

    def test_command_ranges(self):
        np.testing.assert_allclose(
            self.contract.command_ranges,
            [[-0.5, 1.0], [-0.3, 0.3], [-0.2, 0.2]],
        )
        np.testing.assert_allclose(self.contract.clip_command([2.0, -1.0, 0.5]), [1.0, -0.3, 0.2])

    def test_history_is_term_major_and_oldest_first(self):
        initial = {
            name: np.zeros(3 if name in {"base_ang_vel", "projected_gravity", "velocity_commands"} else 29, dtype=np.float32)
            for name in self.contract.observation_order
        }
        history = ObservationHistory(self.contract.history_lengths, initial)
        values = {name: value + index + 1 for index, (name, value) in enumerate(initial.items())}
        history.append(values)
        flat = history.flatten(list(self.contract.observation_order))
        self.assertEqual(flat.shape, (480,))
        self.assertTrue(np.all(flat[12:15] == 1.0))
        self.assertTrue(np.all(flat[27:30] == 2.0))
        self.assertTrue(np.all(flat[42:45] == 3.0))

    def test_action_processing_and_joint_mapping(self):
        raw = np.ones(29, dtype=np.float64)
        processed = self.contract.process_action(raw)
        np.testing.assert_allclose(processed, np.asarray(self.config["default_joint_pos"]) + 0.25)
        motor = self.contract.policy_to_motor(np.arange(29))
        for policy_index, motor_index in enumerate(self.contract.joint_map):
            self.assertEqual(motor[motor_index], policy_index)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from g1_nav.l6_safety import SafetySystem
from isaaclab_nav.contracts import (
    UPPER_FRAME_DIM,
    UPPER_OBSERVATION_DIM,
    BatchSafety,
    UpperObservationHistory,
)
from isaaclab_nav.fallback import BatchedDwaFallback


class IsaacLabContractsTest(unittest.TestCase):
    def test_dwa_fallback_only_updates_selected_environment(self):
        fallback = BatchedDwaFallback(2)
        command = fallback.plan(
            torch.empty((2, 0, 3)),
            torch.full((2,), -0.8),
            torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            torch.tensor([True, False]),
        )
        self.assertGreaterEqual(float(command[0, 0]), 0.25)
        self.assertTrue(torch.all(command[1] == 0.0))

    def test_batch_safety_matches_frozen_scalar_semantics(self):
        desired = torch.tensor([[1.0, 1.0, 1.0], [0.2, -0.04, -0.1]])
        batch = BatchSafety(2, "cpu")
        actual = batch.update(
            desired,
            dt=0.1,
            base_height=torch.tensor([0.8, 0.8]),
            roll=torch.zeros(2),
            pitch=torch.zeros(2),
        )
        expected = np.stack(
            [
                SafetySystem().update(*desired[index].numpy(), 0.1, 0.8, 0.0, 0.0)
                for index in range(2)
            ]
        )
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)

    def test_batch_safety_latches_and_resets_each_environment(self):
        safety = BatchSafety(2, "cpu")
        command = safety.update(
            torch.full((2, 3), 0.1),
            dt=0.1,
            base_height=torch.tensor([0.3, 0.8]),
            roll=torch.zeros(2),
            pitch=torch.zeros(2),
        )
        np.testing.assert_array_equal(command[0].numpy(), np.zeros(3))
        self.assertTrue(safety.emergency_stopped[0])
        safety.reset(torch.tensor([0]))
        self.assertFalse(safety.emergency_stopped[0])

    def test_upper_observation_is_483_and_reset_does_not_leak_history(self):
        num_envs = 2
        history = UpperObservationHistory(num_envs, "cpu")
        zeros3 = torch.zeros((num_envs, 3))
        frame = history.make_frame(
            torch.zeros((num_envs, 10, 10)),
            zeros3,
            zeros3,
            zeros3,
            zeros3,
            zeros3,
            zeros3,
            torch.zeros((num_envs, 2)),
        )
        self.assertEqual(frame.shape, (num_envs, UPPER_FRAME_DIM))
        history.reset(frame)
        changed = frame.clone()
        changed[0, 0] = 1.0
        history.append(changed)
        self.assertEqual(history.buffer[0, -1, 0], 1.0)
        self.assertEqual(history.buffer[1, -1, 0], 0.0)
        history.reset(frame, torch.tensor([0]))
        self.assertTrue(torch.all(history.buffer[0] == 0.0))
        observation = history.observation(torch.zeros((num_envs, 3)))
        self.assertEqual(observation.shape, (num_envs, UPPER_OBSERVATION_DIM))


if __name__ == "__main__":
    unittest.main()

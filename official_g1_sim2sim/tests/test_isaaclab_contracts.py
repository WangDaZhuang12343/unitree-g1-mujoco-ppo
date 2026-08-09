import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from g1_nav.l6_safety import SafetySystem
from isaaclab_nav.contracts import (
    UPPER_FRAME_DIM,
    UPPER_OBSERVATION_DIM,
    BatchSafety,
    UpperObservationHistory,
    collision_body_subset_index,
)
from isaaclab_nav.fallback import BatchedDwaFallback
from isaaclab_nav.perception import LocalDistanceFieldConfig, local_distance_field
from isaaclab_nav.pretraining import (
    ACTOR_INPUT_DIM,
    NavigationActor,
    actor_from_rsl_rl_checkpoint,
    load_teacher_datasets,
    load_actor_initialization,
    make_actor_checkpoint,
    normalized_action_to_physical_command,
    observation_moments,
    physical_command_to_normalized_action,
)
from isaaclab_nav.walking_compatibility import (
    POLICY_RELEASE_COMMIT,
    POLICY_TRAINING_EFFORT_LIMITS,
    POLICY_TRAINING_PROFILE,
    WALKING_ONNX_SHA256,
)
from navigation.scenarios import get_scenario, scenario_names


class IsaacLabContractsTest(unittest.TestCase):
    def test_walking_actuator_profile_is_versioned_to_released_policy(self):
        self.assertEqual(
            POLICY_TRAINING_EFFORT_LIMITS, {"legs": 300, "feet": 20, "arms": 300}
        )
        self.assertEqual(len(WALKING_ONNX_SHA256), 64)
        self.assertEqual(POLICY_RELEASE_COMMIT[:8], "e3c0fe49")
        self.assertEqual(POLICY_TRAINING_PROFILE, "policy_training_2025_07")

    def test_filtered_contact_flat_index_recovers_body_axis(self):
        flat_index = torch.arange(12)
        actual = collision_body_subset_index(flat_index, num_filters=2, num_bodies=3)
        torch.testing.assert_close(actual, torch.tensor([0, 0, 1, 1, 2, 2] * 2))

    def test_navigation_action_conversion_round_trip(self):
        command = torch.tensor([[0.0, -0.1, -0.2], [0.225, 0.0, 0.0], [0.45, 0.1, 0.2]])
        action = physical_command_to_normalized_action(command)
        torch.testing.assert_close(normalized_action_to_physical_command(action), command)

    def test_actor_only_checkpoint_loads_normalizer_without_critic(self):
        class Normalizer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("_mean", torch.zeros(1, ACTOR_INPUT_DIM))
                self.register_buffer("_var", torch.ones(1, ACTOR_INPUT_DIM))
                self.register_buffer("_std", torch.ones(1, ACTOR_INPUT_DIM))
                self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = NavigationActor()
                self.critic = torch.nn.Linear(ACTOR_INPUT_DIM, 1)
                self.actor_obs_normalizer = Normalizer()
                self.std = torch.nn.Parameter(torch.ones(3))

        observations = torch.randn(32, ACTOR_INPUT_DIM)
        mean, std = observation_moments(observations)
        source = NavigationActor()
        payload = make_actor_checkpoint(source, mean, std, sample_count=32, validation_mae=0.1)
        target = Policy()
        critic_before = target.critic.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(payload, path)
            metadata = load_actor_initialization(target, path)
        self.assertEqual(metadata["observation_dim"], ACTOR_INPUT_DIM)
        torch.testing.assert_close(target.actor[0].weight, source[0].weight)
        torch.testing.assert_close(target.actor_obs_normalizer._mean, mean)
        torch.testing.assert_close(target.critic.weight, critic_before)
        torch.testing.assert_close(target.std, torch.full((3,), 0.2))

    def test_rsl_rl_export_uses_actor_and_actor_normalizer_only(self):
        actor = NavigationActor()
        mean = torch.randn(1, ACTOR_INPUT_DIM)
        std = torch.rand(1, ACTOR_INPUT_DIM) + 0.1
        state = {f"actor.{key}": value for key, value in actor.state_dict().items()}
        state.update({
            "actor_obs_normalizer._mean": mean,
            "actor_obs_normalizer._std": std,
            "critic.0.weight": torch.full((1, ACTOR_INPUT_DIM), float("nan")),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_state_dict": state, "iter": 99}, path)
            exported, metadata = actor_from_rsl_rl_checkpoint(path)
        observation = torch.randn(4, ACTOR_INPUT_DIM)
        expected = actor((observation - mean) / (std + 1.0e-2))
        torch.testing.assert_close(exported(observation), expected)
        self.assertEqual(metadata["iteration"], 99)

    def test_teacher_dataset_loader_offsets_trajectory_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(2):
                path = Path(directory) / f"teacher_{index}.npz"
                np.savez(
                    path,
                    format=np.asarray("g1_isaaclab_dwa_teacher_v1"),
                    observation=np.full((2, ACTOR_INPUT_DIM), index, dtype=np.float32),
                    action=np.zeros((2, 3), dtype=np.float32),
                    trajectory_id=np.asarray([3, 4], dtype=np.int64),
                )
                paths.append(path)
            observation, action, trajectory_id = load_teacher_datasets(paths)
        self.assertEqual(observation.shape, (4, ACTOR_INPUT_DIM))
        self.assertEqual(action.shape, (4, 3))
        np.testing.assert_array_equal(trajectory_id.numpy(), [0, 1, 2, 3])

    def test_teacher_dataset_rejects_wrong_actuator_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_v2.npz"
            np.savez(
                path,
                format=np.asarray("g1_isaaclab_dwa_teacher_v2"),
                walking_actuator_profile=np.asarray("current"),
                observation=np.zeros((2, ACTOR_INPUT_DIM), dtype=np.float32),
                action=np.zeros((2, 3), dtype=np.float32),
                trajectory_id=np.asarray([0, 1], dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "expected actuator profile"):
                load_teacher_datasets(
                    [path], expected_actuator_profile="policy_training_2025_07"
                )

    def test_benchmark_contract_covers_all_frozen_scenarios(self):
        names = scenario_names()
        self.assertEqual(len(names), 11)
        layouts = [get_scenario(name, seed=7) for name in names]
        self.assertEqual([layout.name for layout in layouts], list(names))
        self.assertEqual(layouts[0].goal, (5.0, 0.0))
        self.assertTrue(layouts[-1].dynamic)

    def test_cartesian_distance_field_tracks_known_obstacle_distance(self):
        cfg = LocalDistanceFieldConfig(inflation_radius=0.28, max_clearance=5.0)
        points = torch.tensor(
            [
                [[2.0, 0.0, 0.5], [4.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.5], [4.0, 1.0, 0.0]],
            ]
        )
        obstacle = torch.tensor([[True, False], [True, False]])
        field, clearance = local_distance_field(points, obstacle, cfg)
        self.assertEqual(field.shape, (2, 10, 10))
        self.assertAlmostEqual(float(clearance[0]), 1.72, places=5)
        self.assertAlmostEqual(float(clearance[1]), 0.72, places=5)
        self.assertEqual(float(field.amin()), 0.0)
        self.assertNotEqual(int(field[0].argmin()), int(field[1].argmin()))

    def test_distance_field_is_free_without_obstacle_hits(self):
        points = torch.zeros((2, 3, 3))
        field, clearance = local_distance_field(
            points, torch.zeros((2, 3), dtype=torch.bool), LocalDistanceFieldConfig()
        )
        self.assertTrue(torch.all(field == 1.0))
        self.assertTrue(torch.all(clearance == 5.0))

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

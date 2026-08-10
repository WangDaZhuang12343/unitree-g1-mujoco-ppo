import csv
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

from isaaclab_nav.benchmark_gate import (
    BenchmarkGateConfig,
    evaluate_benchmark,
    load_summary,
    wilson_interval,
)
from isaaclab_nav.curriculum import NavigationCurriculumConfig, NavigationCurriculumState
from isaaclab_nav.experimental_teacher import RecoveryAugmentedTeacher, RecoveryTeacherConfig
from isaaclab_nav.pretraining import ACTOR_INPUT_DIM, load_teacher_datasets


class _ConstantTeacher:
    def __init__(self, num_envs: int):
        self.command = torch.tensor([[0.3, 0.0, 0.0]]).repeat(num_envs, 1)
        self.reset_ids = []
        self.closed = False

    def plan(self, points_body, ground_z_body, goal_body):
        del points_body, ground_z_body, goal_body
        return self.command.clone()

    def reset(self, env_ids):
        self.reset_ids.extend(env_ids.tolist())

    def close(self):
        self.closed = True


class IsaacLabTrainingControlsTest(unittest.TestCase):
    def test_curriculum_promotes_success_and_demotes_repeated_failure(self):
        state = NavigationCurriculumState(
            2,
            "cpu",
            NavigationCurriculumConfig(promotion_successes=2, demotion_failures=2),
        )
        env_ids = torch.tensor([0, 1])
        completed = torch.tensor([True, True])
        up, down = state.update(
            env_ids, completed, torch.tensor([True, False]), torch.tensor([False, True])
        )
        self.assertFalse(up.any())
        self.assertFalse(down.any())
        up, down = state.update(
            env_ids, completed, torch.tensor([True, False]), torch.tensor([False, True])
        )
        torch.testing.assert_close(up, torch.tensor([True, False]))
        torch.testing.assert_close(down, torch.tensor([False, True]))
        self.assertTrue(torch.all(state.success_streak == 0))
        self.assertTrue(torch.all(state.failure_streak == 0))

    def test_curriculum_ignores_initial_reset(self):
        state = NavigationCurriculumState(1, "cpu")
        up, down = state.update(
            torch.tensor([0]),
            torch.tensor([False]),
            torch.tensor([True]),
            torch.tensor([True]),
        )
        self.assertFalse(up.item())
        self.assertFalse(down.item())

    def test_recovery_teacher_is_bounded_and_resets_independently(self):
        base = _ConstantTeacher(2)
        teacher = RecoveryAugmentedTeacher(
            base,
            2,
            "cpu",
            RecoveryTeacherConfig(stagnant_steps=2),
        )
        points = torch.empty((2, 0, 3))
        ground = torch.zeros(2)
        goal = torch.tensor([[2.0, 1.0, 0.0], [2.0, -1.0, 0.0]])
        teacher.plan(points, ground, goal)
        teacher.plan(points, ground, goal)
        command = teacher.plan(points, ground, goal)
        torch.testing.assert_close(command[:, 0], torch.full((2,), 0.08))
        torch.testing.assert_close(command[:, 1], torch.tensor([0.04, -0.04]))
        torch.testing.assert_close(command[:, 2], torch.tensor([0.20, -0.20]))
        teacher.reset(torch.tensor([0]))
        self.assertFalse(teacher.recovery_used[0])
        self.assertTrue(teacher.recovery_used[1])
        teacher.close()
        self.assertTrue(base.closed)

    def test_benchmark_gate_rejects_collision_heavy_candidate(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "status": "completed",
                    "success": index < 3,
                    "collision_count": 1 if index >= 3 else 0,
                    "termination_reason": "collision" if index >= 3 else "success",
                }
            )
        decision = evaluate_benchmark(rows)
        self.assertFalse(decision.accepted)
        self.assertAlmostEqual(decision.metrics.success_rate, 0.3)
        self.assertAlmostEqual(decision.metrics.collision_rate, 0.7)
        self.assertIn("collision rate", decision.reasons[0])

    def test_benchmark_gate_accepts_candidate_above_frozen_dwa_floor(self):
        rows = [
            {
                "status": "completed",
                "success": index < 3,
                "collision_count": 0,
                "termination_reason": "success" if index < 3 else "timeout",
            }
            for index in range(10)
        ]
        decision = evaluate_benchmark(rows, BenchmarkGateConfig(maximum_fall_rate=0.0))
        self.assertTrue(decision.accepted)
        low, high = wilson_interval(3, 10)
        self.assertLess(low, 0.3)
        self.assertGreater(high, 0.3)

    def test_benchmark_csv_loader_preserves_frozen_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("status", "success", "collision_count", "termination_reason"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "completed",
                        "success": True,
                        "collision_count": 0,
                        "termination_reason": "success",
                    }
                )
            rows = load_summary(path)
        self.assertEqual(rows[0]["success"], "True")

    def test_teacher_loader_rejects_mixed_frozen_and_recovery_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for mode in ("frozen_dwa", "recovery_dwa"):
                path = Path(directory) / f"{mode}.npz"
                np.savez(
                    path,
                    format=np.asarray("g1_isaaclab_dwa_teacher_v2"),
                    walking_actuator_profile=np.asarray("policy_training_2025_07"),
                    teacher_mode=np.asarray(mode),
                    observation=np.zeros((1, ACTOR_INPUT_DIM), dtype=np.float32),
                    action=np.zeros((1, 3), dtype=np.float32),
                    trajectory_id=np.zeros(1, dtype=np.int64),
                )
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "mix teacher modes"):
                load_teacher_datasets(paths)


if __name__ == "__main__":
    unittest.main()

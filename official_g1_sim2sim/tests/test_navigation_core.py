import unittest

import numpy as np

from g1_nav.l2_costmap import CostMapConfig, LocalCostMap
from g1_nav.l3_dwa import DWANavigator
from g1_nav.l6_safety import SafetySystem
from planner import (
    LearnedNavigator, LocalNavigator, RandomFeatureNavigationPolicy,
    RidgeNavigationPolicy, compact_features,
)


class NavigationCoreTest(unittest.TestCase):
    def test_costmap_detects_and_clears_obstacle(self):
        costmap = LocalCostMap(CostMapConfig(inflation_radius=0.20))
        y = np.linspace(-0.5, 0.5, 41)
        points = np.column_stack([np.full_like(y, 1.2), y, np.full_like(y, 0.12)])
        costmap.update(points, ground_z_body=0.0)
        self.assertTrue(costmap.occupied(1.2, 0.0))
        self.assertFalse(costmap.occupied(0.2, 0.0))
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        self.assertFalse(costmap.occupied(1.2, 0.0))

    def test_dwa_goes_straight_on_clear_map(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        result = DWANavigator().plan(costmap, (3.0, 0.0))
        self.assertGreaterEqual(result.vx, 0.25)
        self.assertAlmostEqual(result.vy, 0.0, places=5)
        self.assertAlmostEqual(result.omega, 0.0, places=5)

    def test_dwa_turns_around_bar(self):
        costmap = LocalCostMap(CostMapConfig(inflation_radius=0.20))
        y = np.linspace(-0.65, 0.65, 80)
        points = np.column_stack([np.full_like(y, 2.2), y, np.full_like(y, 0.15)])
        costmap.update(points, ground_z_body=0.0)
        result = DWANavigator().plan(costmap, (3.0, 0.0))
        self.assertGreater(result.vx, 0.0)
        self.assertGreater(abs(result.omega), 0.01)

    def test_dwa_debug_candidates_are_optional(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        navigator = DWANavigator()
        navigator.plan(costmap, (3.0, 0.0))
        self.assertIsNone(navigator.last_debug)
        selected = navigator.plan(costmap, (3.0, 0.0), collect_debug=True)
        self.assertIsNotNone(navigator.last_debug)
        self.assertEqual(navigator.last_debug.trajectories.shape[0], 331)
        self.assertEqual(navigator.last_debug.valid.shape, (331,))
        self.assertTrue(np.any(navigator.last_debug.valid))
        self.assertGreater(selected.trajectory.shape[0], 1)

    def test_debug_collection_does_not_change_selection(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        regular = DWANavigator().plan(costmap, (3.0, 0.0))
        debug = DWANavigator().plan(costmap, (3.0, 0.0), collect_debug=True)
        self.assertEqual((regular.vx, regular.vy, regular.omega, regular.score),
                         (debug.vx, debug.vy, debug.omega, debug.score))
        np.testing.assert_array_equal(regular.trajectory, debug.trajectory)

    def test_safety_limits_and_stops(self):
        safety = SafetySystem()
        first = safety.update(1.0, 1.0, 1.0, 0.1, 0.8, 0.0, 0.0)
        np.testing.assert_allclose(first, [0.06, 0.03, 0.06], atol=1e-6)
        stopped = safety.update(0.3, 0.0, 0.0, 0.1, 0.3, 0.0, 0.0)
        np.testing.assert_allclose(stopped, [0.0, 0.0, 0.0])

    def test_dwa_satisfies_replaceable_navigator_contract(self):
        self.assertIsInstance(DWANavigator(), LocalNavigator)

    def test_learned_navigator_encodes_goal_map_and_limits_output(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        seen = []

        def predictor(observation):
            seen.append(observation.copy())
            return np.array([1.0, -1.0, 1.0], dtype=np.float32)

        navigator = LearnedNavigator(predictor)
        result = navigator.plan(costmap, (8.0, -4.0), collect_debug=True)
        np.testing.assert_allclose(seen[0][:2], [1.0, -1.0])
        self.assertEqual(seen[0].size, 2 + costmap.obstacle_map.size)
        np.testing.assert_allclose((result.vx, result.vy, result.omega), (0.45, -0.10, 0.20))
        self.assertIsNotNone(navigator.last_debug)
        self.assertEqual(navigator.last_debug.valid.tolist(), [True])

    def test_learned_navigator_vetoes_colliding_trajectory(self):
        costmap = LocalCostMap(CostMapConfig(inflation_radius=0.0))
        y = np.linspace(-0.5, 0.5, 41)
        points = np.column_stack([np.full_like(y, 0.5), y, np.full_like(y, 0.12)])
        costmap.update(points, ground_z_body=0.0)
        navigator = LearnedNavigator(lambda _: np.array([0.45, 0.0, 0.0]))
        result = navigator.plan(costmap, (3.0, 0.0), collect_debug=True)
        self.assertEqual((result.vx, result.vy, result.omega), (0.0, 0.0, 0.0))
        self.assertEqual(navigator.last_debug.valid.tolist(), [False])
        self.assertTrue(navigator.last_command_vetoed)

    def test_learned_navigator_shapes_forward_command_around_walk_deadzone(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        stopped = LearnedNavigator(lambda _: np.array([0.19, 0.0, 0.0])).plan(costmap, (3.0, 0.0))
        walking = LearnedNavigator(lambda _: np.array([0.21, 0.0, 0.0])).plan(costmap, (3.0, 0.0))
        self.assertEqual(stopped.vx, 0.0)
        self.assertEqual(walking.vx, 0.25)

    def test_learned_navigator_rejects_invalid_policy_output(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        with self.assertRaisesRegex(ValueError, "三个有限数值"):
            LearnedNavigator(lambda _: np.array([np.nan, 0.0])).plan(costmap, (3.0, 0.0))

    def test_compact_features_and_ridge_policy_round_trip(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        observations = []
        targets = []
        for goal_y, omega in [(-0.5, -0.1), (0.0, 0.0), (0.5, 0.1)]:
            observation = LearnedNavigator.encode_observation(costmap, (3.0, goal_y))
            observations.append(observation)
            targets.append([0.3, 0.0, omega])
        self.assertEqual(compact_features(observations[0]).shape, (452,))
        policy = RidgeNavigationPolicy.fit(np.stack(observations), np.asarray(targets))
        prediction = policy(observations[1])
        np.testing.assert_allclose(prediction, targets[1], atol=1e-3)

        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy.save(path)
            loaded = RidgeNavigationPolicy.load(path)
            np.testing.assert_allclose(loaded(observations[2]), policy(observations[2]))

    def test_teacher_case_sampling_is_reproducible(self):
        from train_navigation_policy import sample_case
        first_map, first_goal = sample_case(20260806)
        second_map, second_goal = sample_case(20260806)
        third_map, third_goal = sample_case(20260807)
        self.assertEqual(first_goal, second_goal)
        np.testing.assert_array_equal(first_map.obstacle_map, second_map.obstacle_map)
        self.assertFalse(
            first_goal == third_goal and np.array_equal(first_map.obstacle_map, third_map.obstacle_map)
        )

    def test_random_feature_policy_is_reproducible_and_round_trips(self):
        costmap = LocalCostMap()
        costmap.update(np.empty((0, 3)), ground_z_body=0.0)
        observations = np.stack([
            LearnedNavigator.encode_observation(costmap, (3.0, y))
            for y in (-0.8, -0.4, 0.0, 0.4, 0.8)
        ])
        targets = np.asarray([[0.3, 0.0, y * 0.1] for y in (-0.8, -0.4, 0.0, 0.4, 0.8)])
        first = RandomFeatureNavigationPolicy.fit(observations, targets, hidden_features=8)
        second = RandomFeatureNavigationPolicy.fit(observations, targets, hidden_features=8)
        np.testing.assert_allclose(first(observations[2]), second(observations[2]))

        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nonlinear.json"
            first.save(path)
            loaded = RandomFeatureNavigationPolicy.load(path)
            np.testing.assert_allclose(first(observations[4]), loaded(observations[4]))


if __name__ == "__main__":
    unittest.main()

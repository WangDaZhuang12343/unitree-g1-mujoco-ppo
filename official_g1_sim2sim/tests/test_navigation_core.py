import unittest

import numpy as np

from g1_nav.l2_costmap import CostMapConfig, LocalCostMap
from g1_nav.l3_dwa import DWANavigator
from g1_nav.l6_safety import SafetySystem


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


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from benchmark import write_benchmark_reports
from navigation.run_log import NAVIGATION_LOG_FIELDS, write_navigation_log
from navigation.runtime import NavigationRunResult, obstacle_clearance
from navigation.scenarios import ObstacleBox, get_scenario, scenario_names
from simulate import build_model


class NavigationBenchmarkTest(unittest.TestCase):
    def test_required_scenarios_exist(self):
        required = {
            "single_obstacle", "double_obstacle", "triple_obstacle", "random_boxes",
            "narrow_corridor", "wide_wall", "l_wall", "u_wall", "dead_end", "maze",
            "dynamic_obstacle",
        }
        self.assertEqual(set(scenario_names()), required)
        self.assertNotIn("dynamic_obstacle", scenario_names(include_dynamic=False))

    def test_random_boxes_are_reproducible(self):
        first = get_scenario("random_boxes", seed=11)
        second = get_scenario("random_boxes", seed=11)
        third = get_scenario("random_boxes", seed=12)
        self.assertEqual(first.obstacles, second.obstacles)
        self.assertNotEqual(first.obstacles, third.obstacles)

    def test_all_scenarios_compile_expected_geoms(self):
        for name in scenario_names():
            scenario = get_scenario(name)
            args = SimpleNamespace(
                terrain="flat", obstacle_height=0.15, obstacle_x=2.5, obstacle_width=0.8,
                slope_angle=5.0, roughness=0.02, terrain_seed=7,
                scene_obstacles=scenario.obstacles,
            )
            model = build_model(args)
            for obstacle in scenario.obstacles:
                geom_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, f"course_{obstacle.name}"
                )
                self.assertGreaterEqual(geom_id, 0, msg=f"{name}/{obstacle.name}")

    def test_clearance_uses_box_footprint(self):
        obstacle = ObstacleBox("box", 1.0, 0.0, 0.4, 0.6)
        self.assertAlmostEqual(obstacle_clearance(np.array([0.0, 0.0]), (obstacle,), 0.0), 0.58)
        self.assertLess(obstacle_clearance(np.array([1.0, 0.0]), (obstacle,), 0.0), 0.0)

    def test_report_files_are_generated(self):
        result = NavigationRunResult(
            scenario="single_obstacle", run_id="run_001", success=True, reached=True,
            survived=True, collision_count=0, elapsed_sim_s=12.0, elapsed_wall_s=20.0,
            path_length_m=5.4, min_clearance_m=0.31, average_velocity_mps=0.45,
            cpu_usage_percent=95.0, fps=500.0, depth_fps=10.0, planning_fps=10.0,
            average_planning_ms=12.0, final_distance_m=0.2, output=Path("run.csv"),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_benchmark_reports(output, [result])
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "success_rate.csv").is_file())
            report = (output / "navigation_report.md").read_text(encoding="utf-8")
            self.assertIn("总成功率：100.0%", report)

    def test_debug_log_schema_covers_priority_five_metrics(self):
        required = {
            "depth_fps", "planning_fps", "planning_ms", "control_latency_ms",
            "cmd_vx", "cmd_vy", "cmd_omega", "measured_vx_mps", "measured_vy_mps",
            "x_m", "y_m", "yaw_rad", "goal_distance_m", "collision_event",
            "collision_count",
        }
        self.assertTrue(required.issubset(NAVIGATION_LOG_FIELDS))
        row = list(range(len(NAVIGATION_LOG_FIELDS)))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "navigation.csv"
            write_navigation_log(output, [row])
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(","), list(NAVIGATION_LOG_FIELDS))
            self.assertEqual(len(lines[1].split(",")), len(NAVIGATION_LOG_FIELDS))


if __name__ == "__main__":
    unittest.main()

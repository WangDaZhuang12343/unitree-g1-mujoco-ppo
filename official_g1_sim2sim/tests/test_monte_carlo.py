from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.monte_carlo import (
    MonteCarloRecord,
    PROFILES,
    append_checkpoint,
    generate_samples,
    load_checkpoint,
    write_monte_carlo_report,
)
from navigation.runtime import NavigationRunResult


class MonteCarloTest(unittest.TestCase):
    def test_default_suite_is_balanced_and_reproducible(self):
        first = generate_samples()
        second = generate_samples()
        self.assertEqual(len(first), 100)
        self.assertEqual(Counter(sample.profile for sample in first), {profile: 20 for profile in PROFILES})
        self.assertEqual(first, second)
        self.assertEqual(len({sample.run_id for sample in first}), 100)
        self.assertEqual(len({sample.seed for sample in first}), 100)

    def test_parameter_ranges_and_obstacles(self):
        for sample in generate_samples(10, base_seed=42):
            self.assertGreaterEqual(sample.scenario.goal[0], 4.8)
            self.assertGreater(len(sample.scenario.obstacles), 0)
            for obstacle in sample.scenario.obstacles:
                self.assertGreater(obstacle.size_x, 0.0)
                self.assertGreater(obstacle.size_y, 0.0)
                self.assertGreater(obstacle.height, 0.0)

    def test_total_runs_must_be_balanced(self):
        with self.assertRaises(ValueError):
            generate_samples(99)

    def test_checkpoint_roundtrip_and_report(self):
        sample = generate_samples(5, base_seed=100)[0]
        result = NavigationRunResult(
            scenario=sample.scenario.name, run_id=sample.run_id, success=True, reached=True,
            survived=True, collision_count=0, elapsed_sim_s=15.0, elapsed_wall_s=30.0,
            path_length_m=5.5, min_clearance_m=0.3, average_velocity_mps=0.36,
            cpu_usage_percent=99.0, fps=250.0, depth_fps=5.0, planning_fps=5.0,
            average_planning_ms=180.0, final_distance_m=0.2, output=Path("run.csv"),
        )
        record = MonteCarloRecord.from_result(sample, result)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoint.csv"
            append_checkpoint(checkpoint, record)
            loaded = load_checkpoint(checkpoint)
            self.assertEqual(loaded, [record])
            write_monte_carlo_report(output, loaded, expected_runs=5)
            report = (output / "navigation_report.md").read_text(encoding="utf-8")
            self.assertIn("已完成：1 / 5", report)
            self.assertIn("95% Wilson", report)
            self.assertTrue((output / "success_rate.csv").is_file())


if __name__ == "__main__":
    unittest.main()

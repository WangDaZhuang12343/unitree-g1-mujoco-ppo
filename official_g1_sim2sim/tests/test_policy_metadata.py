import unittest

from simulate import MODEL_PATH, OrtRunner


class PolicyMetadataTest(unittest.TestCase):
    def test_model_names_and_dimensions(self):
        runner = OrtRunner(MODEL_PATH)
        try:
            self.assertEqual(runner.input_name, "obs")
            self.assertEqual(runner.input_size, 480)
            self.assertEqual(runner.output_name, "actions")
            self.assertEqual(runner.output_size, 29)
        finally:
            runner.close()


if __name__ == "__main__":
    unittest.main()

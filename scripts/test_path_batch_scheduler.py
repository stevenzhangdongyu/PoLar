import unittest

from polar.path_batch_scheduler import (
    dynamic_layer_schedule,
    identical_path_schedule,
    serial_schedule,
    validate_paths,
)
from scripts.simulate_path_batching import extract_paths


class PathBatchSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.paths = validate_paths([[0, 1, 2], [0, 1, 2], [0, 3, 2]])

    def test_serial_has_one_request_per_layer_batch(self):
        result = serial_schedule(self.paths)
        self.assertEqual(result.layer_executions, 9)
        self.assertTrue(all(len(batch.request_ids) == 1 for batch in result.batches))

    def test_exact_path_batches_only_identical_paths(self):
        result = identical_path_schedule(self.paths)
        self.assertEqual(result.layer_executions, 9)
        self.assertEqual(result.max_batch_size, 2)
        self.assertEqual(result.batch_count, 6)
        self.assertEqual(sorted(len(batch.request_ids) for batch in result.batches), [1, 1, 1, 2, 2, 2])

    def test_dynamic_batches_matching_ready_layer_across_paths(self):
        result = dynamic_layer_schedule(self.paths)
        self.assertEqual(result.layer_executions, 9)
        self.assertGreater(result.max_batch_size, 2)
        self.assertEqual(result.batches[0].layer, 0)
        self.assertEqual(len(result.batches[0].request_ids), 3)

    def test_depth_validation(self):
        with self.assertRaises(ValueError):
            validate_paths([[0, 4]], depth=4)

    def test_extracts_merged_mcts_valid_paths(self):
        data = {"samples": [{"final_valid_transitions": [[0, 1], [2, 3]]}]}
        self.assertEqual(
            extract_paths(data, "final_valid_transitions"), [[0, 1], [2, 3]]
        )


if __name__ == "__main__":
    unittest.main()

"""Paired KV comparison checks use synthetic evaluation records only."""
import unittest

from compare_kv_cache_eval import compare


class Tests(unittest.TestCase):
    def test_repeated_path_improvement_and_fixed_paths(self):
        base = {"completion": "same", "exact_match": 0.0}
        layer = {"q": {"task_id": "q", "base": base, "candidates": [
            {"path": [0, 1], "completion": "same", "exact_match": 0.0},
            {"path": [0, 1, 1], "completion": "wrong", "exact_match": 0.0},
        ]}}
        execution = {"q": {"task_id": "q", "base": base, "candidates": [
            {"path": [0, 1], "completion": "same", "exact_match": 0.0},
            {"path": [0, 1, 1], "completion": "right", "exact_match": 1.0},
        ]}}
        result = compare(layer, execution, "qa")
        self.assertEqual(result["repeated_candidates"], 1)
        self.assertEqual(result["repeated_candidate_improved"], 1)
        self.assertEqual(result["nonrepeated_candidate_output_changes"], 0)
        self.assertEqual(result["execution_topk_oracle_accuracy"], 1.0)
        self.assertEqual(result["layer_topk_oracle_accuracy"], 0.0)

        execution["q"]["candidates"][1]["path"] = [0, 2, 2]
        with self.assertRaisesRegex(ValueError, "paths differ"):
            compare(layer, execution, "qa")


if __name__ == "__main__":
    unittest.main()

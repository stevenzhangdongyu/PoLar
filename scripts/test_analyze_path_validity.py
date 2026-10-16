import tempfile
import unittest
from pathlib import Path

from analyze_path_validity import path_features, path_signature, main


class Tests(unittest.TestCase):
    def test_features(self):
        f = path_features([0, 1, 1, 2, 4], 5)
        self.assertEqual(f["skipped_layers"], [3])
        self.assertEqual(f["repeated_layers"], [1])
        self.assertEqual(f["repeat_exec_count"], 1)
        self.assertTrue(f["has_repeat"])
        self.assertFalse(f["is_base"])
        self.assertEqual(path_signature([0, 1, 2], 3), "base")
        self.assertIn("repeat", path_signature([0, 1, 1, 2], 3))

    def test_cli_eval(self):
        data = {
            "results": [{
                "task_id": "x",
                "base": {"exact_match": 0},
                "candidates": [
                    {"path": [0, 1, 2], "exact_match": 0, "score": 1.0},
                    {"path": [0, 1, 1, 2], "exact_match": 1, "score": .5},
                ],
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "x.json"
            prefix = Path(td) / "out"
            source.write_text(__import__("json").dumps(data), encoding="utf-8")
            import sys
            old = sys.argv
            try:
                sys.argv = ["analyze", "--json", str(source), "--depth", "3",
                            "--min_n", "1", "--out_prefix", str(prefix)]
                main()
            finally:
                sys.argv = old
            self.assertTrue(Path(str(prefix) + "_groups.csv").exists())
            self.assertTrue(Path(str(prefix) + "_paths.csv").read_text().count("\n") >= 2)


if __name__ == "__main__":
    unittest.main()

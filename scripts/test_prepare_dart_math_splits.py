import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_dart_math_splits as splitter


class Tests(unittest.TestCase):
    def test_five_difficulty_files_do_not_need_level_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = []
            for difficulty in range(1, 6):
                path = root / f"diff{difficulty}.json"
                path.write_text(json.dumps([
                    {"query": f"d{difficulty}-q{i}", "gt_ans": f"a{i}"}
                    for i in range(10)
                ]), encoding="utf-8")
                files.append(str(path))
            output = root / "splits"
            argv = ["prepare", "--input_jsons_by_diff", *files,
                    "--output_dir", str(output), "--max_unique_per_diff", "0"]
            with patch("sys.argv", argv):
                splitter.main()
            summary = json.loads((output / "split_summary.json").read_text())
            self.assertEqual(summary["difficulties"]["4"]["selected_unique"], 10)
            train = json.loads((output / "diff4" / "train.json").read_text())
            self.assertTrue(all(row["difficulty"] == 4 for row in train))

    def test_unique_cap_and_disjoint_801010_splits(self):
        rows = []
        for difficulty in range(1, 6):
            for index in range(20):
                row = {"query": f"q{difficulty}-{index}", "gt_ans": f"a{index}",
                       "query_metadata": {"level": difficulty}}
                rows.extend([row, dict(row)])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input.json", root / "splits"
            source.write_text(json.dumps(rows), encoding="utf-8")
            argv = ["prepare", "--input_json", str(source), "--output_dir", str(output),
                    "--max_unique_diff1", "13", "--max_unique_diff2", "20",
                    "--max_unique_diff3", "20", "--max_unique_diff4", "20",
                    "--max_unique_diff5", "20"]
            with patch("sys.argv", argv):
                splitter.main()
            parts = [json.loads((output / "diff1" / f"{name}.json").read_text())
                     for name in ("train", "test", "val")]
            self.assertEqual([len(part) for part in parts], [11, 1, 1])
            questions = [{row["question"] for row in part} for part in parts]
            self.assertFalse(questions[0] & questions[1])
            self.assertFalse(questions[0] & questions[2])
            summary = json.loads((output / "split_summary.json").read_text())
            self.assertEqual(summary["difficulties"]["1"]["available_unique"], 20)

    def test_split_counts_preserve_total_and_nonempty(self):
        for n in range(1, 31):
            counts = splitter.split_counts(n, [0.8, 0.1, 0.1])
            self.assertEqual(sum(counts), n)
            self.assertTrue(all(count >= 0 for count in counts))
            if n >= 3:
                self.assertTrue(all(count >= 1 for count in counts))


if __name__ == "__main__":
    unittest.main()

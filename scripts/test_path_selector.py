import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from polar.selector import (
        PathSelectorDataset, build_candidates, multi_positive_listwise_loss, path_features,
        selector_metrics,
    )


@unittest.skipIf(torch is None, "torch is not installed in this local test environment")
class Tests(unittest.TestCase):
    def test_features_and_dataset(self):
        row = {
            "task_id": "q1",
            "predictor_input": "Question",
            "base": {"path": [0, 1, 2], "exact_match": 0},
            "candidates": [
                {"path": [0, 1, 2], "score": 1.0, "exact_match": 0},
                {"path": [0, 1, 1, 2], "score": .5, "exact_match": 1},
            ],
        }
        paths, scores, labels, ranks = build_candidates(row, 3, 5)
        self.assertEqual(len(paths), 3)
        self.assertEqual(labels, [0.0, 0.0, 1.0])
        feat = path_features([0, 1, 1, 2], 3, rank=2, score=.5, is_base=False)
        self.assertEqual(len(feat), 3 * 3 + 11)
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "eval.json"
            source.write_text(json.dumps({"results": [row]}), encoding="utf-8")
            ds = PathSelectorDataset([str(source)], depth=3)
            self.assertEqual(len(ds), 1)
            ex = ds[0]
            self.assertEqual(ex.features.shape, (6, ds.feature_dim))
            self.assertTrue(ex.mask[:3].all())
            self.assertFalse(ex.mask[3:].any())

    def test_loss_and_metrics(self):
        logits = torch.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]])
        labels = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        loss = multi_positive_listwise_loss(logits, labels, mask)
        self.assertGreaterEqual(float(loss), 0.0)
        m = selector_metrics(logits, labels, mask)
        self.assertEqual(m["selector_acc"], 1.0)
        self.assertEqual(m["base_acc"], 0.5)
        empty_labels = torch.zeros_like(labels)
        self.assertEqual(float(multi_positive_listwise_loss(logits, empty_labels, mask)), 0.0)


if __name__ == "__main__":
    unittest.main()

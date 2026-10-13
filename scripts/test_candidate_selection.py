import unittest
from candidate_selection import select_candidate, paired_metrics, sampling_summary


class Tests(unittest.TestCase):
    def test_reference_free(self):
        base = {"path": [0, 1], "normalized_answer": "B", "exact_match": 1}
        candidates = [
            dict(base, exact_match=0),
            {"path": [1], "normalized_answer": "A", "exact_match": 0},
            {"path": [0], "normalized_answer": "A", "exact_match": 1},
        ]
        self.assertIs(select_candidate(base, candidates, "nonbase_top1")[0], candidates[1])
        self.assertIs(select_candidate(base, candidates, "majority")[0], candidates[1])
        self.assertEqual(select_candidate(base, [candidates[0]], "nonbase_top1"),
                         (base, "base_fallback"))

    def test_denominators(self):
        metrics = paired_metrics([False, False, True, True], [True, False, False, True])
        self.assertEqual(metrics["repair_rate_all"], .25)
        self.assertEqual(metrics["repair_rate_given_base_wrong"], .5)
        self.assertEqual(metrics["degradation_rate_given_base_correct"], .5)
        self.assertEqual(metrics["final_accuracy"], .5)
        self.assertIsNone(paired_metrics([True], [True])["repair_rate_given_base_wrong"])

    def test_best_of_k(self):
        rows = [{"base_samples": [{"passed": False}, {"passed": True}],
                 "candidates": [{"passed": False}, {"passed": False}]}]
        result = sampling_summary(rows, lambda c: c["passed"])
        self.assertEqual(result["base_sampling_best_of_actual_k"], 1)
        self.assertEqual(result["polar_greedy_best_of_actual_k"], 0)
        self.assertEqual(result["actual_k_counts"], {"2": 1})


if __name__ == "__main__":
    unittest.main()

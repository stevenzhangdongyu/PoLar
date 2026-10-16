"""CPU-only interactive chat tests; no model downloads, torch or code execution."""
import unittest

from chat_polar import build_parser
from polar_chat_utils import (
    chat_prompt, predictor_text, unique_paths, grade_manual, grade_reference, chat_summary,
)


class Tests(unittest.TestCase):
    def test_parser(self):
        args = build_parser().parse_args(
            ["--model_path", "m", "--checkpoint_path", "c", "--session_json", "s"])
        self.assertEqual(args.top_k_paths, 5)
        self.assertEqual(args.selection_strategy, "nonbase_top1")

    def test_predictor_input(self):
        messages = [{"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Reply"},
                    {"role": "user", "content": "Next"}]
        self.assertEqual(predictor_text(messages, "question"), "Next")
        self.assertIn("assistant: Reply", predictor_text(messages, "history"))

    def test_template(self):
        class Tokenizer:
            chat_template = "present"

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return messages

        tok = Tokenizer()
        messages = [{"role": "user", "content": "Question"}]
        self.assertEqual(chat_prompt(tok, messages, "Qwen3-8B"), messages)
        self.assertFalse(tok.kwargs["enable_thinking"])

    def test_unique(self):
        beams = [([0, 1], -.1), ([0, 1], -.2), ([], -.3), ([0, 0, 1], -.4)]
        # Empty action metadata to isolate deduplication.
        paths = [[0, 1], [0, 1], [], [0, 0, 1]]
        it = iter(paths)
        rows = unique_paths([([], s) for _, s in beams], 2, lambda a, d: next(it), 5)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["path"], [0, 0, 1])
        with self.assertRaises(ValueError):
            unique_paths([([], 0)], 2, lambda a, d: [2], 5)

    def test_grading(self):
        turn = {"base": {"completion": "London"},
                "candidates": [{"completion": "Paris"}, {"completion": "London"}]}
        ref = grade_reference(turn, ["Paris"])
        self.assertEqual(ref["actual_k"], 2)
        self.assertTrue(ref["best_of_k_correct"])
        self.assertFalse(ref["base"]["exact_match"])
        self.assertNotIn("references", turn)
        manual = grade_manual(turn, [False, True])
        self.assertTrue(manual["best_of_k_correct"])
        for invalid in ([1, 0], [True], "true", None):
            with self.assertRaises(ValueError):
                grade_manual(turn, invalid)
        with self.assertRaises(ValueError):
            grade_reference(turn, [])
        turn["reference_grade"] = ref
        turn["manual_grade"] = manual
        summary = chat_summary([turn, {"ungraded": True}])
        self.assertEqual(summary["turns"], 2)
        self.assertEqual(summary["manual_grade"]["graded_turns"], 1)
        self.assertEqual(summary["manual_grade"]["actual_k_counts"], {"2": 1})
        self.assertIsNone(chat_summary([])["manual_grade"]["best_of_actual_k_rate"])


if __name__ == "__main__":
    unittest.main()

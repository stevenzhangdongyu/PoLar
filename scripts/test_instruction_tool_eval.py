import unittest

from eval_polar_toolcall import extract_calls, score_calls, prompt_for_task, expected_calls
from eval_polar_instruction import load_tasks as _unused  # import smoke


class Tests(unittest.TestCase):
    def test_extract_calls(self):
        text = '```json\n[{"name":"search","arguments":{"q":"polar"}}]\n```'
        calls = extract_calls(text)
        self.assertEqual(calls[0]["name"], "search")
        self.assertEqual(calls[0]["arguments"]["q"], "polar")
        score = score_calls(calls, [{"name": "search", "arguments": {"q": "polar"}}])
        self.assertEqual(score["call_exact"], 1.0)
        self.assertEqual(score_calls([], [])["has_gold"], False)

    def test_bfcl_like(self):
        row = {
            "question": "Find weather in Paris",
            "function": [{"name": "get_weather", "parameters": {"type": "object"}}],
            "possible_answer": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        }
        self.assertIn("Available functions", prompt_for_task(row))
        self.assertEqual(expected_calls(row)[0]["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()

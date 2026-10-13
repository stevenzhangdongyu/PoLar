import unittest

from dedup_qa_jsonl import deduplicate


class Tests(unittest.TestCase):
    def test_duplicates(self):
        a = {"id": "q1", "question": "Who?", "answers": ["Ada", "Ada Lovelace"]}
        b = dict(a, answers=["Ada Lovelace", "Ada"])
        kept, duplicates, conflicts = deduplicate([a, b])
        self.assertEqual(kept, [a])
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(conflicts, [])

    def test_conflicts(self):
        a = {"id": "q1", "question": "Who?", "answers": ["Ada"]}
        for b in (dict(a, question="Where?"), dict(a, answers=["Bob"]),
                  dict(a, context="different context")):
            kept, duplicates, conflicts = deduplicate([a, b])
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(duplicates, [])

    def test_distinct_ids(self):
        a = {"id": "q1", "question": "Who?", "answers": ["Ada"]}
        self.assertEqual(len(deduplicate([a, dict(a, id="q2")])[0]), 2)


if __name__ == "__main__":
    unittest.main()

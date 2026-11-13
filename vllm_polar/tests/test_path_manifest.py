import unittest

from polar_vllm.path_manifest import build_manifest, resolve_path, token_ids_key


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(char) for char in text]


class PathManifestTests(unittest.TestCase):
    def test_manifest_resolves_exact_prompt_tokens(self):
        rows = [
            {"request_id": "a", "path": [0, 1, 0]},
            {"request_id": "b", "path": [1, 2]},
        ]
        manifest, prompts = build_manifest(rows, FakeTokenizer())
        tokens = FakeTokenizer().encode(prompts[0]["prompt"])
        self.assertEqual(resolve_path(manifest, tokens), [0, 1, 0])
        self.assertEqual(manifest["request_count"], 2)

    def test_token_key_is_order_sensitive(self):
        self.assertNotEqual(token_ids_key([4, 5]), token_ids_key([5, 4]))

    def test_duplicate_tokenized_prompts_are_rejected(self):
        rows = [
            {"request_id": "same", "path": [0]},
            {"request_id": "same", "path": [1]},
        ]
        with self.assertRaisesRegex(ValueError, "collision"):
            build_manifest(rows, FakeTokenizer())


if __name__ == "__main__":
    unittest.main()

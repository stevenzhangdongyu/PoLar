import json
import os
import tempfile
import unittest

from polar_vllm.path_config import (
    execution_cache_indices,
    load_runtime_config,
    validate_path,
)


class PathConfigTests(unittest.TestCase):
    def test_validate_path_and_repeats(self):
        self.assertEqual(validate_path([0, 3, 0], 4), [0, 3, 0])
        with self.assertRaises(ValueError):
            validate_path([4], 4)
        with self.assertRaises(ValueError):
            validate_path([True], 4)

    def test_cache_indices(self):
        self.assertEqual(execution_cache_indices([0, 3, 0], "execution"), [0, 1, 2])
        self.assertEqual(execution_cache_indices([0, 3, 0], "layer"), [0, 3, 0])

    def test_load_env_config(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "path.json")
            with open(filename, "w", encoding="utf-8") as stream:
                json.dump([0, 2, 0], stream)
            self.assertEqual(
                load_runtime_config({"POLAR_PATH_JSON": filename}),
                ([0, 2, 0], "execution"),
            )


if __name__ == "__main__":
    unittest.main()

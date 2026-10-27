import unittest

from llm_depth_router.patches.cache_mode import (
    EXECUTION_CACHE_MODE,
    LAYER_CACHE_MODE,
    cache_layer_index,
    normalize_cache_mode,
    temporary_attention_layer_idx,
)


class _DummyAttention:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx


class _DummyLayer:
    def __init__(self, layer_idx):
        self.self_attn = _DummyAttention(layer_idx)


class PolarKvCacheModeTest(unittest.TestCase):
    def test_cache_layer_index_modes(self):
        self.assertEqual(cache_layer_index(LAYER_CACHE_MODE, physical_layer_idx=7, execution_idx=3), 7)
        self.assertEqual(cache_layer_index(EXECUTION_CACHE_MODE, physical_layer_idx=7, execution_idx=3), 3)
        self.assertEqual(cache_layer_index("exec", physical_layer_idx=7, execution_idx=3), 3)
        self.assertEqual(normalize_cache_mode("physical_layer"), LAYER_CACHE_MODE)

    def test_temporary_attention_layer_idx_restores_original(self):
        layer = _DummyLayer(layer_idx=7)
        with temporary_attention_layer_idx(layer, 3):
            self.assertEqual(layer.self_attn.layer_idx, 3)
        self.assertEqual(layer.self_attn.layer_idx, 7)

    def test_repeated_physical_layer_gets_distinct_execution_slots(self):
        path = [0, 1, 3, 3, 4]
        execution_slots = [
            cache_layer_index(EXECUTION_CACHE_MODE, physical_layer_idx=p, execution_idx=i)
            for i, p in enumerate(path)
        ]
        legacy_slots = [
            cache_layer_index(LAYER_CACHE_MODE, physical_layer_idx=p, execution_idx=i)
            for i, p in enumerate(path)
        ]
        self.assertEqual(execution_slots, [0, 1, 2, 3, 4])
        self.assertEqual(legacy_slots, [0, 1, 3, 3, 4])


if __name__ == "__main__":
    unittest.main()

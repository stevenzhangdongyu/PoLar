from contextlib import contextmanager
from typing import Iterator, Optional


LAYER_CACHE_MODE = "layer"
EXECUTION_CACHE_MODE = "execution"
SUPPORTED_CACHE_MODES = {LAYER_CACHE_MODE, EXECUTION_CACHE_MODE}


def normalize_cache_mode(mode: Optional[str]) -> str:
    mode = (mode or LAYER_CACHE_MODE).strip().lower()
    aliases = {
        "physical": LAYER_CACHE_MODE,
        "physical_layer": LAYER_CACHE_MODE,
        "layer_idx": LAYER_CACHE_MODE,
        "exec": EXECUTION_CACHE_MODE,
        "execution_step": EXECUTION_CACHE_MODE,
        "step": EXECUTION_CACHE_MODE,
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_CACHE_MODES:
        raise ValueError(
            f"Unsupported POLAR KV cache mode `{mode}`. "
            f"Expected one of: {sorted(SUPPORTED_CACHE_MODES)}."
        )
    return mode


def get_custom_path_cache_mode(model) -> str:
    return normalize_cache_mode(getattr(model, "custom_path_cache_mode", LAYER_CACHE_MODE))


def cache_layer_index(cache_mode: str, physical_layer_idx: int, execution_idx: int) -> int:
    cache_mode = normalize_cache_mode(cache_mode)
    if cache_mode == EXECUTION_CACHE_MODE:
        return int(execution_idx)
    return int(physical_layer_idx)


@contextmanager
def temporary_attention_layer_idx(decoder_layer, layer_idx: int) -> Iterator[None]:
    """Temporarily route a decoder layer's attention cache writes to `layer_idx`.

    Hugging Face decoder attention modules use `self.layer_idx` as the index into
    the `Cache` object. POLAR's old behavior reused the physical transformer
    layer index, so repeated execution of the same layer wrote into the same
    cache slot. In execution-cache mode we instead assign each path execution
    step a distinct cache slot while still reusing the same physical weights.
    """

    attention = getattr(decoder_layer, "self_attn", None)
    if attention is None or not hasattr(attention, "layer_idx"):
        yield
        return

    old_idx = attention.layer_idx
    attention.layer_idx = int(layer_idx)
    try:
        yield
    finally:
        attention.layer_idx = old_idx


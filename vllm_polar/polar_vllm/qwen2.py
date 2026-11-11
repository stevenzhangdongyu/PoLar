"""Qwen2 fixed-path executor. Pin and validate against vLLM 0.26.x."""

from __future__ import annotations

from torch import nn

from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .path_config import load_runtime_config, validate_path


class _PathVisit(nn.Module):
    def __init__(self, physical_layer: nn.Module, attention: nn.Module):
        super().__init__()
        self.input_layernorm = physical_layer.input_layernorm
        self.post_attention_layernorm = physical_layer.post_attention_layernorm
        self.mlp = physical_layer.mlp
        self.self_attn = attention

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class PolarQwen2Model(Qwen2Model):
    def __init__(self, *, vllm_config, prefix=""):
        if get_pp_group().world_size != 1:
            raise ValueError("Polar vLLM adapter currently requires pipeline-parallel size 1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        raw_path, self.cache_mode = load_runtime_config()
        self.path = validate_path(raw_path, self.config.num_hidden_layers)

        visits = []
        seen: dict[int, nn.Module] = {}
        for visit_idx, layer_idx in enumerate(self.path):
            physical = self.layers[layer_idx]
            if self.cache_mode == "layer" or layer_idx not in seen:
                attention = physical.self_attn
                seen[layer_idx] = attention
            else:
                source = physical.self_attn
                attention = Qwen2Attention(
                    hidden_size=source.hidden_size,
                    num_heads=source.total_num_heads,
                    num_kv_heads=source.total_num_kv_heads,
                    max_position=self.config.max_position_embeddings,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    rope_parameters=self.config.rope_parameters,
                    prefix=f"{prefix}.polar_execution_layers.{visit_idx}.self_attn",
                    dual_chunk_attention_config=getattr(
                        self.config, "dual_chunk_attention_config", None
                    ),
                    qk_norm=source.qk_norm,
                    rms_norm_eps=self.config.rms_norm_eps,
                )
                # Reuse the physical layer's learned projections and rotary table;
                # only the Attention object (and therefore its vLLM KV slot) differs.
                attention.qkv_proj = source.qkv_proj
                attention.o_proj = source.o_proj
                attention.rotary_emb = source.rotary_emb
                if source.qk_norm:
                    attention.q_norm = source.q_norm
                    attention.k_norm = source.k_norm
            visits.append(_PathVisit(physical, attention))
        self.polar_execution_layers = nn.ModuleList(visits)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("Pipeline parallelism is unsupported by the fixed-path adapter")
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        for layer in self.polar_execution_layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class PolarQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = PolarQwen2Model(
            vllm_config=vllm_config, prefix=f"{prefix}.model".strip(".")
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

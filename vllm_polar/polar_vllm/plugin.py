"""vLLM general plugin registration hook."""


def register() -> None:
    # Keep imports lazy: vLLM loads general plugins in API and worker processes.
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Qwen2ForCausalLM", "polar_vllm.qwen2:PolarQwen2ForCausalLM"
    )

# PoLar on vLLM (fixed path prototype)

This subproject registers a vLLM general plugin that executes one fixed PoLar
layer path per server process. It targets Qwen2/Qwen2.5 decoder-only models,
including Qwen2.5-3B-Instruct. It does not yet implement predictor-driven,
per-request paths or mixed-path continuous batching.

## Install

Use the CUDA/PyTorch-compatible vLLM 0.26.0 wheel in the serving environment,
then install this plugin. Do not install a generic vLLM wheel over a vendor
build such as vLLM-Ascend without following that backend's installation guide.

```bash
python -m pip install -r vllm_polar/requirements-vllm.txt
python -m pip install -e ./vllm_polar
```

The model's `config.json` must include `"architectures": ["Qwen2ForCausalLM"]`.
The plugin replaces that architecture registration within this vLLM process.

## Select a path and serve

`path.json` is a JSON array of physical layer indices, zero-based. Repeats are
allowed. Example path `[0, 1, 2, 2, 3, 4]` repeats physical layer 2 once.

```bash
cat > path.json <<'JSON'
[0, 1, 2, 2, 3, 4]
JSON

export POLAR_PATH_JSON="$PWD/path.json"
export POLAR_KV_CACHE_MODE=execution
export VLLM_PLUGINS=polar
vllm serve /models/Qwen2.5-3B-Instruct \
  --served-model-name polar-qwen \
  --tensor-parallel-size 1
```

The OpenAI-compatible endpoint uses the normal vLLM API, for example
`POST /v1/chat/completions`. `POLAR_PATH_JSON` and cache mode are read at model
initialization, so use one server instance per fixed path/configuration.

## KV behavior and current constraints

- `execution`: each repeated visit has its own `Attention` object/KV slot. The
  initial visit of each physical layer uses its normal slot; later visits get
  additional slots. This gives each visit independent prefill/decode KV state.
- `layer`: repeated visits share the physical layer's KV slot, matching the
  original overwrite/reuse behavior.
- Pipeline parallelism is rejected; tensor parallelism follows vLLM's Qwen2
  implementation. Validate the chosen attention backend and quantization on
  the target machine before benchmarking.
- In `execution` mode, normal physical-layer attention modules that are not
  visited remain registered, so vLLM may reserve unused KV capacity for them.
  This prototype prioritizes correctness of visit isolation over cache-memory
  efficiency; measure `--gpu-memory-utilization` and `--max-model-len` on the
  target system.
- One model process has one path. Requests in that process cannot select
  different paths. Efficient mixed-path serving requires scheduler/model-runner
  changes and is intentionally not claimed here.

## Local checks

```bash
cd vllm_polar
PYTHONPATH="$PWD" python -m unittest discover -s tests -v
python -m compileall -q polar_vllm tests
```

The repository development machine may not have CUDA, PyTorch, or vLLM; the
checks above only validate configuration helpers and syntax. A real generation
smoke test must be run with the pinned vLLM build and model weights on the
serving GPU.

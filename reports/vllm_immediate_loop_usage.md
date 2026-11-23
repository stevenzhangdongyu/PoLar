# Dynamic layer: immediate same-layer loops

Sync both PoLar and the modified vLLM repository before running. No CUDA
recompilation is needed. The benchmark starts and stops its own server.

`--modes dynamic_layer` selects only dynamic layer execution. The new
`--loop-policy immediate` sets `VLLM_POLAR_LOOP_POLICY=immediate` in the worker.
The default `greedy` preserves the original largest-ready-group policy.
Both the policy and selected modes are recorded in config.json and checked
when resuming.

Immediate mode drains consecutive visits to the same physical layer before
advancing other requests. It supports different repeat counts, but does not
prioritize arbitrary backward jumps or repeated multi-layer segments.
KV semantics and per-request path order are unchanged. Execution-index KV
can still split planner batches; layer KV can merge different path positions.
This is a candidate heuristic, not a proven optimal scheduler.

```bash
cd /home/z00894149/polar_dev/PoLar
PYTHON=/root/miniconda3/envs/zdy_vllm_polar/bin/python \
POLAR_DEBUG=0 POLAR_DEBUG_BATCH=0 \
KV_CACHE_MODE=layer GPU_ID=0 CONCURRENCIES="8" \
INPUT_LEN=512 OUTPUT_LEN=256 NUM_REQUESTS=512 NUM_WARMUPS=20 \
OUTPUT_DIR="/home/z00894149/polar_dev/dynamic_immediate_c8_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_vllm_aligned_benchmark.sh \
  --modes dynamic_layer --loop-policy immediate
```

No profiler is enabled by this benchmark. For a fresh control run use a new
output directory and `--loop-policy greedy`; keep all other settings identical.
Compare output tokens/s, TTFT, TPOT, E2E mean/P99, and request-ID-matched generated
text in results.jsonl. Deterministic output mismatches need investigation,
although floating-point batch-shape changes may also alter generated text.

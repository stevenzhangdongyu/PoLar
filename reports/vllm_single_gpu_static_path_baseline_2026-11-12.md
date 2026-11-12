# vLLM Single-GPU Static-Path Baseline

## Run 001: concurrency 1

Recorded: 2026-11-12 (Asia/Shanghai)  
Status: benchmark completed successfully; 512/512 requests succeeded.

### Configuration

| Field | Value |
|---|---|
| Serving stack | vLLM OpenAI-compatible server; exact vLLM version not included in captured output |
| Model | Qwen2.5-3B-Instruct, served name `polar-qwen` |
| Execution path | Static path; user reports this is the Base path. Exact `path.json` contents were not included in the capture |
| GPU count | 1 (`--tensor-parallel-size 1`); physical GPU ID/model must be confirmed from the server launch command and `nvidia-smi` |
| KV cache mode | `execution` per the preceding static-path server setup; not shown in the benchmark command itself |
| Endpoint | `http://127.0.0.1:8000/v1/completions` |
| Dataset | vLLM `random` synthetic workload |
| Input length | Exactly 512 tokens (`random_range_ratio=0.0`) |
| Requested output length | 256 tokens |
| Number of prompts | 512 |
| Warmup requests | 20 |
| Request rate | `inf` (benchmark submits as fast as possible) |
| Maximum concurrency | 1 |
| Sampling | `temperature` not specified by benchmark; server default was used |
| Benchmark result file | `./vllm_bench/polar_c1.json` |

### Results

| Metric | Result |
|---|---:|
| Successful requests | 512 |
| Failed requests | 0 |
| Benchmark duration | 834.12 s |
| Total input tokens | 262,144 |
| Total generated tokens | 131,072 |
| Request throughput | 0.61 req/s |
| Output token throughput | 157.14 tok/s |
| Peak output token throughput | 162.00 tok/s |
| Total token throughput (input + output) | 471.41 tok/s |
| Mean TTFT | 52.33 ms |
| Median TTFT | 51.99 ms |
| P99 TTFT | 59.29 ms |
| Mean TPOT | 6.18 ms/token |
| Median TPOT | 6.17 ms/token |
| P99 TPOT | 6.22 ms/token |
| Mean ITL | 6.18 ms |
| P99 ITL | 6.39 ms |
| Peak concurrent requests reported | 2 |

Derived check: `512 requests * 256 output tokens / 834.12 s = 157.14 output tok/s`. Average wall time per request is approximately `834.12 / 512 = 1.63 s`, consistent with roughly 256 decode steps at 6.18 ms/token plus TTFT.

## Run 002: concurrency 2

Recorded: 2026-11-12 (Asia/Shanghai)  
Status: benchmark completed successfully; 512/512 requests succeeded. The user confirms this is still the Base layer path.

### Configuration

Same model, synthetic workload, endpoint, tokenizer, input/output lengths, warmups, request rate, and single-GPU server setup as Run 001; only `--max-concurrency` changed from 1 to 2. Result file: `./vllm_bench/polar_c2.json`.

| Field | Value |
|---|---|
| Dataset | vLLM `random` synthetic workload |
| Input length | Exactly 512 tokens (`random_range_ratio=0.0`) |
| Requested output length | 256 tokens |
| Number of prompts | 512 |
| Warmup requests | 20 |
| Request rate | `inf` |
| Maximum concurrency | 2 |
| Sampling | `temperature` not specified; server default used |

### Results

| Metric | Result |
|---|---:|
| Successful requests | 512 |
| Failed requests | 0 |
| Benchmark duration | 398.17 s |
| Total input tokens | 262,144 |
| Total generated tokens | 131,072 |
| Request throughput | 1.29 req/s |
| Input token throughput (derived) | 658.37 tok/s |
| Output token throughput | 329.18 tok/s |
| Peak output token throughput | 338.00 tok/s |
| Total token throughput (input + output) | 987.55 tok/s |
| Mean TTFT | 44.53 ms |
| Median TTFT | 44.52 ms |
| P99 TTFT | 52.52 ms |
| Mean TPOT | 5.92 ms/token |
| Median TPOT | 5.92 ms/token |
| P99 TPOT | 5.96 ms/token |
| Mean ITL | 5.92 ms |
| P99 ITL | 6.16 ms |
| Peak concurrent requests reported | 4 (reported value exceeds configured maximum concurrency 2; verify metric semantics/version) |

Derived check: `131,072 / 398.17 = 329.18 output tok/s`; input throughput is `262,144 / 398.17 = 658.37 input tok/s`.

### Change vs. Run 001

| Metric | C=1 | C=2 | Change |
|---|---:|---:|---:|
| Benchmark duration | 834.12 s | 398.17 s | -52.3% |
| Request throughput | 0.61 req/s | 1.29 req/s | about 2.1x |
| Output token throughput | 157.14 tok/s | 329.18 tok/s | 2.09x |
| Input token throughput | 314.30 tok/s (derived) | 658.37 tok/s (derived) | 2.09x |
| Mean TTFT | 52.33 ms | 44.53 ms | -14.9% |
| Mean TPOT | 6.18 ms/token | 5.92 ms/token | -4.2% |
| Failures | 0 | 0 | unchanged |

The run suggests that moving from one to two concurrent requests substantially improves GPU utilization/throughput on this workload without a latency penalty in the reported averages. This is not yet the saturation point; continue the sweep, checking memory, failures, and tail latency. The reported peak concurrency of 4 despite a configured limit of 2 should be investigated before treating that field as an actual scheduler batch size.

## Run 003: concurrency 4

Recorded: 2026-11-12 (Asia/Shanghai)  
Status: benchmark completed successfully; 512/512 requests succeeded. The user confirms the Base layer order/path.

### Configuration

Same model, endpoint, tokenizer, one-GPU server, synthetic workload, input/output lengths, prompt count, warmups, and request rate as Runs 001-002; only `--max-concurrency` changed to 4. Result file: `./vllm_bench/polar_c4.json`.

| Field | Value |
|---|---|
| Dataset | vLLM `random` synthetic workload |
| Input length | Exactly 512 tokens (`random_range_ratio=0.0`) |
| Requested output length | 256 tokens |
| Number of prompts | 512 |
| Warmup requests | 20 |
| Request rate | `inf` |
| Maximum concurrency | 4 |
| Sampling | `temperature` not specified; server default used |

### Results

| Metric | Result |
|---|---:|
| Successful requests | 512 |
| Failed requests | 0 |
| Benchmark duration | 206.16 s |
| Total input tokens | 262,144 |
| Total generated tokens | 131,072 |
| Request throughput | 2.48 req/s |
| Input token throughput (derived) | 1,271.61 tok/s |
| Output token throughput | 635.77 tok/s |
| Peak output token throughput | 656.00 tok/s |
| Total token throughput (input + output) | 1,907.32 tok/s |
| Mean TTFT | 52.16 ms |
| Median TTFT | 52.05 ms |
| P99 TTFT | 62.85 ms |
| Mean TPOT | 6.11 ms/token |
| Median TPOT | 6.11 ms/token |
| P99 TPOT | 6.16 ms/token |
| Mean ITL | 6.11 ms |
| P99 ITL | 6.37 ms |
| Peak concurrent requests reported | 8 (reported value exceeds configured maximum concurrency 4; verify metric semantics/version) |

Derived check: `131,072 / 206.16 = 635.77 output tok/s`; input throughput is `262,144 / 206.16 = 1,271.61 input tok/s`.

### Change vs. Run 002

| Metric | C=2 | C=4 | Change |
|---|---:|---:|---:|
| Benchmark duration | 398.17 s | 206.16 s | -48.2% |
| Request throughput | 1.29 req/s | 2.48 req/s | about 1.92x |
| Output token throughput | 329.18 tok/s | 635.77 tok/s | 1.93x |
| Input token throughput | 658.37 tok/s | 1,271.61 tok/s | 1.93x |
| Mean TTFT | 44.53 ms | 52.16 ms | +17.1% |
| P99 TTFT | 52.52 ms | 62.85 ms | +19.7% |
| Mean TPOT | 5.92 ms/token | 6.11 ms/token | +3.2% |
| Failures | 0 | 0 | unchanged |

Throughput nearly doubles again from concurrency 2 to 4, with no failed requests. Unlike the C=2 run, TTFT and TPOT rise modestly; this is an early throughput/latency tradeoff, not yet evidence of saturation. Continue the sweep and compare tail latency and GPU memory at each concurrency. The reported peak concurrent count is again twice the configured maximum, so do not interpret it as batch size without confirming the benchmark's metric definition.

## Runs 004-008: concurrency 8-96

Recorded: 2026-11-12 (Asia/Shanghai)  
Status: all five benchmark runs completed successfully; each processed 512 requests with 0 failures. The user identifies these as Base-path tests.

All five use the same workload and server configuration as Runs 001-003; only `--max-concurrency` varies. Result files are `polar_c8.json`, `polar_c16.json`, `polar_c32.json`, `polar_c64.json`, and `polar_c96.json` under `./vllm_bench/`.

| Concurrency | Duration (s) | Req/s | Input tok/s* | Output tok/s | Peak output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | P99 ITL (ms) | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 105.00 | 4.88 | 2,496.61 | 1,248.30 | 1,308 | 3,744.91 | 54.71 | 87.44 | 6.22 | 6.28 | 8.17 | 0 |
| 16 | 55.72 | 9.19 | 4,704.67 | 2,352.42 | 2,496 | 7,057.25 | 70.30 | 128.70 | 6.54 | 6.65 | 14.58 | 0 |
| 32 | 31.23 | 16.40 | 8,393.98 | 4,197.47 | 4,576 | 12,592.42 | 120.75 | 202.07 | 7.15 | 7.28 | 14.35 | 0 |
| 64 | 18.51 | 27.66 | 14,162.29 | 7,081.04 | 7,861 | 21,243.13 | 143.65 | 365.55 | 8.44 | 8.61 | 13.81 | 0 |
| 96 | 14.96 | 34.21 | 17,522.99 | 8,758.68 | 10,981 | 26,276.05 | 182.61 | 660.49 | 9.42 | 9.93 | 17.60 | 0 |

\* Input token throughput is derived as `262,144 / benchmark duration`; the benchmark summary does not print it as a separate field.

### Interpretation

- Output throughput continues rising through the highest tested concurrency: `1,248 tok/s` at C=8 to `8,759 tok/s` at C=96. C=96 is the best observed point, not a proven saturation limit.
- Scaling is sublinear: C=64 to C=96 increases output throughput by about 23.7% while concurrency rises 50%, so gains are diminishing.
- Tail latency grows sharply at high concurrency. P99 TTFT rises from 87.44 ms (C=8) to 660.49 ms (C=96); P99 TPOT rises from 6.28 to 9.93 ms/token. The best production point depends on the latency SLO, not only peak throughput.
- No request failures are reported. Peak concurrent requests is exactly twice the configured maximum in every run (16, 32, 64, 128, 192); treat this metric cautiously until its version-specific definition is verified.
- These are synthetic fixed-length requests with `request-rate=inf`, a saturation-style workload rather than a production arrival pattern. This is one static Base path, not a mixed-path serving result.

### Interpretation and caveats

- This is a **single-request-at-a-time static-path measurement**, not the single-GPU saturation limit. Higher concurrency runs are needed to locate the throughput/latency knee.
- It is a useful candidate baseline for the Base path, but should only be treated as an apples-to-apples reference after recording the exact path JSON, physical GPU model/ID, server launch flags, vLLM version, and KV-cache mode.
- The server logs captured separately showed varying prefix-cache hit rates (including high values). Prefix caching can alter prefill work and TTFT. Confirm whether prefix caching was enabled and use the same setting for Base and every PoLar path; preferably disable it for a clean compute comparison if the benchmark workload does not represent production reuse.
- The benchmark left temperature unspecified, so decoding used the server's default sampling configuration. For fixed-length throughput, output length is controlled by the benchmark workload, but record server sampling settings for repeatability.
- The reported `Total token throughput` includes input and generated tokens; use `Output token throughput` when comparing decode capacity, and report both when useful.
- Benchmark client wall duration includes request orchestration and HTTP overhead. It is not identical to the server's periodic engine throughput log.

### Comparison protocol for future runs

Keep GPU, model weights, vLLM build, server flags, path-specific KV mode, tokenizer, input/output lengths, request count, warmup count, request rate, and concurrency identical. Run Base, Skip, Loop, and Skip+Loop paths separately, then repeat each at concurrency `1, 2, 4, 8, 16, 32, ...` until throughput stops improving or latency/errors exceed the target. Save each raw result JSON next to the run notes.

Suggested next records:

| Run | Path type | Concurrency | Output tok/s | Request tok/s | P99 TTFT | P99 TPOT | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| 001 | Base/static | 1 | 157.14 | 0.61 req/s | 59.29 ms | 6.22 ms/token | 0 |
| 002 | Base/static | 2 | 329.18 | 1.29 req/s | 52.52 ms | 5.96 ms/token | 0 |
| 003 | Base/static | 4 | 635.77 | 2.48 req/s | 62.85 ms | 6.16 ms/token | 0 |
| 004 | Base/static | 8 | 1,248.30 | 4.88 req/s | 87.44 ms | 6.28 ms/token | 0 |
| 005 | Base/static | 16 | 2,352.42 | 9.19 req/s | 128.70 ms | 6.65 ms/token | 0 |
| 006 | Base/static | 32 | 4,197.47 | 16.40 req/s | 202.07 ms | 7.28 ms/token | 0 |
| 007 | Base/static | 64 | 7,081.04 | 27.66 req/s | 365.55 ms | 8.61 ms/token | 0 |
| 008 | Base/static | 96 | 8,758.68 | 34.21 req/s | 660.49 ms | 9.93 ms/token | 0 |
| 009 | Skip/static | 1 | pending | pending | pending | pending | pending |
| 010 | Loop/static | 1 | pending | pending | pending | pending | pending |

# Heterogeneous Layer-Path Scheduling Demo

`simulate_path_batching.py` accepts an input JSON containing 128 (or any number
of) integer layer paths and builds schedules for three policies:

1. `serial`: one request/layer operation at a time.
2. `identical_path_batch`: batch requests only when their full paths match.
3. `dynamic_layer_batch`: among currently ready requests, batch the largest
   group whose next physical layer is the same, then advance only those
   requests and continue.

## Input format

Plain path list:

```json
{
  "paths": [
    [0, 1, 2, 3],
    [0, 1, 2, 3],
    [0, 1, 5, 3],
    [7, 8, 9]
  ]
}
```

It also accepts MCTS result records containing `final_valid_transitions`:

```bash
python scripts/simulate_path_batching.py paths.json \
  --path-field final_valid_transitions \
  --depth 36 \
  --output path_schedule_results.json
```

If reading MCTS data, each valid transition becomes one request path. For a
controlled workload, prepare a flat `paths` list so repeated path frequencies
match the intended serving trace.

## Calibrated timing

Without `--batch-cost-json`, each layer-batch invocation costs one normalized
unit. Those results compare schedule shapes only; they are not milliseconds or
GPU throughput. To produce modeled time estimates, supply measured latency for
one physical layer forward at different batch sizes:

```json
{
  "batch_latency_ms": {
    "1": 0.7,
    "2": 0.8,
    "4": 1.0,
    "8": 1.4,
    "16": 2.2
  }
}
```

Then run:

```bash
python scripts/simulate_path_batching.py paths.json \
  --path-field final_valid_transitions \
  --depth 36 \
  --batch-cost-json layer_batch_latency.json \
  --output path_schedule_results.json
```

The latency curve must be measured on the target GPU with the same model,
precision, attention/cache mode, token shape, and kernel/backend as the serving
experiment. The current estimator sums layer-batch costs and omits queueing,
prefill/decode interaction, KV memory pressure, CUDA graph effects, and HTTP
overhead.

## Scope

This is a scheduling-plan simulator, not yet an executor that runs heterogeneous
paths inside vLLM. The current fixed-path vLLM plugin only supports one path per
server process. The emitted `batches` in the JSON are a replayable schedule
specification for the next model-runner integration stage. Its dynamic policy
uses a deterministic greedy largest-ready-layer rule, not an optimal scheduler.

The reported lower/upper comparison is deliberately scoped: serial is the
no-batching reference; the fastest of the three tested schedules is the
empirical upper bound for this scheduler set. It is not a theoretical hardware
maximum. A true serving lower/upper bound must come from replaying the same
request/path trace through a real executor and an idealized oracle scheduler.

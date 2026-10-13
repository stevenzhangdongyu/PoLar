# Base sampling versus POLAR greedy paths

Both QA and HumanEval evaluators now save `selected`, `selection_strategy`,
`selection_reason`, `base_samples`, and `timings` per task. Existing greedy Base
and top-1/top-k outputs are retained. Use a new output directory after changing
settings; old GPU results cannot supply missing timings.

## Run

```bash
GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=1 \
K=5 BEAM_SIZE=10 TEMPERATURE=0.7 TOP_P=0.95 \
MAX_NEW_TOKENS=2048 SELECTION_STRATEGY=nonbase_top1 \
HF_LOCAL_FILES_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run_sampling_vs_polar.sh qa \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /actual/predictor.pt /actual/qa_test.jsonl /actual/qa_sampling_comparison
```

Add `--limit 200` before `qa`/`code` or after the output path to evaluate only
the first 200 tasks **in total**, not 200 per worker. The default is all tasks;
`--limit 0` explicitly selects all. Existing `LIMIT=200` is also supported;
the command-line option takes precedence. If fewer tasks exist, all available
tasks are used. Resume must retain the original limit and output directory.

For code use `code` in place of `qa`, HumanEval input, and an isolated environment
with `ALLOW_CODE_EXECUTION=1`. The subprocess code runner is not a security sandbox.
All existing launcher environment variables, resume and merge-only remain supported.

POLAR generates greedily on up to K distinct decoded paths. Base generates the
same actual number of samples on the full-depth path with `do_sample=True`.
Repeated sampled outputs are retained: repeated answers are real sampling outcomes.
If beam decoding produces fewer than K paths, no arbitrary paths are invented.
The actual k histogram and requested-k shortfall count are reported.
Base greedy is additionally generated as a paired diagnostic baseline.
Stable per-task RNG seeds make sampling independent of shard layout and completed
task resume. Do not reset the seed between samples. Exact GPU reproducibility
across environments is not guaranteed.

## Selection

- `top1`: first predictor candidate, Base fallback if no candidate.
- `nonbase_top1`: first candidate whose full execution sequence differs from Base.
- `majority`: most frequent normalized final answer (QA) or exact extracted code
  string (code), ties broken by predictor order. Code text voting is a crude
  baseline, not semantic equivalence.

These rules do not read reference scores or hidden HumanEval tests. No learned
verifier is implemented. `verifier_seconds=0` explicitly reflects this.
All candidates, including Base-equivalent paths, remain in best-of-k comparison;
nonbase filtering changes only the selected answer.

## Summary

`selector`:

- repair/degradation counts, rates over all tasks, and conditional rates over
  Base-wrong/Base-correct tasks. Zero conditional denominator gives null.
- final_accuracy, net_accuracy_change, fallback_count.
- timing_totals_seconds and mean_end_to_end_seconds.

`sampling_comparison`:

- base_sampling_best_of_actual_k and polar_greedy_best_of_actual_k.
- actual_k_counts, requested_k, tasks_with_fewer_than_requested_k.

Best-of-k uses reference grading and is NOT a production selector or unbiased
pass@k estimator. Scores use completed tasks; check missing tasks before reporting.

Timings synchronize CUDA before/after generation. `selected_pipeline_seconds`
sums Base greedy generation, predictor/decode, all POLAR generations and selector;
it excludes Base sampling and offline reference tests. `end_to_end_seconds`
includes both experiment arms, trace IO and reference grading, but not model
loading or final result writes. Worker-time totals are NOT parallel wall-clock.
Multiworker contention, cold-start and warmup can bias latency; use one worker
per GPU for cleaner measurements. Equal candidate count does not imply equal
token/layer compute.

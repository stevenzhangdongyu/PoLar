GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=8 MAX_NEW_TOKENS=128 \
bash scripts/run_mcts_multiworker.sh \
  Qwen/Qwen2.5-3B-Instruct \
  hkust-nlp/dart-math-pool-math \
  train \
  1 \
  2000 \
  64 \
  ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-1/multiworker_unique
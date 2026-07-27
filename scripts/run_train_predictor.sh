FORCE_LINK_DATA=1 NUM_EPOCHS=20 BATCH_SIZE=16 LEARNING_RATE=1e-4 \
bash scripts/train_predictor.sh \
  Qwen/Qwen2.5-3B-Instruct \
  ./data/Qwen/Qwen2.5-3B-Instruct \
  1 \
  ./outputs \
  ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-1/multiworker/merged_mcts_samples.json
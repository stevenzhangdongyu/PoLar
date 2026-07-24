TOKENIZERS_PARALLELISM=false python3 scripts/verify_valid_paths.py \
  --model_path "Qwen/Qwen2.5-3B-Instruct" \
  --merged_json ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-1/merged_mcts_samples.json \
  --sample_limit 3 \
  --paths_per_sample 2 \
  --include_invalid \
  --invalid_paths_per_sample 1 \
  --max_new_tokens 50 \
  --debug_io \
  --debug_initial_path
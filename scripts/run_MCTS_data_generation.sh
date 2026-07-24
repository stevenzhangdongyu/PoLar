python3 scripts/generate_mcts_samples.py \
  --model_path "meta-llama/Llama-3.2-3B-Instruct" \
  --hf_dataset "hkust-nlp/dart-math-pool-math" \
  --hf_split train \
  --difficulty 1 \
  --output ./data/meta-llama/Llama-3.2-3B-Instruct/dart-math-diff-1/merged_mcts_samples.json \
  --limit 10 \
  --simulations 64 \
  --resume
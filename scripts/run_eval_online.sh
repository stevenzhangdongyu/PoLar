HF_LOCAL_FILES_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python3 run_polar.py \
  --eval \
  --model_path Qwen/Qwen2.5-3B-Instruct \
  --data_root ./data/Qwen/Qwen2.5-3B-Instruct \
  --save_dir ./outputs \
  --target_diff 1 \
  --checkpoint_path ./outputs/polar/target_diff1/polar_policy_epochs20_bs16_lr0.0001_amp_mpps50_seed42_target_diff1.pt \
  --eval_start_idx 0 \
  --num_samples 50 \
  --no_trust_valid_cache \
  --max_new_tokens 128 \
  --top_k_paths 5 \
  --beam_size 5 \
  --top_k_ops 2
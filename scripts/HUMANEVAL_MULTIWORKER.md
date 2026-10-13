# HumanEval 多 worker 评测

## 前提

- 本地 JSON/JSONL 含 `task_id`（可选）、`prompt`、`test`、`entry_point`。
- 每个 worker 单独加载一份 base LLM 和 predictor；不是张量并行。
- `EMBEDDING_MODEL_NAME`、`POLAR_D_MODEL`、`POLAR_HEADS`、`POLAR_LAYERS`
  必须与 checkpoint 的训练设置一致。不要为了离线方便随意换 encoder。
- 执行生成的 Python 有风险。必须在隔离环境中运行；超时和临时目录不是沙箱。
- base 和 POLAR 均为 greedy generation。POLAR top-k 指 k 个不同执行路径
  中至少一条通过测试，并非随机采样的无偏 pass@k 估计。

## 命令

```bash
GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=2 \
MAX_NEW_TOKENS=1024 TOP_K_PATHS=5 \
ALLOW_CODE_EXECUTION=1 \
HF_LOCAL_FILES_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run_humaneval_multiworker.sh \
  /path/to/Qwen2.5-3B-Instruct \
  /path/to/predictor.pt \
  /path/to/humaneval.jsonl \
  /path/to/humaneval_eval
```

`LIMIT=0` 默认全部；正整数为最多评测的题目数。
`MAX_TOTAL_WORKERS=0` 默认 GPU 数 × 每卡 worker 数；正整数为 worker 上限。
GPU 轮流分配，每个 worker 的题目数最多相差 1，无重叠。
例如 164 题、16 worker，每个 worker 10 或 11 题。

同一输出目录已存在时拒绝覆盖。中断后，原命令增加 `RESUME=1` 可续跑。
输入、worker 布局和参数必须完全相同；只跳过整题结果已经落盘的题目，
未完成题目会重新生成（trace 日志保留历史候选事件）。
如需改变 worker 数，使用新输出目录，当前版本不跨布局续跑。

原命令增加 `MERGE_ONLY=1` 可重新统计现有分片，不加载模型或启动 worker。

## 文件

- `manifest.json`：输入哈希、评测配置、GPU/rank/子集分配；
- `input_tasks.json`：选定输入快照；
- `shard_XXX_gpu_Y.log`：对应 worker 的运行日志；
- `shard_XXX_gpu_Y.json`：每完成一题原子保存；
- `shard_XXX_gpu_Y.trace.jsonl`：每完成 Base 或一条候选 path 后追加，
  包含代码、path、通过情况、错误；resume 时可能出现同题重复历史事件；
- `merged_results.json`：全部已完成整题结果及汇总；
- `summary.json`：通过数、准确率、缺失题目、失败分片、平均执行层数。

整题结果保存原始 prompt、predictor 输入、chat-template 输入、seg/op logits。
每个候选保存 segment_actions、完整 path、skipped_layers、repeated_layers、
layer_visit_counts、每个执行位置的 operation、raw completion、extracted_code、
passed、error、score、unique_depth、total_layer_executions。
layer ID 和 visit index 均从 0 开始；segment_actions 按 segment 顺序排列，
path 为最终展开后的实际层序列。

失败时仍聚合已落盘整题结果，summary 会列出缺失题目。
缺失未补齐时的准确率仅针对已完成题目，不能当作完整测试集准确率。
trace 保存的半题结果不计入汇总，避免漏掉较慢或失败候选造成选择偏差。

## 本地验证

```bash
python3 scripts/test_humaneval_multiworker.py
```

此测试不需要 GPU/torch，不执行生成代码，验证均匀分片、GPU 分配、
聚合、缺失统计、续跑参数和层操作描述。实际 GPU 运行请先用 `LIMIT=8`。

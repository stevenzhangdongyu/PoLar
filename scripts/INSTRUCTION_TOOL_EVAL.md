# 指令遵循与工具调用评测框架

这两个框架复用已有 POLAR predictor，输出 Base 与 top-k POLAR path 的生成结果。

## 1. 指令遵循 / IFEval

输入支持 IFEval 风格 JSONL，每条至少有：

```json
{"key": 1, "prompt": "...", "instruction_id_list": [...], "kwargs": [...]}
```

运行：

```bash
cd /home/z00894149/polar_dev/PoLar

GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=1 LIMIT=0 \
MAX_NEW_TOKENS=1024 K=5 HF_LOCAL_FILES_ONLY=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run_instruction_multiworker.sh \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /home/z00894149/polar_policy_epochs20_bs16_lr0.0001_amp_mpps50_seed42_all_diffs.pt \
  /home/z00894149/raw_data/ifeval/ifeval_input_data.jsonl \
  /home/z00894149/polar_dev/ifeval_polar_k5
```

输出：

- `merged_results.json`：完整 Base/POLAR candidates、path、logits、生成文本。
- `summary.json`：只含生成统计，不宣称自动评分。
- `*.ifeval_predictions.jsonl`：每条 Base/POLAR 回复一行，保留 prompt、instruction_id_list、kwargs。

注意：脚本不内置官方 IFEval checker，因为不同下载版本的 evaluator 文件路径可能不同。
请用官方 IFEval 评测代码读取 `ifeval_predictions.jsonl`。若要对比 Base/POLAR，需要按
`method` 和 `candidate_rank` 分组分别跑 checker。POLAR top-k oracle 需要定义“任一候选通过”
的聚合方式。

## 2. 工具调用 / BFCL-like

输入支持 BFCL 常见字段：

- `question` / `prompt` / `query`
- `function` / `functions` / `tools`
- `possible_answer` / `answer` / `ground_truth`

运行：

```bash
cd /home/z00894149/polar_dev/PoLar

GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=1 LIMIT=200 \
MAX_NEW_TOKENS=512 HF_LOCAL_FILES_ONLY=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run_toolcall_multiworker.sh \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /home/z00894149/polar_policy_epochs20_bs16_lr0.0001_amp_mpps50_seed42_all_diffs.pt \
  /home/z00894149/raw_data/bfcl/BFCL_v3_simple.json \
  /home/z00894149/polar_dev/bfcl_simple_polar_k5
```

内置评分是近似 checker：

- 解析模型输出里的 JSON function calls。
- `name_exact`：函数名序列是否完全一致。
- `args_exact`：参数 dict 是否完全一致。
- `call_exact`：调用数量、函数名、参数均一致。

这不是官方 BFCL scoring，不能直接报成 BFCL leaderboard 结果。它适合快速判断
POLAR path 是否破坏 function calling 格式。正式结果应接入 BFCL 官方 evaluator。

## 3. 通用参数

```bash
LIMIT=64              # 0 表示全部
GPUS="0 1 2 3 4 5 6 7"
WORKERS_PER_GPU=1
MAX_TOTAL_WORKERS=0
MAX_NEW_TOKENS=1024   # toolcall 默认 512
BEAM_SIZE=10
TOP_K_PATHS=5
TOP_K_OPS=2
RESUME=1              # 断点续跑
MERGE_ONLY=1          # 只聚合已有 shard
```

建议先 `LIMIT=20` 做 smoke test，确认数据字段和输出格式，再跑全量。

## 4. 结果解释

对这两类任务建议报告：

- Base greedy
- POLAR top-1
- POLAR top-k oracle
- nonbase/top selector 若后续接入
- 平均执行层数、unique depth、输出 token 数

指令遵循特别关注格式和约束，工具调用特别关注 JSON 可解析率、函数名与参数准确性。
开放文本不要用简单 EM/F1；工具调用也不要只看文本相似度。

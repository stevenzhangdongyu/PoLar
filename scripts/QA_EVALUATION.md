# POLAR 跨领域 QA 评测

无需重新训练 predictor。加载数学 MCTS 训练的 checkpoint，在 QA 测试集比较
Base 全层 greedy、POLAR top-1 和最多 k 条路径的 oracle EM/F1。
默认只读取本地 JSON/JSONL，不下载数据，不执行生成代码。

## 支持的输入

短答案 QA：

```json
[
  {"id": "q1", "question": "Who wrote Hamlet?", "answers": ["William Shakespeare", "Shakespeare"]},
  {"id": "q2", "question": "Where was Alice born?", "context": "Alice was born in London.", "answers": ["London"]}
]
```

`answers` 是一个答案的多个等价别名，**不是**多个必须同时答出的答案。
也支持 `answer`/`gt_ans` 字符串、扁平的 `answers.text` 列表、
TriviaQA 风格 `answer.aliases`/`answer.value`。
问题字段可用 `question` 或 `query`。ID 可用 `task_id`/`id`；缺失时生成稳定 ID。
容器格式支持纯列表、`{"tasks":[...]}`、`{"samples":[...]}`、`{"data":[...]}`。
原始嵌套 SQuAD `articles→paragraphs→qas` 要先展开为上述一题一行格式。
空答案/不可回答题暂不支持，必须先过滤或另写专门的 no-answer 评测。

选择题，设置 `QA_MODE=multiple_choice`：

```json
[
  {"id": "mc1", "question": "Which planet is known as the Red Planet?",
   "choices": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
   "answer": "B"}
]
```

choices 也支持字符串列表（自动 A/B/C...）和 `{"label":[...],"text":[...]}`。
标签必须单个字母，参考答案必须是标签；也支持字段 `answerKey`。
输出只接受单个标签或 `Answer: B` 等明确格式；冲突标签判失败，不猜选项。

## 启动

```bash
GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=2 \
LIMIT=0 MAX_NEW_TOKENS=1024 TOP_K_PATHS=5 \
HF_LOCAL_FILES_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run_qa_multiworker.sh \
  /models/Qwen2.5-3B-Instruct \
  /outputs/predictor.pt \
  /datasets/qa_test.jsonl \
  /outputs/qa_eval
```

- `LIMIT=0` 全部；正整数表示最大题目数。
- 每卡 worker 数 × GPU 数为总 worker 数，最多不超过题目数。
- `MAX_TOTAL_WORKERS` 可限制总 worker；默认 0 不额外限制。
- 连续子集均匀切分，题目数相差至多 1，GPU 轮流分配。
- 每 worker 单独加载一份 base LLM 和 predictor；不是数据并行训练或张量并行。
- `MAX_NEW_TOKENS` 同时用于 Base 和所有 POLAR 路径，默认 1024。
- `EMBEDDING_MODEL_NAME`、`POLAR_D_MODEL/HEADS/LAYERS` 必须与训练一致。
  默认 embedding 为 `Qwen/Qwen3-Embedding-0.6B`；离线需要缓存或本地路径。
- `PREDICTOR_INPUT=question` 默认只编码问题；
  `PREDICTOR_INPUT=prompt` 则编码完整指令、context 和 choices（不含参考答案）。
  对阅读理解/选择题，这两种输入是不同实验设定，应分别报告。
- prompt 相同，Base/POLAR 都 greedy，chat template 相同。
- 不做隐式截断；超出模型上下文长度的题目报错，需要预先筛选/缩短 context。
- 每次更新 checkpoint 后使用新的 output_dir，resume 不检测 checkpoint 内容改变。

可先用 `LIMIT=8 WORKERS_PER_GPU=1` 和新的输出目录试跑。
演示数据在 `scripts/qa_example.json`，不是基准测试集。

## 结果和过程

```text
manifest.json                   配置、数据哈希、GPU 和子集分配
input_tasks.json                标准化输入快照
shard_XXX_gpu_Y.log             worker 日志
shard_XXX_gpu_Y.json            每完成整题原子更新
shard_XXX_gpu_Y.trace.jsonl      每完成一个答案追加
merged_results.json            聚合后的每题详细结果
summary.json                   全局统计和缺失题目
```

每题记录：

- question/context/references、predictor 输入、chat template 后的模型输入；
- seg/op logits、beam 分数、segment_actions、展开后的 path；
- skipped_layers/repeated_layers、每层执行次数、每个 execution visit；
- 原始回答、提取答案、normalized_answer、EM/F1、生成 token 数；
- hit_token_limit（达到预算，不保证一定发生语义截断）。

每道题全部候选完成才计入统计。中途失败时该题已完成的候选在 trace 中，
但不计入全局准确率，避免把只完成部分候选的题目当成完整评测。

## 指标含义与限制

`summary.json` 的 Base/top-1/oracle-top-k 各自有 `correct`、`exact_match`、`f1`。
EM/F1 为 0–1 范围，乘 100 是百分比；多个答案别名分别取最高匹配分数。

- **EM**：NFKC、小写、去标点、去英文 a/an/the、合并空白后精确匹配。
- **F1**：英文词/数字按 token，中文汉字逐字，计算重叠 precision/recall。
- 这是透明的自定义评分规则，不宣称等同每个 benchmark 的官方评测器。
  正式复现某个 benchmark，应适配其官方 normalization/scoring。
- 选择题 EM/F1 均等于选项是否命中。
- 提取仅清理格式，不做“答案出现在输出中就算对”的宽松匹配。
- 不适合开放式长答案的语义评分；需另行设计人工/可信 judge 评测。
- top-k oracle 利用参考答案取最高分，不是线上答案选择器，
  也不是随机采样的无偏 pass@k。与单次 Base 比较时必须标明增加了候选预算。
  直接衡量单次泛化收益应看 POLAR top-1 vs Base。
- 此评测沿用仓库的 custom-path/KV 实现；指标不能单独证明底层 repeat/KV
  语义正确，正式系统实验仍需要另做运行时正确性验证。

## 恢复和重新聚合

原命令增加 `RESUME=1`：参数、输入、worker 布局保持完全一致。
已落盘整题跳过，半题重跑；trace 会保留重复历史事件。

原命令增加 `MERGE_ONLY=1`：不启动 GPU，只聚合现有分片。
失败分片/缺失题目会列出；未补齐的指标只对已完成题目成立。

单 worker 也可直接：

```bash
PYTHONPATH="$PWD" python3 scripts/eval_polar_qa.py \
  --model_path /models/Qwen2.5-3B-Instruct \
  --checkpoint_path /outputs/predictor.pt \
  --input_json /datasets/qa_test.jsonl \
  --save_json /outputs/qa_single.json \
  --max_new_tokens 1024
```

测试：`python3 scripts/test_qa_evaluation.py`。标准库测试不需要 GPU，
不加载真实模型。真实 GPU smoke test 仍需要在你的服务器上执行。

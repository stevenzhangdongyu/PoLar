# POLAR 实时终端对话

模型与 predictor 常驻单张 GPU，每轮输入生成 Base greedy + 最多 K 条不同
POLAR 执行路径的 greedy 回复，默认 K=5。token 流式输出；候选依次生成，
不是五路并行，也不是 token beam search。路径 beam search 仅用于 predictor。

## 启动（服务器）

先同步新脚本 `chat_polar.py`、`polar_chat_utils.py`、`run_polar_chat.sh`；
依赖仓库现有 `candidate_selection.py`、`qa_eval_utils.py`。

```bash
cd /home/z00894149/polar_dev/PoLar
GPU=0 K=5 BEAM_SIZE=10 MAX_NEW_TOKENS=2048 \
PREDICTOR_INPUT=question SELECTION_STRATEGY=nonbase_top1 \
HF_LOCAL_FILES_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run_polar_chat.sh \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /home/z00894149/polar_policy_epochs20_bs16_lr0.0001_amp_mpps50_seed42_all_diffs.pt
```

第三个位置参数可指定新的 session JSON 文件，不传则自动生成唯一名称。
已有文件拒绝覆盖，不支持隐式恢复历史。embedding 路径/名称及 predictor 结构
必须和训练一致，可设置 `EMBEDDING_MODEL_NAME`、`POLAR_D_MODEL/HEADS/LAYERS`。
默认使用本地 GPU 0，可通过 `GPU` 修改。`STREAM=0` 关闭流式显示。

## 多轮对话与控制

```text
你> 用一句话解释什么是 KV cache。
... Base 与 POLAR 1..5 依次输出 ...
你> /use 2
你> 请结合刚才的解释举个例子。
你> /summary
你> /new
你> /quit
```

所有候选看到相同上下文。默认按 `nonbase_top1` 选择一条回复写入共同历史，
不代表该回复最好；特别是代码任务，此策略在既有实验上会退化。
`SELECTION_STRATEGY=top1` 可改为原始首选策略。
`/use base` 或 `/use N` 可在输入下一道问题之前切换最近一轮的历史回复。
不会维护五条独立分支对话。`/new` 清空模型对话上下文，历史实验日志与评分保留。
`/multi` 支持多行问题/代码输入，单独一行 `/end` 结束。

每轮/每条路径重新 prefill，不跨路径/轮次复用 KV，不意味着修复了循环层
patch 内部的 KV 语义。输入与输出预算超过上下文长度会报错，不会静默截断。
`PREDICTOR_INPUT=question` 仅使用最新用户问题（接近数学训练条件）；
`history` 使用带角色的共同历史文本，是另一种实验设置。
predictor 内部 tokenizer 仍最多截断到 512 tokens，长历史不保证全部被看到。
系统提示可用 `SYSTEM_PROMPT` 设置，不会自动注入“只给短答案”的约束。

## pass@5 的边界与评分

普通对话没有标准答案。默认只展示候选，不声称算出 pass@5。
无学习型验证器、外部 API 判分或代码执行。

可用手工标签判断每条候选：

```text
/grade [false,true,false,false,true]
```

标签长度必须等于实际 POLAR 候选数（不含 Base）；本轮记为“任一候选正确”。
代码题请自行在安全环境测试后标注；脚本不执行生成的代码。

若任务是短答案 QA，可在生成后输入参考答案：

```text
/reference ["Paris","巴黎"]
```

只在生成后做现有框架的自定义短答案 EM/F1，不把参考答案放入模型输入。
不是语义判分：长解释即使含正确答案也可能 EM=0。开放对话请优先人工评分。
同一轮重评分会替换对应评分；人工与参考评分分开汇总，不混合分母。
`/summary` 显示有评分轮次中 best-of-实际K 的成功率，未评分轮次排除。
此数是人工/答案辅助的 best-of-k，不是无偏随机 pass@k 或生产自动选择准确率。
若 beam 只生成不足五条独特路径，明确显示 actual_k，不补造路径、不标成 pass@5。

## 保存与性能

- session JSON：配置、当前共同历史、所有已完成轮次、实际 k、路径分数、
  segment 操作、skip/repeat、输入/完整输出、token 数、logits、耗时和评分。
- `.trace.jsonl`：逐条 generation_start/end、错误和命令；中途失败时已完成输出保留。
- 不自动恢复失败轮次；中断保存完整轮次，未完成轮次不写入共同历史。
- 控制台流式显示但日志只在候选生成结束后保存完整文本，不逐 token 持久化。
- CUDA 同步计时；流式打印会包含在生成耗时里，不能当作优化服务性能结果。
- 日志含原始对话和参考答案，请注意敏感信息。

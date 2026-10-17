# MLP path selector 训练框架

目标：从 `[Base + top-5 POLAR paths]` 中选一条 path。第一版 selector 是
**生成前 path-only selector**，不看候选 completion 文本，只看问题 embedding 和
path 结构特征，因此未来可在生成前决定只执行一条 path。

## 数据

输入使用已经跑完的 QA/HumanEval `merged_results.json`，其中每个样本包含：

- `base.path` 与 `base.exact_match` / `base.passed`
- `candidates[i].path`、`score`、`exact_match` / `passed`
- `predictor_input` / `question` / `original_prompt`

一个样本会构造成固定候选集合：

```text
candidate 0 = Base
candidate 1..5 = POLAR paths
```

如果一个样本不足 5 条 POLAR path，则 padding 并 mask。Base 永远作为安全候选。
同一个 question 的所有候选不会被拆开；split 是按样本整体随机切分。

## 特征

每条 path 的结构特征包括：

- 每层 visit count，长度 D。
- skip mask，长度 D。
- repeat mask，长度 D。
- length ratio、unique-depth ratio、skip/repeat 数量、repeat 执行次数。
- max visit count、non-monotonic edge 数、jump edge 数。
- POLAR predictor path score。
- candidate rank。
- 是否 Base。

问题侧使用 frozen `Qwen/Qwen3-Embedding-0.6B` mean pooling，再过一个投影层。

## 模型

```text
q_emb = frozen_text_encoder(question)
path_feat_j = handcrafted_features(path_j)
score_j = MLP([q_emb, path_feat_j])
```

这是一版简单 MLP，没有候选间 attention。后续如果 MLP 不够，可以把 6 个候选
的 representation 再送入小 Transformer 做 cross-candidate ranking。

## Loss

使用 multi-positive listwise loss：

```text
L = -log sum_{j in correct candidates} softmax(score)_j
```

如果多个候选都正确，不强迫模型选某一个；只要把概率分给任意正确候选即可。
默认保留 all-wrong 样本，但它们对当前 loss 没有梯度，只用于评估 oracle gap；
可以设置 `SKIP_ALL_WRONG=1` 跳过。

## 训练命令

```bash
cd /home/z00894149/polar_dev/PoLar

HF_LOCAL_FILES_ONLY=1 TRANSFORMERS_OFFLINE=1 \
NUM_EPOCHS=5 BATCH_SIZE=32 LEARNING_RATE=1e-4 \
bash scripts/run_train_path_selector.sh \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /home/z00894149/polar_dev/selector_outputs/triviaqa_mlp \
  /home/z00894149/polar_dev/qa_all_triviaqa_sampling_vs_polar_k5_alldiff_magen50/merged_results.json
```

多领域联合训练：

```bash
bash scripts/run_train_path_selector.sh \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /home/z00894149/polar_dev/selector_outputs/qa_code_mlp \
  /path/to/triviaqa/merged_results.json \
  /path/to/squad/merged_results.json \
  /path/to/humaneval/merged_results.json
```

输出：

- `selector_config.json`
- `selector_history.json`
- `selector_mlp_best.pt`

## 离线评估命令

```bash
HF_LOCAL_FILES_ONLY=1 TRANSFORMERS_OFFLINE=1 \
python3 scripts/eval_path_selector.py \
  --checkpoint /home/z00894149/polar_dev/selector_outputs/triviaqa_mlp/selector_mlp_best.pt \
  --eval_json /home/z00894149/polar_dev/qa_all_triviaqa_sampling_vs_polar_k5_alldiff_magen50/merged_results.json \
  --output_json /home/z00894149/polar_dev/selector_outputs/triviaqa_mlp/eval_on_triviaqa.json \
  --local_files_only
```

评估不会重新生成答案，只在已有 candidates 上选择一条，报告：

- Base accuracy
- selector accuracy
- oracle accuracy
- repair count
- degradation count

## 注意

1. 如果用 TriviaQA 训练，就不能再用同一批 TriviaQA 当最终泛化测试。
2. 这是 path-only selector，不看 completion；它更适合生产，但上限可能低于 answer-aware verifier。
3. 默认随机 split，不是严格按数据源/文章分组。正式实验应显式准备 train/val/test 文件。
4. 当前 loss 对 all-wrong 样本无梯度。若要学习“回退/拒绝”，需要增加 reject 头或 margin loss。
5. 训练出的 selector 目前只做离线选择；要真正接入在线推理，需要在生成前用相同 feature 构造逻辑对 paths 打分。

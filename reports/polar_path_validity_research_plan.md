# 如何提高 POLAR top-5 路径 valid 密度：统计框架与研究思路

目标不是再训练一个工程 selector，而是理解“什么样的 layer execution program
更可能正确”，再反过来改进 predictor 解码、MCTS 数据生成和训练目标，使 top-5
候选里正确路径密度更高。

## 1. 统计框架

新增脚本：

```text
scripts/analyze_path_validity.py
```

它支持两类输入：

- MCTS 数据：`merged_mcts_samples.json`，使用 `final_valid_transitions` 和
  `final_invalid_transitions` 作为 valid/invalid 标签。
- 下游评测结果：QA/HumanEval 的 `merged_results.json`，使用 `exact_match` 或
  `passed` 作为 correct 标签。

示例：

```bash
cd /home/z00894149/polar_dev/PoLar

python3 scripts/analyze_path_validity.py \
  --json /home/z00894149/data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-5/merged_mcts_samples.json \
  --depth 36 \
  --min_n 50 \
  --out_prefix /home/z00894149/analysis/path_stats/diff5_mcts

python3 scripts/analyze_path_validity.py \
  --json /home/z00894149/polar_dev/qa_all_triviaqa_sampling_vs_polar_k5_alldiff_magen50/merged_results.json \
  --depth 36 \
  --min_n 100 \
  --out_prefix /home/z00894149/analysis/path_stats/triviaqa_eval
```

输出：

- `<prefix>_summary.json`：总体 valid/correct 密度、样本级有无正例比例。
- `<prefix>_groups.csv`：按结构分桶后的 valid/correct rate。
- `<prefix>_paths.csv`：每条 path 的结构特征与 label，方便后续回归或作图。

重点查看 group：

- `rank`：正确路径在 top-5 的第几位。
- `signature`：base / skip / repeat / nonmonotonic 组合。
- `length_ratio_bucket`、`extra_exec_bucket`：执行次数是否越多越好。
- `repeat_layer_count`、`repeat_exec_count`、`repeat_run_len`：
  哪种 loop 密度更高。
- `repeated_layer`、`skipped_layer`：哪些具体层被 repeat/skip 时更容易正确。
- `max_repeat_run_len`、`max_skip_run_len`：连续 segment 粒度。

## 2. 应先回答的研究问题

### 2.1 valid 路径是否集中在某些结构族

看 `signature` 和 `length_ratio_bucket`：

- 如果 `repeat` 或 `repeat+skip` 的 correct rate 高于 base-like 路径，说明 loop
  是有效探索方向。
- 如果长度过长的 path correct rate 高但成本高，需要研究准确率/执行成本 Pareto。
- 如果某些结构 n 很小，不能直接相信 rate，要提高 `--min_n` 或做置信区间。

### 2.2 哪些层/segment 的 loop 最有效

看 `repeated_layer`、`repeat_run_start`、`repeat_run_len`：

- 如果中后层 repeat 明显更高，可能说明“推理整合层”值得循环。
- 如果 repeat run length=1 明显优于长 segment，当前 MCTS 最大 block 可以收紧；
  反之如果 length=2/3 更好，可以鼓励 segment 级 recurrence。
- 如果某些层 repeat 高但同时 skip 某些层高，需要分析它们是否形成“替代计算”：
  例如跳过低贡献层，把预算给更有用的层。

### 2.3 正确候选在 top-5 中的位置

看 `rank`：

- 如果 correct 主要在 rank 2-5，说明 predictor 有发现能力但排序差，需要改训练目标。
- 如果 rank 1 基本是 base 而 rank 2-5 有正例，说明训练中原始路径权重过强。
- 如果 top-5 oracle 高但每题平均正例少，说明路径密度低，需要改解码约束和训练目标。

### 2.4 数学训练路径和下游正确路径是否同构

分别统计数学 MCTS 和 TriviaQA/HumanEval eval 的结构分布：

- 数学中 valid 的 loop 模式是否也在 QA/code 中 correct？
- 如果 QA 的 correct 路径偏好与数学完全不同，说明 predictor 的候选生成需要领域自适应。
- 如果结构模式相同但排序不好，selector 或 rank loss 可能更关键。

## 3. 可能的理论解释

POLAR 的 path 可以看成对 transformer 深度维度做离散程序搜索。不同 loop/skip
不是随意扰动，而是在改变 representation refinement 的次数和位置：

1. **早层更多是 lexical/局部特征抽取。** 重复早层可能收益小，甚至会过度处理局部模式。
2. **中层可能承载推理状态变换。** 对中层 segment recurrence 可能相当于给问题状态多次迭代。
3. **后层更接近输出对齐和格式化。** 重复后层可能提高答案形式稳定性，也可能放大错误偏置。
4. **skip 与 repeat 共同构成预算重分配。** 有效路径未必更短，而是把层执行预算从低收益层转移到高收益层。
5. **valid 密度取决于任务类型。** 数学可能偏好反复中后层推理，代码可能对完整层序更敏感，所以强制 non-base 会退化。

这解释了你目前的观察：

- TriviaQA top-5 oracle 高于 Base sampling：路径扰动能产生额外正确答案。
- nonbase-top1 几乎无收益：目前排序没学会哪些扰动是真正有益。
- HumanEval 非 Base 退化：代码生成对程序结构和格式更脆弱，错误 loop 更容易破坏可执行性。

## 4. 如何提高 top-5 valid 密度

### 4.1 改解码约束

基于统计结果，限制低效结构：

- 禁止或降低过长 path。
- 只允许高收益 layer band 的 repeat。
- 控制 repeat segment length，例如只保留统计上有效的 1/2/3。
- 明确保留一条 base path，但让其不占满 top-1/top-5 多样性。

### 4.2 改训练目标

从“学习所有 valid path”改成“提高 top-k 正例密度”：

- top-k listwise loss：让每题 top-k 中至少有多个 valid。
- hard negative：加入与 valid 结构相似但错误的 paths。
- length-aware positive weighting：奖励正确且成本低的 path，但不要过度偏短。
- diversity regularization：top-5 不要全是 base 附近微扰。

### 4.3 改 MCTS 数据生成

如果统计发现某些 loop 类型 valid rate 高，应在 MCTS 中提高这些邻居的 proposal
概率；如果某些结构几乎无效，则减少展开。

这相当于从 uniform-ish path search 变成 structure-aware search：

```text
MCTS proposal prior ∝ historical valid density(structure)
```

但要避免把统计偏差写死。最好分 difficulty、任务类型、模型层数分别统计。

## 5. 最小研究闭环

1. 在数学 MCTS train/val 上跑 `analyze_path_validity.py`。
2. 在 TriviaQA/HumanEval eval 上跑同样统计。
3. 比较 valid/correct 的结构分布差异。
4. 选 2-3 个简单解码约束，例如“只 repeat 中后层、repeat_run_len≤2、保留 base 但 top-5 去重”。
5. 不重新训练，直接改解码/MCTS proposal，看 top-5 oracle 和 path density 是否提高。
6. 再决定是否修改 predictor 训练目标。

如果这个闭环能让 top-5 oracle 不变或上升，同时平均正确候选数增加，就说明你找到了比 selector 更基础的研究方向。

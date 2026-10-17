# Diff5 MCTS path valid-density 统计报告

日期：2026-10-16  
数据来源：用户提供的 Diff5 MCTS path 结构统计表  
统计对象：MCTS 生成的 `final_valid_transitions` 与 `final_invalid_transitions`  
模型深度：从表中 `is_base=true` 路径看，Base depth 为 36

## 1. 总体结论

Diff5 的 MCTS 统计呈现出一个很清晰的结构偏好：

> **有效路径更像“保留完整 36 层，然后适度增加 repeat/loop”，而不是通过 skip 缩短模型。**

几个最关键的信号：

- Base path 本身 valid rate 很高：`893/1867 = 47.83%`。
- 非 Base path 总体 valid rate 只有 `31605/186247 = 16.97%`。
- 但只要路径包含 repeat/loop，valid rate 上升到 `24.10%`。
- 如果路径不 skip 任何层，valid rate 达到 `31.74%`。
- 反过来，包含 skip 的路径 valid rate 只有 `8.26%`。
- 执行层数略多于 36 层时，valid rate 明显高于短路径。

这说明 Diff5 上，MCTS 找到的正确路径更多来自 **depth scaling / recurrence over layers**，而不是 **layer pruning / shorter execution**。

## 2. 总体 valid density

| 路径类型 | n | valid | valid rate |
|---|---:|---:|---:|
| Base path | 1,867 | 893 | **47.83%** |
| 非 Base path | 186,247 | 31,605 | 16.97% |
| 全部 path | 188,114 | 32,498 | 17.28% |

解释：

Base valid rate 高，说明在 Diff5 数据里，原始完整模型本身已经能解出相当一部分题。这会带来两个影响：

1. predictor 训练时容易偏向 Base/full-depth path。
2. 如果想提高 top-5 的 valid 密度，不能简单排除 Base；应把 Base 作为安全 anchor，同时鼓励其他高质量 recurrence path。

## 3. Repeat/loop 明显比 skip 更有价值

| 结构 | n | valid | valid rate | avg executions | avg unique depth |
|---|---:|---:|---:|---:|---:|
| has_loop / has_repeat = True | 104,007 | 25,062 | **24.10%** | 37.80 | 35.19 |
| has_loop / has_repeat = False | 84,107 | 7,436 | 8.84% | 32.93 | 32.93 |
| has_skip = False | 72,257 | 22,932 | **31.74%** | 38.65 | 36.00 |
| has_skip = True | 115,857 | 9,566 | 8.26% | 33.73 | 33.04 |

最强结论是：

```text
repeat/loop 是正向信号；
skip 是强负向信号；
不 skip 且适度 repeat 的路径最值得优先探索。
```

尤其是 `has_skip=false` 的平均 unique depth 为 36，说明这些路径基本覆盖所有层，只是在某些层上增加 repeat。它们的 valid rate 达到 31.74%，远高于整体非 Base 平均。

这和 POLAR 论文里 “segment recurrence improves execution programs” 的直觉一致：对 Diff5 来说，有效路径不是少算，而是**在保留完整表示流水线的基础上，多做几步 refinement**。

## 4. 执行层数越短越差，适度增加深度更好

### 4.1 extra executions

这里 `extra_exec = total_layer_executions - 36`。

| extra_exec_bucket | n | valid | valid rate | avg executions | avg unique depth |
|---|---:|---:|---:|---:|---:|
| <= -6 | 8,272 | 38 | **0.46%** | 29.33 | 29.33 |
| <= -1 | 87,709 | 7,104 | 8.10% | 33.42 | 33.17 |
| <= 0 | 10,279 | 1,739 | 16.92% | 36.00 | 34.04 |
| <= 1 | 21,673 | 5,583 | 25.76% | 37.00 | 35.47 |
| <= 4 | 55,064 | 15,960 | 28.98% | 38.93 | 35.87 |
| <= 8 | 5,117 | 2,074 | **40.53%** | 41.39 | 36.00 |

短路径非常差：

- 少 6 层以上几乎无效，valid rate 只有 0.46%。
- 少 1 层以内的短路径整体也只有 8.10%。

而增加执行层数后，valid rate 明显上升：

- 37 层左右：25.76%。
- 39 层左右：28.98%。
- 41 层左右：40.53%。

注意 `<=8` 桶样本量是 5,117，已经不是极小样本，但仍应和具体 repeat 层位置一起分析，避免误解成“越长越好”。

### 4.2 length ratio

| length ratio bucket | n | valid | valid rate |
|---|---:|---:|---:|
| <= 0.9 | 32,171 | 712 | 2.21% |
| <= 1.0 | 74,089 | 8,169 | 11.03% |
| <= 1.05 | 21,673 | 5,583 | 25.76% |
| <= 1.15 | 58,186 | 17,173 | 29.51% |
| <= 1.3 | 1,995 | 861 | **43.16%** |

趋势和 extra executions 一致：**压缩深度会显著损害正确率，适度加深会提高 valid density。**

这对后续设计有直接意义：

- 如果目标是提升 top-5 valid 密度，不应优先追求更短路径。
- 可以把候选空间集中在 `36~42` 次 layer execution 附近。
- 特别值得研究 `40~42` 层的路径结构：到底 repeat 了哪些层、是否集中在中后层、是否是连续 segment recurrence。

## 5. Skip 连续长度越长，valid rate 越低

| max_skip_run_len | n | valid | valid rate |
|---|---:|---:|---:|
| 0 | 72,257 | 22,932 | **31.74%** |
| 1 | 25,517 | 4,938 | 19.35% |
| 2 | 27,251 | 2,495 | 9.16% |
| 3 | 29,851 | 1,352 | 4.53% |
| 4 | 31,557 | 765 | 2.42% |
| 5 | 566 | 12 | 2.12% |
| 6 | 485 | 1 | 0.21% |
| 7 | 407 | 1 | 0.25% |
| 8 | 223 | 2 | 0.90% |

这是全表里最稳定的负相关之一：

```text
不 skip：31.74%
skip 1 层连续段：19.35%
skip 2 层连续段：9.16%
skip 3~4 层连续段：快速掉到 2~5%
```

所以在 Diff5 上，skip 并不是一个好的探索方向。尤其是连续跳过多个层，会严重破坏路径有效性。

研究建议：

- MCTS proposal 可以大幅降低 skip 操作概率，特别是连续 skip。
- predictor 解码阶段可以对 `max_skip_run_len >= 2` 加惩罚。
- 如果要保留 skip，优先只允许 isolated single-layer skip，并且最好搭配 repeat 作为补偿。

## 6. Repeat segment length 的信号

| max_repeat_run_len | n | valid | valid rate |
|---|---:|---:|---:|
| 0 | 84,107 | 7,436 | 8.84% |
| 1 | 27,045 | 6,191 | 22.89% |
| 2 | 26,347 | 6,389 | 24.25% |
| 3 | 26,284 | 6,522 | **24.81%** |
| 4 | 23,970 | 5,807 | 24.23% |
| 5 | 276 | 117 | 42.39% |
| 6 | 85 | 36 | 42.35% |

主要结论：

- 有 repeat 明显好于没有 repeat。
- repeat run length 1~4 的 valid rate 比较稳定，约 22.9%~24.8%。
- run length 5/6 的 rate 很高，但样本量分别只有 276 和 85，不能直接下结论。

一个合理解释是：

```text
长度 1~4 的 repeat segment 是稳定有效区域；
长度 5/6 可能代表某些特殊高收益结构，但需要更多采样验证。
```

后续可以针对 run length 5/6 做定向 MCTS 扩展，看高 valid rate 是否可复现。如果复现，说明较长连续 segment recurrence 可能是 Diff5 的关键模式。

## 7. 单层最多访问次数：适度重复有益，过度重复可能变差

| max_visit_count | n | valid | valid rate |
|---|---:|---:|---:|
| 1 | 84,107 | 7,436 | 8.84% |
| 2 | 102,622 | 24,601 | 23.97% |
| 3 | 1,119 | 426 | **38.07%** |
| 4 | 266 | 35 | 13.16% |

含义：

- 某些层访问 2 次，比完全不 repeat 好很多。
- 访问 3 次的 valid rate 更高，但样本量只有 1,119，需要进一步验证。
- 访问 4 次反而下降到 13.16%，说明过度循环可能破坏表示。

因此不建议无脑增大 `max_repeat`。更好的方向是：

```text
允许少数层/segment repeat 1~2 次；
谨慎探索 max_visit_count=3；
默认不鼓励 max_visit_count>=4。
```

## 8. rank 表不能当作 predictor top-k 排序

表中 `rank` 需要特别小心。

对 MCTS 数据而言，当前统计脚本里的 rank 是：

```text
valid list 内部的第 r 个 path
invalid list 内部的第 r 个 path
```

不是 predictor top-k 排名，也不是 MCTS 节点 UCB 排名。

所以不能说：

```text
rank 1 valid rate 45.31%，说明 predictor 第一名很准
```

这是错误解读。

更合理的解读是：rank 曲线反映了每个样本中 valid/invalid path 数量尾部分布的差异。前面的 rank rate 高，是因为很多样本有至少 1 个 valid path；后面 rank rate 逐渐归零，是因为很少有样本拥有那么多 valid paths，而 invalid paths 更长尾。

从趋势看：

- rank 1 rate 45.31%。
- rank 10 rate 35.11%。
- rank 30 rate 19.75%。
- rank 50 rate 6.77%。
- rank 77 以后 valid 基本为 0。

这说明 Diff5 的 valid paths 在每题内部并不均匀丰富：有些题 valid path 很多，但多数题 valid path 数量有限。要提高 top-5 valid 密度，需要把搜索和解码集中到上面几类高密度结构上，而不是扩大低质量尾部。

## 9. 对提高 top-5 valid 密度的启发

### 9.1 解码阶段可以加入结构先验

目前 predictor beam search 主要根据模型分数选 path。基于这份 MCTS 统计，可以引入结构 rerank：

```text
score_final(path)
  = score_predictor(path)
  + alpha * repeat_bonus(path)
  - beta  * skip_penalty(path)
  - gamma * overlong_or_overskip_penalty(path)
```

推荐先验：

- 奖励 `has_repeat=true`。
- 奖励 `has_skip=false`。
- 奖励 `total_layer_executions` 在 37~42 附近。
- 惩罚 `max_skip_run_len >= 2`。
- 惩罚 `max_visit_count >= 4`。

这个方向不需要重新训练 predictor，可以先做快速 ablation。

### 9.2 MCTS proposal 可以从 uniform search 改为 structure-aware search

Diff5 中最值得提高采样概率的结构：

```text
full-depth + repeat
no-skip repeat
repeat_run_len = 1~4
total executions = 37~42
max_visit_count = 2 或 3
```

最应该降低采样概率的结构：

```text
short path
skip-heavy path
max_skip_run_len >= 2
max_visit_count >= 4
```

这会让数据生成阶段产生更高密度的 valid paths，从而改善 predictor 的训练样本质量。

### 9.3 训练目标应从“拟合 valid path”转为“提高 top-k density”

现在的 predictor 容易学到 Base/full-depth 偏好，而不是学到“哪些 recurrence 真的有效”。可以考虑：

- 对 valid but non-base/no-skip/repeat path 提高权重。
- 对 skip-heavy invalid path 做 hard negative。
- 做 listwise top-k density loss：让每个问题的 top-5 包含尽可能多 valid paths。
- 保留 Base 作为候选，但不要让 Base 独占 top-1/top-5。

## 10. 还需要补的统计

这张表还缺几个关键维度。建议继续跑：

1. `repeated_layer`：具体 repeat 哪些层最有效。
2. `repeat_run_start`：有效 repeat segment 从哪里开始。
3. `skipped_layer`：哪些层被 skip 后最容易失败。
4. valid path 与 invalid path 的 path edit distance to Base。
5. 按题目难度、Base 是否正确分组统计。

尤其是 Base 是否正确分组非常重要：

```text
Base 已正确的题：repeat path 是否只是冗余？
Base 错误的题：哪些 repeat path 能真正修复？
```

这会直接决定后续研究方向：是做 “safe recurrence”，还是做 “repair-oriented recurrence”。

## 11. 一句话总结

Diff5 的 MCTS 统计强烈支持这样一个研究假设：

> 对难数学题来说，POLAR 的有效路径主要来自 **保留完整深度并在少数层/连续 segment 上做适度 recurrence**；skip/prune 尤其是连续 skip 会显著降低 valid density。后续提升 top-5 valid 密度的关键，不是扩大搜索预算，而是把 MCTS proposal、predictor 解码和训练目标都向 “full-depth + controlled repeat” 的高密度区域偏置。


# POLAR predictor 跨领域泛化性验证报告

日期：2026 年 10 月 13 日  
更新：2026 年 10 月 15 日（纳入 TriviaQA 9960 条全量结果）  
实验阶段：SQuAD 小规模探索性验证 + TriviaQA/HumanEval 全量验证  
依据：提供的 SQuAD、TriviaQA、HumanEval 评测汇总与运行命令；未独立检查服务器上的逐题结果或训练数据。

## 1. 核心结论

本轮实验表明，数学任务训练的 POLAR predictor 生成的层执行路径，在非数学任务中具有一定候选探索价值，但尚不能证明其具备稳定、可直接部署的跨领域路径选择能力。

- **SQuAD：**最高分的非 Base 路径将准确率从 79.69% 提升至 93.75%，是本轮最积极的结果。
- **TriviaQA（全量 9960 题）：**non-base top-1 几乎不提升，准确率由 35.19% 降至 35.12%，修复 491 题、退化 498 题，净损失 7 题；但 POLAR 五路径 oracle 达到 46.06%，明显高于 Base 五次随机采样的 41.98%。
- **HumanEval（全量 164 题）：**POLAR 五路径 oracle 为 82.93%，Base 五次随机采样为 83.54%，前者少答对一题；强制选择非 Base 路径将准确率从 68.29% 降至 62.20%，修复 12 题、退化 22 题，净损失 10 题。
- 三个任务的原始 POLAR top-1 均与 Base 准确率一致，平均执行次数和 unique depth 都为 36。这与完整路径偏好一致，但是否逐题等于 Base 路径仍需检查完整执行序列。

因此，应分别评价 **候选生成能力** 与 **候选选择能力**。现阶段不建议跨任务统一删除 Base 路径，也不能根据 oracle 提升直接宣称生产准确率提升。

## 2. 验证目标与实验设置

### 2.1 验证目标

验证数学领域训练的层执行策略，能否迁移到阅读理解、知识问答和代码生成场景，而无需针对目标领域重新训练 predictor。

测试使用 checkpoint：

```text
/home/z00894149/polar_policy_epochs20_bs16_lr0.0001_amp_mpps50_seed42_all_diffs.pt
```

按照提供的实验背景，这是 all-diffs 数学训练 checkpoint。仅凭文件名不能独立证明训练数据组成、生成长度或训练/测试无交叉，正式报告前应保留训练配置与数据清单。

### 2.2 对照方法

| 方法 | 执行方式 | 是否使用标准答案进行选择 |
|---|---|---|
| Base greedy | 原始完整路径，greedy 生成一次 | 否 |
| POLAR top-1 | predictor 排名第一的路径，greedy 生成 | 否 |
| POLAR non-base top-1 | 按 predictor 顺序选第一条不等于 Base 的执行序列 | 否 |
| Base sampling best-of-5 | 完整路径随机采样五次，判断是否任一答案正确 | 是，oracle |
| POLAR five-path oracle | 五条不同执行路径各 greedy 生成，判断是否任一答案正确 | 是，oracle |

non-base top-1 不判断候选答案质量，也没有训练好的验证器；它只是一个参考答案无关的路径过滤规则。不能把它表述为“选择真正最优候选”。

SQuAD 完成 64 个用例，TriviaQA 完成全量 9960 个用例，HumanEval 完成全量 164 个用例，均无缺失或不可读取 worker。所有题均获得五条候选路径，Base 对应采样五次。相同候选数量不意味着相同计算预算。

### 2.3 配置一致性

| 配置 | SQuAD | TriviaQA | HumanEval |
|---|---|---|---|
| Base 模型 | Qwen2.5-3B-Instruct | Qwen2.5-3B-Instruct | 命令指向相同模型 |
| checkpoint | all_diffs | all_diffs | 命令指向相同 checkpoint |
| 样本数 | 64 | 9960（全量） | 164（全量） |
| 候选数 | 5 | 5 | 5 |
| 选择策略 | nonbase_top1 | nonbase_top1 | nonbase_top1 |
| max_new_tokens | 2048 | 2048 | 2048（启动命令） |
| beam_size | 10 | 10 | 10（启动命令） |
| Base temperature / top_p | 0.7 / 0.95 | 0.7 / 0.95 | 0.7 / 0.95（启动命令） |
| workers_per_gpu | 8 | 8 | 8（启动命令） |
| predictor 输入 | question | question | 当前代码使用任务提示，需以实际版本核实 |

以结果 JSON 中的 configuration 为准，不以外层命令或输出目录名称推断配置。目录名中的 `magen50` 不代表本轮输出上限为 50，也不足以确认 MCTS 训练数据的生成上限。

本报告的 TriviaQA 数据已替换为全量结果；本次设置为 nonbase_top1、2048 输出上限、beam_size=10、每卡八个 worker、limit=0。此前 64 条 top1/1024/beam_size=5 和 64 条 nonbase_top1 结果不作为当前主结论。

TriviaQA 的“全量”指本地转换后输入文件的全部 9960 条记录，不据此认定与官方原始分割逐条一致。输入路径为 `/home/z00894149/QA/triviaqa_validation.jsonl`，结果目录为 `/home/z00894149/polar_dev/qa_all_triviaqa_sampling_vs_polar_k5_alldiff_magen50`，输入 SHA256 为 `5b47e4150f6a61e84d27807525303c7c94ad1ab69d9dd9c21b9b49d50f1e259a`。

HumanEval 已使用全量 164 题结果替换此前前 64 题的结果。启动命令显式 export 参数，与两组 QA 的主要参数一致；仍建议归档 manifest 核实最终生效配置和输入哈希。全量结果与此前子集结果不能仅作为“样本规模”单因素对照，除非确认逐题输入与执行配置一致。

## 3. 准确率与候选探索结果

### 3.1 总体结果

QA 使用当前框架的自定义归一化 Exact Match；HumanEval 使用代码测试通过率。跨数据集分数不应混合平均，也不能直接和官方榜单数字比较。

| 方法 | SQuAD | TriviaQA | HumanEval |
|---|---:|---:|---:|
| Base greedy | 51/64，79.69% | 3505/9960，35.19% | 112/164，68.29% |
| POLAR 原始 top-1 | 51/64，79.69% | 3505/9960，35.19% | 112/164，68.29% |
| 当前选择器（nonbase_top1） | **60/64，93.75%** | **3498/9960，35.12%** | **102/164，62.20%** |
| Base sampling best-of-5 oracle | 57/64，89.06% | 4181/9960，41.98% | **137/164，83.54%** |
| POLAR five-path oracle | **62/64，96.88%** | **4588/9960，46.06%** | 136/164，82.93% |
| POLAR oracle − Base sampling oracle | +7.81 个百分点 | +4.09 个百分点 | −0.61 个百分点 |

Base sampling 与 POLAR oracle 是固定五候选集合上的 best-of-5 成功率，不是生产选择器表现，也不是不加说明的标准 pass@k 估计。

### 3.2 QA 的 F1

| 方法 | SQuAD F1 | TriviaQA F1 |
|---|---:|---:|
| Base / POLAR top-1 | 90.48% | 41.97% |
| POLAR top-k oracle | 98.77% | 53.76% |

当前 oracle 的 EM 与 F1 分别取各候选最大值，可能对应不同候选，不能解释为同一个实际选中答案同时取得这两个分数。提供的 summary 未包含 non-base 选择器的 F1，不能用 oracle F1 代替。

## 4. 修复与退化分析

定义：

- 修复：Base 答错，当前选择器答对。
- 退化：Base 答对，当前选择器答错。
- 条件修复率 = 修复数 / Base 答错数。
- 条件退化率 = 退化数 / Base 答对数。
- 最终准确率 =（Base 答对数 + 修复数 − 退化数）/ 总题数。

| 指标 | SQuAD nonbase_top1 | TriviaQA nonbase_top1 | HumanEval nonbase_top1 |
|---|---:|---:|---:|
| Base 答错数 | 13 | 6455 | 52 |
| 修复数 | 11 | 491 | 12 |
| 条件修复率 | **84.62%** | 7.61% | 23.08% |
| Base 答对数 | 51 | 3505 | 112 |
| 退化数 | 2 | 498 | 22 |
| 条件退化率 | 3.92% | 14.21% | **19.64%** |
| 净正确题数变化 | +9 | −7 | −10 |
| 净准确率变化 | **+14.06 个百分点** | **−0.07 个百分点** | **−6.10 个百分点** |

### 4.1 SQuAD

非 Base 首选路径改善了大部分 Base 错题，同时仅破坏两道 Base 正确题。它证明本批样本上存在无需标准答案参与选择的实际提升，不仅是 oracle 提升。

但这是前 64 个用例上的探索性结果。如果样本按文章排列，多个问题可能共享上下文，64 道题不一定等于 64 个独立场景。策略又是在前期结果观察后提出的，应使用未参与策略设计的样本确认。

### 4.2 TriviaQA

全量 TriviaQA 上，nonbase_top1 修复 491 道 Base 错题，但破坏 498 道 Base 正确题，最终答对 3498/9960，较 Base 净损失 7 题（−0.070 个百分点）。这说明此前 64 条样本中净提升两题的现象不能代表全量表现。

从条件率看，TriviaQA 的 Base 错题中只有 7.61% 被 nonbase_top1 修复，而 Base 正确题中有 14.21% 被破坏。虽然总体净变化只有 −7/9960，几乎不改变总准确率，但对于生产部署来说，退化风险并没有被当前规则有效控制。

另一方面，POLAR 五路径 oracle 为 4588/9960，较 Base greedy 净多答对 1083 题，较 Base 五次采样 oracle 净多答对 407 题。这说明 POLAR 的路径候选集合在 TriviaQA 上包含额外正确答案，但 407 是总正确数差，不是 POLAR 独有成功题数，仍需逐题比较交集。当前 nonbase_top1 选择规则不能把这些 oracle 潜力转化成稳定收益。现有选择器距离 POLAR oracle 上限相差 1090 题；这不保证可由参考答案无关的验证器全部恢复。

### 4.3 HumanEval

在全量 164 题上，强制非 Base 路径修复 12 题，但破坏 22 题，最终正确数为 112 + 12 − 22 = 102，净损失 10 题。原始 top-1 与 Base 均答对 112 题。代码领域的候选生成有潜力，但现有排序不能可靠识别更好的非 Base 路径，无条件删除 Base 的策略不适合直接部署。

POLAR oracle 为 136/164，相比 Base greedy 多答对 24 题，但 Base sampling oracle 为 137/164，反而多于 POLAR 一题。因此代码场景没有观察到路径探索相对当前输出采样配置的准确率优势。一题差距不能证明两者统计等价或前者显著更差；还需多种子与逐题配对分析。两种候选集合可能答对不同题，需检查交集与互补性。

## 5. 计算开销

下表是各 worker 的逐题时间相加，不是八卡并行总墙钟时间，也不包含模型加载。

| 累计耗时，秒 | SQuAD | TriviaQA | HumanEval |
|---|---:|---:|---:|
| Base greedy | 35.01 | 4852.78 | 949.76 |
| predictor 与路径解码 | 36.91 | 945.06 | 64.23 |
| 五条 POLAR 路径生成 | 135.18 | 47308.40 | 4891.92 |
| Base 五次采样 | 91.01 | 25080.58 | 3965.35 |
| 规则选择 | 0.0023 | 0.1737 | 0.0046 |
| 学习型验证器 | 0，未实现 | 0，未实现 | 0，未实现 |
| 当前 selected pipeline | 207.10 | 53106.42 | 5905.91 |
| 完整实验逐题累计时间 | 298.92 | 78254.12 | 9965.75 |

解释：

1. selected pipeline 包含 Base greedy、predictor、全部 POLAR 候选生成与选择，不包含 Base 随机采样及离线判分。它不是优化后的单候选生产实现。
2. 完整实验时间还包括两条对照分支、trace 写入和参考答案/代码测试判分，但不包含最终结果写盘。
3. SQuAD 和全量 TriviaQA 均为每卡八个 worker，但仍存在 GPU 竞争、冷启动和输出长度差异。不能直接用这组时间代表单请求生产时延，也不能断言 predictor 开销恒定或可忽略。TriviaQA 完整实验平均逐题时间为 7.86 秒，包含两条对照分支及离线判分，不是优化后选择策略的响应时间。
4. nonbase_top1 只依赖路径顺序，可在答案生成前确定；后续应只执行选中路径，重新测部署时延。运行全部五个候选是当前分析实验的成本。
5. 全量 HumanEval 所有候选平均执行 39.1805 层、平均 unique depth 为 35.9902。相对 36 层 Base，平均执行次数约增加 8.83%，并没有显示明显的独特层数节省。所有候选的均值不代表最终选中路径的成本。完整实验平均逐题耗时为 60.77 秒，不是单次部署响应时间。
6. TriviaQA 五路径生成累计 47308.40 秒，明显高于五次 Base 采样的 25080.58 秒；加上 predictor/解码约为 48253.46 秒。HumanEval 五路径生成累计 4891.92 秒，高于五次 Base 采样的 3965.35 秒；加上 predictor/解码约为 4956.15 秒。但这仍是当前并发实验的 worker 时间统计，不能直接推断优化后的服务时延。

## 6. 当前证据的边界

- SQuAD 仍只有 64 个用例；TriviaQA 已覆盖全量 9960 题，HumanEval 已覆盖全量 164 题，但单一知识问答/代码基准仍不能代表整个 QA 或代码领域。
- 三组启动参数中的主要配置已统一，HumanEval 参数有显式 export，但尚未提供完整 manifest；应核实最终生效参数，不能将全部差异归因于领域。
- 只测一个随机种子、一个采样温度，尚未确认优于经过合理调参的 Base sampling。
- equal-k 不是 equal-compute：还需报告输出 token、执行层数、真实 wall-clock 与吞吐。
- QA 指标为自定义规则；答案抽取、参考别名和代码提取可能影响判分，需人工复核修复/退化题。
- 原始 top-1 等于 Base 尚未逐题核实，平均层数相同不能替代序列相等检查。
- 训练数据来自数学任务是本实验背景；基础模型预训练是否包含测试数据、是否存在数据污染，当前未核实。
- 循环路径 KV cache 的正确性应独立验证；当前结果沿用代码仓实现，不能以准确率提升反推缓存语义正确。
- 当前没有验证器，更没有保证“只提升不退化”的生产选择机制。

## 7. 下一步实验

### 第一阶段：统一配置与扩大测试

- 归档全量 HumanEval manifest、逐题结果与输入哈希，核实主要参数统一。
- 扩大 SQuAD 的未参与策略设计样本；在全量 TriviaQA 和 HumanEval 上增加采样种子重复实验，并扩展到其他 QA/代码基准。
- 明确样本选择与去重规则；SQuAD 按文章分组分析，必要时做分组抽样。
- 冻结 predictor 和选择策略；若训练目标领域选择器，须单独命名为“数学 predictor + 目标领域选择器”，不再称整个系统零样本泛化。

### 第二阶段：分析路径与答案

- 统计 top-1 完整序列等于 Base 的比例，以及 selected 的执行次数/unique depth。
- 输出 Base sampling 与 POLAR oracle 的逐题交集、各自独有成功题。
- 人工检查所有修复与退化题，排除格式抽取和判分误差。
- 对每题修复/退化构成做配对统计；增加样本和随机种子后报告不确定性，避免只比较总百分比。

### 第三阶段：可部署选择与效率

- 实现生成前 nonbase_top1 筛选，只执行最终选中路径，测真实单请求与并发时延。
- 保留 Base 为可选路径，而不是跨领域无条件剔除。
- 研究参考答案无关的质量预测/验证器及置信度回退，重点控制退化率。
- 以独立开发集选择阈值，在未参与调参的最终测试集报告准确率、修复/退化率与总开销。
- 构建等候选数和等计算预算两种对照，避免将更多计算误解释为更好的策略。

## 8. 汇报用结论

> 数学任务训练的 POLAR predictor 在 QA 和代码场景中能够生成有价值的非标准层执行候选，但候选生成能力与实际路径选择能力存在明显差距。64 条 SQuAD 探索实验中，非 Base 首选策略将准确率由 79.69% 提升至 93.75%；但在全量 9960 条 TriviaQA 上，同一策略由 35.19% 微降至 35.12%，修复 491 题、退化 498 题，基本抵消；在全量 164 题 HumanEval 上则由 68.29% 降至 62.20%，净损失 10 题。POLAR 五路径 oracle 在 TriviaQA 上为 46.06%，高于 Base 五次采样的 41.98%，说明路径候选集合存在额外正确答案；但 HumanEval 为 82.93%，略低于 Base sampling 的 83.54%，未观察到代码场景的候选准确率优势。现阶段证据支持进一步研究跨领域候选探索与质量感知选择，不支持直接宣称原始 top-1 泛化成功、统一删除 Base 路径或生产时延改善。

## 附录：全量 HumanEval 启动配置

以下按提供的命令整理；额外明确 `LIMIT=0`（全量）和 `MAX_TOTAL_WORKERS=0`（不加总 worker 上限），避免继承旧环境设置。代码执行必须在隔离环境进行，现有 subprocess 执行器不是安全沙箱。

```bash
cd /home/z00894149/polar_dev/PoLar
export PYTHONPATH="/home/z00894149/polar_dev/PoLar:${PYTHONPATH:-}"
export GPUS="0 1 2 3 4 5 6 7"
export WORKERS_PER_GPU=8
export MAX_TOTAL_WORKERS=0
export K=5
export BEAM_SIZE=10
export TEMPERATURE=0.7
export TOP_P=0.95
export MAX_NEW_TOKENS=2048
export SELECTION_STRATEGY=nonbase_top1
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ALLOW_CODE_EXECUTION=1

bash scripts/run_sampling_vs_polar.sh --limit 0 code \
  /home/z00894149/Qwen2.5-3B-Instruct \
  /home/z00894149/polar_policy_epochs20_bs16_lr0.0001_amp_mpps50_seed42_all_diffs.pt \
  /home/z00894149/polar_dev/human_eval.json \
  /home/z00894149/polar_dev/code_all_sampling_vs_polar_k5_alldiff_magen50
```

原始提供的命令未显式设置 `ALLOW_CODE_EXECUTION`，实际成功运行可能继承了该环境变量。上述命令是整理后的复现实例，不是已核验的原始 manifest；复跑已有结果时应使用新的输出目录，或保留原配置按既有 resume 机制继续。

# PPTX 内容提取

来源：/Users/stevendongyu/Downloads/Layer-Transformer.pptx

按文件对象顺序提取文字；图片保留原始字节。该结果不表示页面的实际排版或阅读顺序。

## 第 1 页

### 正文

AI INFRA & SYSTEMS ARCHITECTURE
动态 Layer Transformer 的高性能服务化推理挑战
POLAR · Loop Transformer · Skip-layer · Mixture-of-Depth 打破了固定层序 Serving 基础假设
EXEC_STATUS // BREAKING_CHANGES
PARADIGM: DYNAMIC_TOPOLOGY
IMPACT: BATCH / KV / SLA COLLAPSE
01 · 传统固定层 SERVING 假设
固定算子图：整齐竖向拓扑
同步执行：Batch 内同层锁步推进
静态存储：KV Cache 按 Layer ID 直排
简单度量：调度仅看 Token 长度与内存
基石: CONTINUOUS BATCHING · PAGED ATTENTION · CUDA GRAPH
02 · 动态 EXECUTION 打破假设
核心冲突：算法精度收益来自动态路径选择，但在工程侧直接打散 Batch 同步、破坏 KV Cache 层序语义，并让单个 Token 计算开销与时延不可预测。
03 · 四个核心系统挑战
1. Path-Aware Batching
请求路径严重离散，按 Layer、Segment 或 Prefix 聚合组 Batch 难度大，硬件算力利用率暴跌。
2. Execution-Aware KV Cache
Repeat Layer 第二次计算时 Hidden State 已改变，无法以静态 Layer ID 寻址，需按 Execution Step 路由。
3. Cost-Aware Scheduling
各请求总 Layer 数差异显著，长尾效应剧烈，极难同时兼顾系统吞吐、P99 Latency 与生产级 SLA。
4. Top-k Candidate Serving
多路径候选虽提升准确率但乘倍放大开销，急需 Path 共享复用、Early-stop 及轻量级 Verifier 折中。
[MISSION] 核心挑战定义：             构建软硬协同运行时，让动态层执行 Transformer 具备与固定层 Transformer 同等高效、稳定与可预测的服务化吞吐与延迟表现。

### 备注

向 AI Infra 负责人系统汇报：阐述动态层模型（如 POLAR、MoD 等）在算法精度收益背后的服务化代价。通过左中右架构逻辑，直观展示为何既有 Continuous Batching 和静态 KV Cache 体系在面对动态路径时全面失效，并点明四大关键攻坚方向与核心使命。

### 原始图片

- [ppt/media/image-1-1.png](media/b081666c55aee44cd3dfd549b7205816862600ca58c7f5ce53bc7b21c81d4e59.png)（来源：slide）
- [ppt/media/image-1-2.png](media/90df0b0b7b7caaab708706312a2f8aa544c0a4c71f157f36d382106b6f26a9f8.png)（来源：slide）
- [ppt/media/image-1002-1.png](media/591accd6ecdb20315c1ce0017f70029388994ee11bc6fba05a1a53441c6c0240.png)（来源：layout）

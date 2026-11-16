# 单卡 Base / PoLar 对齐实测

## KV Cache 模式

修改版 vLLM 核心现在支持 `VLLM_POLAR_KV_CACHE_MODE=execution|layer`，默认 execution：每个路径位置独立 KV。layer：按物理层共享 KV，第二次进入覆盖当前 scheduled token 位置的 KV，不清空历史。layer 模式不申请 loop 的额外 execution slots，且同物理层的不同路径位置可合批（首次执行与后续执行仍分开，保证 residual 语义）。不依赖旧静态插件的 `POLAR_KV_CACHE_MODE`。

调试命令：

```bash
bash scripts/debug_vllm_paths.sh data/path_batching/debug_paths_example.json \
  --kv-cache-mode layer --concurrency 3 --max-tokens 8
```

全流程吞吐测试：

```bash
KV_CACHE_MODE=layer GPU_ID=0 CONCURRENCIES="8" \
OUTPUT_DIR=/home/z00894149/polar_dev/vllm_aligned_layer_c8 \
bash scripts/run_vllm_aligned_benchmark.sh
```

execution 对照使用 `KV_CACHE_MODE=execution` 和另一个新目录。恢复时必须保持 KV 模式相同；此前未记录模式的配置按 execution 处理。需要同步两个仓库的 Python 源码并重启服务，不涉及 CUDA 重编译。

覆盖模式的历史 KV 是上次最后一次访问物理层留下的值；prefill 内先写整个 chunk 再读 attention，结果可能随 chunk 边界改变。保留现有 chunked-prefill 设置以便与已有实验对照，不宣称等价于独立缓存或先前 HF 后端。修改缓存模式同时改变模型语义、KV 内存和可合批机会，因此吞吐差值不是纯调度收益。须在服务器验证无 loop 路径两模式输出、loop 路径各调度策略输出与精度；本地仅验证了 CPU gather/scatter、slot 选择分组和覆盖模拟，未运行真实 CUDA attention。

在服务器已安装修改版 vLLM 的 `zdy_vllm_polar` 环境运行。脚本不安装软件、不编译源码，也不会连接远端机器代跑。先同步 PoLar 的两个新脚本和原有 `data/path_batching/synthetic_paths_128.json`，并确保服务器 vLLM 含已有 PoLar 适配代码。

```bash
conda activate zdy_vllm_polar
cd /home/z00894149/polar_dev/PoLar
GPU_ID=0 bash scripts/run_vllm_aligned_benchmark.sh
```

默认顺序：原生 Qwen2 Base、serial、identical_path、dynamic_layer。每种服务加载一次，分别扫描并发 `1 2 4 8 16 32 64 96`；每个并发先进行 20 个完整长度请求预热，再实测 512 个请求。输入严格 512 token，输出严格 256 token，temperature=0、ignore_eos=true。默认整套耗时可能很长，建议先完成 C=8 对照：

```bash
GPU_ID=0 CONCURRENCIES="8" \
OUTPUT_DIR=/home/z00894149/polar_dev/vllm_aligned_c8 \
bash scripts/run_vllm_aligned_benchmark.sh
```

可配置环境变量：`VLLM_ROOT`、`MODEL_PATH`、`PATHS_JSON`、`OUTPUT_DIR`、`GPU_ID`、`PORT`、`INPUT_LEN`、`OUTPUT_LEN`、`NUM_REQUESTS`、`NUM_WARMUPS`、`CONCURRENCIES`、`REPEATS`、`SEED`、`GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN`、`MAX_NUM_BATCHED_TOKENS`、`MAX_NUM_SEQS`、`PYTHON`。默认 vLLM 和模型路径分别是 `/home/z00894149/polar_dev/vllm` 和 `/home/z00894149/Qwen2.5-3B-Instruct`。修改这些设置会改变对照条件。

所有服务统一 V1 runner、eager、FlashAttention、BF16、单卡、prefix cache 关闭、相同调度 token/sequence 预算。脚本要求端口空闲；只清理自己启动的服务进程组，Ctrl+C 会保留已完成结果。新实验目录必须不存在；恢复时显式使用 `RESUME=1`。

## 断点恢复

同步更新后的两个脚本，指定原来的目录和相同参数：

```bash
conda activate zdy_vllm_polar
cd /home/z00894149/polar_dev/PoLar
RESUME=1 GPU_ID=0 CONCURRENCIES="8" \
OUTPUT_DIR=/home/z00894149/polar_dev/vllm_aligned_c8 \
bash scripts/run_vllm_aligned_benchmark.sh
```

恢复单位是 `repeat × 策略 × 并发` 的完整轮次。完整的 summary、逐请求记录、token 数和服务日志均通过检查才跳过；兼容本脚本此前版本生成的完整轮次。中断或失败的轮次归档为 `.interrupted_*`，重新预热后整轮测试，不拼接旧吞吐。已有结果会重新纳入 comparison/report。已完成整个策略时不启动该策略服务。

恢复校验模型路径、GPU 编号、长度/请求数/并发/重复/seed/预算、数据哈希、vLLM commit/dirty 状态和 Python 包列表；不匹配时拒绝合并。请保持模型权重不变（脚本未计算大体积权重哈希）。原配置保留，恢复环境写入 `resume_*.json`。同目录加文件锁，防止两个脚本同时修改结果。SIGKILL/机器掉电不能保证服务自动清理；重新执行前确认原服务已停止、端口已释放。恢复不支持自动接管遗留服务。

服务子进程显式设置 `VLLM_PLUGINS=""`，禁用外部插件自动注册。此前安装的 `polar_vllm` 静态路径插件会覆盖原生 Qwen2，并要求 `POLAR_PATH_JSON`；它不参与本次混合路径实验。这里使用修改版 vLLM 自带的模型注册，不需要卸载旧插件。

## 输出

- `config.json`：完整实验参数、源码 commit/dirty 状态、包版本、GPU 信息、输入哈希及指标定义。
- `prompts.jsonl`、`path_manifest.json`：相同输入与混合路径映射；默认把原有 128 行路径重复四次后固定种子打乱，生成 512 个唯一随机 token 输入。
- `repeat_N/<mode>/launch.json`、`server.log`：真实启动命令与服务日志。
- `repeat_N/<mode>/cC/results.jsonl`、`summary.json`：逐请求结果、流式延迟和实际成功 token 计数。预热单独存放，不计入吞吐。
- `comparison.json`、`report.md`：逐组吞吐/延迟，以及不同策略、并发、重复之间的输出文本一致性检查。

输出吞吐 = 实际输出 token / 整轮 wall time。输入吞吐也使用整轮 wall time，不是纯 prefill 性能。TTFT 是首个非空文本 SSE chunk 的到达时间；TPOT 用首末文本 chunk 时间差除以输出 token 数减一。网络可能合并 token，因此不报告精确逐 token ITL。E2E 包含接收完整流的时间。并发表示客户端同时在途请求上限，不等于每个层的 batch size。

该实验复现历史报告的请求数和 token 长度，但使用固定随机种子的新 token 数据、统一新版本服务参数和新流式客户端；历史报告没有完整记录服务端参数，不能称为对旧结果的逐项复现。严谨对照使用本次重测 Base。旧报告仅作参考。

原生 Base 执行固定完整层序，混合路径的执行工作量和 KV 占用不同，因此它是服务能力参照，不是理论上界。serial 是 forward 内串行执行层子批次，不代表全请求严格逐个执行。三种策略结果也不构成数学上下界。随机路径不评价模型精度；文本一致性异常需要排查，文本一致也不证明 KV 语义正确。当前动态执行原型含 eager Python 调度和按执行位置拆批开销。

OOM、请求失败或未观察到 PoLar hook 时脚本停止并保留日志，不会降低并发后偷偷继续。显存不足时可以明确修改预算后新建实验，但修改后的配置需重新建立 Base 对照。

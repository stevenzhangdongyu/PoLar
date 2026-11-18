# PoLar 路径回放 profiling

同步 PoLar 新脚本和 vLLM 的 polar_profile.py、polar_batch_ops.py、polar_layer_scheduler.py 后运行。不依赖 predictor；每条 path 生成唯一 token 输入，通过 manifest 匹配到实际请求。默认读取原 128-path JSON，每条执行一次，C=8、输入512 token、输出256 token；不同于吞吐基线的512请求。若指定其他 path 文件，则按其实际条数执行，不自动重复或截断。

```bash
cd /home/z00894149/polar_dev/PoLar
PYTHON=/root/miniconda3/envs/zdy_vllm_polar/bin/python \
KV_CACHE_MODE=execution GPU_ID=0 \
bash scripts/profile_polar_paths.sh \
  data/path_batching/synthetic_paths_128.json \
  --output /home/z00894149/polar_dev/polar_profile_execution
```

默认完成20个短输出预热请求，随后开启 worker CPU/CUDA profiler，再跑全部路径并停止采集、等待落盘。完整采集可能产生很大的 trace；建议首次使用 `PROFILE_ITERATIONS=8` 限定为最初若干 engine forward，或添加 `--max-tokens 16` 缩短全量运行。限制 iterations 时所有请求仍执行完，但 trace 只覆盖开始的窗口，不能用于全128路径占比统计。8并发是客户端在途请求上限，不保证每次forward都有8条。

覆盖模式设置 `KV_CACHE_MODE=layer` 并使用另一个新输出目录。其他参数：`STRATEGY=serial|identical_path|dynamic_layer`，以及 `--port`、`--input-len`、`--max-tokens`。未提供CUDA GPU实测，服务器需确保CUPTI可用；仅靠CPU trace不能判断GPU瓶颈。

输出：`launch.json`、`prompts.jsonl`、`path_manifest.json`、`warmup.json`、`results.jsonl`、`server.log`、`traces/*.pt.trace.json.gz`。停止 profiler 可能较慢，请等待落盘，不要直接kill。将 worker trace 复制到本机，用 https://ui.perfetto.dev/ 打开。

标注（需同步修改版vLLM）包括：`polar/plan/*` 调度计划生成；`polar/execute_plan` 整体执行；`polar/gather`、`polar/scatter` 激活搬运；`polar/attention_metadata` attention元数据；`polar/layer_N/requests_K` 物理层及实际子批次数。层区间内包括metadata准备和完整decoder计算，区间嵌套不可直接相加；查看self CPU、关联CUDA kernel和GPU时间线，CPU区间长度不等于CUDA执行时间。Attention和MLP/Linear通过其内部算子/kernel辨识。

禁用现有断点和大量print；脚本自动关闭 POLAR_DEBUG/POLAR_DEBUG_BATCH 两个既有开关。profiler会扰动性能，不能把此次时延/吞吐直接与无profiling基线比较。若需前端或主进程scheduler时间，应另采集对应进程；当前主要针对EngineCore路径调度及GPU worker。

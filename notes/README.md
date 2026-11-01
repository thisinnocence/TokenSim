# TokenSim 快速启动与架构学习

本文使用仓库 README 推荐的 Llama-7B Quick start，在 CPU 上运行 serving simulation。
TokenSim 根据模型、硬件、负载及服务配置预测性能，无需下载 weights 或运行真实模型推理。
主要输入如下（部分举例）：

- Model Geometry（模型结构参数）
- Device Capabilities（硬件能力参数）
- Request Lengths（请求的输入与输出 token 数）
- Workload（负载配置）：请求数量、到达速率和分布。
- Serving Configuration（服务配置）：batching、调度、并行方式与 KV cache 策略。
- Latency Backend（延迟估算配置）：analytical formulas 或 measured operator tables。

主要输出如下（部分举例）：

- TTFT（首 token 延迟）：从请求到达到生成首个 output token 的时间。
- TPOT（后续 token 平均生成间隔）：反映单个请求在 decode 阶段的生成速度。
- Throughput（整体吞吐）：每个模拟秒完成的请求数和生成的 output token 数。
- Request Latency（端到端请求延迟）与完成状态：请求从到达到全部输出完成的时间，以及是否存在未完成或失败的请求。
- 调度与缓存统计：prefill 分块、抢占重算、KV 容量与复用，以及延迟估算来源。

TokenSim 整理逻辑架构分析的入口为 [architecture.md](architecture.md).

## 环境搭建和运行

在 repo root 执行以下命令，使用 uv 创建 Python 环境, `uv pip` 安装依赖，具体操作：

```bash
bash notes/scripts/setup.sh
source .venv/bin/activate
python --version
tokensim --help
```

运行 README 默认示例:

```bash
bash notes/scripts/run-example.sh
.venv/bin/python notes/scripts/inspect-result.py \
  results/example/result_1.0.json
```

运行脚本直接调用 `.venv/bin/tokensim`，无需激活环境，参数与 README 的基础 Quick start 一致，具体操作：

```bash
tokensim --latency_backend analytical \
  --request_count 100 \
  --results_path results/example
```

其中未显式指定的配置使用仓库默认值。

| 配置 | 默认值 | 来源 | 备注 |
| --- | --- | --- | --- |
| Model | Llama-7B | [llama-7b.yaml](../data/models/llama-7b.yaml) | 定义模型结构与精度，用于估算计算量和存储需求 |
| Cluster | 单个 A100-40G hybrid worker | [h1.json](../data/clusters/1_a100/h1.json) | 定义硬件与部署；同一 worker 执行 prefill 和 decode |
| Batching | `dynamic` | [CLI arguments](../TokenSim/cli/arguments.py) | 决定请求如何组成 batch；完成的请求腾出位置后补入等待请求 |
| Arrival process | `uniform`，1 query per second（QPS） | [WorkloadConfig](../TokenSim/config/workload_config.py) | 控制请求到达间隔与负载，本例每秒到达一个请求 |
| Request lengths | 512 prompt tokens、128 output tokens | `WorkloadConfig` | 决定 prefill 和 decode 工作量，以及 KV cache 需求 |
| Random seed | 0 | CLI arguments | 固定随机采样，使相同配置的仿真可复现 |

本次 100 个 requests 全部完成，simulated time 约 100.57 秒，`simulator_wall_time` 约 0.38 秒。

- Simulated time（模拟时间）：模拟系统内请求到达、排队和执行所经历的虚拟时间，本例约 100.57 秒，用于计算预测 throughput。
- `simulator_wall_time`（实际执行耗时）：host 执行仿真循环所花的真实时间，本例约 0.38 秒，不包含初始化，也不是整条命令的耗时。

输出 throughput（输出吞吐）约为 127.28 tokens/s，表示整个模拟系统平均每秒输出多少个 tokens。
这个指标按“全部请求生成的 output tokens 总数 ÷ simulated time”计算，统计的是整批请求，不是单个请求的生成速度。

本例的输入和统计过程如下：

- `1 QPS` 表示每个模拟秒送入 1 个新请求，即第 0 秒送入第 1 个、第 1 秒送入第 2 个，直到第 99 秒送入第 100 个。
- 每个请求要求生成 128 个 output tokens，因此 100 个请求总共生成 `100 × 128 = 12,800` 个 output tokens。
- 请求会在系统内排队或并发执行，不要求某个请求在下一请求到达前完成。
- 最后一个请求完成时，模拟时间约为 100.57 秒，因此整体输出吞吐为 `12,800 ÷ 100.57 ≈ 127.28 tokens/s`。

所谓『受输入速率影响』，是指输入端每秒只增加 1 个请求，也就只增加 128 个 output tokens 的生成任务。
只要系统能持续跟上这个输入速率、不让等待队列越来越长，长期平均输出就会接近 `1 × 128 = 128 tokens/s`。
即使模拟硬件还有余力，输入端也没有提供更多请求供它处理，所以这个数值不能说明硬件最多能输出多少 tokens/s。
要观察处理能力上限，需要逐步提高 `--qps`，对比 output throughput 是否继续增长，以及 request latency 和排队是否明显增加。

原始 result JSON 位于 Git 忽略的 `results/` 下。
成功运行时，`termination_reason` 为 `completed`，`notdone` 为空。
[run-summary.json](run-summary.json) 是从本次原始结果中选取字段形成的精简归档，并非 TokenSim 的原生输出格式。

## 验证

- [architecture.md](architecture.md) 说明 request source、engine、worker、scheduler、latency backend 和 result export 如何连接。
- [inspect-result.py](scripts/inspect-result.py) 读取原生 result JSON，提取完成状态、latency provenance 和 scheduler 统计。
- [run-example.sh](scripts/run-example.sh) 仅封装默认 Quick start，便于追加参数开展实验。

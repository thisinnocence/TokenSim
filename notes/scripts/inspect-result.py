"""Print completion, latency provenance and scheduler evidence from a result JSON."""

# 本脚本用于查看和解析 TokenSim 已生成的原生 result JSON 输出。
# 用法：.venv/bin/python notes/scripts/inspect-result.py results/example/result_1.0.json
# 输入应为单次运行的结果对象，不是 notes/run-summary.json 这样的多次运行归档。
# 脚本按 FIELDS 选取关键字段，将格式化后的 JSON 打印到标准输出；缺失的字段会跳过。
# 完成状态包括 termination_reason 和未完成请求列表 notdone。
# 性能字段包括模拟时间、实际执行耗时、请求与 token 吞吐，以及请求、prefill 和 decode 延迟。
# 延迟来源字段用于判断估算使用了公式、插值还是外推，并查看缺失的 operator shapes 数量。
# 调度字段用于观察 prefill 分块、抢占、重算及 scheduler 统计。
# 文件读取失败、JSON 无法解析或缺少 termination_reason 时，打印错误并以状态码 1 退出。
# 结果成功打印后，若运行状态不是 completed，或 notdone 非空，也会以状态码 1 退出。
# 正常完成且没有未完成请求时，以状态码 0 退出，便于在 shell 中检查运行结果。

import argparse
import json
from pathlib import Path

FIELDS = (
    "termination_reason",
    "failure",
    "request_count",
    "notdone",
    "simulated_time",
    "simulator_wall_time",
    "output_qps",
    "output_token_ps",
    "request_time",
    "prefill_time",
    "decode_time",
    "prefill_chunk_count",
    "preemption_count",
    "recomputation_count",
    "latency_backends",
    "operator_query_count",
    "operator_match_type_counts",
    "operator_missing_shape_count",
    "scheduler_stats",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="TokenSim result JSON path")
    args = parser.parse_args()
    try:
        result = json.loads(args.result.read_text())
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot read result: {error}\n")
    if not isinstance(result, dict) or "termination_reason" not in result:
        parser.exit(1, "Expected a TokenSim result object with termination_reason.\n")
    print(json.dumps({key: result[key] for key in FIELDS if key in result}, indent=2))
    if result.get("termination_reason") != "completed" or result.get("notdone"):
        parser.exit(1, "Simulation did not complete every request.\n")


if __name__ == "__main__":
    main()

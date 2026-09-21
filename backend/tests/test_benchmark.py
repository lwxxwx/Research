"""
tests/test_benchmark.py
Phase G · Rule-Only Benchmark 入口 (v2)

执行（宿主机 powershell）:
    docker compose `
        -f infra/docker/docker-compose.yml `
        -f infra/docker/docker-compose.dev.yml `
        exec backend uv run python -m tests.test_benchmark

输出: out/bench_ruleonly_sprint0.csv
"""
import os
import sys

# 支持 `python -m tests.test_benchmark` 与 `python tests/test_benchmark.py` 两种调用
sys.path.insert(0, "/app")

from app.services.benchmark_service import (
    run_single_case,
    write_csv,
    print_console_summary,
    BENCH_OUT_DIR,
)


def main() -> None:
    # Sprint0 基线只跑 case001；Sprint1 扩至 10 case 时在此追加路径
    case_list = [
        "/data/cases/case001",
    ]

    results = []
    for case_path in case_list:
        print(f"\n>>> Running benchmark for {case_path}")
        res = run_single_case(case_path)
        print_console_summary(res)
        results.append(res)

    out_csv = os.path.join(BENCH_OUT_DIR, "bench_ruleonly_sprint0.csv")
    write_csv(results, out_csv)
    print(f"\n✅ Benchmark finished. Output: {out_csv}")


if __name__ == "__main__":
    main()
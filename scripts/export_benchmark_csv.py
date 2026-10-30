# backend/scripts/export_benchmark_csv.py
"""导出 Benchmark 结果到 CSV。"""
import argparse
import sys
from pathlib import Path

from app.services.benchmark_service import (
    MODE_FULL_REVIEW,
    MODE_RULE_ONLY,
    run_all_benchmark,
    write_csv,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Export benchmark results to CSV")
    parser.add_argument("--mode", choices=[MODE_RULE_ONLY, MODE_FULL_REVIEW], default=MODE_RULE_ONLY)
    parser.add_argument("--cases", nargs="+", required=True, help="Case IDs to run")
    parser.add_argument("--out", default="/out/benchmark_result.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Running benchmark mode={args.mode} cases={args.cases} ...")
    results = run_all_benchmark(args.cases, args.mode)
    
    # 提前建目录（write_csv 内部也会建，但这里显式确保）
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    write_csv(results, args.out)
    print(f"✅ CSV exported to {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
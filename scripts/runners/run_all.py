from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))

from cache_npu_scheduling.cases import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
from cache_npu_scheduling.problem1_scheduler import run as run_problem1
from cache_npu_scheduling.problem2_allocator import run as run_problem2
from cache_npu_scheduling.problem3_pipeline import run as run_problem3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all scheduling stages.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    args = parser.parse_args()
    cases = parse_cases(args.cases)

    print("== Problem 1 ==")
    for result in run_problem1(cases=cases, data_dir=args.data_dir, output_dir=args.output_dir):
        print(f"{result.case}: nodes={len(result.schedule)} peak_ub_l1={result.peak_ub_l1}")

    print("== Problem 2 ==")
    for result in run_problem2(cases=cases, problem="problem2", data_dir=args.data_dir, output_dir=args.output_dir):
        print(f"{result.case}: spill_cost={result.total_spill_cost}")

    print("== Problem 3 ==")
    for result in run_problem3(cases=cases, data_dir=args.data_dir, output_dir=args.output_dir):
        print(f"{result.case}: cycles={result.original_cycles}->{result.compressed_cycles}")


if __name__ == "__main__":
    main()

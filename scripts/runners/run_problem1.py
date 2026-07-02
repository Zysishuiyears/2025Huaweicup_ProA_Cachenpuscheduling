from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cache_npu_scheduling.cases import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
from cache_npu_scheduling.problem1_scheduler import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Problem 1 heuristic scheduling.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    args = parser.parse_args()

    results = run(cases=parse_cases(args.cases), data_dir=args.data_dir, output_dir=args.output_dir)
    for result in results:
        print(f"{result.case}: nodes={len(result.schedule)} peak_ub_l1={result.peak_ub_l1} -> {result.output_path}")


if __name__ == "__main__":
    main()

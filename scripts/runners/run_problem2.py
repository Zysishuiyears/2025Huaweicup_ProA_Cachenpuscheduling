from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))

from cache_npu_scheduling.cases import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
from cache_npu_scheduling.problem2_allocator import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Problem 2 spill-aware allocation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    args = parser.parse_args()

    results = run(cases=parse_cases(args.cases), problem="problem2", data_dir=args.data_dir, output_dir=args.output_dir)
    for result in results:
        print(
            f"{result.case}: nodes={len(result.schedule)} spill_cost={result.total_spill_cost} "
            f"cycles={result.original_cycles}->{result.compressed_cycles} -> {result.output_dir}"
        )


if __name__ == "__main__":
    main()

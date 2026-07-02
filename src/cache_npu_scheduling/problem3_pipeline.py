"""Problem 3 ASAP-style pipeline compression entrypoint.

The final competition artifact used the same spill-aware schedule and memory
outputs for Problems 2 and 3, while computing an ASAP-style conservative
left-slide timing comparison. This module keeps that behavior but writes into
the `problem3` output namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cache_npu_scheduling.cases import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
    from cache_npu_scheduling.problem2_allocator import AllocationResult, run as run_allocator
else:
    from .cases import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
    from .problem2_allocator import AllocationResult, run as run_allocator


def run(
    cases: Iterable[str] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[AllocationResult]:
    return run_allocator(cases=cases, problem="problem3", data_dir=data_dir, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Problem 3 ASAP-style pipeline compression artifact.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    args = parser.parse_args()

    results = run(cases=parse_cases(args.cases), data_dir=args.data_dir, output_dir=args.output_dir)
    for result in results:
        print(
            f"{result.case}: nodes={len(result.schedule)} spill_cost={result.total_spill_cost} "
            f"cycles={result.original_cycles}->{result.compressed_cycles} -> {result.output_dir}"
        )


if __name__ == "__main__":
    main()

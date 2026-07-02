from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core"))

from cache_npu_scheduling.cases import CASES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
from cache_npu_scheduling.submission import DEFAULT_SUBMISSION_READY_DIR, run_and_export_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cleaned code and export competition-style submission files."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUBMISSION_READY_DIR)
    parser.add_argument("--package-name", default="A25100550012")
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    parser.add_argument("--no-code", action="store_true", help="Do not include cleaned source code in the package.")
    parser.add_argument("--no-zip", action="store_true", help="Only write the folder tree, not the zip archive.")
    args = parser.parse_args()

    selected_cases = parse_cases(args.cases)
    if set(selected_cases) != set(CASES):
        print(
            "Warning: exporting a partial submission tree. Full competition submission requires all official cases.",
            file=sys.stderr,
        )

    result = run_and_export_submission(
        cases=selected_cases,
        data_dir=args.data_dir,
        run_output_dir=args.run_output_dir,
        output_dir=args.output_dir,
        package_name=args.package_name,
        include_code=not args.no_code,
        make_zip=not args.no_zip,
    )
    print(f"Submission tree: {result.package_dir}")
    if not args.no_zip:
        print(f"Submission zip:  {result.archive_path}")
    print(f"Files copied:    {len(result.copied_files)}")


if __name__ == "__main__":
    main()

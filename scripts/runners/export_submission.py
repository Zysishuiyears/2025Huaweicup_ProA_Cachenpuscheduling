from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cache_npu_scheduling.cases import parse_cases
from cache_npu_scheduling.submission import DEFAULT_SUBMISSION_READY_DIR, export_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export competition-style submission attachment files from existing outputs."
    )
    parser.add_argument(
        "--source",
        default="canonical",
        help="canonical, reconstructed, or a path containing problem1/problem2/problem3 outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUBMISSION_READY_DIR)
    parser.add_argument("--package-name", default="A25100550012")
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    parser.add_argument("--no-code", action="store_true", help="Do not include cleaned source code in the package.")
    parser.add_argument("--no-zip", action="store_true", help="Only write the folder tree, not the zip archive.")
    args = parser.parse_args()

    result = export_submission(
        source=args.source,
        output_dir=args.output_dir,
        package_name=args.package_name,
        include_code=not args.no_code,
        make_zip=not args.no_zip,
        cases=parse_cases(args.cases),
    )
    print(f"Submission tree: {result.package_dir}")
    if not args.no_zip:
        print(f"Submission zip:  {result.archive_path}")
    print(f"Files copied:    {len(result.copied_files)}")


if __name__ == "__main__":
    main()

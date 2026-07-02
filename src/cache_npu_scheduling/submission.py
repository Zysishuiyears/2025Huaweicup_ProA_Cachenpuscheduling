"""Build a competition-style submission attachment tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import zipfile

from .cases import CASES, DEFAULT_OUTPUT_DIR, REPO_ROOT
from .problem1_scheduler import run as run_problem1
from .problem2_allocator import run as run_problem2
from .problem3_pipeline import run as run_problem3


CANONICAL_SUBMISSION_DIR = REPO_ROOT / "outputs" / "submission"
DEFAULT_SUBMISSION_READY_DIR = REPO_ROOT / "outputs" / "submission_ready"


@dataclass(frozen=True)
class SubmissionExportResult:
    package_dir: Path
    archive_path: Path
    copied_files: list[Path]


def export_submission(
    source: str = "canonical",
    output_dir: Path = DEFAULT_SUBMISSION_READY_DIR,
    package_name: str = "A25100550012",
    include_code: bool = True,
    make_zip: bool = True,
    cases: list[str] | None = None,
) -> SubmissionExportResult:
    """Export results using the original competition attachment layout.

    `source="canonical"` uses the official submitted results preserved under
    outputs/submission. `source="reconstructed"` uses outputs/reconstructed.
    The default is intentionally canonical so the exported package aligns with
    the final competition attachment baseline.
    """

    source_root = _resolve_source_root(source)
    selected_cases = cases or CASES
    package_dir = output_dir / package_name
    attachment_dir = package_dir / "Attachment"
    copied_files: list[Path] = []

    if package_dir.exists():
        shutil.rmtree(package_dir)
    attachment_dir.mkdir(parents=True, exist_ok=True)

    copied_files.extend(_copy_problem1(source_root, attachment_dir / "Problem1", selected_cases))
    copied_files.extend(_copy_problem23(source_root, "problem2", "Q2", attachment_dir / "Problem2", selected_cases))
    copied_files.extend(_copy_problem23(source_root, "problem3", "Q3", attachment_dir / "Problem3", selected_cases))

    if include_code:
        copied_files.extend(_copy_code(package_dir / "code"))

    validate_submission_tree(package_dir, include_code=include_code, cases=selected_cases)
    archive_path = output_dir / f"{package_name}_submission_ready.zip"
    if make_zip:
        _write_zip(package_dir, archive_path)
    return SubmissionExportResult(package_dir=package_dir, archive_path=archive_path, copied_files=copied_files)


def run_and_export_submission(
    cases: list[str] | None = None,
    data_dir: Path | None = None,
    run_output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = DEFAULT_SUBMISSION_READY_DIR,
    package_name: str = "A25100550012",
    include_code: bool = True,
    make_zip: bool = True,
) -> SubmissionExportResult:
    """Run the cleaned code, then export a competition-style attachment tree.

    This is the "code-generated submission" path. It writes intermediate
    problem outputs under `run_output_dir`, then converts them into the required
    `Attachment/Problem*/Q*_...txt` layout.
    """

    selected_cases = cases or CASES
    data_root = data_dir or (REPO_ROOT / "data" / "raw" / "csv")
    run_problem1(cases=selected_cases, data_dir=data_root, output_dir=run_output_dir)
    run_problem2(cases=selected_cases, problem="problem2", data_dir=data_root, output_dir=run_output_dir)
    run_problem3(cases=selected_cases, data_dir=data_root, output_dir=run_output_dir)
    return export_submission(
        source=str(run_output_dir),
        output_dir=output_dir,
        package_name=package_name,
        include_code=include_code,
        make_zip=make_zip,
        cases=selected_cases,
    )


def _resolve_source_root(source: str) -> Path:
    if source == "canonical":
        return CANONICAL_SUBMISSION_DIR
    if source == "reconstructed":
        return DEFAULT_OUTPUT_DIR
    source_path = Path(source)
    if source_path.exists():
        return source_path
    raise ValueError("source must be 'canonical', 'reconstructed', or an existing path")


def _copy_problem1(source_root: Path, target_dir: Path, cases: list[str]) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for case in cases:
        src = source_root / "problem1" / f"{case}_schedule.txt"
        dst = target_dir / f"Q1_{case}_schedule.txt"
        _copy_required(src, dst)
        copied.append(dst)
    return copied


def _copy_problem23(source_root: Path, problem: str, prefix: str, target_dir: Path, cases: list[str]) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for case in cases:
        for suffix in ("schedule", "memory", "spill"):
            src = source_root / problem / f"{case}_{suffix}.txt"
            dst = target_dir / f"{prefix}_{case}_{suffix}.txt"
            _copy_required(src, dst)
            copied.append(dst)
    return copied


def _copy_code(target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    src_dir = REPO_ROOT / "src" / "cache_npu_scheduling"
    scripts_dir = REPO_ROOT / "scripts"
    files = [
        src_dir / "cases.py",
        src_dir / "problem1_scheduler.py",
        src_dir / "problem2_allocator.py",
        src_dir / "problem3_pipeline.py",
        scripts_dir / "run_problem1.py",
        scripts_dir / "run_problem2.py",
        scripts_dir / "run_problem3.py",
    ]
    copied: list[Path] = []
    for src in files:
        dst = target_dir / src.name
        _copy_required(src, dst)
        copied.append(dst)
    return copied


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required submission source file is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def validate_submission_tree(package_dir: Path, include_code: bool = True, cases: list[str] | None = None) -> None:
    selected_cases = cases or CASES
    expected = []
    for case in selected_cases:
        expected.append(package_dir / "Attachment" / "Problem1" / f"Q1_{case}_schedule.txt")
        for problem, prefix in (("Problem2", "Q2"), ("Problem3", "Q3")):
            for suffix in ("schedule", "memory", "spill"):
                expected.append(package_dir / "Attachment" / problem / f"{prefix}_{case}_{suffix}.txt")

    if include_code:
        expected.extend(
            [
                package_dir / "code" / "cases.py",
                package_dir / "code" / "problem1_scheduler.py",
                package_dir / "code" / "problem2_allocator.py",
                package_dir / "code" / "problem3_pipeline.py",
            ]
        )

    missing = [path for path in expected if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Submission tree is incomplete:\n{formatted}")


def _write_zip(package_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))

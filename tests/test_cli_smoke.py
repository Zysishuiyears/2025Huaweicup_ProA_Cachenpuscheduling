from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "data" / "fixtures" / "mini_case"


def run_script(script: str, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "runners" / script),
            "--data-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(output_dir),
            "--case",
            "Mini_Case0",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def test_problem1_cli_smoke(tmp_path: Path) -> None:
    run_script("run_problem1.py", tmp_path)
    schedule = tmp_path / "problem1" / "Mini_Case0_schedule.txt"
    assert schedule.exists()
    assert schedule.read_text(encoding="utf-8").splitlines() == ["0", "1", "2", "3", "4"]


def test_problem2_cli_smoke(tmp_path: Path) -> None:
    run_script("run_problem2.py", tmp_path)
    assert (tmp_path / "problem2" / "Mini_Case0_schedule.txt").exists()
    assert (tmp_path / "problem2" / "Mini_Case0_memory.txt").exists()
    assert (tmp_path / "problem2" / "Mini_Case0_spill.txt").exists()


def test_problem3_cli_smoke(tmp_path: Path) -> None:
    run_script("run_problem3.py", tmp_path)
    assert (tmp_path / "problem3" / "Mini_Case0_schedule.txt").exists()
    assert (tmp_path / "problem3" / "Mini_Case0_memory.txt").exists()
    assert (tmp_path / "problem3" / "Mini_Case0_spill.txt").exists()


def test_submission_export_uses_competition_attachment_layout(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "runners" / "export_submission.py"),
            "--output-dir",
            str(tmp_path),
            "--no-zip",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    package_dir = tmp_path / "A25100550012"
    assert (package_dir / "Attachment" / "Problem1" / "Q1_FlashAttention_Case0_schedule.txt").exists()
    assert (package_dir / "Attachment" / "Problem2" / "Q2_FlashAttention_Case0_schedule.txt").exists()
    assert (package_dir / "Attachment" / "Problem2" / "Q2_FlashAttention_Case0_memory.txt").exists()
    assert (package_dir / "Attachment" / "Problem2" / "Q2_FlashAttention_Case0_spill.txt").exists()
    assert (package_dir / "Attachment" / "Problem3" / "Q3_FlashAttention_Case0_schedule.txt").exists()


def test_run_submission_exports_code_generated_competition_layout(tmp_path: Path) -> None:
    run_output = tmp_path / "run_outputs"
    package_output = tmp_path / "submission_ready"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "runners" / "run_submission.py"),
            "--data-dir",
            str(FIXTURE_DIR),
            "--run-output-dir",
            str(run_output),
            "--output-dir",
            str(package_output),
            "--package-name",
            "MiniSubmission",
            "--case",
            "Mini_Case0",
            "--no-zip",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    package_dir = package_output / "MiniSubmission"
    assert (package_dir / "Attachment" / "Problem1" / "Q1_Mini_Case0_schedule.txt").exists()
    assert (package_dir / "Attachment" / "Problem2" / "Q2_Mini_Case0_schedule.txt").exists()
    assert (package_dir / "Attachment" / "Problem2" / "Q2_Mini_Case0_memory.txt").exists()
    assert (package_dir / "Attachment" / "Problem2" / "Q2_Mini_Case0_spill.txt").exists()
    assert (package_dir / "Attachment" / "Problem3" / "Q3_Mini_Case0_schedule.txt").exists()

"""Shared case and path configuration."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "reconstructed"

CASES = [
    "FlashAttention_Case0",
    "FlashAttention_Case1",
    "Matmul_Case0",
    "Matmul_Case1",
    "Conv_Case0",
    "Conv_Case1",
]

CAPACITIES = {
    "L1": 4096,
    "UB": 1024,
    "L0A": 256,
    "L0B": 256,
    "L0C": 512,
}


def parse_cases(case_args: list[str] | None) -> list[str]:
    if not case_args or "all" in {case.lower() for case in case_args}:
        return CASES
    return case_args

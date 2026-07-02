"""Problem 1 heuristic scheduler.

This module keeps the competition implementation intentionally simple:
read a DAG case, produce a topological schedule, and write one node id per
line. The heuristic follows the original project direction of reducing cache
residency pressure while avoiding simultaneous L0 allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import sys

import networkx as nx
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cache_npu_scheduling.cases import CASES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
else:
    from .cases import CASES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases


@dataclass(frozen=True)
class Problem1Result:
    case: str
    schedule: list[object]
    peak_ub_l1: int
    output_path: Path


def read_case(case: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = data_dir / f"{case}_Nodes.csv"
    edges_path = data_dir / f"{case}_Edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing input files for case {case!r} under {data_dir}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    if "NodeId" in nodes.columns:
        nodes = nodes.set_index("NodeId")
    elif "Id" in nodes.columns:
        nodes = nodes.set_index("Id")

    for column, default in {"Op": "", "Type": "", "Size": 0}.items():
        if column not in nodes.columns:
            nodes[column] = default
    nodes["Size"] = nodes["Size"].fillna(0).astype(int)
    nodes["Cost"] = nodes.apply(_cache_pressure_cost, axis=1)
    return nodes, edges


def _cache_pressure_cost(row: pd.Series) -> int:
    op = str(row["Op"]).upper()
    buf_type = str(row["Type"]).upper()
    if buf_type in {"UB", "L1"}:
        if op == "ALLOC":
            return int(row["Size"])
        if op == "FREE":
            return -int(row["Size"])
    return 0


def build_dag(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.index)
    graph.add_edges_from((int(row.StartNodeId), int(row.EndNodeId)) for _, row in edges.iterrows())
    return graph


def greedy_schedule(nodes: pd.DataFrame, graph: nx.DiGraph) -> list[object]:
    """Return a topological order using the original cache-pressure heuristic.

    The priority is dominated by the signed UB/L1 pressure cost: FREE nodes
    reduce residency pressure, ordinary compute/data movement nodes are neutral,
    and ALLOC nodes increase pressure. L0A/L0B/L0C are constrained so that the
    heuristic avoids selecting a second live buffer of the same L0 type when a
    feasible alternative is ready.
    """

    indegree = {node: graph.in_degree(node) for node in graph.nodes}
    ready = [node for node, degree in indegree.items() if degree == 0]
    order: list[object] = []
    live_l0 = {"L0A": 0, "L0B": 0, "L0C": 0}

    while ready:
        selected = _select_ready_node(nodes, ready, live_l0)
        row = nodes.loc[selected]
        op = str(row["Op"]).upper()
        buf_type = str(row["Type"]).upper()

        if op == "ALLOC" and buf_type in live_l0:
            live_l0[buf_type] += 1
        elif op == "FREE" and buf_type in live_l0:
            live_l0[buf_type] = max(0, live_l0[buf_type] - 1)

        ready.remove(selected)
        order.append(selected)
        for successor in graph.successors(selected):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)

    return order


def _select_ready_node(nodes: pd.DataFrame, ready: list[object], live_l0: dict[str, int]) -> object:
    feasible: list[object] = []
    for node in ready:
        row = nodes.loc[node]
        op = str(row["Op"]).upper()
        buf_type = str(row["Type"]).upper()
        if op == "ALLOC" and buf_type in live_l0 and live_l0[buf_type] >= 1:
            continue
        feasible.append(node)

    candidates = feasible or ready
    return min(candidates, key=lambda node: (int(nodes.loc[node, "Cost"]), _stable_node_key(node)))


def _stable_node_key(node: object) -> tuple[int, str]:
    try:
        return (0, f"{int(node):020d}")
    except (TypeError, ValueError):
        return (1, str(node))


def calculate_peak_ub_l1(nodes: pd.DataFrame, schedule: Iterable[object]) -> int:
    current = 0
    peak = 0
    for node_id in schedule:
        row = nodes.loc[node_id]
        op = str(row["Op"]).upper()
        buf_type = str(row["Type"]).upper()
        if op == "ALLOC" and buf_type in {"UB", "L1"}:
            current += int(row["Size"])
            peak = max(peak, current)
        elif op == "FREE" and buf_type in {"UB", "L1"}:
            current -= int(row["Size"])
    return peak


def write_schedule(schedule: Iterable[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for node_id in schedule:
            file.write(f"{node_id}\n")


def run_case(case: str, data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Problem1Result:
    nodes, edges = read_case(case, data_dir)
    graph = build_dag(nodes, edges)
    schedule = greedy_schedule(nodes, graph)
    peak = calculate_peak_ub_l1(nodes, schedule)
    output_path = output_dir / "problem1" / f"{case}_schedule.txt"
    write_schedule(schedule, output_path)
    return Problem1Result(case=case, schedule=schedule, peak_ub_l1=peak, output_path=output_path)


def run(
    cases: Iterable[str] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Problem1Result]:
    selected_cases = list(cases or CASES)
    return [run_case(case, data_dir=data_dir, output_dir=output_dir) for case in selected_cases]


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

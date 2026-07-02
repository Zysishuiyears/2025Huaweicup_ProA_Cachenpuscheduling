"""Problem 2 spill-aware allocator and shared scheduling machinery.

The implementation is a cleaned version of the final competition script. It
keeps the original heuristic structure: greedy topological scheduling,
best-fit allocation per cache pool, WCB-style victim scoring, and explicit
SPILL marker generation in the emitted schedule.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import argparse
import sys

import networkx as nx
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cache_npu_scheduling.cases import CAPACITIES, CASES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases
else:
    from .cases import CAPACITIES, CASES, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, parse_cases


@dataclass(frozen=True)
class AllocationResult:
    case: str
    problem: str
    schedule: list[object]
    memory_alloc: dict[object, int]
    spill_log: list[tuple[object, int, int]]
    total_spill_cost: int
    output_dir: Path
    original_cycles: int | None = None
    compressed_cycles: int | None = None


class CompactPool:
    def __init__(self, pool_type: str, capacity: int):
        self.type = pool_type
        self.capacity = capacity
        self.used_blocks: list[tuple[int, int, object]] = []
        self.free_blocks: list[tuple[int, int]] = [(0, capacity)]
        self.buf_to_offset: dict[object, int] = {}
        self.spill_log: list[tuple[object, int, int]] = []
        self.spill_cost = 0
        self.total_allocated = 0
        self.spill_counter = 0

    def alloc(
        self,
        buf_id: object,
        size: int,
        allow_spill: bool = True,
        current_time: int = 0,
        free_times: dict[object, int] | None = None,
        buf_has_copy_in: dict[object, bool] | None = None,
        w1: float = 1.0,
        w2: float = 1.0,
        nodes: pd.DataFrame | None = None,
        schedule: list[object] | None = None,
    ) -> tuple[bool, str]:
        if size > self.capacity:
            return False, f"request size {size} exceeds {self.type} capacity {self.capacity}"

        best_idx = -1
        best_size = float("inf")
        for i, (start, end) in enumerate(self.free_blocks):
            block_size = end - start
            if size <= block_size < best_size:
                best_idx = i
                best_size = block_size

        if best_idx != -1:
            start, end = self.free_blocks.pop(best_idx)
            self.used_blocks.append((start, start + size, buf_id))
            self.buf_to_offset[buf_id] = start
            self.total_allocated += size
            if start + size < end:
                self.free_blocks.append((start + size, end))
            self.free_blocks.sort()
            return True, "allocated"

        if allow_spill and self.type in {"UB", "L1"}:
            _, victim_bufs, _ = self._find_best_spill_position(
                size,
                current_time,
                free_times or {},
                buf_has_copy_in or {},
                w1,
                w2,
            )
            if victim_bufs:
                self.spill_victims(
                    victim_bufs,
                    current_time,
                    free_times or {},
                    buf_has_copy_in or {},
                    nodes=nodes,
                    schedule=schedule,
                )
                return self.alloc(
                    buf_id,
                    size,
                    allow_spill,
                    current_time,
                    free_times,
                    buf_has_copy_in,
                    w1,
                    w2,
                    nodes=nodes,
                    schedule=schedule,
                )
        return False, "allocation failed"

    def _find_best_spill_position(
        self,
        required_size: int,
        current_time: int,
        free_times: dict[object, int],
        buf_has_copy_in: dict[object, bool],
        w1: float,
        w2: float,
    ) -> tuple[int, list[object], float]:
        candidate_positions = {start for start, _ in self.free_blocks}
        candidate_positions.update(start for start, _, _ in self.used_blocks)

        best_position = -1
        best_victims: list[object] = []
        best_total_cost = float("inf")

        for pos in sorted(candidate_positions):
            if pos + required_size > self.capacity:
                continue
            victims: list[object] = []
            total_cost = 0.0
            for start, end, buf_id in self.used_blocks:
                if end <= pos or start >= pos + required_size:
                    continue
                size = end - start
                free_time = free_times.get(buf_id, current_time + 1000)
                copy_tag = 1 if buf_has_copy_in.get(buf_id, False) else 2
                remaining_time = max(1, free_time - current_time)
                total_cost += copy_tag * w1 / max(1, size) + w2 / remaining_time
                victims.append(buf_id)
            if victims and total_cost < best_total_cost:
                best_position = pos
                best_victims = victims
                best_total_cost = total_cost

        return best_position, best_victims, best_total_cost

    def spill_victims(
        self,
        victim_bufs: Iterable[object],
        current_time: int,
        free_times: dict[object, int],
        buf_has_copy_in: dict[object, bool],
        nodes: pd.DataFrame | None = None,
        schedule: list[object] | None = None,
    ) -> None:
        for victim_buf in victim_bufs:
            victim_block = next((block for block in self.used_blocks if block[2] == victim_buf), None)
            if victim_block is None:
                continue
            start, end, buf_id = victim_block
            size = end - start
            cost_coeff = 1 if buf_has_copy_in.get(buf_id, False) else 2
            cost = cost_coeff * size
            self.spill_cost += cost
            self.spill_log.append((victim_buf, cost, current_time))

            if nodes is not None and schedule is not None:
                new_id = f"spill_{self.type}_{self.spill_counter}"
                self.spill_counter += 1
                nodes.loc[new_id] = {
                    "Op": "SPILL",
                    "Type": self.type,
                    "BufId": buf_id,
                    "Size": size,
                    "Cost": cost,
                    "Cycles": 1,
                    "Pipe": "DMA",
                    "Bufs": str([buf_id]),
                }
                schedule.append(new_id)

            self.free(victim_buf)

    def free(self, buf_id: object) -> None:
        block = next((item for item in self.used_blocks if item[2] == buf_id), None)
        if block is None:
            return
        self.used_blocks.remove(block)
        start, end, _ = block
        self.total_allocated -= end - start
        self.free_blocks.append((start, end))
        self.free_blocks.sort()
        self.free_blocks = _merge_intervals(self.free_blocks)

    def get_offset(self, buf_id: object) -> int:
        return self.buf_to_offset.get(buf_id, -1)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class MultiPoolManager:
    def __init__(self, capacities: dict[str, int], buf_has_copy_in: dict[object, bool]):
        self.pools = {pool_type: CompactPool(pool_type, capacities[pool_type]) for pool_type in capacities}
        self.buf_has_copy_in = buf_has_copy_in
        self.l0_alloc_failed = False

    def alloc(
        self,
        buf_id: object,
        size: int,
        buf_type: str,
        current_time: int,
        free_time: int,
        nodes: pd.DataFrame | None = None,
        schedule: list[object] | None = None,
    ) -> tuple[bool, str]:
        if buf_type not in self.pools:
            return False, f"unknown pool {buf_type}"
        allow_spill = buf_type not in {"L0A", "L0B", "L0C"}
        success, message = self.pools[buf_type].alloc(
            buf_id,
            int(size),
            allow_spill=allow_spill,
            current_time=current_time,
            free_times={buf_id: free_time},
            buf_has_copy_in=self.buf_has_copy_in,
            nodes=nodes,
            schedule=schedule,
        )
        if not success and not allow_spill:
            self.l0_alloc_failed = True
        return success, message

    def free(self, buf_id: object, buf_type: str) -> None:
        if buf_type in self.pools:
            self.pools[buf_type].free(buf_id)

    def get_offset(self, buf_id: object, buf_type: str) -> int:
        if buf_type not in self.pools:
            return -1
        return self.pools[buf_type].get_offset(buf_id)

    def collect_spill_info(self) -> tuple[int, list[tuple[object, int, int]]]:
        spill_log: list[tuple[object, int, int]] = []
        total_cost = 0
        for pool in self.pools.values():
            spill_log.extend(pool.spill_log)
            total_cost += pool.spill_cost
        return total_cost, spill_log


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

    for column, default in {
        "Op": "",
        "Type": "",
        "BufId": -1,
        "Size": 0,
        "Pipe": "",
        "Cycles": 0,
        "Bufs": "",
    }.items():
        if column not in nodes.columns:
            nodes[column] = default

    nodes["Size"] = nodes["Size"].fillna(0).astype(int)
    nodes["Cycles"] = nodes.apply(_cycle_rule, axis=1).astype(int)
    nodes["Cost"] = nodes.apply(_cache_cost, axis=1).astype(int)
    return nodes, edges


def _cache_cost(row: pd.Series) -> int:
    op = str(row["Op"]).upper()
    if op == "ALLOC":
        return int(row["Size"])
    if op == "FREE":
        return -int(row["Size"])
    return 0


def _cycle_rule(row: pd.Series) -> int:
    op = str(row["Op"]).upper()
    if op in {"ALLOC", "FREE", "COPY_IN", "COPY_OUT"}:
        return 0
    if pd.notna(row.get("Cycles", None)):
        try:
            return int(row["Cycles"])
        except (TypeError, ValueError):
            pass
    return 1


def build_dag(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.index)
    graph.add_edges_from((int(row.StartNodeId), int(row.EndNodeId)) for _, row in edges.iterrows())
    return graph


def greedy_topo_with_l0_pool(
    nodes: pd.DataFrame,
    graph: nx.DiGraph,
    capacities: dict[str, int] = CAPACITIES,
) -> list[object]:
    node_attrs = {node_id: row.to_dict() for node_id, row in nodes.iterrows()}
    indegree = {node: graph.in_degree(node) for node in graph.nodes}
    ready = deque([node for node in graph.nodes if indegree[node] == 0])
    order: list[object] = []
    l0_pools = {pool_type: CompactPool(pool_type, capacities[pool_type]) for pool_type in ["L0A", "L0B", "L0C"]}

    while ready:
        selected = None
        selected_cost = float("inf")
        for node in list(ready):
            attrs = node_attrs[node]
            op = str(attrs["Op"]).upper()
            buf_type = str(attrs["Type"]).upper()
            if op == "ALLOC" and buf_type in l0_pools:
                ok, _ = l0_pools[buf_type].alloc(attrs["BufId"], int(attrs["Size"]), allow_spill=False)
                if ok:
                    l0_pools[buf_type].free(attrs["BufId"])
                else:
                    continue
            if int(attrs["Cost"]) < selected_cost:
                selected = node
                selected_cost = int(attrs["Cost"])

        if selected is None:
            selected = ready[0]

        attrs = node_attrs[selected]
        op = str(attrs["Op"]).upper()
        buf_type = str(attrs["Type"]).upper()
        if op == "ALLOC" and buf_type in l0_pools:
            l0_pools[buf_type].alloc(attrs["BufId"], int(attrs["Size"]), allow_spill=False)
        elif op == "FREE" and buf_type in l0_pools:
            l0_pools[buf_type].free(attrs["BufId"])

        for successor in graph.successors(selected):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        order.append(selected)
        ready.remove(selected)

    return order


def build_buf_free_times(nodes: pd.DataFrame, schedule: list[object]) -> dict[object, int]:
    free_times: dict[object, int] = {}
    for i, node_id in enumerate(schedule):
        row = nodes.loc[node_id]
        if str(row["Op"]).upper() == "FREE":
            free_times.setdefault(row["BufId"], i)
    return free_times


def detect_copy_in_buffers(nodes: pd.DataFrame) -> dict[object, bool]:
    result: dict[object, bool] = {}
    for _, row in nodes.iterrows():
        op = str(row["Op"]).upper()
        pipe = str(row["Pipe"]).upper()
        if "COPY_IN" not in op and pipe != "FIXP":
            continue
        for buf_id in _parse_bufs(row.get("Bufs", "")):
            result[buf_id] = True
    return result


def _parse_bufs(value: object) -> list[object]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        import ast

        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def allocate_schedule(
    nodes: pd.DataFrame,
    schedule: list[object],
    capacities: dict[str, int] = CAPACITIES,
) -> tuple[dict[object, int], list[tuple[object, int, int]], int]:
    free_times = build_buf_free_times(nodes, schedule)
    manager = MultiPoolManager(capacities, detect_copy_in_buffers(nodes))
    memory_alloc: dict[object, int] = {}

    i = 0
    while i < len(schedule):
        node_id = schedule[i]
        row = nodes.loc[node_id]
        op = str(row["Op"]).upper()
        buf_type = str(row["Type"]).upper()
        if op == "ALLOC":
            free_time = free_times.get(row["BufId"], i + 1000)
            ok, _ = manager.alloc(
                row["BufId"],
                int(row["Size"]),
                buf_type,
                i,
                free_time,
                nodes=nodes,
                schedule=schedule,
            )
            if ok:
                offset = manager.get_offset(row["BufId"], buf_type)
                if offset != -1:
                    memory_alloc[row["BufId"]] = offset
        elif op == "FREE":
            manager.free(row["BufId"], buf_type)
        i += 1

    total_cost, spill_log = manager.collect_spill_info()
    return memory_alloc, spill_log, total_cost


def compute_clock(schedule: list[object], nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[dict[object, int], dict[object, int], int]:
    pred_end: defaultdict[object, int] = defaultdict(int)
    pipe_last: defaultdict[str, int] = defaultdict(int)
    edge_map: defaultdict[object, list[object]] = defaultdict(list)
    for _, row in edges.iterrows():
        edge_map[int(row.StartNodeId)].append(int(row.EndNodeId))

    start: dict[object, int] = {}
    end: dict[object, int] = {}
    for node_id in schedule:
        row = nodes.loc[node_id]
        pipe = str(row["Pipe"])
        cycles = int(row["Cycles"])
        earliest = max(pred_end[node_id], pipe_last[pipe])
        start[node_id] = earliest
        end[node_id] = earliest + cycles
        pipe_last[pipe] = end[node_id]
        for successor in edge_map[node_id]:
            pred_end[successor] = max(pred_end[successor], end[node_id])

    return start, end, max(end.values(), default=0)


def conservative_left_slide(
    schedule: list[object],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[dict[object, int], dict[object, int], int]:
    predecessors: defaultdict[object, list[object]] = defaultdict(list)
    for _, row in edges.iterrows():
        predecessors[int(row.EndNodeId)].append(int(row.StartNodeId))

    pipe_last: defaultdict[str, int] = defaultdict(int)
    start: dict[object, int] = {}
    end: dict[object, int] = {}
    for node_id in schedule:
        row = nodes.loc[node_id]
        pipe = str(row["Pipe"])
        cycles = int(row["Cycles"])
        pred_ready = max((end[pred] for pred in predecessors[node_id] if pred in end), default=0)
        earliest = max(pred_ready, pipe_last[pipe])
        start[node_id] = earliest
        end[node_id] = earliest + cycles
        pipe_last[pipe] = end[node_id]
    return start, end, max(end.values(), default=0)


def write_outputs(
    case: str,
    problem: str,
    schedule: Iterable[object],
    memory_alloc: dict[object, int],
    spill_log: Iterable[tuple[object, int, int]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    problem_dir = output_dir / problem
    problem_dir.mkdir(parents=True, exist_ok=True)
    prefix = problem_dir / case

    with (prefix.with_name(f"{case}_schedule.txt")).open("w", encoding="utf-8", newline="\n") as file:
        for node_id in schedule:
            file.write(f"{node_id}\n")

    with (prefix.with_name(f"{case}_memory.txt")).open("w", encoding="utf-8", newline="\n") as file:
        for buf_id, offset in memory_alloc.items():
            file.write(f"{buf_id}:{offset}\n")

    with (prefix.with_name(f"{case}_spill.txt")).open("w", encoding="utf-8", newline="\n") as file:
        for buf_id, cost, _ in spill_log:
            file.write(f"{buf_id}:{cost}\n")

    return problem_dir


def run_case(
    case: str,
    problem: str = "problem2",
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> AllocationResult:
    nodes, edges = read_case(case, data_dir)
    graph = build_dag(nodes, edges)
    schedule = greedy_topo_with_l0_pool(nodes, graph)
    memory_alloc, spill_log, total_cost = allocate_schedule(nodes, schedule)
    _, _, original_cycles = compute_clock(schedule, nodes, edges)
    _, _, compressed_cycles = conservative_left_slide(schedule, nodes, edges)
    written_dir = write_outputs(case, problem, schedule, memory_alloc, spill_log, output_dir)
    return AllocationResult(
        case=case,
        problem=problem,
        schedule=schedule,
        memory_alloc=memory_alloc,
        spill_log=spill_log,
        total_spill_cost=total_cost,
        output_dir=written_dir,
        original_cycles=original_cycles,
        compressed_cycles=compressed_cycles,
    )


def run(
    cases: Iterable[str] | None = None,
    problem: str = "problem2",
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[AllocationResult]:
    selected_cases = list(cases or CASES)
    return [run_case(case, problem=problem, data_dir=data_dir, output_dir=output_dir) for case in selected_cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Problem 2 spill-aware allocation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", dest="cases", help="Case name. Repeat or use all.")
    parser.add_argument("--problem", choices=["problem2", "problem3"], default="problem2")
    args = parser.parse_args()

    results = run(cases=parse_cases(args.cases), problem=args.problem, data_dir=args.data_dir, output_dir=args.output_dir)
    for result in results:
        print(
            f"{result.case}: nodes={len(result.schedule)} spill_cost={result.total_spill_cost} "
            f"cycles={result.original_cycles}->{result.compressed_cycles} -> {result.output_dir}"
        )


if __name__ == "__main__":
    main()

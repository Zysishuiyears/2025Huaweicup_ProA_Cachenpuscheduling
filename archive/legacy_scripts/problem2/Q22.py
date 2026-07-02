# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:40:44 2025

@author: JZX
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Problem2: Spill-aware scheduling & allocation
改进版 Victim 策略 (Weighted Cost-Benefit)
"""

import pandas as pd
import networkx as nx
import heapq, os, time
from collections import defaultdict

# ===================== 基础函数 =====================

def read_csv(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    if "NodeId" in nodes.columns:
        nodes.set_index("NodeId", inplace=True)
    # 安全处理 Size 列
    if "Size" in nodes.columns:
        nodes["Size"] = nodes["Size"].fillna(0).astype(int)
    else:
        nodes["Size"] = 0
    return nodes, edges


def build_dag(edges):
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(int(r.StartNodeId), int(r.EndNodeId))
    return G

def key(n, nodes):
    row = nodes.loc[n]
    if str(row.Op).upper() == "FREE":
        return (0, 0, n)
    if str(row.Pipe).upper() == "FIXP":
        return (0, 1, n)
    if str(row.Op).upper() in {"MMAD", "CUBE"}:
        return (1, 0, n)
    if str(row.Pipe).upper() == "MTE1":
        return (2, 0, n)
    if str(row.Op).upper() != "ALLOC" and str(row.Type).upper() in {"UB", "L1"}:
        return (3, n)
    return (4, row.Size, n)

def greedy_topo(nodes, G):
    indeg = {n: G.in_degree(n) for n in G.nodes}
    ready = [(key(n, nodes), n) for n in G.nodes if indeg[n] == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for succ in G.successors(node):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                heapq.heappush(ready, (key(succ, nodes), succ))
    return order

# ===================== 内存管理器 =====================
class MemoryManager:
    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
        self.alloc_map = {}   # nid -> {"size":, "spilled":, "copy_in":}
        self.spill_log = []
        self.total_cost = 0
        self.addr_offset = {}  # BufId -> offset
        self.next_free = 0     # 当前空闲起始地址

    def add_alloc_record(self, nid, size, copy_in=False):
        self.alloc_map[nid] = {"size": size, "spilled": False, "copy_in": copy_in}

    def is_allocated(self, nid):
        return nid in self.alloc_map and not self.alloc_map[nid]["spilled"]

    def alloc(self, nid, size, copy_in, current_index, future_use_func, allow_spill=True):
        if nid in self.alloc_map and not self.alloc_map[nid]["spilled"]:
            return 0, []

        if nid not in self.alloc_map:
            self.alloc_map[nid] = {"size": size, "spilled": False, "copy_in": copy_in}

        need = max(0, size - (self.capacity - self.used))
        victims = []
        cost = 0
        if need > 0:
            if not allow_spill:
                return float("inf"), []
            victims = self.choose_victims(need, current_index, future_use_func)
            cost = self.spill(victims)

        # 分配地址 (简单线性分配)
        self.addr_offset[nid] = self.next_free
        self.next_free += size

        self.alloc_map[nid]["spilled"] = False
        self.used += size
        return cost, victims

    def free(self, nid):
        if nid in self.alloc_map and not self.alloc_map[nid]["spilled"]:
            self.used -= self.alloc_map[nid]["size"]
        if nid in self.alloc_map:
            del self.alloc_map[nid]

    # -------- Victim 策略 (替换过的部分) --------
    def choose_victims(self, need_bytes, current_index, future_use_func, w1=1.0, w2=1.0):
        candidates = [(nid, meta) for nid, meta in self.alloc_map.items() if not meta["spilled"]]
        scored = []
        for nid, meta in candidates:
            size = meta["size"]
            cost = 1 if meta.get("copy_in", False) else 2
            future_pos = future_use_func(nid, current_index) if future_use_func else float("inf")
            next_use_dist = float("inf") if future_pos == float("inf") else (future_pos - current_index)
            score = (cost / max(size, 1)) * w1 + (1.0 / (next_use_dist + 1)) * w2
            scored.append((score, nid, size))
        scored.sort(key=lambda x: x[0])
        victims, freed = [], 0
        for score, nid, size in scored:
            victims.append(nid)
            freed += size
            if freed >= need_bytes:
                break
        return victims

    def spill(self, victims):
        cost = 0
        for v in victims:
            if v not in self.alloc_map:
                continue
            meta = self.alloc_map[v]
            if meta["spilled"]:
                continue
            meta["spilled"] = True
            self.used -= meta["size"]
            c = 1 if meta.get("copy_in", False) else 2
            cost += c
            self.total_cost += c
            self.spill_log.append({"victim": v, "size": meta["size"], "copy_in": meta["copy_in"], "cost": c})
        return cost

# ===================== 工具函数 =====================
def build_successor_map(edges):
    succ = defaultdict(list)
    for _, r in edges.iterrows():
        succ[int(r.StartNodeId)].append(int(r.EndNodeId))
    return succ

def make_future_use_lookup(order, succ_map):
    pos = {nid: i for i, nid in enumerate(order)}
    direct_min_pos = {}
    for nid, succs in succ_map.items():
        minpos = float("inf")
        for s in succs:
            if s in pos:
                minpos = min(minpos, pos[s])
        direct_min_pos[nid] = minpos

    def future_use(nid, current_index):
        mp = direct_min_pos.get(nid, float("inf"))
        return mp if mp > current_index else float("inf")
    return future_use

# ===================== 调度模拟 =====================
def simulate_with_spill(nodes, edges, initial_order, mem_capacity):
    succ_map = build_successor_map(edges)
    future_use_func = make_future_use_lookup(initial_order, succ_map)
    mm = MemoryManager(mem_capacity)

    output_order, spill_events = [], []
    executed = set()

    for idx, nid in enumerate(initial_order):
        # 检查前驱是否需要 SPILL_IN
        preds = edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist()
        for p in preds:
            if p in mm.alloc_map and mm.alloc_map[p]["spilled"]:
                size = mm.alloc_map[p]["size"]
                copy_in_flag = mm.alloc_map[p]["copy_in"]
                cost, victims = mm.alloc(p, size, copy_in_flag, idx, future_use_func)
                output_order.append(f"SPILL_IN_{p}")
                spill_events.append({"type": "SPILL_IN", "node": p, "at_idx": idx})

        row = nodes.loc[nid]
        op, ntype, size, pipe = str(row.Op).upper(), str(row.Type).upper(), int(row.Size), str(row.Pipe).upper()
        is_copy_in = ("COPY" in op) or (pipe == "FIXP")

        if op == "ALLOC":
            cost, victims = mm.alloc(nid, size, is_copy_in, idx, future_use_func)
            for v in victims:
                output_order.append(f"SPILL_OUT_{v}")
                spill_events.append({"type": "SPILL_OUT", "node": v, "at_idx": idx})
            output_order.append(nid)
        elif op == "FREE":
            mm.free(nid)
            output_order.append(nid)
        else:
            output_order.append(nid)
        executed.add(nid)

    return {
        "final_order": output_order,
        "spill_log": mm.spill_log,
        "total_spill_cost": mm.total_cost,
        "addr_offset": mm.addr_offset,
    }

# ===================== 输出函数 =====================
def write_outputs(case, result):
    os.makedirs("Problem2", exist_ok=True)

    with open(f"Problem2/{case}_schedule.txt", "w", encoding="utf-8") as f:
        for item in result["final_order"]:
            f.write(f"{item}\n")

    with open(f"Problem2/{case}_memory.txt", "w", encoding="utf-8") as f:
        f.write("BufId,Offset\n")
        for buf, off in result["addr_offset"].items():
            f.write(f"{buf},{off}\n")

    with open(f"Problem2/{case}_spill.txt", "w", encoding="utf-8") as f:
        f.write("victim,size,copy_in,cost\n")
        for rec in result["spill_log"]:
            f.write(f"{rec['victim']},{rec['size']},{rec['copy_in']},{rec['cost']}\n")
        f.write(f"\nTOTAL_SPILL_COST,{result['total_spill_cost']}\n")

    print(f"-> 写出 {case} 到 Problem2 (schedule / memory / spill)")
    print(f"  总额外搬运量 (bytes): {result['total_spill_cost']}")

# ===================== 主程序 =====================
def main():
    cases = ["FlashAttention_Case0", "FlashAttention_Case1",
             "Matmul_Case0", "Matmul_Case1",
             "Conv_Case0", "Conv_Case1"]

    DEFAULT_MEMORY_CAPACITY = 4096  # ⚠️ 示例：L1 容量，可根据题目表调整

    print("=== Problem2: Spill-aware scheduling & allocation ===")
    t0 = time.time()
    for case in cases:
        if not (os.path.exists(f"{case}_Nodes.csv") and os.path.exists(f"{case}_Edges.csv")):
            print(f"[跳过] {case} 缺少输入文件")
            continue
        nodes, edges = read_csv(case)
        G = build_dag(edges)
        order = greedy_topo(nodes, G)
        result = simulate_with_spill(nodes, edges, order, DEFAULT_MEMORY_CAPACITY)
        write_outputs(case, result)
    print(f"完成。总耗时 {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()

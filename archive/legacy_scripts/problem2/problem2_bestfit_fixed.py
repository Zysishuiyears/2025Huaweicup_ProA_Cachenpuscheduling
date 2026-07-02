# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:50:47 2025

@author: JZX
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Problem2: 多内存池 + Best-Fit 地址分配 + WCB Victim (防碎片修复版)
"""

import pandas as pd
import networkx as nx
import heapq, os, time
from collections import defaultdict

# ---------- 基础工具 ----------
def read_csv(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    if "NodeId" in nodes.columns:
        nodes.set_index("NodeId", inplace=True)
    nodes["Size"] = nodes["Size"].fillna(0).astype(int)
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

# ---------- 多内存池管理 ----------
CAPACITY = {
    "UB": 1024,
    "L1": 4096,
    "L0A": 256,
    "L0B": 256,
    "L0C": 512,
}

class SinglePool:
    """单个物理池（UB/L1/L0A/L0B/L0C），Best-Fit 地址管理"""
    def __init__(self, capacity):
        self.cap = capacity
        self.used = 0
        self.alloc_map = {}       # nid -> {"size":, "copy_in":, "spilled":}
        self.spill_log = []
        self.total_cost = 0
        self.offset = {}          # nid -> offset
        self.free_list = [(0, capacity)]  # 空闲区间表 (start, length)

    # ---------- Best-Fit 地址分配 ----------
    def _best_fit_alloc(self, nid, size):
        best_idx, best_gap = None, None
        for i, (start, length) in enumerate(self.free_list):
            if length >= size:
                if best_gap is None or length < best_gap:
                    best_idx, best_gap = i, length
        if best_idx is None:
            return None
        start, length = self.free_list.pop(best_idx)
        self.offset[nid] = start
        if length > size:
            self.free_list.insert(best_idx, (start + size, length - size))
        return start

    def _best_fit_free(self, nid):
        if nid not in self.offset:
            return
        start = self.offset.pop(nid)
        size = self.alloc_map[nid]["size"]
        self.free_list.append((start, size))
        self.free_list.sort()
        # 合并相邻区间
        merged = []
        for s, l in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + l)
            else:
                merged.append((s, l))
        self.free_list = merged

    # ---------- Victim 选择 (WCB) ----------
    def choose_victims(self, need_bytes, cur_idx, future_func, w1=2.0, w2=1.0):
        alive = [(buf, meta) for buf, meta in self.alloc_map.items() if not meta["spilled"]]
        scored = []
        for buf, meta in alive:
            size   = meta["size"]
            cost   = 1 if meta.get("copy_in", False) else 2
            nxt    = future_func(buf, cur_idx) if future_func else float("inf")
            next_u = float("inf") if nxt == float("inf") else (nxt - cur_idx)
            score  = (cost / max(size, 1)) * w1 + (1.0 / (next_u + 1)) * w2
            scored.append((score, buf, size))
        scored.sort(key=lambda x: x[0])
        victims, freed = [], 0
        for _, buf, sz in scored:
            victims.append(buf)
            freed += sz
            if freed >= need_bytes:
                break
        return victims

    # ---------- 分配 / 释放 ----------
    def alloc(self, nid, size, copy_in, cur_idx, future_func):
        if nid in self.alloc_map and not self.alloc_map[nid]["spilled"]:
            return 0, []
        if nid not in self.alloc_map:
            self.alloc_map[nid] = {"size": size, "copy_in": copy_in, "spilled": False}
        need = max(0, size - (self.cap - self.used))
        victims = []
        if need > 0:
            victims = self.choose_victims(need, cur_idx, future_func)
            self._spill(victims)

        # 尝试 Best-Fit 分配
        addr = self._best_fit_alloc(nid, size)
        if addr is None:
            # 再 spill 一些
            extra_victims = self.choose_victims(size, cur_idx, future_func)
            self._spill(extra_victims)
            addr = self._best_fit_alloc(nid, size)

        if addr is None:
            # fallback: 放到池子末尾
            max_end = max([s + l for s, l in self.free_list], default=0)
            if max_end + size <= self.cap:
                addr = max_end
                self.offset[nid] = addr
            else:
                raise RuntimeError(f"Still cannot alloc: nid={nid}, size={size}, free_list={self.free_list}")

        self.alloc_map[nid]["spilled"] = False
        self.used += size
        return 0, victims

    def _spill(self, victims):
        for v in victims:
            if self.alloc_map[v]["spilled"]:
                continue
            self.alloc_map[v]["spilled"] = True
            self.used -= self.alloc_map[v]["size"]
            cost = 1 if self.alloc_map[v]["copy_in"] else 2
            self.total_cost += cost
            self.spill_log.append({
                "victim": v, "size": self.alloc_map[v]["size"],
                "copy_in": self.alloc_map[v]["copy_in"], "cost": cost
            })
            # ✅ 回收地址，避免碎片化
            self._best_fit_free(v)

    def free(self, nid):
        if nid in self.alloc_map and not self.alloc_map[nid]["spilled"]:
            self.used -= self.alloc_map[nid]["size"]
            self._best_fit_free(nid)
        self.alloc_map.pop(nid, None)

class MultiPoolManager:
    """分池管理器：根据 ALLOC.Type 路由到对应 SinglePool"""
    def __init__(self):
        self.pools = {k: SinglePool(v) for k, v in CAPACITY.items()}

    def _pool_of(self, nid, nodes):
        ty = str(nodes.loc[nid].Type).upper()
        return self.pools.get(ty, self.pools["L1"])

    def alloc(self, nid, size, copy_in, cur_idx, future_func, nodes):
        pool = self._pool_of(nid, nodes)
        return pool.alloc(nid, size, copy_in, cur_idx, future_func)

    def free(self, nid, nodes):
        pool = self._pool_of(nid, nodes)
        pool.free(nid)

    @property
    def total_spill_cost(self):
        return sum(p.total_cost for p in self.pools.values())

    @property
    def spill_log(self):
        return [rec for p in self.pools.values() for rec in p.spill_log]

    @property
    def addr_offset(self):
        return {buf: off for p in self.pools.values() for buf, off in p.offset.items()}

# ---------- 后继/未来使用 ----------
def build_successor_map(edges):
    succ = defaultdict(list)
    for _, r in edges.iterrows():
        succ[int(r.StartNodeId)].append(int(r.EndNodeId))
    return succ

def make_future_use_lookup(order, succ_map):
    pos = {nid: i for i, nid in enumerate(order)}
    direct = {}
    for nid, sucs in succ_map.items():
        mp = min((pos[s] for s in sucs if s in pos), default=float("inf"))
        direct[nid] = mp
    def future_use(nid, cur):
        f = direct.get(nid, float("inf"))
        return f if f > cur else float("inf")
    return future_use

# ---------- 带 SPILL 的调度模拟 ----------
def simulate_with_spill(nodes, edges, initial_order):
    succ_map = build_successor_map(edges)
    future_func = make_future_use_lookup(initial_order, succ_map)
    mm = MultiPoolManager()

    output_order, spill_events = [], []
    for idx, nid in enumerate(initial_order):
        preds = edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist()
        for p in preds:
            pool = mm._pool_of(p, nodes)
            if pool.alloc_map.get(p, {}).get("spilled"):
                size = pool.alloc_map[p]["size"]
                copy_in = pool.alloc_map[p]["copy_in"]
                pool.alloc(p, size, copy_in, idx, future_func)
                output_order.append(f"SPILL_IN_{p}")
                spill_events.append({"type": "SPILL_IN", "node": p, "at_idx": idx})

        row = nodes.loc[nid]
        op, ntype, size, pipe = str(row.Op).upper(), str(row.Type).upper(), int(row.Size), str(row.Pipe).upper()
        is_copy_in = ("COPY" in op) or (pipe == "FIXP")

        if op == "ALLOC":
            cost, victims = mm.alloc(nid, size, is_copy_in, idx, future_func, nodes)
            for v in victims:
                output_order.append(f"SPILL_OUT_{v}")
                spill_events.append({"type": "SPILL_OUT", "node": v, "at_idx": idx})
            output_order.append(nid)
        elif op == "FREE":
            mm.free(nid, nodes)
            output_order.append(nid)
        else:
            output_order.append(nid)
    return {
        "final_order": output_order,
        "spill_log": mm.spill_log,
        "total_spill_cost": mm.total_spill_cost,
        "addr_offset": mm.addr_offset,
    }

# ---------- 输出 ----------
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

# ---------- 主程序 ----------
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cases = ["FlashAttention_Case0", "FlashAttention_Case1",
             "Matmul_Case0", "Matmul_Case1",
             "Conv_Case0", "Conv_Case1"]
    print("=== Problem2: 多池 + Best-Fit 地址分配 + WCB Victim (防碎片修复版) ===")
    t0 = time.time()
    summary = []
    for case in cases:
        if not (os.path.exists(f"{case}_Nodes.csv") and os.path.exists(f"{case}_Edges.csv")):
            print(f"[跳过] {case} 缺少输入文件")
            continue
        nodes, edges = read_csv(case)
        G = build_dag(edges)
        order = greedy_topo(nodes, G)
        result = simulate_with_spill(nodes, edges, order)
        write_outputs(case, result)
        summary.append((case, result['total_spill_cost']))
    print("\n=== 总结 ===")
    for case, cost in summary:
        print(f"{case:<20} 搬运量: {cost} bytes")
    print(f"完成。总耗时 {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()




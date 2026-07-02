# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:14:03 2025

@author: JZX
"""

# Q2_compact_wcb_full.py  紧凑+WCB完整可跑版
import pandas as pd, networkx as nx, heapq, os, time
from collections import defaultdict

# ---------- 1. 读数据 ----------
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

# ---------- 2. 紧凑池 ----------
CAPACITY = {"UB": 1024, "L1": 4096, "L0A": 256, "L0B": 256, "L0C": 512}

class CompactPool:
    def __init__(self, cap):
        self.cap = cap
        self.free = [(0, cap)]          # (start,len)
        self.alloc = {}                 # buf→(start,size,ci)
        self._spill_log = []
        self._total_cost = 0

    @property
    def spill_log(self):
        return self._spill_log

    @property
    def total_spill_cost(self):
        return self._total_cost

    @property
    def addr_offset(self):
        return {b: st for b, (st, _, _) in self.alloc.items() if st != -1}

    # ---- 对外 ----
    def alloc(self, nid, size, copy_in, cur_idx, future_func, w1=1.0, w2=1.0):
        if nid in self.alloc and self.alloc[nid][0] != -1:
            return 0, []
        if nid not in self.alloc:
            self.alloc[nid] = (-1, size, copy_in)
        need = max(0, size - self._total_free())
        victims = []
        if need > 0:
            victims = self._choose_wcb(need, cur_idx, future_func, w1, w2)
            self._spill(victims)
        idx, start, blk = self._best_fit(size)
        if idx is None:
            raise RuntimeError("紧凑池仍不够！")
        self._split(idx, start, blk, size, nid)
        return 0, victims

    def free(self, nid):
        if nid not in self.alloc or self.alloc[nid][0] == -1:
            return
        st, sz, _ = self.alloc[nid]
        self._coalesce(st, sz)

    # ---- 内部 ----
    def _total_free(self):
        return sum(l for _, l in self.free)

    def _best_fit(self, size):
        best = None
        for i, (s, l) in enumerate(self.free):
            if l >= size and (best is None or l < best[2]):
                best = (i, s, l)
        return best if best else (None, None, None)

    def _split(self, idx, start, blk, need, nid):
        self.alloc[nid] = (start, need, self.alloc[nid][2])
        rem = blk - need
        if rem > 0:
            self.free[idx] = (start + need, rem)
        else:
            self.free.pop(idx)

    def _coalesce(self, start, sz):
        self.free.append((start, sz))
        self.free.sort(key=lambda x: x[0])
        merged = []
        for s, l in self.free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                s0, l0 = merged.pop()
                merged.append((s0, l0 + l))
            else:
                merged.append((s, l))
        self.free = merged

    def _choose_wcb(self, need_bytes, cur_idx, future_func, w1, w2):
        alive = [(b, st, sz, ci) for b, (st, sz, ci) in self.alloc.items() if st != -1]
        scored = []
        for b, st, sz, ci in alive:
            cost = 1 if ci else 2
            nxt = future_func(b, cur_idx) if future_func else float("inf")
            next_u = float("inf") if nxt == float("inf") else (nxt - cur_idx)
            score = (cost / max(sz, 1)) * w1 + (1.0 / (next_u + 1)) * w2
            scored.append((score, b, sz))
        scored.sort(key=lambda x: x[0])
        victims, freed = [], 0
        for _, b, sz in scored:
            victims.append(b)
            freed += sz
            if freed >= need_bytes:
                break
        return victims

    def _spill(self, victims):
        c = 0
        for b in victims:
            st, sz, ci = self.alloc[b]
            if st == -1:
                continue
            self._coalesce(st, sz)  # 立即归还并合并
            cost = 1 if ci else 2
            c += cost
            self._total_cost += cost
            self._spill_log.append({"victim": b, "size": sz, "copy_in": ci, "cost": cost})
            self.alloc[b] = (-1, sz, ci)  # 标记在 DDR
        return c

# ---------- 3. 多池管理 ----------
# ---------- 修正版 MultiPoolManager ----------
class MultiPoolManager:
    def __init__(self):
        #self.pools = {k: SinglePool(v) for k, v in CAPACITY.items()}
        self.pools = {k: CompactPool(v) for k, v in CAPACITY.items()}

    def _pool_of(self, nid, nodes):
        ty = str(nodes.loc[nid].Type).upper()
        return self.pools.get(ty, self.pools["L1"])

    # ****** 显式接收 nodes ******
    def alloc(self, nid, size, copy_in, cur_idx, future_func, nodes):
        pool = self._pool_of(nid, nodes)          # 方法调用
        #return pool.alloc(nid, size, copy_in, cur_idx, future_func)
        return pool.alloc(nid, size, copy_in, cur_idx, future_func, w1, w2)

    def free(self, nid, nodes):
        pool = self._pool_of(nid, nodes)          # 方法调用
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
# class MultiPoolManager:
#     def __init__(self):
#         self.pools = {k: CompactPool(v) for k, v in CAPACITY.items()}

#     def _pool_of(self, nid, nodes):
#         ty = str(nodes.loc[nid].Type).upper()
#         return self.pools.get(ty, self.pools["L1"])

#     # ****** 显式接收 nodes ******
#     def alloc(self, nid, size, copy_in, cur_idx, future_func, nodes, w1=1.0, w2=1.0):
#         return self._pool_of(nid, nodes).alloc(nid, size, copy_in, cur_idx, future_func, w1, w2)

#     def free(self, nid, nodes):
#         self._pool_of(nid, nodes).free(nid)

#     @property
#     def spill_log(self):
#         return [item for p in self.pools.values() for item in p.spill_log]

#     @property
#     def total_spill_cost(self):
#         return sum(p.total_spill_cost for p in self.pools.values())

#     @property
#     def addr_offset(self):
#         return {buf: off for p in self.pools.values() for buf, off in p.addr_offset.items()}
# class MultiPoolManager:
#     def __init__(self):
#         self.pools = {k: CompactPool(v) for k, v in CAPACITY.items()}

#     def _pool_of(self, nid, nodes):
#         ty = str(nodes.loc[nid].Type).upper()
#         return self.pools.get(ty, self.pools["L1"])

#     def alloc(self, nid, size, copy_in, cur_idx, future_func, w1=1.0, w2=1.0):
#         return self._pool_of(nid, nodes).alloc(nid, size, copy_in, cur_idx, future_func, w1, w2)

#     def free(self, nid, nodes):
#         self._pool_of(nid, nodes).free(nid)

#     @property
#     def spill_log(self):
#         return [item for p in self.pools.values() for item in p.spill_log]

#     @property
#     def total_spill_cost(self):
#         return sum(p.total_spill_cost for p in self.pools.values())

#     @property
#     def addr_offset(self):
#         return {buf: off for p in self.pools.values() for buf, off in p.addr_offset.items()}

# ---------- 4. 工具 ----------
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

# ---------- 5. 主程序 ----------
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

def main():
    cases = ["FlashAttention_Case0", "FlashAttention_Case1",
             "Matmul_Case0", "Matmul_Case1",
             "Conv_Case0", "Conv_Case1"]
    print("=== Problem2: 紧凑地址 + WCB  victim ===")
    t0 = time.time()
    for case in cases:
        if not (os.path.exists(f"{case}_Nodes.csv") and os.path.exists(f"{case}_Edges.csv")):
            print(f"[跳过] {case} 缺少输入文件")
            continue
        nodes, edges = read_csv(case)
        order = greedy_topo(nodes, build_dag(edges))
        result = simulate_with_spill(nodes, edges, order, w1=1.0, w2=1.0)
        write_outputs(case, result)
    print(f"完成。总耗时 {time.time()-t0:.2f}s")
    # ---------- 修正版 simulate_with_spill ----------
def simulate_with_spill(nodes, edges, initial_order, w1=1.0, w2=1.0):

    succ_map   = build_successor_map(edges)
    future_fun = make_future_use_lookup(initial_order, succ_map)
    mm         = MultiPoolManager()
    out_order  = []

    for idx, nid in enumerate(initial_order):
        # 前驱 SPILL_IN 检查
        for p in edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist():
            p_pool = mm._pool_of(p, nodes)        # 方法调用
            if p in mm.addr_offset and p_pool.alloc_map.get(p, {}).get("spilled"):
                sz, ci = p_pool.alloc_map[p]["size"], p_pool.alloc_map[p]["copy_in"]
                p_pool.alloc(p, sz, ci, idx, future_fun)
                out_order.append(f"SPILL_IN_{p}")

        # 正常节点处理
        row  = nodes.loc[nid]
        op   = str(row.Op).upper()
        size = int(row.Size)
        pipe = str(row.Pipe).upper()
        is_ci = ("COPY" in op) or (pipe == "FIXP")

        if op == "ALLOC":
            pool = mm._pool_of(nid, nodes)        # 方法调用
            #cost, victims = pool.alloc(nid, size, is_ci, idx, future_fun)
            #cost, victims = pool.alloc(nid, size, is_ci, idx, future_fun, nodes)
            cost, victims = pool.alloc(nid, size, is_ci, idx, future_fun, nodes=nodes)
            for v in victims:
                out_order.append(f"SPILL_OUT_{v}")
            out_order.append(nid)
        elif op == "FREE":
            mm.free(nid, nodes)                   # 显式传 nodes
            out_order.append(nid)
        else:
            out_order.append(nid)

    return {
        "final_order": out_order,
        "spill_log": mm.spill_log,
        "total_spill_cost": mm.total_spill_cost,
        "addr_offset": mm.addr_offset,
}

# def simulate_with_spill(nodes, edges, initial_order, w1=1.0, w2=1.0):
#     succ_map   = build_successor_map(edges)
#     future_fun = make_future_use_lookup(initial_order, succ_map)
#     mm         = MultiPoolManager()
#     out_order  = []

#     for idx, nid in enumerate(initial_order):
#         # -------- 前驱 SPILL_IN 检查 --------
#         for p in edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist():
#             p_pool = mm._pool_of(p, nodes)        # 取池对象
#             if p in mm.addr_offset and p_pool.alloc[p][0] == -1:   # spilled
#                 sz, ci = p_pool.alloc[p][1], p_pool.alloc[p][2]
#                 p_pool.alloc(p, sz, ci, idx, future_fun, w1, w2)
                
#                 out_order.append(f"SPILL_IN_{p}")

#         # -------- 正常节点处理 --------
#         row  = nodes.loc[nid]
#         op   = str(row.Op).upper()
#         size = int(row.Size)
#         pipe = str(row.Pipe).upper()
#         is_ci = ("COPY" in op) or (pipe == "FIXP")

#         if op == "ALLOC":
#             pool = mm._pool_of(nid, nodes)        # 取池对象
#            # cost, victims = pool.alloc(nid, size, is_ci, idx, future_fun, w1, w2)
#             cost, victims = pool.alloc(nid, size, is_ci, idx, future_fun, nodes, w1, w2)
#             for v in victims:
#                 out_order.append(f"SPILL_OUT_{v}")
#             out_order.append(nid)
#         elif op == "FREE":
#             mm.free(nid, nodes)
#             out_order.append(nid)
#         else:
#             out_order.append(nid)

#     return {
#         "final_order": out_order,
#         "spill_log": mm.spill_log,
#         "total_spill_cost": mm.total_spill_cost,
#         "addr_offset": mm.addr_offset,
# }

if __name__ == "__main__":
    main()
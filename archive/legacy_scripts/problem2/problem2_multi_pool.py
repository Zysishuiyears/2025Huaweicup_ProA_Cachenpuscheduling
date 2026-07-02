# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:41:09 2025

@author: JZX
"""

"""
Problem2: Spill-aware scheduling & allocation
多内存池版本（UB/L1/L0A/L0B/L0C 独立容量）
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

# ---------- 多内存池管理器 ----------
CAPACITY = {
    "UB": 1024,
    "L1": 4096,
    "L0A": 256,
    "L0B": 256,
    "L0C": 512,
}

# ============ CompactPool ==============
class CompactPool:
    """
    真·紧凑管理：
    1. 空闲块用 (start,len) 列表，按地址排序
    2. free 时立即与邻洞合并
    3. alloc 用 best-fit 在 0~cap-1 内挑洞
    4. 地址即真实物理偏移，永不 > capacity
    """
    def __init__(self, capacity):
        self.cap = capacity
        # 空闲块：按 start 排序，用于合并
        self.free = [(0, capacity)]
        # 已分配：buf → (start, size)
        self.alloc = {}
        self.spill_log = []
        self.total_cost = 0

    # ---------- 对外接口 ----------
    def alloc(self, nid, size, copy_in, cur_idx, future_func,
              w1=1.0, w2=1.0):
        if nid in self.alloc:
            return 0, []
        # 1. best-fit 找洞
        idx, start, blk_sz = self._best_fit(size)
        if idx is None:
            # 2. 没洞就踢人
            need = size - self._total_free()
            victims = self._choose_victims_compact(need, cur_idx,
                                                   future_func,w1,w2)
            self._spill(victims)
            idx, start, blk_sz = self._best_fit(size)
            if idx is None:
                raise RuntimeError("紧凑池仍不够！")
        # 3. 分配 / 分裂
        self._split_and_insert(idx, start, blk_sz, size, nid)
        return 0, []

    def free(self, nid):
        if nid not in self.alloc:
            return
        start, sz = self.alloc.pop(nid)
        self._coalesce(start, sz)

    # ---------- 内部工具 ----------
    def _total_free(self):
        return sum(l for _, l in self.free)

    def _best_fit(self, size):
        """返回 (free列表下标, 起始, 长度) 或 None"""
        best_idx, best_start, best_sz = None, None, None
        for idx, (start, sz) in enumerate(self.free):
            if sz >= size and (best_sz is None or sz < best_sz):
                best_idx, best_start, best_sz = idx, start, sz
        return best_idx, best_start, best_sz

    def _split_and_insert(self, idx, start, blk_sz, need, nid):
        self.alloc[nid] = (start, need)
        rem = blk_sz - need
        if rem > 0:
            self.free[idx] = (start + need, rem)
        else:
            self.free.pop(idx)

    def _coalesce(self, start, sz):
        # 把 [start, start+sz) 插回并合并邻洞
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

    def _choose_victims_compact(self, need_bytes, cur_idx, future_func,
                                w1, w2):
        alive = [(buf, meta) for buf, meta in self.alloc.items()]
        scored = []
        for buf, meta in alive:
            size = meta["size"]
            cost = 1 if meta.get("copy_in", False) else 2
            nxt = future_func(buf, cur_idx) if future_func else float("inf")
            next_u = float("inf") if nxt == float("inf") else (nxt - cur_idx)
            score = (cost / max(size, 1)) * w1 + (1.0 / (next_u + 1)) * w2
            scored.append((score, buf, size))
        scored.sort(key=lambda x: x[0])
        victims, freed = [], 0
        for _, buf, sz in scored:
            victims.append(buf)
            freed += sz
            if freed >= need_bytes:
                break
        return victims

    # ---------- spill 账本 ----------
    def _spill(self, victims):
        c = 0
        for v in victims:
            start, sz = self.alloc[v]
            self._coalesce(start, sz)  # 立即把物理区归还空闲表
            cost = 1 if self.alloc[v].get("copy_in", False) else 2
            c += cost
            self.total_cost += cost
            self.spill_log.append({"victim": v, "size": sz,
                                   "copy_in": self.alloc[v].get("copy_in", False),
                                   "cost": cost})
            # 逻辑上仍占坑，只是标记 spilled
            self.alloc[v] = (-1, sz)  # start=-1 表示在 DDR
        return c

    # ---------- 属性导出 ----------
    @property
    def total_spill_cost(self):
        return self.total_cost

    @property
    def spill_log(self):
        return self.spill_log

    @property
    def addr_offset(self):
        return {buf: off for buf, (off, _) in self.alloc.items() if off != -1}

# class SinglePool:
#     """单个物理池（UB/L1/L0A/L0B/L0C）"""
#     def __init__(self, capacity):
#         self.cap = capacity
#         self.used = 0
#         self.alloc_map = {}       # nid -> {"size":, "copy_in":, "spilled":}
#         self.spill_log = []
#         self.total_cost = 0
#         self.offset = {}          # nid -> offset
#         self.next_free = 0

#     def _score(self, nid, size, copy_in, future):
#         cost = 1 if copy_in else 2
#         dist = float("inf") if future == float("inf") else future
#         return (cost / max(size, 1)) + (1.0 / (dist + 1))
    
#     ### WCB评分选择spill位置
#     def choose_victims(self, need_bytes, cur_idx, future_func,
#                         w1=2.0, w2=1.0):
#         alive = [(buf, meta) for buf, meta in self.alloc_map.items()
#                   if not meta["spilled"]]
#         scored = []
#         for buf, meta in alive:
#             size   = meta["size"]
#             cost   = 1 if meta.get("copy_in", False) else 2
#             nxt    = future_func(buf, cur_idx) if future_func else float("inf")
#             next_u = float("inf") if nxt == float("inf") else (nxt - cur_idx)
#             score  = (cost / max(size, 1)) * w1  +  (1.0 / (next_u + 1)) * w2
#             scored.append((score, buf, size))
#         scored.sort(key=lambda x: x[0])
#         victims, freed = [], 0
#         for _, buf, sz in scored:
#             victims.append(buf)
#             freed += sz
#             if freed >= need_bytes:
#                 break
#         return victims
    
    ### LRU原则选择spill位置
    
    # def choose_victims(self, need, cur_idx, future_func):
    #     candidates = [(nid, meta) for nid, meta in self.alloc_map.items() if not meta["spilled"]]
    #     scored = []
    #     for nid, meta in candidates:
    #         f = future_func(nid, cur_idx)
    #         scored.append((self._score(nid, meta["size"], meta["copy_in"], f),
    #                        nid, meta["size"]))
    #     scored.sort(key=lambda x: x[0])
    #     victims, freed = [], 0
    #     for _, nid, sz in scored:
    #         victims.append(nid)
    #         freed += sz
    #         if freed >= need:
    #             break
    #     return victims

    def alloc(self, nid, size, copy_in, cur_idx, future_func):
        if nid in self.alloc_map and not self.alloc_map[nid]["spilled"]:
            return 0, []
        if nid not in self.alloc_map:
            self.alloc_map[nid] = {"size": size, "copy_in": copy_in, "spilled": False}
        need = max(0, size - (self.cap - self.used))
        victims = []
        cost = 0
        if need > 0:
            victims = self.choose_victims(need, cur_idx, future_func)
            cost = self._spill(victims)
        # 线性地址分配
        self.offset[nid] = self.next_free
        self.next_free += size
        self.alloc_map[nid]["spilled"] = False
        self.used += size
        return cost, victims

    def _spill(self, victims):
        c = 0
        for v in victims:
            if self.alloc_map[v]["spilled"]:
                continue
            self.alloc_map[v]["spilled"] = True
            self.used -= self.alloc_map[v]["size"]
            cost = 1 if self.alloc_map[v]["copy_in"] else 2
            c += cost
            self.total_cost += cost
            self.spill_log.append({"victim": v, "size": self.alloc_map[v]["size"],
                                   "copy_in": self.alloc_map[v]["copy_in"], "cost": cost})
        return c

    def free(self, nid):
        if nid in self.alloc_map and not self.alloc_map[nid]["spilled"]:
            self.used -= self.alloc_map[nid]["size"]
        self.alloc_map.pop(nid, None)

class MultiPoolManager:
    """分池管理器：根据 ALLOC.Type 路由到对应 SinglePool"""
    def __init__(self):
        self.pools = {k: SinglePool(v) for k, v in CAPACITY.items()}

    def _pool_of(self, nid, nodes):
        ty = str(nodes.loc[nid].Type).upper()
        return self.pools.get(ty, self.pools["L1"])

    # 对外接口保持与旧 MemoryManager 一致
    def alloc(self, nid, size, copy_in, cur_idx, future_func, allow_spill=True):
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
        return [item for p in self.pools.values() for item in p.spill_log]

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
            if p in mm.addr_offset and p not in mm.pools["L1"].alloc_map:
                # 需要 SPILL_IN
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
            pool = mm._pool_of(nid, nodes)
            cost, victims = pool.alloc(nid, size, is_copy_in, idx, future_func)
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
    cases = ["FlashAttention_Case0", "FlashAttention_Case1",
             "Matmul_Case0", "Matmul_Case1",
             "Conv_Case0", "Conv_Case1"]
    print("=== Problem2: 多内存池 Spill-aware scheduling & allocation ===")
    t0 = time.time()
    for case in cases:
        if not (os.path.exists(f"{case}_Nodes.csv") and os.path.exists(f"{case}_Edges.csv")):
            print(f"[跳过] {case} 缺少输入文件")
            continue
        nodes, edges = read_csv(case)
        G = build_dag(edges)
        order = greedy_topo(nodes, G)
        result = simulate_with_spill(nodes, edges, order)
        write_outputs(case, result)
    print(f"完成。总耗时 {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
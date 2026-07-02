# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:40:05 2025

@author: JZX
"""

# Q2_l0b_block.py  第二问：L0B 满时阻塞节点 + Best-Fit + WCB 紧凑地址
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
    
    if str(row.Op).upper() == "FREE" and str(row.Type).upper() == "L0B":
        return (-1, 0, n)   # 比普通 FREE 更优先
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

# ---------- 2+. 具体池子 ----------
class CompactPool:
    """
    紧凑池（Best-Fit + WCB victim + L0 阻塞支持）
    - alloc(...) 总是返回 (ret, victims)
      ret == 0 -> 成功分配
      ret == None -> L0 阻塞（调用方应重试/跳过）
    - alloc_map: nid -> (start, size, copy_in)  start == -1 表示已 spill（不驻留）
    - _free_blocks: list of (start, length), 保持不重叠且按 start 排序
    - _spill_log / _total_cost 记录 spill 事件
    """
    def __init__(self, cap):
        self.cap = cap
        self._free_blocks = [(0, cap)]
        self.alloc_map = {}            # nid -> (start, size, copy_in)
        self._spill_log = []
        self._total_cost = 0
        # 判断是否为 L0（L0 不允许真实 spill，改为阻塞）
        # 你也可以针对 L0B 做特殊判断，当前用 cap<=512 简单区分
        self.is_l0 = cap <= 512

    # ---------- 外部接口 ----------
    def alloc(self, nid, size, copy_in, cur_idx, future_func, nodes, w1=1.0, w2=1.0):
        """
        返回 (ret, victims)
        - ret == 0: 成功
        - ret == None: L0 阻塞（caller should delay/retry）
        - victims: list of victim nids that were spilled to free space
        """
        # 已经在池内且驻留
        if nid in self.alloc_map and self.alloc_map[nid][0] != -1:
            return 0, []

        # 初始化 alloc_map 记录（-1 表示当前未驻留）
        if nid not in self.alloc_map:
            self.alloc_map[nid] = (-1, size, copy_in)

        victims = []
        # 1) 如果总体空闲不足，先选择 victim 并 spill（WCB）
        need = max(0, size - self._total_free())
        if need > 0:
            victims = self._choose_wcb(need, cur_idx, future_func, w1, w2, nodes)
            self._spill(victims, nodes)

        # 2) L0 特殊策略：如果是 L0（如 L0B）且当前没有足够连续洞 => 阻塞（不强制 spill）
        if self.is_l0:
            idx, start, blk = self._best_fit(size)
            if idx is None:
                # 返回二元组，表示调用方应把该节点标记为阻塞并跳过
                return None, []

        # 3) 正常 Best-Fit 分配；如果失败则尝试再次 spill 一批并重试
        idx, start, blk = self._best_fit(size)
        if idx is None:
            # 再次挑 victim（更激进）并 spill
            victims2 = self._choose_wcb(size, cur_idx, future_func, w1, w2, nodes)
            # 可能 victims2 会包含已 spill 的项，_spill 会安全地跳过
            self._spill(victims2, nodes)
            # 再试一次 best-fit
            idx, start, blk = self._best_fit(size)

        if idx is None:
            # 仍然失败 —— 极端情况下说明碎片/容量都不能满足
            raise RuntimeError(f"Still cannot alloc: nid={nid}, size={size}, free_blocks={self._free_blocks}")

        # 执行分割并记录驻留地址
        self._split(idx, start, blk, size, nid)
        return 0, victims

    def free(self, nid):
        """
        释放驻留的 buf（若已 spill 则不做操作）
        释放后将 alloc_map[nid] 标记为 (-1,size,copy_in) 表示曾存在但已不驻留
        """
        if nid not in self.alloc_map:
            return
        st, sz, ci = self.alloc_map[nid]
        if st == -1:
            # 已经是 spill 状态，无需释放
            return
        # 回收区间并合并
        self._coalesce(st, sz)
        # 标记为已 spill（未驻留）
        self.alloc_map[nid] = (-1, sz, ci)

    # ---------- 内部工具方法 ----------
    def _total_free(self):
        return sum(l for _, l in self._free_blocks)

    def _best_fit(self, size):
        best = None
        for i, (s, l) in enumerate(self._free_blocks):
            if l >= size and (best is None or l < best[2]):
                best = (i, s, l)
        if best is None:
            return (None, None, None)
        return best

    def _split(self, idx, start, blk, need, nid):
        """
        把 free_blocks[idx] 拆分成已分配段和剩余空洞
        记录 alloc_map[nid] = (start, need, copy_in)
        """
        _, _, ci = self.alloc_map[nid]
        self.alloc_map[nid] = (start, need, ci)
        rem = blk - need
        if rem > 0:
            self._free_blocks[idx] = (start + need, rem)
        else:
            # exact fit
            self._free_blocks.pop(idx)

    def _coalesce(self, start, sz):
        """回收区间并合并相邻 free blocks"""
        self._free_blocks.append((start, sz))
        self._free_blocks.sort(key=lambda x: x[0])
        merged = []
        for s, l in self._free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == s:
                prev_s, prev_l = merged.pop()
                merged.append((prev_s, prev_l + l))
            else:
                merged.append((s, l))
        self._free_blocks = merged

    def _choose_wcb(self, need_bytes, cur_idx, future_func, w1, w2, nodes):
        """
        Weighted Cost-Benefit victim selection (WCB)
        返回要 spill 的 victims（按 score 升序选择直到释放 >= need_bytes）
        """
        alive = [(b, st, sz, ci) for b, (st, sz, ci) in self.alloc_map.items() if st != -1]
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

    def _spill(self, victims, nodes):
        """执行 spill：回收空间、记录日志、标记 alloc_map 为 (-1, size, copy_in)"""
        for b in victims:
            if b not in self.alloc_map:
                continue
            st, sz, ci = self.alloc_map[b]
            if st == -1:
                # 已经处于 spilled 状态
                continue
            # 回收实际地址区间
            self._coalesce(st, sz)
            # 记录成本（题目规则：COPY_IN -> cost 1，否则 2）
            cost = 1 if ci else 2
            self._total_cost += cost
            self._spill_log.append({"victim": b, "size": sz, "copy_in": ci, "cost": cost})
            # 标记为 spilled（不驻留）
            self.alloc_map[b] = (-1, sz, ci)

    # ---------- 属性访问 ----------
    @property
    def spill_log(self):
        return self._spill_log

    @property
    def total_spill_cost(self):
        return self._total_cost

    @property
    def addr_offset(self):
        # 返回所有当前驻留的 buf 的 offset（不包含已 spill 的）
        return {nid: st for nid, (st, sz, ci) in self.alloc_map.items() if st != -1}


# ---------- 2+. 具体池子 ----------
class UBPool(CompactPool): pass
class L1Pool(CompactPool): pass
class L0Pool(CompactPool): pass


# ---------- 3. 多池管理 ----------
class MultiPoolManager:
    def __init__(self):
        # 用类名区分 L0 / 非 L0
        self.pools = {
            "UB":   UBPool(CAPACITY["UB"]),
            "L1":   L1Pool(CAPACITY["L1"]),
            "L0A":  L0Pool(CAPACITY["L0A"]),
            "L0B":  L0Pool(CAPACITY["L0B"]),
            "L0C":  L0Pool(CAPACITY["L0C"]),
        }

    def _pool_of(self, nid, nodes):
        ty = str(nodes.loc[nid].Type).upper()
        return self.pools.get(ty, self.pools["L1"])

    def alloc(self, nid, size, copy_in, cur_idx, future_func, nodes, w1=1.0, w2=1.0):
        return self._pool_of(nid, nodes).alloc(nid, size, copy_in, cur_idx, future_func, nodes, w1, w2)

    def free(self, nid, nodes):
        self._pool_of(nid, nodes).free(nid)

    @property
    def spill_log(self):
        return [rec for p in self.pools.values() for rec in p.spill_log]

    @property
    def total_spill_cost(self):
        return sum(p.total_spill_cost for p in self.pools.values())

    @property
    def addr_offset(self):
        return {buf: off for p in self.pools.values() for buf, off in p.addr_offset.items()}


# ---------- 4. 阻塞感知调度 ----------
# # ---------- 4. 工具 ----------
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

def simulate_with_spill(nodes, edges, initial_order, w1=1.0, w2=1.0):
    
    nodes.index = nodes.index.astype(int)
    edges = edges.astype(int)
    
    succ_map   = build_successor_map(edges)
    future_fun = make_future_use_lookup(initial_order, succ_map)
    mm         = MultiPoolManager()
    out_order  = []
    block_reason = {}          # nid -> "L0B_full" / None

    l0b_ready = []                      # 小根堆，只放 L0B-ALLOC
    l0b_indeg = {}                      # 记录 L0B-ALLOC 的剩余入度
    l0b_cap_left = CAPACITY["L0B"]      # 当前剩余 L0B 字节
    
    # 初始就绪队列
    indeg = {n: 0 for n in nodes.index}
    for _, r in edges.iterrows():
        indeg[int(r.EndNodeId)] += 1
    # ready = [(key(n, nodes), n) for n in nodes.index if indeg[n] == 0]
    ready = []
    for n in nodes.index:
        if indeg[n] == 0:
            if str(nodes.loc[n].Op).upper() == "ALLOC" and str(nodes.loc[n].Type).upper() == "L0B":
                heapq.heappush(l0b_ready, (key(n, nodes), n))
                l0b_indeg[n] = 0
            else:
                heapq.heappush(ready, (key(n, nodes), n))
    heapq.heapify(ready)
    
    # ------------- 主循环（带死锁检测/恢复） -------------
    no_progress = 0
    last_progress_len = -1
    MAX_NO_PROGRESS = max(100, len(nodes) // 2)  # 超参：无进展次数阈值，可调整

    while ready:
        while l0b_ready and int(nodes.loc[l0b_ready[0][1]].Size) <= l0b_cap_left:
            _, nid = heapq.heappop(l0b_ready)
            heapq.heappush(ready, (key(nid, nodes), nid))

        _, nid = heapq.heappop(ready)

        # 如果当前节点被标记为 L0B 阻塞，则把它放回队列（后续会处理）
        if block_reason.get(nid) == "L0B_full":
            heapq.heappush(ready, (key(nid, nodes), nid))
            no_progress += 1
        else:
            # 正常尝试执行节点（原有逻辑）
            # 前驱 SPILL_IN 检查
            for p in edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist():
                p_pool = mm._pool_of(p, nodes)
                st, sz, ci = p_pool.alloc_map.get(p, (-1, 0, False))
                if st == -1:
                    p_pool.alloc(p, sz, ci, len(out_order), future_fun, nodes, w1, w2)
                    out_order.append(f"SPILL_IN_{p}")

            # 节点处理
            row  = nodes.loc[nid]
            op   = str(row.Op).upper()
            size = int(row.Size)
            pipe = str(row.Pipe).upper()
            is_ci = ("COPY" in op) or (pipe == "FIXP")

            if op == "ALLOC":
                pool = mm._pool_of(nid, nodes)
                ret, victims = pool.alloc(nid, size, is_ci, len(out_order), future_fun, nodes, w1, w2)
                if ret is None:                      # L0B 满且洞不够
                    block_reason[nid] = "L0B_full"
                    heapq.heappush(ready, (key(nid, nodes), nid))
                    no_progress += 1
                    # 不 append out_order
                    pass
                else:
                    # 成功：输出 spill_out 并把节点加入 out_order
                    for v in victims:
                        out_order.append(f"SPILL_OUT_{v}")
                    out_order.append(nid)
                    # 进展了
                    no_progress = 0
            elif op == "FREE":
                mm.free(nid, nodes)
                
                # ★★★ 新增：如果是 FREE-L0B，把容量还回来并批量转移 ★★★
                if str(nodes.loc[nid].Type).upper() == "L0B":
                    # 找到对应 ALLOC 的大小
                    alloc_nid = int(edges[edges.EndNodeId == nid].StartNodeId.iloc[0])
                    sz = int(nodes.loc[alloc_nid].Size)
                    l0b_cap_left += sz
                    # 立即再触发一次“批量转移”
                    while l0b_ready and int(nodes.loc[l0b_ready[0][1]].Size) <= l0b_cap_left:
                        _, nid2 = heapq.heappop(l0b_ready)
                        heapq.heappush(ready, (key(nid2, nodes), nid2))
                
                if str(nodes.loc[nid].Type).upper() in {"L0B"}:
                    block_reason.pop(nid, None)   # 解除阻塞（如果有）
                out_order.append(nid)
                no_progress = 0
            else:
                out_order.append(nid)
                no_progress = 0
                
                # ★★★ 新增：L0A-FREE 同样可能堵死后续 L0B，一并批量唤醒 ★★★
                if str(nodes.loc[nid].Type).upper() == "L0A":
                    alloc_nid = int(edges[edges.EndNodeId == nid].StartNodeId.iloc[0])
                    sz = int(nodes.loc[alloc_nid].Size)
                    # 把 L0A 容量还回（虽然代码里没显式计数，但逻辑一致）
                    # 更重要的是：立即再触发一次 L0B 唤醒
                    while l0b_ready and int(nodes.loc[l0b_ready[0][1]].Size) <= l0b_cap_left:
                        _, nid2 = heapq.heappop(l0b_ready)
                        heapq.heappush(ready, (key(nid2, nodes), nid2))

        # 死锁 / 长期无进展检测
        if no_progress >= MAX_NO_PROGRESS:
            # 检测 ready 中是否全部被阻塞（L0B_full）
            ready_items = [n for _, n in ready]
            if ready_items and all(block_reason.get(x) == "L0B_full" for x in ready_items):
                # 选一个阻塞节点对应的池（优先处理导致阻塞的 L0 池）
                target_nid = ready_items[0]
                target_pool = mm._pool_of(target_nid, nodes)

                # 计算需要释放的空间（按第一个阻塞节点的需求）
                needed = int(nodes.loc[target_nid].Size) - target_pool._total_free()
                if needed <= 0:
                    # 虽然检测到阻塞，但空间充足（可能碎片问题），尝试 aggressive spill
                    needed = int(nodes.loc[target_nid].Size)
               
                # ★★★ 合规救援：只从 L1/UB 里 spill，绝不碰 L0 ★★★
                for pool_name in ["L1", "UB"]:
                    pool_other = mm.pools.get(pool_name)
                    if not pool_other:
                        continue
                    victims = pool_other._choose_wcb(needed, len(out_order), future_fun, w1, w2, nodes)
                    if victims:
                        pool_other._spill(victims, nodes)
                        print(f"[DEADLOCK-RESOLVE] spilled from {pool_name} victims={victims}")
                        break
                
                # 清理阻塞标记（让被阻塞的节点有机会再次分配）
                for x in ready_items:
                    block_reason.pop(x, None)

                # 重置 no_progress
                no_progress = 0
            else:
                # 不是全阻塞（有其它 ready），继续等待／循环，但也防止无限增长
                no_progress = 0

    # end while

    # while ready:
    #     _, nid = heapq.heappop(ready)
    #     # 跳过仍阻塞的节点
    #     if block_reason.get(nid) == "L0B_full":
    #         heapq.heappush(ready, (key(nid, nodes), nid))
    #         continue

    #     # 前驱 SPILL_IN 检查
    #     for p in edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist():
    #         p_pool = mm._pool_of(p, nodes)
    #         st, sz, ci = p_pool.alloc_map.get(p, (-1, 0, False))
    #         if st == -1:
    #             p_pool.alloc(p, sz, ci, len(out_order), future_fun, nodes, w1, w2)
    #             out_order.append(f"SPILL_IN_{p}")

    #     # 正常节点处理
    #     row  = nodes.loc[nid]
    #     op   = str(row.Op).upper()
    #     size = int(row.Size)
    #     pipe = str(row.Pipe).upper()
    #     is_ci = ("COPY" in op) or (pipe == "FIXP")

    #     if op == "ALLOC":
    #         pool = mm._pool_of(nid, nodes)
    #         ret, victims = pool.alloc(nid, size, is_ci, len(out_order), future_fun, nodes, w1, w2)
    #         if ret is None:                      # L0B 满且洞不够
    #             block_reason[nid] = "L0B_full"
    #             heapq.heappush(ready, (key(nid, nodes), nid))
    #             continue
    #         for v in victims:
    #             out_order.append(f"SPILL_OUT_{v}")
    #         out_order.append(nid)
    #     elif op == "FREE":
    #         mm.free(nid, nodes)
    #         if str(nodes.loc[nid].Type).upper() in {"L0B"}:
    #             block_reason.pop(nid, None)   # 解除阻塞
    #         out_order.append(nid)
    #     else:
    #         out_order.append(nid)
    # 保证所有 L0B-ALLOC 都被执行完
    assert not l0b_ready, f"还有 {len(l0b_ready)} 个 L0B-ALLOC 没调度！"

    return {
        "final_order": out_order,
        "spill_log": mm.spill_log,
        "total_spill_cost": mm.total_spill_cost,
        "addr_offset": mm.addr_offset,
    }


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
    print("=== Problem2: L0B 阻塞 + 紧凑地址 + WCB  victim ===")
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

if __name__ == "__main__":
    main()
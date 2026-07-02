# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:34:26 2025

@author: JZX
"""

# problem3_pipeline.py
"""
第三问实验：受约束的流水调度（多池 + Best-Fit + WCB）
- 先用 greedy_topo 得到 baseline 的 spill cost（相当于题目2结果）
- 然后运行 pipeline scheduler，在 spill_cost <= baseline*(1+epsilon) 的约束下
  最小化 makespan（finish time）
可配置参数：lambda_time, epsilon, duration defaults
"""

import pandas as pd, networkx as nx, heapq, os, time, copy
from collections import defaultdict

# ------------------ utils / topology ------------------
def read_csv(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    if "NodeId" in nodes.columns:
        nodes.set_index("NodeId", inplace=True)
    nodes["Size"] = nodes["Size"].fillna(0).astype(int)
    # Duration optional
    if "Duration" not in nodes.columns:
        nodes["Duration"] = nodes.apply(lambda r: default_duration(str(r.Op).upper()), axis=1)
    else:
        nodes["Duration"] = nodes["Duration"].fillna(0).astype(float)
    return nodes, edges

def build_dag(edges):
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(int(r.StartNodeId), int(r.EndNodeId))
    return G

# a simple default duration mapping if CSV lacks Duration
def default_duration(op):
    op = op.upper()
    if "ALLOC" in op: return 0.1
    if "FREE" in op:  return 0.05
    if "MMAD" in op or "CUBE" in op: return 1.0
    if "FIXP" in op: return 0.5
    if "MTE1" in op or "MTE2" in op or "MTE3" in op: return 0.2
    if "COPY" in op: return 0.2
    return 0.5

# your key (keeps your original behavior; you can swap)
def key(n, nodes):
    row = nodes.loc[n]
    if str(row.Op).upper() == "FREE" and str(row.Type).upper() == "L0B":
        return (-1, 0, n)
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

# ------------------ CompactPool & MultiPoolManager (same semantics as Q2) ------------------
CAPACITY = {"UB": 1024, "L1": 4096, "L0A": 256, "L0B": 256, "L0C": 512}

class CompactPool:
    def __init__(self, cap):
        self.cap = cap
        self._free_blocks = [(0, cap)]
        self.alloc_map = {}            # nid -> (start, size, copy_in)
        self._spill_log = []
        self._total_cost = 0
        self.is_l0 = cap <= 512


    def alloc(self, nid, size, copy_in, cur_idx, future_func, nodes, w1=1.0, w2=1.0):
        """
        Enhanced Best-Fit alloc that tries controlled victim spilling to resolve fragmentation.
        Returns (ret, victims) like before:
          - ret == 0: success
          - ret == None: L0-blocked (no way to create contiguous space without touching L0)
        Victims are those actually spilled to make room (already applied to pool).
        """
        # Already present and resident
        if nid in self.alloc_map and self.alloc_map[nid][0] != -1:
            return 0, []

        # Initialize record
        if nid not in self.alloc_map:
            self.alloc_map[nid] = (-1, size, copy_in)

        # Quick path: if total free < size, we must pick victims (WCB) to free capacity
        need_total = max(0, size - self._total_free())
        victims = []
        if need_total > 0:
            # Choose victims by WCB to free total capacity (may not produce contiguous block)
            victims = self._choose_wcb(need_total, cur_idx, future_func, w1, w2, nodes)
            # If pool is L0, do NOT spill (L0 cannot be spilled) -> block upstream caller
            if self.is_l0:
                # cannot free by spilling in L0
                return None, []
            # Spill chosen victims (this will free their blocks and update free blocks)
            self._spill(victims, nodes)

        # Try best-fit now
        idx, start, blk = self._best_fit(size)
        if idx is not None:
            # success
            self._split(idx, start, blk, size, nid)
            return 0, victims

        # Fragmentation case: total free might be >= size but no single block >= size.
        # We'll attempt controlled victim selection targeted to create a large enough contiguous block.
        # Compute current largest block
        max_block = 0
        for s, l in self._free_blocks:
            if l > max_block:
                max_block = l

        # If pool is L0 -> cannot spill its contents, so must block caller
        if self.is_l0:
            # nothing to do here (no allowed spills to defragment)
            return None, []

        # Need to free at least this much contiguous space
        need_contiguous = max(0, size - max_block)

        if need_contiguous == 0:
            # unexpected, but try best-fit again
            idx, start, blk = self._best_fit(size)
            if idx is None:
                # fallback: attempt to pick some victims equal to size and spill them
                victims2 = self._choose_wcb(size, cur_idx, future_func, w1, w2, nodes)
                self._spill(victims2, nodes)
                idx, start, blk = self._best_fit(size)
                if idx is None:
                    raise RuntimeError(f"Still cannot alloc after aggressive spill: nid={nid}, size={size}, free_blocks={self._free_blocks}")
                else:
                    self._split(idx, start, blk, size, nid)
                    return 0, victims + victims2
            else:
                self._split(idx, start, blk, size, nid)
                return 0, victims

        # Now iteratively pick victims targeted by WCB until contiguous requirement met
        freed = 0
        victims_target = []
        # Note: choose victims from current alive (not yet spilled) list
        candidates = self._choose_wcb(need_contiguous, cur_idx, future_func, w1, w2, nodes)
        # If no candidates available, give up
        if not candidates:
            return None, victims
        # Spill candidates and check
        self._spill(candidates, nodes)
        victims_target.extend(candidates)

        # After spilling, try best-fit
        idx, start, blk = self._best_fit(size)
        if idx is not None:
            self._split(idx, start, blk, size, nid)
            return 0, victims + victims_target

        # If still not enough (rare), keep spilling more (aggressive phase)
        max_iterations = 5
        it = 0
        while it < max_iterations:
            # Determine remaining contiguous shortage
            max_block = max((l for _, l in self._free_blocks), default=0)
            need_contiguous = max(0, size - max_block)
            if need_contiguous == 0:
                break
            more = self._choose_wcb(need_contiguous, cur_idx, future_func, w1, w2, nodes)
            # remove already spilled ones
            more = [m for m in more if self.alloc_map.get(m, (-1,0,False))[0] != -1]
            if not more:
                break
            self._spill(more, nodes)
            victims_target.extend(more)
            idx, start, blk = self._best_fit(size)
            if idx is not None:
                self._split(idx, start, blk, size, nid)
                return 0, victims + victims_target
            it += 1

        # final attempt: if still cannot, try a last-resort aggressive spill of many small buffers
        # (this is invasive — we keep it as last resort)
        aggressive = self._choose_wcb(size, cur_idx, future_func, w1, w2, nodes)
        aggressive = [a for a in aggressive if self.alloc_map.get(a, (-1,0,False))[0] != -1]
        if aggressive:
            self._spill(aggressive, nodes)
            victims_target.extend(aggressive)
            idx, start, blk = self._best_fit(size)
            if idx is not None:
                self._split(idx, start, blk, size, nid)
                return 0, victims + victims_target

        # give up (should be rare). Return None for caller to handle or raise to crash.
        # We return None to indicate caller may block (for L0-like semantics), but here pool is non-L0.
        # To be conservative, raise an error so you see it during tests.
        raise RuntimeError(f"Still cannot alloc after all defrag/spill attempts: nid={nid}, size={size}, free_blocks={self._free_blocks}")


    def free(self, nid):
        if nid not in self.alloc_map: return
        st, sz, ci = self.alloc_map[nid]
        if st == -1: return
        self._coalesce(st, sz)
        self.alloc_map[nid] = (-1, sz, ci)

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
        _, _, ci = self.alloc_map[nid]
        self.alloc_map[nid] = (start, need, ci)
        rem = blk - need
        if rem > 0:
            self._free_blocks[idx] = (start + need, rem)
        else:
            self._free_blocks.pop(idx)

    def _coalesce(self, start, sz):
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
        for b in victims:
            if b not in self.alloc_map: continue
            st, sz, ci = self.alloc_map[b]
            if st == -1: continue
            self._coalesce(st, sz)
            cost = 1 if ci else 2
            self._total_cost += cost
            self._spill_log.append({"victim": b, "size": sz, "copy_in": ci, "cost": cost})
            self.alloc_map[b] = (-1, sz, ci)

    @property
    def spill_log(self):
        return self._spill_log

    @property
    def total_spill_cost(self):
        return self._total_cost

    @property
    def addr_offset(self):
        return {nid: st for nid, (st, sz, ci) in self.alloc_map.items() if st != -1}

class UBPool(CompactPool): pass
class L1Pool(CompactPool): pass
class L0Pool(CompactPool): pass

class MultiPoolManager:
    def __init__(self):
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

# ------------------ baseline simulator (reuse typical greedy schedule) ------------------
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

def simulate_with_spill_baseline(nodes, edges, order):
    # simple reuse of earlier simulate: allocate/free in order, record spill cost
    succ_map = build_successor_map(edges)
    future_func = make_future_use_lookup(order, succ_map)
    mm = MultiPoolManager()
    out_order = []
    for idx, nid in enumerate(order):
        # check predecessors spilled -> SPILL_IN (not affecting spill cost here)
        for p in edges[edges["EndNodeId"] == nid]["StartNodeId"].tolist():
            p_pool = mm._pool_of(p, nodes)
            st, sz, ci = p_pool.alloc_map.get(p, (-1, 0, False))
            if st == -1:
                p_pool.alloc(p, sz, ci, idx, future_func, nodes)
                out_order.append(f"SPILL_IN_{p}")
        row = nodes.loc[nid]
        op = str(row.Op).upper()
        size = int(row.Size)
        pipe = str(row.Pipe).upper()
        is_ci = ("COPY" in op) or (pipe == "FIXP")
        if op == "ALLOC":
            pool = mm._pool_of(nid, nodes)
            cost, victs = pool.alloc(nid, size, is_ci, idx, future_func, nodes)
            for v in victs:
                out_order.append(f"SPILL_OUT_{v}")
            out_order.append(nid)
        elif op == "FREE":
            mm.free(nid, nodes)
            out_order.append(nid)
        else:
            out_order.append(nid)
    return {"spill_cost": mm.total_spill_cost, "spill_log": mm.spill_log, "order": out_order}

# ------------------ pipeline scheduler ------------------
def pipeline_schedule(nodes, edges, lambda_time=1.0, epsilon=0.1, SCALE=1000.0):
    """
    lambda_time: weight for time in combined score; spill weight is 1-lambda_time (implicitly)
    epsilon: allowed relative degradation over baseline spill (e.g. 0.1 => allow 10% more)
    SCALE: scale factor to convert spill bytes into time-like units (heuristic)
    """
    # baseline
    G = build_dag(edges)
    base_order = greedy_topo(nodes, G)
    base = simulate_with_spill_baseline(nodes, edges, base_order)
    baseline_spill = base["spill_cost"]
    budget = baseline_spill * (1.0 + epsilon)
    print(f"[Baseline] spill_cost={baseline_spill}  budget={budget}")

    # prepare structures for scheduling
    indeg = {n: 0 for n in nodes.index}
    preds = defaultdict(list)
    succs = defaultdict(list)
    for _, r in edges.iterrows():
        s = int(r.StartNodeId); t = int(r.EndNodeId)
        preds[t].append(s); succs[s].append(t)
        indeg[t] += 1

    # ready set initial
    ready = [n for n in nodes.index if indeg[n] == 0]
    # track finish times of all nodes
    finish_time = {}
    start_time = {}
    # per-pipe availability (earliest time pipe free)
    pipe_ready = defaultdict(float)
    # pool manager state (we will mutate)
    mm = MultiPoolManager()
    total_spill = 0.0

    # future lookup for WCB decisions
    succ_map = build_successor_map(edges)
    future_func = make_future_use_lookup(base_order, succ_map)

    # we will iterate until all nodes scheduled
    scheduled = set()
    order_sched = []

    # helper to try schedule one node and return (predicted_finish, added_spill, pool_state_after)
    def evaluate_node(nid, cur_time, mm_state):
        # shallow copy mm_state
        mm_sim = copy.deepcopy(mm_state)
        row = nodes.loc[nid]
        op = str(row.Op).upper()
        size = int(row.Size)
        pipe = str(row.Pipe).upper()
        dur = float(row.Duration)
        is_ci = ("COPY" in op) or (pipe == "FIXP")
        # earliest based on preds
        preds_finish = max([finish_time[p] for p in preds[nid]] or [0.0])
        start = max(preds_finish, pipe_ready[pipe], cur_time)
        added_spill = 0
        # simulate allocation/free
        if op == "ALLOC":
            ret, victims = mm_sim.alloc(nid, size, is_ci, 0, future_func, nodes)
            if ret is None:
                # blocked in L0: represent as huge added finish (unfavorable)
                return (float("inf"), float("inf"), None)
            # compute added spill cost (sum cost)
            for v in victims:
                # victim cost was recorded in pool; we can sum sizes/costs:
                # but mm_sim pools already updated totals; we can compute delta by comparing later
                pass
            # we compute added_spill by checking mm_sim.total_spill_cost - mm_state.total_spill_cost
            added_spill = mm_sim.total_spill_cost - mm_state.total_spill_cost
        elif op == "FREE":
            mm_sim.free(nid, nodes)
        else:
            # compute nothing for non-alloc/free
            pass
        finish = start + dur
        return (finish, added_spill, mm_sim)

    # main loop: greedy selection by score
    cur_time = 0.0
    while len(scheduled) < len(nodes):
        # build candidate list
        candidates = [n for n in ready if n not in scheduled]
        if not candidates:
            # advance time to earliest pipe or earliest pred finish
            next_time = min([pipe_ready[p] for p in pipe_ready] + [min([finish_time[x] for x in finish_time]) if finish_time else float('inf')])
            if next_time == float('inf'):
                raise RuntimeError("No candidates and cannot advance time -> deadlock")
            cur_time = max(cur_time, next_time)
            continue

        best = None
        best_score = float("inf")
        best_eval = None

        for nid in candidates:
            preds_finish = max([finish_time[p] for p in preds[nid]] or [0.0])
            # earliest we could try (we allow scheduling starting now)
            eval_start_time = max(cur_time, preds_finish)
            finish, add_spill, mm_after = evaluate_node(nid, cur_time, mm)
            if finish == float('inf'):
                continue
            # compute score: time-weighted finish + normalized spill
            time_term = finish
            spill_term = (total_spill + add_spill) / SCALE
            # prohibit exceeding budget
            if (total_spill + add_spill) > budget:
                score = float('inf')
            else:
                score = lambda_time * time_term + (1.0 - lambda_time) * spill_term
            if score < best_score:
                best_score = score
                best = nid
                best_eval = (finish, add_spill, mm_after)

        if best is None:
            # can't schedule any candidate under budget; relax by choosing earliest finish ignoring budget (but warn)
            for nid in candidates:
                finish, add_spill, mm_after = evaluate_node(nid, cur_time, mm)
                if finish == float('inf'): continue
                # pick minimal finish
                if best is None or finish < best_eval[0]:
                    best = nid; best_eval = (finish, add_spill, mm_after)
            if best is None:
                raise RuntimeError("No feasible candidate found even ignoring budget")

        # commit best
        finish, add_spill, mm_new = best_eval
        row = nodes.loc[best]
        op = str(row.Op).upper()
        size = int(row.Size)
        pipe = str(row.Pipe).upper()
        dur = float(row.Duration)
        preds_finish = max([finish_time[p] for p in preds[best]] or [0.0])
        start = max(preds_finish, pipe_ready[pipe], cur_time)
        # actually perform on real mm (not copy) to update pools
        if op == "ALLOC":
            ret, victims = mm.alloc(best, size, ("COPY" in op) or (pipe == "FIXP"), 0, future_func, nodes)
            if ret is None:
                # should not happen because evaluate prevented it; skip and mark scheduled to avoid infinite loop
                print("[WARN] alloc turned blocked at commit time:", best)
            for v in victims:
                total_spill += (1 if nodes.loc[v].Pipe and ("FIXP" in str(nodes.loc[v].Pipe).upper()) else 2) if False else 0
                # above is placeholder; actual mm.total_spill_cost tracks it
        elif op == "FREE":
            mm.free(best, nodes)
        # update times
        start_time[best] = start
        finish_time[best] = start + dur
        pipe_ready[pipe] = finish_time[best]
        scheduled.add(best)
        order_sched.append(best)
        # update ready set (decrement indeg)
        for s in succs[best]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
        # update current time to smallest next event
        cur_time = min([pipe_ready[p] for p in pipe_ready])
        # update total_spill from mm
        total_spill = mm.total_spill_cost

    makespan = max(finish_time.values()) if finish_time else 0.0
    return {
        "order": order_sched,
        "start": start_time,
        "finish": finish_time,
        "makespan": makespan,
        "spill_cost": total_spill,
    }

# ------------------ main driver ------------------
def main():
    cases = ["Conv_Case1"]  # replace with list or single case you want to test
    for case in cases:
        print("Running case:", case)
        nodes, edges = read_csv(case)
        t0 = time.time()
        result = pipeline_schedule(nodes, edges, lambda_time=0.9, epsilon=0.1, SCALE=1000.0)
        t1 = time.time()
        print(f"Done in {t1-t0:.2f}s; makespan={result['makespan']:.3f}; spill_cost={result['spill_cost']}")
        # write schedule
        os.makedirs("Problem3", exist_ok=True)
        with open(f"Problem3/{case}_schedule_pipeline.txt", "w", encoding="utf-8") as f:
            for nid in result["order"]:
                st = result["start"].get(nid, 0.0)
                ft = result["finish"].get(nid, 0.0)
                f.write(f"{nid},{st:.6f},{ft:.6f}\n")
        print("Schedule written to Problem3/")

if __name__ == "__main__":
    main()

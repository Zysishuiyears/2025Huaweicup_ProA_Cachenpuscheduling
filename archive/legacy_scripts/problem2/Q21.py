# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:47:41 2025

@author: JZX
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem2.py  —— 2025 A题 第二问完整主流程
1. 读入第一问拓扑序（或换成你自己的）
2. 按 Type 分组、内存分拨（公用+私用）、小→大首次放置
3. 线性扫描 + Belady 选牺牲页（最远再用 + COPY_IN 优先）
4. 自动插入 SPILL 节点、加依赖边、重新拓扑
5. 输出三文件 + 打印额外搬运量
"""
import pandas as pd, networkx as nx, os, sys, heapq, math
from collections import defaultdict

CASES = ["FlashAttention_Case0", "FlashAttention_Case1",
         "Matmul_Case0", "Matmul_Case1",
         "Conv_Case0", "Conv_Case1"]

CAP = {"UB": 1024, "L1": 4096,
       "L0A": 256, "L0B": 256, "L0C": 512}

# ---------- 工具 ----------
def read_csv(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv", dtype={"Id": int})
    edges = pd.read_csv(f"{case}_Edges.csv", dtype={"StartNodeId": int, "EndNodeId": int})
    nodes.set_index("Id", inplace=True)
    return nodes, edges

def load_order(case):
    with open(f"{case}_schedule.txt") as f:
        return [int(x) for x in f]

# def build_next_use(sched, nodes):
#     nu = defaultdict(dict)
#     last = defaultdict(list)
#     for pos in reversed(range(len(sched))):
#         nid = sched[pos]
#         for buf in nodes.loc[nid].get("Bufs", []):
#             if last[buf]:
#                 nu[buf][pos] = last[buf][-1]
#             else:
#                 nu[buf][pos] = math.inf
#         for buf in nodes.loc[nid].get("Bufs", []):
#             last[buf].append(pos)
#     return nu

def build_next_use(sched, nodes):
    nu = defaultdict(dict)
    last = defaultdict(list)
    for pos in reversed(range(len(sched))):
        nid = sched[pos]
        bufs = nodes.loc[nid].get("Bufs", [])
        if pd.isna(bufs) or bufs == "":
            bufs = []
        else:
            if isinstance(bufs, str):
                bufs = [int(x) for x in bufs.replace(',', ' ').split()]
            elif isinstance(bufs, (int, float)):
                bufs = [int(bufs)]
            else:  # 已经是 list
                bufs = [int(x) for x in bufs]
        for buf in bufs:
            if last[buf]:
                nu[buf][pos] = last[buf][-1]
            else:
                nu[buf][pos] = math.inf
        for buf in bufs:
            last[buf].append(pos)
    return nu

def copy_in_bufs(nodes):
    s = set()
    for nid, row in nodes.iterrows():
        if row.Op == "COPY_IN":
            s.update(row.get("Bufs", []))
    return s

# ---------- 内存分拨 ----------
def first_fit_alloc(lst, total, ratio=0.25):
    pub_top = int(total * ratio)
    prv_top = total
    pub_used, prv_used = [], []
    alloc = []
    for buf, size in lst:
        off = find_gap(pub_used, size, pub_top)
        if off is not None:
            pub_used.append((off, off + size))
            alloc.append((buf, off, off + size))
            continue
        off = find_gap(prv_used, size, prv_top)
        if off is not None:
            prv_used.append((off, off + size))
            alloc.append((buf, off, off + size))
            continue
        alloc.append((buf, None, None))  # 触发 SPILL
    return alloc

def find_gap(used, size, top):
    used.sort()
    prev = 0
    for s, e in used:
        if s - prev >= size:
            return prev
        prev = e
    if top - prev >= size:
        return prev
    return None

# ---------- Belady 选牺牲 ----------
def pick_victim(buffers_in_mem, nu, pos, copy_in_set):
    best_i, best_prio = -1, (-1, math.inf)
    for i, (buf, sz, _) in enumerate(buffers_in_mem):
        nu_pos = nu[buf].get(pos, math.inf)
        prio = (buf in copy_in_set, nu_pos)
        if prio > best_prio:
            best_i, best_prio = i, prio
    return best_i

# ---------- 插入 SPILL ----------
def insert_spill(G, nodes, sched, buf, spill_pos, nu, copy_in_set):
    N = len(nodes)
    so_id, si_id = N, N + 1
    sz = nodes.loc[buf].Size
    so_node = {"Id": so_id, "Op": "SPILL_OUT", "Bufs": [buf], "Type": None, "Size": sz}
    si_node = {"Id": si_id, "Op": "SPILL_IN",  "Bufs": [buf], "Type": None, "Size": sz}
 #   new_nodes = nodes.append(pd.DataFrame([so_node, si_node]).set_index("Id"))
    spill_df = pd.DataFrame([so_node, si_node]).set_index("Id")
    new_nodes = pd.concat([nodes, spill_df])
    new_G = G.copy()
    # 基本依赖
    alloc_id = nodes.index[nodes.Op == "ALLOC"][0]
    free_id  = nodes.index[nodes.Op == "FREE"][0]
    new_G.add_edge(alloc_id, so_id)
    new_G.add_edge(so_id, si_id)
    new_G.add_edge(si_id, free_id)
    # 已执行/未执行边
    for nid in sched:
        if new_nodes.loc[nid].Op == "COPY_IN":   # 关键跳过
            continue
        bufs = new_nodes.loc[nid].get("Bufs", [])

        if pd.isna(bufs) or bufs == "":
            bufs = []
        else:
            if isinstance(bufs, str):
                bufs = [int(x) for x in bufs.replace(',', ' ').split()]
            elif isinstance(bufs, (int, float)):
                bufs = [int(bufs)]
            else:
                bufs = [int(x) for x in bufs]
        if buf not in bufs:
            continue
        idx = sched.index(nid)

    # for nid in sched:
    #     if buf not in new_nodes.loc[nid].get("Bufs", []):
    #         continue
    #     idx = sched.index(nid)
        if idx <= spill_pos:
            new_G.add_edge(nid, so_id)
        else:
            new_G.add_edge(si_id, nid)
    # 新拓扑序：插入 so_id, si_id
    new_sched = sched[:spill_pos+1] + [so_id, si_id] + sched[spill_pos+1:]
    extra = sz if buf in copy_in_set else 2 * sz
    return new_G, new_nodes, new_sched, so_id, si_id, extra

# ---------- 主流程 ----------
def run_case(case):
    nodes, edges = read_csv(case)
    nodes["Size"] = nodes["Size"].fillna(0).astype(int)
    # ****** 用原图重新拓扑，不要读旧 schedule.txt ******
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(int(r.StartNodeId), int(r.EndNodeId))
    sched = list(nx.topological_sort(G))  # 或你自己的贪心函数
    # ****************************************************
    nu = build_next_use(sched, nodes)

    cset = copy_in_bufs(nodes)

    # 按 Type 收集 ALLOC
    mem_stat = defaultdict(list)
    for nid in sched:
        row = nodes.loc[nid]
        if row.Op == "ALLOC":
            mem_stat[row.Type].append((row.BufId, row.Size))

    alloc_map = {}  # buf -> (off, end)
    spill_list = []
    extra_total = 0

    for typ, lst in mem_stat.items():
        total = CAP[typ]
        lst.sort(key=lambda x: x[1])  # 小→大
        allocs = first_fit_alloc(lst, total)
        for buf, off, end in allocs:
            if off is not None:
                alloc_map[buf] = (off, end)
            else:
                # 需要 SPILL
                pos = sched.index(buf)
                victim_idx = pick_victim([(b, s, None) for b, s in lst], nu, pos, cset)
                victim = lst[victim_idx][0]
                new_G, new_nodes, new_sched, so_id, si_id, extra = insert_spill(G, nodes, sched, victim, pos, nu, cset)
                extra_total += extra
                spill_list.append((victim, so_id, si_id))
                # 更新
                G, nodes, sched = new_G, new_nodes, new_sched
                alloc_map[victim] = (None, None)
                allocs = first_fit_alloc(lst, total)  # 重试

    # 输出三文件
    with open(f"{case}_schedule.txt", "w") as f:
        for nid in sched:
            f.write(f"{nid}\n")
    with open(f"{case}_memory.txt", "w") as f:
        for buf, (off, _) in alloc_map.items():
            f.write(f"{buf}:{off}\n")
    with open(f"{case}_spill.txt", "w") as f:
        for buf, so, si in spill_list:
            f.write(f"{buf}:{alloc_map.get(si, (0, 0))[0]}\n")

    print(f"{case}: 额外搬运量 = {extra_total} 字节")

def main():
    for case in CASES:
        run_case(case)

if __name__ == "__main__":
    main()
    

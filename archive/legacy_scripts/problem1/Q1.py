# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:20:29 2025

@author: JZX
"""

# #!/usr/bin/env python3
# """
# 新贪心拓扑：FREE > Cube/MMAD > MTE1 > 普算(UB/L1) > ALLOC(小→大)
# 全部顶点保留；Vstay 只计 UB/L1；一键跑 6 图
# """
# import pandas as pd, networkx as nx, time, os, sys

# def read_csv(case):
#     return pd.read_csv(f"{case}_Nodes.csv"), pd.read_csv(f"{case}_Edges.csv")

# def build_dag(edges):
#     G = nx.DiGraph()
#     for _, r in edges.iterrows():
#         G.add_edge(int(r.StartNodeId), int(r.EndNodeId))
#     return G

# def key(n, nodes):
#     row = nodes.loc[n]
#     if row.Op == "FREE":                                    # 0 级
#         return (0, 0)
#     if row.Op in {"MMAD", "CUBE"}:                          # 1 级
#         return (1, 0)
#     if row.Pipe == "MTE1":                                  # 2 级
#         return (2, 0)
#     if row.Op != "ALLOC" and row.Type in {"UB", "L1"}:      # 3 级
#         return (3, n)
#     return (4, row.Size)                                    # 4 级 小→大

# def greedy_topo(nodes, G):
#     indeg = {n: G.in_degree(n) for n in G.nodes}
#     ready = [n for n in G.nodes if indeg[n] == 0]
#     order = []
#     while ready:
#         ready.sort(key=lambda n: key(n, nodes))
#         node = ready.pop(0)
#         order.append(node)
#         for succ in G.successors(node):
#             indeg[succ] -= 1
#             if indeg[succ] == 0:
#                 ready.append(succ)
#     return order

# def max_stay(nodes, order):
#     cur = peak = 0
#     for nid in order:
#         row = nodes.loc[nid]
#         if row.Op == "ALLOC" and row.Type in {"UB", "L1"}:
#             cur += row.Size
#             peak = max(peak, cur)
#         elif row.Op == "FREE" and row.Type in {"UB", "L1"}:
#             cur -= row.Size
#     return peak

# def write(case, order, peak):
#     print(f"{case:<20} 节点数：{len(order):<6}  峰值(UB+L1)：{peak:<6} 字节")
#     with open(f"{case}_schedule.txt", "w") as f:
#         for nid in order:
#             f.write(f"{nid}\n")

# def main():
#     cases = ["FlashAttention_Case0", "FlashAttention_Case1",
#              "Matmul_Case0", "Matmul_Case1",
#              "Conv_Case0", "Conv_Case1"]
#     print("==== 新贪心拓扑（FREE>Cube>MTE1>普算>ALLOC小→大）====")
#     t0 = time.time()
#     for case in cases:
#         nodes, edges = read_csv(case)
#         G   = build_dag(edges)
#         order = greedy_topo(nodes, G)
#         peak  = max_stay(nodes, order)
#         write(case, order, peak)
#     print(f"总耗时：{time.time() - t0:.2f} s")
#     print("✅ 六个 schedule.txt 已生成！")

# if __name__ == "__main__":
#     main()

"""
新贪心拓扑：FREE > Fixp >Cube/MMAD > MTE1 > 普算(UB/L1) > ALLOC(小→大)
全部顶点保留；Vstay 只计 UB/L1；一键跑 6 图
优化版：使用 heapq 代替 sort，将复杂度降至 O((n+m) log n)
"""
import pandas as pd, networkx as nx, time, os, sys, heapq

def read_csv(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    # 优化：以 NodeId 作为索引，加速 loc 查询
    if "NodeId" in nodes.columns:
        nodes.set_index("NodeId", inplace=True)
    return nodes, edges

def build_dag(edges):
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(int(r.StartNodeId), int(r.EndNodeId))
    return G


def key(n, nodes):
    row = nodes.loc[n]
    if row.Op == "FREE":                                    # 0 级
        return (0, 0, n)
    if row.Pipe == "FIXP":                                  # 0.5 级：FIXP 搬运
        return (0, 1, n)
    if row.Op in {"MMAD", "CUBE"}:                          # 1 级
        return (1, 0, n)
    if row.Pipe == "MTE1":                                  # 2 级
        return (2, 0, n)
    if row.Op != "ALLOC" and row.Type in {"UB", "L1"}:      # 3 级
        return (3, n)
    return (4, row.Size, n)   # 4 级 小→大

def greedy_topo(nodes, G):
    indeg = {n: G.in_degree(n) for n in G.nodes}
    # 初始化 ready 队列，用 heapq 管理 (优先级, 节点)
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

def max_stay(nodes, order):
    cur = peak = 0
    for nid in order:
        row = nodes.loc[nid]
        if row.Op == "ALLOC" and row.Type in {"UB", "L1"}:
            cur += row.Size
            peak = max(peak, cur)
        elif row.Op == "FREE" and row.Type in {"UB", "L1"}:
            cur -= row.Size
    return peak

def write(case, order, peak):
    print(f"{case:<20} 节点数：{len(order):<6}  峰值(UB+L1)：{peak:<6} 字节")
    with open(f"{case}_schedule.txt", "w") as f:
        for nid in order:
            f.write(f"{nid}\n")

def main():
    cases = ["FlashAttention_Case0", "FlashAttention_Case1",
             "Matmul_Case0", "Matmul_Case1",
             "Conv_Case0", "Conv_Case1"]
    print("==== 新贪心拓扑（FREE>Cube>MTE1>普算>ALLOC小→大，heap 优化版）====")
    t0 = time.time()
    for case in cases:
        nodes, edges = read_csv(case)
        G   = build_dag(edges)
        order = greedy_topo(nodes, G)
        peak  = max_stay(nodes, order)
        write(case, order, peak)
    print(f"总耗时：{time.time() - t0:.2f} s")
    print("✅ 六个 schedule.txt 已生成！")

if __name__ == "__main__":
    main()

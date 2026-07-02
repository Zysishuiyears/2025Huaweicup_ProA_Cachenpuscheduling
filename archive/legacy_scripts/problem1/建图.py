# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 13:41:01 2025

@author: JZX
"""


# import pandas as pd
# import networkx as nx

# # 1. 读 CSV
# nodes = pd.read_csv("FlashAttention_Case0_Nodes.csv")
# edges = pd.read_csv("FlashAttention_Case0_Edges.csv")

# # 2. 建图（全部节点 + 全部边）
# G = nx.DiGraph()
# for _, row in edges.iterrows():
#     G.add_edge(int(row.StartNodeId), int(row.EndNodeId))

# # 3. 贪心拓扑序：优先 FREE 节点
# indeg = {n: G.in_degree(n) for n in G.nodes}
# ready = [n for n in G.nodes if indeg[n] == 0]
# ready.sort(key=lambda n: 0 if nodes.loc[n, "Op"] == "FREE" else 1)

# order = []
# while ready:
#     node = ready.pop(0)
#     order.append(node)
#     for succ in G.successors(node):
#         indeg[succ] -= 1
#         if indeg[succ] == 0:
#             ready.append(succ)
#             ready.sort(key=lambda n: 0 if nodes.loc[n, "Op"] == "FREE" else 1)

# # 4. 模拟 V_stay（只算 UB / L1）
# current = 0
# max_stay = 0
# for node_id in order:
#     row = nodes.loc[node_id]
#     if row.Op == "ALLOC" and row.Type in {"UB", "L1"}:
#         current += row.Size
#         max_stay = max(max_stay, current)
#     elif row.Op == "FREE" and row.Type in {"UB", "L1"}:
#         current -= row.Size

# print("最大缓存驻留量（UB+L1）：", max_stay, "字节")

# # 5. 写问题一提交文件
# with open("FlashAttention_Case0_schedule.txt", "w") as f:
#     for nid in order:
#         f.write(f"{nid}\n")
        

        
#!/usr/bin/env python3
# """
# 问题一 完整流水线
# 读 CSV → 建图 → 贪心拓扑（FREE最前→解锁大操作→ALLOC）→ 计量V_stay → 写调度序
# """
# import pandas as pd
# import networkx as nx
# import sys
# import os

# # ---------- 1. 读 CSV ----------
# def read_csv(case: str):
#     node_file = f"{case}_Nodes.csv"
#     edge_file = f"{case}_Edges.csv"
#     nodes = pd.read_csv(node_file)
#     edges = pd.read_csv(edge_file)
#     return nodes, edges

# # ---------- 2. 建全图 ----------
# def build_dag(edges):
#     G = nx.DiGraph()
#     for _, row in edges.iterrows():
#         G.add_edge(int(row.StartNodeId), int(row.EndNodeId))
#     return G

# # ---------- 3. 贪心拓扑 ----------
# def greedy_topological_order(nodes, G):
#     indeg = {n: G.in_degree(n) for n in G.nodes}
#     ready = [n for n in G.nodes if indeg[n] == 0]

#     # 关键：贪心关键字
#     def key(n):
#         row = nodes.loc[n]
#         if row.Op == "FREE":
#             return (0, 0)
#         if row.Op != "ALLOC":
#             unlock = sum(nodes.loc[s, "Size"] for s in G.successors(n)
#                          if nodes.loc[s, "Op"] == "FREE")
#             return (1, -unlock)          # 解锁越大越靠前
#         return (2, n)                    # ALLOC 放最后

#     order = []
#     while ready:
#         ready.sort(key=key)              # 每次重排保证优先级
#         node = ready.pop(0)
#         order.append(node)
#         for succ in G.successors(node):
#             indeg[succ] -= 1
#             if indeg[succ] == 0:
#                 ready.append(succ)
#     return order

# # ---------- 4. 计量 V_stay（只算 UB/L1） ----------
# def compute_max_stay(nodes, order):
#     current = 0
#     max_stay = 0
#     for nid in order:
#         row = nodes.loc[nid]
#         if row.Op == "ALLOC" and row.Type in {"UB", "L1"}:
#             current += row.Size
#             max_stay = max(max_stay, current)
#         elif row.Op == "FREE" and row.Type in {"UB", "L1"}:
#             current -= row.Size
#     return max_stay

# # ---------- 5. 写结果 ----------
# def write_output(case, order, max_stay):
#     print(f"{case} 最大缓存驻留量（UB+L1）：{max_stay} 字节")
#     with open(f"{case}_schedule.txt", "w") as f:
#         for nid in order:
#             f.write(f"{nid}\n")

# # ---------- 6. 一键跑 ----------
# def main():
#     if len(sys.argv) != 2:
#         print("用法: python prob1_pipeline.py <case名>")
#         print("例: python prob1_pipeline.py Matmul_Case0")
#         sys.exit(1)

#     case = sys.argv[1]
#     nodes, edges = read_csv(case)
#     G = build_dag(edges)
#     order = greedy_topological_order(nodes, G)
#     max_stay = compute_max_stay(nodes, order)
#     write_output(case, order, max_stay)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
问题一 完整流水线 - 六级图通用
FlashAttention / Matmul / Conv 全部适配
-----------------------------------
贪心优先级（Key 小→大）：
  0  FREE（立即释放）
  1  操作节点（解锁 FREE 越多越前）
  2  L0A/B/C-ALLOC（硬件互斥，但不算峰值）
  3  UB/L1-ALLOC（会叠加峰值，放最后）
Vstay 只累加 UB / L1 的 Size；全部顶点都进拓扑序。
"""
import pandas as pd
import networkx as nx
import os, sys, time

# ---------- 工具：读 CSV ----------
def read_csv(case: str):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    return nodes, edges

# ---------- 工具：建全图 ----------
def build_dag(edges):
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(int(r.StartNodeId), int(r.EndNodeId))
    return G

# ---------- 核心：四级贪心拓扑 ----------
def greedy_topological_order(nodes, G):
    indeg = {n: G.in_degree(n) for n in G.nodes}
    ready = [n for n in G.nodes if indeg[n] == 0]

    def key(n):
        row = nodes.loc[n]
        if row.Op == "FREE":                       # ① 真 FREE
            return (0, 0)
        if row.Op != "ALLOC":                      # ② 操作节点
            unlock = sum(nodes.loc[s, "Size"] for s in G.successors(n)
                         if nodes.loc[s, "Op"] == "FREE")
            return (1, -unlock)                    # 解锁越大越前
        # 是 ALLOC
        if row.Type in {"L0A", "L0B", "L0C"}:      # ③ L0 系列
            return (2, 0)
        return (3, n)                              # ④ UB/L1 ALLOC

    order = []
    while ready:
        ready.sort(key=key)        # 每次重排保证优先级
        node = ready.pop(0)
        order.append(node)
        for succ in G.successors(node):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                ready.append(succ)
    return order

# ---------- 核心：Vstay 只算 UB/L1 ----------
def compute_max_stay(nodes, order):
    current = max_stay = 0
    for nid in order:
        row = nodes.loc[nid]
        if row.Op == "ALLOC" and row.Type in {"UB", "L1"}:
            current += row.Size
            max_stay = max(max_stay, current)
        elif row.Op == "FREE" and row.Type in {"UB", "L1"}:
            current -= row.Size
    return max_stay

# ---------- 工具：写官方格式 ----------
def write_output(case, order, max_stay):
    print(f"{case:<20} 总节点数：{len(order):<6}  峰值(UB+L1)：{max_stay:<6} 字节")
    with open(f"{case}_schedule.txt", "w") as f:
        for nid in order:
            f.write(f"{nid}\n")

# ---------- 一键批跑 ----------
def main():
    cases = ["FlashAttention_Case0", "FlashAttention_Case1",
             "Matmul_Case0", "Matmul_Case1",
             "Conv_Case0", "Conv_Case1"]
    print("==== 问题一 批量运行 ====")
    t0 = time.time()
    for case in cases:
        nodes, edges = read_csv(case)
        G   = build_dag(edges)
        order = greedy_topological_order(nodes, G)
        peak  = compute_max_stay(nodes, order)
        write_output(case, order, peak)
    print(f"总耗时：{time.time() - t0:.2f} s")
    print("✅ 六个 *_schedule.txt 已生成，直接打包附件！")

if __name__ == "__main__":
    main()
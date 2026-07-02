# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 03:16:32 2025

@author: JZX
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque, defaultdict
import time
import ast
import os

        

# ==================== 内存池管理类 ====================
class CompactPool:
    def __init__(self, pool_type, capacity):
        self.type = pool_type
        self.capacity = capacity
        self.used_blocks = []  # (start, end, buf_id)
        self.free_blocks = [(0, capacity)]
        self.buf_to_offset = {}
        self.spill_log = []
        self.spill_cost = 0
        self.total_allocated = 0

    def alloc(self, buf_id, size, allow_spill=True, current_time=0,
              free_times=None, buf_has_copy_in_dict=None,
              W1=1.0, W2=1.0):
        if size > self.capacity:
            return False, f"请求大小 {size} 超过内存池容量 {self.capacity}"

        # Best-Fit
        best_idx, best_size = -1, float('inf')
        for i, (start, end) in enumerate(self.free_blocks):
            block_size = end - start
            if block_size >= size and block_size < best_size:
                best_idx, best_size = i, block_size

        if best_idx != -1:
            start, end = self.free_blocks[best_idx]
            self.free_blocks.pop(best_idx)
            self.used_blocks.append((start, start + size, buf_id))
            self.buf_to_offset[buf_id] = start
            self.total_allocated += size
            if start + size < end:
                self.free_blocks.append((start + size, end))
            self.free_blocks.sort()
            return True, "分配成功"

        # 如果失败，尝试 spill（仅 UB/L1）
        if allow_spill and self.type in ['UB', 'L1']:
            best_pos, victim_bufs, total_cost = self._find_best_spill_position(
                size, current_time, free_times or {},
                buf_has_copy_in_dict or {}, W1, W2
            )
            if victim_bufs:
                spill_success, _ = self.spill_victims(
                    victim_bufs, current_time,
                    free_times or {}, buf_has_copy_in_dict or {}, W1, W2
                )
                if spill_success:
                    return self.alloc(buf_id, size, allow_spill,
                                      current_time, free_times,
                                      buf_has_copy_in_dict, W1, W2)

        return False, "分配失败"

    def _find_best_spill_position(self, required_size, current_time,
                                  free_times, buf_has_copy_in_dict,
                                  W1=1.0, W2=1.0):
        candidate_positions = set()
        for start, _ in self.free_blocks:
            candidate_positions.add(start)
        for start, _, _ in self.used_blocks:
            candidate_positions.add(start)

        best_position, best_victims, best_total_cost = -1, [], float('inf')
        for pos in sorted(candidate_positions):
            if pos + required_size > self.capacity:
                continue
            overlapping_blocks, total_cost = [], 0
            for start, end, buf_id in self.used_blocks:
                if not (end <= pos or start >= pos + required_size):
                    size = end - start
                    free_time = free_times.get(buf_id, current_time + 1000)
                    tag = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
                    remaining_time = max(1, free_time - current_time)
                    cost = tag * W1 / size + W2 / remaining_time
                    overlapping_blocks.append((buf_id, cost, size))
                    total_cost += cost
            if overlapping_blocks and total_cost < best_total_cost:
                best_position, best_victims, best_total_cost = (
                    pos, [b for b, _, _ in overlapping_blocks], total_cost)
        return best_position, best_victims, best_total_cost

    def spill_victims(self, victim_bufs, current_time,
                      free_times, buf_has_copy_in_dict,
                      W1=1.0, W2=1.0):
        for victim_buf in victim_bufs:
            victim_block = next((b for b in self.used_blocks if b[2] == victim_buf), None)
            if not victim_block:
                continue
            start, end, buf_id = victim_block
            size = end - start
            cost_coeff = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
            cost = cost_coeff * size
            self.spill_cost += cost
            self.spill_log.append((victim_buf, cost, current_time))
            self.free(victim_buf)
        return True, self.spill_cost

    def free(self, buf_id):
        victim_block = next((b for b in self.used_blocks if b[2] == buf_id), None)
        if not victim_block:
            return
        self.used_blocks.remove(victim_block)
        start, end, _ = victim_block
        self.total_allocated -= (end - start)
        self.free_blocks.append((start, end))
        self.free_blocks.sort()
        merged = []
        if self.free_blocks:
            s, e = self.free_blocks[0]
            for i in range(1, len(self.free_blocks)):
                ns, ne = self.free_blocks[i]
                if ns <= e:
                    e = max(e, ne)
                else:
                    merged.append((s, e))
                    s, e = ns, ne
            merged.append((s, e))
        self.free_blocks = merged

    def get_offset(self, buf_id):
        return self.buf_to_offset.get(buf_id, -1)

# ==================== 多池管理器 ====================
class MultiPoolManager:
    def __init__(self, capacities, buf_has_copy_in_dict):
        self.pools = {t: CompactPool(t, capacities[t]) for t in capacities}
        self.buf_has_copy_in_dict = buf_has_copy_in_dict
        self.global_spill_log, self.total_spill_cost = [], 0
        self.l0_alloc_failed = False

    def alloc(self, buf_id, size, buf_type, current_time,
              free_time, W1=1.0, W2=1.0):
        pool = self.pools[buf_type]
        allow_spill = buf_type not in ['L0A', 'L0B', 'L0C']
        success, message = pool.alloc(buf_id, size, allow_spill,
                                      current_time, {buf_id: free_time},
                                      self.buf_has_copy_in_dict, W1, W2)
        if not success and not allow_spill:
            self.l0_alloc_failed = True
        return success, message

    def free(self, buf_id, buf_type):
        self.pools[buf_type].free(buf_id)

    def get_offset(self, buf_id, buf_type):
        return self.pools[buf_type].get_offset(buf_id)

    def collect_spill_info(self):
        self.global_spill_log, self.total_spill_cost = [], 0
        for p in self.pools.values():
            self.global_spill_log.extend(p.spill_log)
            self.total_spill_cost += p.spill_cost
        return self.total_spill_cost, self.global_spill_log

# ==================== 时钟数赋值 ====================
def assign_cycles(nodes):
    def cycle_rule(row):
        op = str(row['Op']).upper()
        if op in ['ALLOC', 'FREE', 'COPY_IN', 'COPY_OUT']:
            return 0
        if 'Cycles' in row and pd.notna(row['Cycles']):
            return int(row['Cycles'])
        if op in ['MTE1', 'MTE2', 'MTE3', 'MMAD', 'CUBE']:
            return max(1, int(row.get('Size', 1)) // 16)
        return 1
    nodes['Cycles'] = nodes.apply(cycle_rule, axis=1)
    return nodes

# ==================== 拓扑排序 ====================
def build_dag_fast(edges):
    G = nx.DiGraph()
    G.add_edges_from([(int(r.StartNodeId), int(r.EndNodeId)) for _, r in edges.iterrows()])
    return G

def greedy_topo_with_L0_pool(nodes, graph, capacities):
    node_attrs = {nid: row.to_dict() for nid, row in nodes.iterrows()}
    in_degree = {n: graph.in_degree(n) for n in graph.nodes}
    zero_degree_nodes = deque([n for n in graph.nodes if in_degree[n] == 0])
    execution_order = []

    l0_pools = {t: CompactPool(t, capacities[t]) for t in ['L0A', 'L0B', 'L0C']}

    while zero_degree_nodes:
        min_cost, selected_node = float('inf'), None
        for node in list(zero_degree_nodes):
            attr = node_attrs[node]
            if attr['Op'] == "ALLOC" and attr['Type'] in l0_pools:
                pool = l0_pools[attr['Type']]
                success, _ = pool.alloc(attr['BufId'], attr['Size'], allow_spill=False)
                if success:
                    pool.free(attr['BufId'])
                else:
                    continue
            if attr['Cost'] < min_cost:
                min_cost, selected_node = attr['Cost'], node
        if selected_node is None:
            selected_node = zero_degree_nodes[0]
        current_node = selected_node
        current_attr = node_attrs[current_node]
        if current_attr['Op'] == "ALLOC" and current_attr['Type'] in l0_pools:
            l0_pools[current_attr['Type']].alloc(current_attr['BufId'], current_attr['Size'], allow_spill=False)
        elif current_attr['Op'] == "FREE" and current_attr['Type'] in l0_pools:
            l0_pools[current_attr['Type']].free(current_attr['BufId'])
        for succ in graph.successors(current_node):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                zero_degree_nodes.append(succ)
        execution_order.append(current_node)
        if current_node in zero_degree_nodes:
            zero_degree_nodes.remove(current_node)
    return execution_order

# ==================== 时钟调度计算 ====================
def compute_clock(schedule, nodes, edges):
    pred_end, pipe_last = defaultdict(int), defaultdict(int)
    S, E = {}, {}
    edge_map = defaultdict(list)
    for _, r in edges.iterrows():
        edge_map[int(r.StartNodeId)].append(int(r.EndNodeId))
    for nid in schedule:
        row = nodes.loc[nid]
        p = row.Pipe
        cycles = int(row['Cycles'])
        early = max(pred_end[nid], pipe_last[p])
        S[nid], E[nid] = early, early + cycles
        pipe_last[p] = E[nid]
        for succ in edge_map[nid]:
            pred_end[succ] = max(pred_end[succ], E[nid])
    return S, E, max(E.values(), default=0)

# ==================== 数据读取 ====================
'''
def read_csv_optimized(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    if 'Type' not in nodes: nodes['Type'] = ''
    if 'BufId' not in nodes: nodes['BufId'] = -1
    if 'Size' not in nodes: nodes['Size'] = 0
    nodes['Cost'] = nodes.apply(lambda r: r['Size'] if r['Op'] == 'ALLOC' else -r['Size'] if r['Op'] == 'FREE' else 0, axis=1)
    if "NodeId" in nodes: nodes.set_index("NodeId", inplace=True)
    elif "Id" in nodes: nodes.set_index("Id", inplace=True)
    return assign_cycles(nodes), edges
'''

def read_csv_optimized(case):
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")

    if 'Type' not in nodes: 
        nodes['Type'] = ''
    if 'BufId' not in nodes: 
        nodes['BufId'] = -1
    if 'Size' not in nodes: 
        nodes['Size'] = 0

    def compute_cost(r):
        if str(r['Op']).upper() == 'ALLOC':
            return int(r['Size'])
        elif str(r['Op']).upper() == 'FREE':
            return -int(r['Size'])
        else:
            return 0

    nodes['Cost'] = nodes.apply(compute_cost, axis=1)

    if "NodeId" in nodes:
        nodes.set_index("NodeId", inplace=True)
    elif "Id" in nodes:
        nodes.set_index("Id", inplace=True)

    # 加上时钟数赋值
    nodes = assign_cycles(nodes)
    return nodes, edges



def parameter_scan(cases, capacities, W1_list, W2_list, out_dir_scan="ParamScan"):
    os.makedirs(out_dir_scan, exist_ok=True)
    
    for case in cases:
        print(f"\n=== 参数扫描 {case} ===")

        spill_matrix = np.zeros((len(W1_list), len(W2_list)))
        time_matrix = np.zeros((len(W1_list), len(W2_list)))

        for i, W1 in enumerate(W1_list):
            for j, W2 in enumerate(W2_list):
                nodes, edges = read_csv_optimized(case)
                G = build_dag_fast(edges)
                schedule = greedy_topo_with_L0_pool(nodes, G, capacities)

                # 执行内存管理 + 调度
                buf_free_positions = {nodes.loc[nid].BufId: k for k, nid in enumerate(schedule) if nodes.loc[nid].Op == 'FREE'}
                manager = MultiPoolManager(capacities, {})
                for k, nid in enumerate(schedule):
                    row = nodes.loc[nid]
                    if row.Op == 'ALLOC':
                        free_time = buf_free_positions.get(row.BufId, k + 1000)
                        manager.alloc(row.BufId, row.Size, row.Type, k, free_time, W1, W2)
                    elif row.Op == 'FREE':
                        manager.free(row.BufId, row.Type)

                total_cost, _ = manager.collect_spill_info()
                _, _, T_total = compute_clock(schedule, nodes, edges)

                spill_matrix[i, j] = total_cost
                time_matrix[i, j] = T_total

        # ====== 这里加绘图 ======
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Spill cost 热力图 ---
        im1 = axes[0].imshow(spill_matrix, cmap="viridis", origin="lower")
        axes[0].set_title(f"{case} - Spill Cost")
        axes[0].set_xlabel("W2")
        axes[0].set_ylabel("W1")
        axes[0].set_xticks(np.arange(len(W2_list)))
        axes[0].set_yticks(np.arange(len(W1_list)))
        axes[0].set_xticklabels(W2_list)
        axes[0].set_yticklabels(W1_list)
        axes[0].grid(color='white', linestyle='--', linewidth=0.5)
        plt.colorbar(im1, ax=axes[0])

        # 数值写在格子中心
        for i in range(len(W1_list)):
            for j in range(len(W2_list)):
                val = spill_matrix[i, j]
                text_color = "white" if val > spill_matrix.max()/2 else "black"
                axes[0].text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=7)

        # --- Execution time 热力图 ---
        im2 = axes[1].imshow(time_matrix, cmap="plasma", origin="lower")
        axes[1].set_title(f"{case} - Execution Time")
        axes[1].set_xlabel("W2")
        axes[1].set_ylabel("W1")
        axes[1].set_xticks(np.arange(len(W2_list)))
        axes[1].set_yticks(np.arange(len(W1_list)))
        axes[1].set_xticklabels(W2_list)
        axes[1].set_yticklabels(W1_list)
        axes[1].grid(color='white', linestyle='--', linewidth=0.5)
        plt.colorbar(im2, ax=axes[1])

        for i in range(len(W1_list)):
            for j in range(len(W2_list)):
                val = time_matrix[i, j]
                text_color = "white" if val > time_matrix.max()/2 else "black"
                axes[1].text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_scan, f"{case}_heatmap.png"))
        plt.close()

        print(f"完成 {case} 参数扫描图，已保存到 {out_dir_scan}/{case}_heatmap.png")
        

def conservative_left_slide(final_schedule, nodes, edges):
    """
    第三问：零新增 SPILL 前提下的保守时间左滑
    只把节点压到理论最早可启动时刻，不改变顺序、不新增 SPILL、不调地址
    返回：新起止时钟 S_new, E_new, 新总周期 T_new
    """
    from collections import defaultdict
    import pandas as pd

    # 1. 前驱最晚结束表（图依赖）
    pred_end = defaultdict(int)
    for _, r in edges.iterrows():
        u, v = int(r.StartNodeId), int(r.EndNodeId)
        pred_end[v] = max(pred_end[v], 0)  # 后面更新为 E[u]

    # 2. 地址空出表（FREE 序列位置当时刻）
    addr_free = {}  # bufId -> FREE 时刻
    for i, nid in enumerate(final_schedule):
        row = nodes.loc[nid]
        if row.Op == 'FREE':
            addr_free[row.BufId] = i

    # 3. 同列串行尾巴 & 逐节点左滑
    pipe_last = defaultdict(int)  # Pipe -> 列尾时间
    S_new, E_new = {}, {}

    for i, nid in enumerate(final_schedule):
        row = nodes.loc[nid]
        p = row.Pipe
        cycles = int(row.Cycles) if pd.notna(row.get('Cycles', None)) else 0

        # 理论最早可启动 = 图依赖 vs 列尾 vs 地址空出
        early = max(pred_end[nid], pipe_last[p], addr_free.get(row.get('BufId'), 0))
        S_new[nid] = early
        E_new[nid] = early + cycles
        pipe_last[p] = E_new[nid]

    T_new = max(E_new.values(), default=0)
    return S_new, E_new, T_new

def main():
    capacities = {'L1': 4096, 'UB': 1024, 'L0A': 256, 'L0B': 256, 'L0C': 512}
    cases = ["FlashAttention_Case0", "FlashAttention_Case1", 
             "Matmul_Case0", "Matmul_Case1", "Conv_Case0", "Conv_Case1"]

    out_dir2 = "Problem2_Output"
    out_dir3 = "Problem3_Output"
    os.makedirs(out_dir2, exist_ok=True)
    os.makedirs(out_dir3, exist_ok=True)

    print("==== 二问：数据搬运量 + 三问：流水时钟 ====")
    for case in cases:
        print(f"\n=== 处理 {case} ===")
        try:
            # === 读入数据 ===
            nodes, edges = read_csv_optimized(case)
            G = build_dag_fast(edges)
            schedule = greedy_topo_with_L0_pool(nodes, G, capacities)

            # === 第二问：内存池管理模拟 ===
            buf_free_positions = {}
            for i, nid in enumerate(schedule):
                row = nodes.loc[nid]
                if row.Op == 'FREE':
                    buf_free_positions[row.BufId] = i

            buf_has_copy_in_dict = {}
            manager = MultiPoolManager(capacities, buf_has_copy_in_dict)

            for i, nid in enumerate(schedule):
                row = nodes.loc[nid]
                if row.Op == 'ALLOC':
                    free_time = buf_free_positions.get(row.BufId, i + 1000)
                    manager.alloc(row.BufId, row.Size, row.Type, i, free_time)
                elif row.Op == 'FREE':
                    manager.free(row.BufId, row.Type)

            total_cost, spill_log = manager.collect_spill_info()

            # === 第三问：时钟调度 ===
            S, E, T_total = compute_clock(schedule, nodes, edges)

            slide_start = time.time()
            S_new, E_new, T_new = conservative_left_slide(schedule, nodes, edges)
            slide_time = time.time() - slide_start

            print(f"搬运量: {total_cost}, 流水总时钟: {T_total}")
            print(f"搬运量: {total_cost}, SPILL次数: {len(spill_log)}")
            print(f"左滑耗时: {slide_time:.3f}s, 左滑后周期: {T_new}, 时间下降: {(T_total - T_new) / T_total * 100:.2f}%")

            # ========== 文件输出 ==========
            # 第二问输出
            with open(os.path.join(out_dir2, f"{case}_schedule.txt"), "w") as f:
                f.write("\n".join(str(n) for n in schedule))

            with open(os.path.join(out_dir2, f"{case}_memory.txt"), "w") as f:
                for pool in manager.pools.values():
                    for buf, offset in pool.buf_to_offset.items():
                        f.write(f"{buf}:{offset}\n")

            with open(os.path.join(out_dir2, f"{case}_spill.txt"), "w") as f:
                for (buf, _, _) in spill_log:
                    new_offset = (manager.get_offset(buf, 'UB') or 
                                  manager.get_offset(buf, 'L1'))
                    if new_offset != -1:
                        f.write(f"{buf}:{new_offset}\n")

            # 第三问输出
            with open(os.path.join(out_dir3, f"{case}_schedule.txt"), "w") as f:
                f.write("\n".join(str(n) for n in schedule))

            with open(os.path.join(out_dir3, f"{case}_memory.txt"), "w") as f:
                for pool in manager.pools.values():
                    for buf, offset in pool.buf_to_offset.items():
                        f.write(f"{buf}:{offset}\n")

            with open(os.path.join(out_dir3, f"{case}_spill.txt"), "w") as f:
                for (buf, _, _) in spill_log:
                    new_offset = (manager.get_offset(buf, 'UB') or 
                                  manager.get_offset(buf, 'L1'))
                    if new_offset != -1:
                        f.write(f"{buf}:{new_offset}\n")

        except Exception as e:
            print(f"处理 {case} 时出错: {e}")

if __name__ == "__main__":
    capacities = {'L1': 4096, 'UB': 1024, 'L0A': 256, 'L0B': 256, 'L0C': 512}
    cases = ["FlashAttention_Case0", "FlashAttention_Case1", 
             "Matmul_Case0", "Matmul_Case1", "Conv_Case0", "Conv_Case1"]

    # 参数扫描范围（可以改）
    W1_list = [0.5, 1.0, 2.0]
    W2_list = [0.5, 1.0, 2.0]

    parameter_scan(cases, capacities, W1_list, W2_list)


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:36:05 2025

@author: JZX
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import deque, defaultdict
import heapq
import time
import ast
import os
import matplotlib.pyplot as plt
import numpy as np

# ==================== 内存池管理类 ====================
class CompactPool:
    def __init__(self, pool_type, capacity):
        self.type = pool_type
        self.capacity = capacity
        self.used_blocks = []  # (start, end, buf_id)
        self.free_blocks = [(0, capacity)]  # (start, end)
        self.buf_to_offset = {}
        self.spill_log = []
        self.spill_cost = 0
        self.total_allocated = 0

    def alloc(self, buf_id, size, allow_spill=True, current_time=0,
              free_times=None, buf_has_copy_in_dict=None, W1=1.0, W2=1.0):
        """分配内存，返回(成功标志, 消息)"""
        if size > self.capacity:
            return False, f"请求大小 {size} 超过内存池容量 {self.capacity}"

        # Best-Fit: 找最小但能容纳的free块
        best_idx = -1
        best_size = float('inf')
        best_start = -1

        for i, (start, end) in enumerate(self.free_blocks):
            block_size = end - start
            if block_size >= size and block_size < best_size:
                best_idx = i
                best_size = block_size
                best_start = start

        if best_idx != -1:
            start, end = self.free_blocks[best_idx]
            self.free_blocks.pop(best_idx)
            self.used_blocks.append((start, start + size, buf_id))
            # === 修复：维护 offset 与总分配量 ===
            self.buf_to_offset[buf_id] = start
            self.total_allocated += size

            if start + size < end:
                self.free_blocks.insert(best_idx, (start + size, end))
            self.free_blocks.sort()
            return True, "分配成功"

        # 如果分配失败且允许spill，尝试spill操作
        if allow_spill and self.type in ['UB', 'L1']:
            if free_times is None:
                free_times = {}
            if buf_has_copy_in_dict is None:
                buf_has_copy_in_dict = {}

            best_pos, victim_bufs, total_cost = self._find_best_spill_position(
                size, current_time, free_times, buf_has_copy_in_dict, W1, W2
            )

            if victim_bufs:
                spill_success, spill_results = self.spill_victims(
                    victim_bufs, current_time, free_times, buf_has_copy_in_dict, W1, W2
                )

                if spill_success:
                    return self.alloc(buf_id, size, allow_spill, current_time,
                                      free_times, buf_has_copy_in_dict, W1, W2)

        return False, "内存不足，分配失败"

    def _find_best_spill_position(self, required_size, current_time,
                                  free_times, buf_has_copy_in_dict,
                                  W1=1.0, W2=1.0):
        candidate_positions = set()
        for start, end in self.free_blocks:
            candidate_positions.add(start)
        for start, end, buf_id in self.used_blocks:
            candidate_positions.add(start)
        candidate_positions = sorted(candidate_positions)

        best_position = -1
        best_victims = []
        best_total_cost = float('inf')

        for pos in candidate_positions:
            if pos + required_size > self.capacity:
                continue

            overlapping_blocks = []
            total_cost = 0
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
                best_position = pos
                best_victims = [buf_id for buf_id, cost, size in overlapping_blocks]
                best_total_cost = total_cost

        return best_position, best_victims, best_total_cost

    def spill_victims(self, victim_bufs, current_time,
                      free_times, buf_has_copy_in_dict, W1=1.0, W2=1.0):
        total_cost = 0
        spilled_info = []
        for victim_buf in victim_bufs:
            victim_block = None
            for block in self.used_blocks:
                if block[2] == victim_buf:
                    victim_block = block
                    break
            if not victim_block:
                continue
            start, end, buf_id = victim_block
            size = end - start
            tag = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
            cost_coeff = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
            cost = cost_coeff * size
            total_cost += cost
            self.spill_cost += cost
            self.spill_log.append((victim_buf, cost, current_time))
            spilled_info.append((victim_buf, cost, size))
            self.free(victim_buf)
        return True, spilled_info

    def free(self, buf_id):
        block_to_free = None
        for block in self.used_blocks:
            if block[2] == buf_id:
                block_to_free = block
                break
        if block_to_free is None:
            return
        self.used_blocks.remove(block_to_free)
        start, end, _ = block_to_free
        size = end - start
        self.total_allocated -= size
        # === 修复：删除偏移映射 ===
        self.buf_to_offset.pop(buf_id, None)

        self.free_blocks.append((start, end))
        self.free_blocks.sort()
        merged = []
        if self.free_blocks:
            current_start, current_end = self.free_blocks[0]
            for i in range(1, len(self.free_blocks)):
                start, end = self.free_blocks[i]
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            merged.append((current_start, current_end))
        self.free_blocks = merged

    def get_offset(self, buf_id):
        return self.buf_to_offset.get(buf_id, -1)

    def get_usage(self):
        return self.total_allocated, self.capacity


# ==================== 多池管理器 ====================
class MultiPoolManager:
    def __init__(self, capacities, buf_has_copy_in_dict):
        self.pools = {
            'UB': CompactPool('UB', capacities['UB']),
            'L1': CompactPool('L1', capacities['L1']),
            'L0A': CompactPool('L0A', capacities['L0A']),
            'L0B': CompactPool('L0B', capacities['L0B']),
            'L0C': CompactPool('L0C', capacities['L0C']),
        }
        self.buf_has_copy_in_dict = buf_has_copy_in_dict
        self.global_spill_log = []
        self.total_spill_cost = 0
        self.spill_operations = []
        self.l0_alloc_failed = False

    def alloc(self, buf_id, size, buf_type, current_time, free_time, W1=1.0, W2=1.0):
        pool = self.pools.get(buf_type)
        if not pool:
            return False, f"未知的内存池类型: {buf_type}"
        allow_spill = buf_type not in ['L0A', 'L0B', 'L0C']
        free_times = {buf_id: free_time} if allow_spill else {}
        success, message = pool.alloc(
            buf_id, size, allow_spill, current_time,
            free_times, self.buf_has_copy_in_dict, W1, W2
        )
        if not success and not allow_spill and not self.l0_alloc_failed:
            self.l0_alloc_failed = True
        return success, message

    def free(self, buf_id, buf_type):
        pool = self.pools.get(buf_type)
        if pool:
            pool.free(buf_id)

    def get_offset(self, buf_id, buf_type):
        pool = self.pools.get(buf_type)
        return pool.get_offset(buf_id) if pool else -1

    def collect_spill_info(self):
        self.global_spill_log = []
        self.total_spill_cost = 0
        for pool_name, pool in self.pools.items():
            self.global_spill_log.extend(pool.spill_log)
            self.total_spill_cost += pool.spill_cost
        return self.total_spill_cost, self.global_spill_log


# ==================== 贪心拓扑排序（带 L0 检查） ====================
def greedy_topo_with_L0_pool(nodes, graph, capacities):
    node_attrs = {}
    for nid, row in nodes.iterrows():
        node_attrs[nid] = {
            'Op': row['Op'],
            'Type': row.get('Type', ''),
            'Cost': row['Cost'],
            'BufId': row.get('BufId', -1),
            'Size': row.get('Size', 0)
        }
    in_degree = {n: graph.in_degree(n) for n in graph.nodes}
    zero_degree_nodes = deque([n for n in graph.nodes if in_degree[n] == 0])
    execution_order = []
    l0_pools = {
        'L0A': CompactPool('L0A', capacities['L0A']),
        'L0B': CompactPool('L0B', capacities['L0B']),
        'L0C': CompactPool('L0C', capacities['L0C']),
    }
    l0_alloc_failed = False

    while zero_degree_nodes:
        min_cost = float('inf')
        selected_node = None
        for node in list(zero_degree_nodes):
            attr = node_attrs[node]
            can_allocate = True
            if attr['Op'] == "ALLOC" and attr['Type'] in l0_pools:
                buf_type = attr['Type']
                pool = l0_pools[buf_type]
                success, _ = pool.alloc(attr['BufId'], attr['Size'], allow_spill=False)
                if not success:
                    can_allocate = False
                else:
                    current_buf_count = len(pool.used_blocks)
                    current_usage_ratio = pool.total_allocated / pool.capacity
                    if current_buf_count > 2 and current_usage_ratio > 0.5:
                        can_allocate = False
                    pool.free(attr['BufId'])
            if not can_allocate:
                continue
            if attr['Cost'] < min_cost:
                min_cost = attr['Cost']
                selected_node = node
        if selected_node is None:
            selected_node = zero_degree_nodes[0]
            if not l0_alloc_failed:
                l0_alloc_failed = True
        current_node = selected_node
        current_attr = node_attrs[current_node]
        if current_attr['Op'] == "ALLOC" and current_attr['Type'] in l0_pools:
            pool = l0_pools[current_attr['Type']]
            success, _ = pool.alloc(current_attr['BufId'], current_attr['Size'], allow_spill=False)
            if not success and not l0_alloc_failed:
                l0_alloc_failed = True
        elif current_attr['Op'] == "FREE" and current_attr['Type'] in l0_pools:
            pool = l0_pools[current_attr['Type']]
            pool.free(current_attr['BufId'])
        for successor in graph.successors(current_node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                zero_degree_nodes.append(successor)
        execution_order.append(current_node)
        if current_node in zero_degree_nodes:
            zero_degree_nodes.remove(current_node)

    if l0_alloc_failed:
        print("警告：在拓扑排序过程中检测到L0内存分配失败")
    else:
        print("L0内存分配全部成功")
    return execution_order


# ==================== 读取数据 ====================
def read_csv_optimized(case):
    base_path = os.path.dirname(os.path.abspath(__file__))
    case_path = os.path.join(base_path, case)
    nodes = pd.read_csv(f"{case_path}_Nodes.csv")
    edges = pd.read_csv(f"{case_path}_Edges.csv")
    if 'Type' not in nodes.columns:
        nodes['Type'] = ''
    if 'BufId' not in nodes.columns:
        nodes['BufId'] = -1
    if 'Size' not in nodes.columns:
        nodes['Size'] = 0

    def calculate_cost(row):
        if row['Type'] in {'UB', 'L1'}:
            if row['Op'] == 'ALLOC':
                return row['Size']
            elif row['Op'] == 'FREE':
                return -row['Size']
        return 0

    nodes['Cost'] = nodes.apply(calculate_cost, axis=1).astype(int)
    if "NodeId" in nodes.columns:
        nodes.set_index("NodeId", inplace=True)
    elif "Id" in nodes.columns:
        nodes.set_index("Id", inplace=True)
    return nodes, edges


def build_dag_fast(edges):
    edge_tuples = [(int(r.StartNodeId), int(r.EndNodeId)) for _, r in edges.iterrows()]
    G = nx.DiGraph()
    G.add_edges_from(edge_tuples)
    return G


def schedule_with_memory_management(nodes, edges, capacities, W1=1.0, W2=1.0):
    G = build_dag_fast(edges)
    execution_order = greedy_topo_with_L0_pool(nodes, G, capacities)
    buf_free_positions = {}
    buf_has_copy_in_dict = {}
    for i, node_id in enumerate(execution_order):
        if node_id not in nodes.index:
            continue
        row = nodes.loc[node_id]
        if row.Op == 'FREE':
            buf_free_positions[row.BufId] = i
        if row.Op == 'COPY_IN' and pd.notna(row.get('Bufs', None)):
            bufs_value = row['Bufs']
            if isinstance(bufs_value, str):
                try:
                    if bufs_value.startswith('[') and bufs_value.endswith(']'):
                        buf_list = ast.literal_eval(bufs_value)
                    else:
                        buf_list = [int(bufs_value)]
                except:
                    buf_list = [int(bufs_value)]
            else:
                buf_list = [bufs_value]
            for buf_id in buf_list:
                buf_has_copy_in_dict[buf_id] = True
    manager = MultiPoolManager(capacities, buf_has_copy_in_dict)
    final_schedule = []
    memory_alloc = {}
    for i, node_id in enumerate(execution_order):
        if node_id not in nodes.index:
            final_schedule.append(node_id)
            continue
        row = nodes.loc[node_id]
        final_schedule.append(node_id)
        if row.Op == 'ALLOC':
            free_time = buf_free_positions.get(row.BufId, i + 1000)
            success, message = manager.alloc(
                row.BufId, row.Size, row.Type, i, free_time, W1, W2
            )
            if success:
                offset = manager.get_offset(row.BufId, row.Type)
                if offset != -1:
                    memory_alloc[row.BufId] = offset
            else:
                if row.Type not in ['L0A', 'L0B', 'L0C']:
                    print(f"警告: 节点 {node_id} 分配失败: {message}")
        elif row.Op == 'FREE':
            manager.free(row.BufId, row.Type)
    total_cost, spill_log = manager.collect_spill_info()
    if manager.l0_alloc_failed:
        print("警告：在内存管理过程中检测到L0内存分配失败")
    else:
        print("L0内存管理全部成功")
    return final_schedule, memory_alloc, spill_log, total_cost, []


# ==================== 时钟计算 & 保守左滑 ====================
def compute_clock_serial(schedule, nodes):
    """串行计算时钟（只考虑顺序，不考虑流水）"""
    S, E = {}, {}
    time_cursor = 0
    for nid in schedule:
        row = nodes.loc[nid]
        cycles = int(row.Cycles) if pd.notna(row.get('Cycles', None)) else 0
        S[nid] = time_cursor
        E[nid] = time_cursor + cycles
        time_cursor = E[nid]
    T_total = time_cursor
    return S, E, T_total


def conservative_left_slide(schedule, nodes, edges):
    """保守左滑：零新增 SPILL，仅同列内压紧"""
    pred_end = defaultdict(int)
    edge_map = defaultdict(list)
    for _, r in edges.iterrows():
        u, v = int(r.StartNodeId), int(r.EndNodeId)
        edge_map[u].append(v)

    addr_free = {}
    for i, nid in enumerate(schedule):
        row = nodes.loc[nid]
        if row.Op == 'FREE':
            addr_free[row.BufId] = i

    pipe_last = defaultdict(int)
    S_new, E_new = {}, {}
    for i, nid in enumerate(schedule):
        row = nodes.loc[nid]
        p = row.Pipe if 'Pipe' in row else "DEFAULT"
        cycles = int(row.Cycles) if pd.notna(row.get('Cycles', None)) else 0
        early = max(pred_end[nid], pipe_last[p], addr_free.get(row.get('BufId'), 0))
        S_new[nid] = early
        E_new[nid] = early + cycles
        pipe_last[p] = E_new[nid]
        for succ in edge_map[nid]:
            pred_end[succ] = max(pred_end[succ], E_new[nid])
    T_new = max(E_new.values(), default=0)
    return S_new, E_new, T_new


# def sensitivity_analysis(cases, capacities):
#     """扫描 W1, W2 参数组合，输出每个数据集的搬运量变化曲线"""
#     W1_values = [0.5, 1.0, 1.5, 2.0]
#     W2_values = [0.5, 1.0, 1.5, 2.0]
#     combos = [(w1, w2) for w1 in W1_values for w2 in W2_values]

#     results = {case: [] for case in cases}

#     for case in cases:
#         print(f"\n=== 参数扫描 {case} ===")
#         nodes, edges = read_csv_optimized(case)
#         for (w1, w2) in combos:
#             try:
#                 schedule, memory_alloc, spill_log, total_cost, _ = schedule_with_memory_management(
#                     nodes, edges, capacities, W1=w1, W2=w2
#                 )
#                 results[case].append((w1, w2, total_cost))
#                 print(f"W1={w1}, W2={w2} => Spill={total_cost}")
#             except Exception as e:
#                 print(f"  跑 {case} W1={w1}, W2={w2} 出错: {e}")
#                 results[case].append((w1, w2, None))

#     # 绘制图像
#     for case in cases:
#         plt.figure(figsize=(8,5))
#         data = results[case]
#         xs = [f"{w1},{w2}" for w1,w2,_ in data]
#         ys = [spill if spill is not None else 0 for _,_,spill in data]
#         plt.bar(xs, ys, color="skyblue")
#         plt.title(f"{case} - Spill Cost Sensitivity")
#         plt.xlabel("W1, W2 组合")
#         plt.ylabel("总搬运量 (bytes)")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(f"Problem2/{case}_sensitivity.png")
#         plt.close()

#     print("\n灵敏度分析完成，图像已保存到 Problem2/ 目录下。")
    


def sensitivity_analysis_heatmap(cases, capacities):
    """扫描 W1, W2 参数组合，并绘制热力图"""
    W1_values = [0.5, 1.0, 1.5, 2.0]
    W2_values = [0.5, 1.0, 1.5, 2.0]

    results = {case: np.zeros((len(W2_values), len(W1_values))) for case in cases}

    for case in cases:
        print(f"\n=== 参数扫描 {case} ===")
        nodes, edges = read_csv_optimized(case)
        for i, w2 in enumerate(W2_values):
            for j, w1 in enumerate(W1_values):
                try:
                    _, _, _, total_cost, _ = schedule_with_memory_management(
                        nodes, edges, capacities, W1=w1, W2=w2
                    )
                    results[case][i, j] = total_cost
                    print(f"W1={w1}, W2={w2} => Spill={total_cost}")
                except Exception as e:
                    print(f"  跑 {case} W1={w1}, W2={w2} 出错: {e}")
                    results[case][i, j] = np.nan  # 标记错误位置

    # 绘制热力图
    for case in cases:
        plt.figure(figsize=(6,5))
        data = results[case]

        im = plt.imshow(data, cmap="YlOrRd", origin="lower")

        # 标上数字
        for i in range(len(W2_values)):
            for j in range(len(W1_values)):
                val = data[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{int(val)}", ha="center", va="center", color="black")

        plt.colorbar(im, label="总搬运量 (bytes)")
        plt.xticks(range(len(W1_values)), W1_values)
        plt.yticks(range(len(W2_values)), W2_values)
        plt.xlabel("W1")
        plt.ylabel("W2")
        plt.title(f"{case} - Spill Cost Heatmap")

        plt.tight_layout()
        plt.savefig(f"Problem2/{case}_sensitivity_heatmap.png")
        plt.close()

    print("\n灵敏度分析完成，热力图已保存到 Problem2/ 目录下。")

# ==================== 输出 ====================
def write_outputs(case, schedule, memory_alloc, spill_log, total_spill_cost):
    os.makedirs(f"Problem2", exist_ok=True)
    prefix = f"Problem2/{case}"
    with open(f"{prefix}_schedule.txt", "w") as f:
        for node_id in schedule:
            f.write(f"{node_id}\n")
    with open(f"{prefix}_memory.txt", "w") as f:
        for buf_id in sorted(memory_alloc.keys()):
            offset = memory_alloc[buf_id]
            f.write(f"{buf_id}:{offset}\n")
    with open(f"{prefix}_spill.txt", "w") as f:
        for buf_id, cost, spill_time in spill_log:
            f.write(f"{buf_id}:{cost}\n")
        f.write(f"TotalSpillCost:{total_spill_cost}\n")


# ==================== 主函数 ====================
def main():
    capacities = {
        'L1': 4096, 'UB': 1024,
        'L0A': 256, 'L0B': 256, 'L0C': 512
    }
    cases = [
        "FlashAttention_Case0", "FlashAttention_Case1",
        "Matmul_Case0", "Matmul_Case1",
        "Conv_Case0", "Conv_Case1"
    ]
    print("==== 第二问 + 第三问（零SPILL + 保守左滑） ====")
    total_start = time.time()
    for case in cases:
        print(f"\n=== 处理 {case} ===")
        try:
            nodes, edges = read_csv_optimized(case)
            schedule, memory_alloc, spill_log, total_cost, _ = schedule_with_memory_management(
                nodes, edges, capacities, W1=1.0, W2=1.0
            )
            # 第二问：顺序时钟
            S_orig, E_orig, T_orig = compute_clock_serial(schedule, nodes)
            # 第三问：保守左滑
            S_new, E_new, T_new = conservative_left_slide(schedule, nodes, edges)
            write_outputs(case, schedule, memory_alloc, spill_log, total_cost)
            print(f"原总周期: {T_orig} | 左滑后周期: {T_new}")
            print(f"时间下降: {(T_orig - T_new) / max(1,T_orig) * 100:.2f}%")
            print(f"总额外搬运量: {total_cost} | SPILL次数: {len(spill_log)}")
        except Exception as e:
            print(f"处理 {case} 时出错: {e}")
            import traceback; traceback.print_exc()
    total_time = time.time() - total_start
    print(f"\n==== 全部完成，总耗时 {total_time:.2f}s ====")
    # 参数扫描
    sensitivity_analysis_heatmap(cases, capacities)


if __name__ == "__main__":
    main()

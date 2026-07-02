import pandas as pd
import numpy as np
import networkx as nx
from collections import deque, defaultdict
import heapq
import time
import ast
import os

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

    def alloc(self, buf_id, size):
        """分配内存，返回(成功标志, 消息)"""
        # 检查容量限制
        if self.total_allocated + size > self.capacity:
            return False
        
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
            self.buf_to_offset[buf_id] = start
            self.total_allocated += size
            
            if start + size < end:
                self.free_blocks.append((start + size, end))
            self.free_blocks.sort()
            return True
        
        return False

    def _find_best_spill_position(self, required_size, current_time, free_times, buf_has_copy_in_dict, W1=1.0, W2=1.0):
        """找到最优的SPILL位置和受影响的缓冲区"""
        # 收集所有可能的位置（空闲块和已使用块的起始位置）
        candidate_positions = set()
        
        # 添加所有空闲块的起始位置
        for start, end in self.free_blocks:
            candidate_positions.add(start)
        
        # 添加所有已使用块的起始位置
        for start, end, buf_id in self.used_blocks:
            candidate_positions.add(start)
        
        # 按地址排序
        candidate_positions = sorted(candidate_positions)
        
        best_position = -1
        best_victims = []
        best_total_cost = float('inf')
        
        # 评估每个候选位置
        for pos in candidate_positions:
            # 检查这个位置是否有效（不超出内存池边界）
            if pos + required_size > self.capacity:
                continue
                
            # 找出与区间[pos, pos+required_size)相交的所有已使用块
            overlapping_blocks = []
            total_cost = 0
            
            for start, end, buf_id in self.used_blocks:
                # 检查是否有重叠
                if not (end <= pos or start >= pos + required_size):
                    # 有重叠，需要SPILL这个块
                    size = end - start
                    free_time = free_times.get(buf_id, current_time + 1000)
                    
                    # 计算Tag因子：是否被COPY_IN过
                    tag = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
                    
                    # 计算WCB评分
                    remaining_time = max(1, free_time - current_time)
                    cost = tag * W1 / size + W2 / remaining_time
                    
                    overlapping_blocks.append((buf_id, cost, size))
                    total_cost += cost
            
            # 如果这个位置的总代价更优，更新最佳选择
            if overlapping_blocks and total_cost < best_total_cost:
                best_position = pos
                best_victims = [buf_id for buf_id, cost, size in overlapping_blocks]
                best_total_cost = total_cost
        
        return best_position, best_victims, best_total_cost

    def spill_victims(self, victim_bufs, current_time, free_times, buf_has_copy_in_dict, W1=1.0, W2=1.0):
        assert self.type not in {'L0A','L0B','L0C'}, f"L0 池被非法 spill：{self.type}"
        """执行多个缓冲区的SPILL操作"""
        total_cost = 0
        spilled_info = []
        
        for victim_buf in victim_bufs:
            # 找到victim的块信息
            victim_block = None
            for block in self.used_blocks:
                if block[2] == victim_buf:
                    victim_block = block
                    break
            
            if not victim_block:
                continue
            
            start, end, buf_id = victim_block
            size = end - start
            
            # 计算Tag因子：是否被COPY_IN过
            tag = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
            
            # 计算代价系数
            cost_coeff = 1 if buf_has_copy_in_dict.get(buf_id, False) else 2
            cost = cost_coeff * size
            
            # 记录SPILL
            total_cost += cost
            self.spill_cost += cost
            self.spill_log.append((victim_buf, cost, current_time))
            spilled_info.append((victim_buf, cost, size))
            
            # 释放该缓冲区
            self.free(victim_buf)
        
        return True, spilled_info

    def free(self, buf_id):
        """释放缓冲区并合并空闲块"""
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
        
        # 合并相邻空闲块
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

    def alloc(self, buf_id, size, buf_type, current_time, free_time, W1=1.0, W2=1.0):
        pool = self.pools.get(buf_type)
        if not pool:
            return False
        
        # 第一次尝试分配内存
        success = pool.alloc(buf_id, size)
        
        if not success and buf_type in ['UB', 'L1']:  # 只有UB/L1可以SPILL
            # 构建free_times字典
            free_times = {buf_id: free_time}
            
            # 使用新的SPILL策略：找到最优位置和需要SPILL的缓冲区
            best_pos, victim_bufs, total_cost = pool._find_best_spill_position(
                size, current_time, free_times, self.buf_has_copy_in_dict, W1, W2
            )
            
            if victim_bufs:  # 找到了可行的SPILL方案
                spill_success, spill_results = pool.spill_victims(
                    victim_bufs, current_time, free_times, self.buf_has_copy_in_dict, W1, W2
                )
                
                if spill_success:
                    # 记录所有SPILL操作
                    for spill_result in spill_results:
                        self.spill_operations.append({
                            'buf_id': spill_result[0],
                            'time': current_time,
                            'cost': spill_result[1],
                            'size': spill_result[2]
                        })
                    
                    # 重新尝试分配
                    success = pool.alloc(buf_id, size)
        
        return success

    def free(self, buf_id, buf_type):
        pool = self.pools.get(buf_type)
        if pool:
            pool.free(buf_id)

    def get_offset(self, buf_id, buf_type):
        pool = self.pools.get(buf_type)
        return pool.get_offset(buf_id) if pool else -1

    def collect_spill_info(self):
        """收集所有SPILL信息"""
        self.global_spill_log = []
        self.total_spill_cost = 0
        
        for pool_name, pool in self.pools.items():
            self.global_spill_log.extend(pool.spill_log)
            self.total_spill_cost += pool.spill_cost
        
        return self.total_spill_cost, self.global_spill_log

    def get_pool_usage(self, pool_type):
        pool = self.pools.get(pool_type)
        return pool.get_usage() if pool else (0, 0)

# ==================== 其余代码保持不变 ====================
def read_csv_optimized(case):
    """读取节点和边数据"""

    # 尝试使用绝对路径
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    case_path = os.path.join(base_path, case)
    nodes = pd.read_csv(f"{case_path}_Nodes.csv")
    edges = pd.read_csv(f"{case_path}_Edges.csv")
    
    # 确保有必要的列
    if 'Type' not in nodes.columns:
        nodes['Type'] = ''
    if 'BufId' not in nodes.columns:
        nodes['BufId'] = -1
    if 'Size' not in nodes.columns:
        nodes['Size'] = 0
    
    def calculate_cost(row):
        """计算节点Cost"""
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
    """快速构建有向无环图"""
    edge_tuples = [(int(r.StartNodeId), int(r.EndNodeId)) for _, r in edges.iterrows()]
    G = nx.DiGraph()
    G.add_edges_from(edge_tuples)
    return G

def greedy_topo_optimized(nodes, graph, capacities):
    """优化版贪心拓扑排序，考虑L0容量限制"""
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
    
    # L0类型当前使用量
    l0_usage = {'L0A': 0, 'L0B': 0, 'L0C': 0}
    l0_capacities = {
        'L0A': capacities['L0A'],
        'L0B': capacities['L0B'], 
        'L0C': capacities['L0C']
    }
    
    while zero_degree_nodes:
        # 选择Cost最小的可行节点
        min_cost = float('inf')
        selected_node = None
        
        # 遍历所有零入度节点，找到cost最小且满足L0容量限制的节点
        for node in list(zero_degree_nodes):
            attr = node_attrs[node]
            
            # 检查L0容量限制
            can_allocate = True
            if attr['Op'] == "ALLOC" and attr['Type'] in l0_usage:
                buf_type = attr['Type']
                if l0_usage[buf_type] + attr['Size'] > l0_capacities[buf_type]:
                    can_allocate = False
            
            if not can_allocate:
                continue
            
            # 选择cost最小的节点
            if attr['Cost'] < min_cost:
                min_cost = attr['Cost']
                selected_node = node
        
        if selected_node is None:
            # 如果没有满足条件的节点，选择第一个节点
            selected_node = zero_degree_nodes[0]
        
        current_node = selected_node
        current_attr = node_attrs[current_node]
        
        # 更新L0使用量
        if current_attr['Op'] == "ALLOC" and current_attr['Type'] in l0_usage:
            l0_usage[current_attr['Type']] += current_attr['Size']
        elif current_attr['Op'] == "FREE" and current_attr['Type'] in l0_usage:
            l0_usage[current_attr['Type']] = max(0, l0_usage[current_attr['Type']] - current_attr['Size'])
        
        # 更新后继节点入度
        for successor in graph.successors(current_node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                zero_degree_nodes.append(successor)
        
        execution_order.append(current_node)
        if current_node in zero_degree_nodes:
            zero_degree_nodes.remove(current_node)
    
    return execution_order

def schedule_with_memory_management(nodes, edges, capacities, W1=1.0, W2=1.0):

    """带内存管理的调度器"""
    # 构建DAG
    G = build_dag_fast(edges)
    
    # 生成初始拓扑序
    execution_order = greedy_topo_optimized(nodes, G, capacities)
    
    # 预处理：记录每个BufId的FREE节点位置和是否有COPY_IN
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
    
    # 初始化内存管理器，传递buf_has_copy_in_dict
    manager = MultiPoolManager(capacities, buf_has_copy_in_dict)
    
    # 模拟执行过程
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
            
            # 尝试分配内存
            success = manager.alloc(
                row.BufId, row.Size, row.Type, i, free_time, W1, W2
            )
            
            if success:
                offset = manager.get_offset(row.BufId, row.Type)
                if offset != -1:
                    memory_alloc[row.BufId] = offset
        
        elif row.Op == 'FREE':
            manager.free(row.BufId, row.Type)
    
    # 收集SPILL信息
    total_cost, spill_log = manager.collect_spill_info()
    
    return final_schedule, memory_alloc, spill_log, total_cost, []

def conservative_left_slide(final_schedule, nodes, edges):
    """保守左滑：零新增 SPILL，仅同列内压紧"""
    from collections import defaultdict
    import pandas as pd

    # 1. 前驱最晚结束 & 地址空出（Q2 已保证）
    pred_end = defaultdict(int)
    for _, r in edges.iterrows():
        u, v = int(r.StartNodeId), int(r.EndNodeId)
        pred_end[v] = max(pred_end[v], 0)   # 后面更新为 E[u]

    addr_free = {}  # bufId -> FREE 时刻（用序列下标当时钟）
    for i, nid in enumerate(final_schedule):
        row = nodes.loc[nid]
        if row.Op == 'FREE':
            addr_free[row.BufId] = i

    # 2. 同列串行尾巴 & 逐节点左滑
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

# ==================== 计算时钟（零依赖，已处理缺失 Cycles=0） ====================
def compute_clock(final_schedule, nodes, edges):
    """
    输入：final_schedule 节点 ID 列表
          nodes DataFrame（含 Pipe, Cycles, Op, Type, BufId）
          edges DataFrame（含 StartNodeId, EndNodeId）
    返回：S, E, T_total, pipe_gantt（ASCII 甘特图）
    """
    from collections import defaultdict
    import pandas as pd

    # 1. 前驱最晚结束表
    pred_end = defaultdict(int)
    edge_map = defaultdict(list)
    for _, r in edges.iterrows():
        edge_map[int(r.StartNodeId)].append(int(r.EndNodeId))

    # 2. 地址空出表（FREE 节点序列位置当时刻）
    addr_free = {}  # bufId -> FREE 时刻
    for i, nid in enumerate(final_schedule):
        row = nodes.loc[nid]
        if row.Op == 'FREE':
            addr_free[row.BufId] = i  # 用序列下标当“时刻”即可

    # 3. 同列串行尾巴 & 逐节点算 S/E
    pipe_last = defaultdict(int)  # Pipe -> 列尾时间
    S, E = {}, {}
    pipe_gantt = defaultdict(list)  # Pipe -> [(start, end, node)]

    for i, nid in enumerate(final_schedule):
        row = nodes.loc[nid]
        p = row.Pipe
        # 缺失 Cycles → 0（官方设定：管理节点不占周期）
        cycles = int(row.Cycles) if pd.notna(row.get('Cycles', None)) else 0

        # 理论最早可启动 = 图依赖 vs 列尾 vs 地址空出
        early = max(pred_end[nid], pipe_last[p], addr_free.get(row.get('BufId'), 0))
        S[nid] = early
        E[nid] = early + cycles
        pipe_last[p] = E[nid]
        pipe_gantt[p].append((early, E[nid], nid))

    T_total = max(E.values(), default=0)

    # 4. 断言：无 NaN
    assert not any(pd.isna(e) for e in E.values()), "仍有 NaN 节点未处理"

    return S, E, T_total, pipe_gantt


def print_gantt(pipe_gantt, max_width=80):
    """ASCII 甘特图，▓=占用，空格=空闲"""
    for p, bars in pipe_gantt.items():
        bars.sort(key=lambda x: x[0])
        line, last_end = [], 0
        for s, e, nid in bars:
            if s > last_end:
                line.append(' ' * (s - last_end))
            line.append(f'▓{nid}▓')
            last_end = e
        # 截断过长行
        raw = ''.join(line)
        print(f"{p:6} |{raw[:max_width]}{'...' if len(raw) > max_width else ''}")


def write_outputs(case, schedule, memory_alloc, spill_log, total_spill_cost):
    """输出结果文件"""
    import os
    os.makedirs(f"Problem2", exist_ok=True)
    prefix = f"Problem2/{case}"
    
    # schedule.txt
    with open(f"{prefix}_schedule.txt", "w") as f:
        for node_id in schedule:
            f.write(f"{node_id}\n")
    
    # memory.txt
    with open(f"{prefix}_memory.txt", "w") as f:
        for buf_id in sorted(memory_alloc.keys()):
            offset = memory_alloc[buf_id]
            f.write(f"{buf_id}:{offset}\n")
    
    # spill.txt
    with open(f"{prefix}_spill.txt", "w") as f:
        for buf_id, cost, spill_time in spill_log:
            f.write(f"{buf_id}:{cost}\n")
        f.write(f"TotalSpillCost:{total_spill_cost}\n")
        
def compute_clock_serial(final_schedule, nodes):
    """
    串行baseline时钟：所有节点严格按照顺序一个接一个跑
    忽略依赖，只保留 Pipe 内的串行 + 顺序执行
    """
    S, E = {}, {}
    cur_time = 0
    pipe_last = defaultdict(int)
    
    for nid in final_schedule:
        row = nodes.loc[nid]
        cycles = int(row.Cycles) if pd.notna(row.get('Cycles', None)) else 0
        p = row.Pipe
        
        # 串行baseline：按调度顺序排队执行
        start = max(cur_time, pipe_last[p])
        end = start + cycles
        S[nid], E[nid] = start, end
        cur_time = end
        pipe_last[p] = end
    
    T_total = max(E.values(), default=0)
    return S, E, T_total


def main():
    """第二问零-SPILL + 第三问保守左滑（纯数值版）"""
    import os
    capacities = {
        'L1': 4096, 'UB': 1024, 'L0A': 256, 'L0B': 256, 'L0C': 512
    }
    cases = [
        "FlashAttention_Case0", "FlashAttention_Case1",
        "Matmul_Case0", "Matmul_Case1",
        "Conv_Case0", "Conv_Case1"
    ]

    print("==== 第二问零-SPILL + 第三问保守左滑 ====")
    total_start = time.time()

    for case in cases:
        print(f"\n=== 处理 {case} ===")
        try:
            # 1. 数据读取
            data_start = time.time()
            nodes, edges = read_csv_optimized(case)
            data_time = time.time() - data_start

            # 2. 第二问：零-SPILL 调度 + 内存管理
            topo_start = time.time()
            schedule, memory_alloc, spill_log, total_spill_cost, _ = schedule_with_memory_management(
                nodes, edges, capacities, W1=1.0, W2=1.0
            )
            topo_time = time.time() - topo_start
            
            # 3. baseline：串行调度
            clock_start = time.time()
            S_orig, E_orig, T_orig = compute_clock_serial(schedule, nodes)
            clock_time = time.time() - clock_start

            # 4. Q3：ASAP 调度（保守左滑）
            slide_start = time.time()
            S_new, E_new, T_new = conservative_left_slide(schedule, nodes, edges)
            slide_time = time.time() - slide_start


            # 5. 结果输出（第二问原有 + 第三问新增）
            os.makedirs("Problem2", exist_ok=True)
            os.makedirs("Problem3", exist_ok=True)   # 关键：提前创建目录

            write_outputs(case, schedule, memory_alloc, spill_log, total_spill_cost)
            with open(f"Problem2/{case}_timeline.txt", "w") as f:
                for nid in schedule:
                    f.write(f"{nid} {S_orig[nid]} {E_orig[nid]}\n")
            with open(f"Problem3/{case}_timeline.txt", "w") as f:
                for nid in schedule:
                    f.write(f"{nid} {S_new[nid]} {E_new[nid]}\n")

            # 6. 纯数值对比
            print(f"数据: {data_time:.3f}s | 拓扑+内存: {topo_time:.3f}s | 时钟: {clock_time:.3f}s | 左滑: {slide_time:.3f}s")
            print(f"总额外搬运量: {total_spill_cost} | SPILL次数: {len(spill_log)}")
            print(f"原总周期: {T_orig}")
            print(f"左滑后周期: {T_new}")
            print(f"时间下降: {(T_orig - T_new) / T_orig * 100:.2f}%")

        except Exception as e:
            print(f"处理 {case} 时出错: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start
    print(f"\n==== 全部完成 ====")
    print(f"总耗时: {total_time:.2f}s")

    
if __name__ == "__main__":
    main()
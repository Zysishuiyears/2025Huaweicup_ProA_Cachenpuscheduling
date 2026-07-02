import pandas as pd
import numpy as np
import networkx as nx
from collections import deque, defaultdict
import heapq
import time
import ast

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

    def alloc(self, buf_id, size, allow_spill=True, current_time=0, free_times=None, buf_has_copy_in_dict=None, W1=1.0, W2=1.0):
        """分配内存，返回(成功标志, 消息)
        
        Args:
            allow_spill: 是否允许spill操作（L0内存池设置为False）
        """
        # 检查容量限制
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
            self.buf_to_offset[buf_id] = start
            self.total_allocated += size
            
            if start + size < end:
                self.free_blocks.append((start + size, end))
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
                    # 重新尝试分配
                    return self.alloc(buf_id, size, allow_spill, current_time, free_times, buf_has_copy_in_dict, W1, W2)
        
        return False, "内存不足，分配失败"

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
        self.l0_alloc_failed = False  # 记录L0分配是否失败过

    def alloc(self, buf_id, size, buf_type, current_time, free_time, W1=1.0, W2=1.0):
        pool = self.pools.get(buf_type)
        if not pool:
            return False, f"未知的内存池类型: {buf_type}"
        
        # 对于L0内存池，禁止spill操作
        allow_spill = buf_type not in ['L0A', 'L0B', 'L0C']
        free_times = {buf_id: free_time} if allow_spill else {}
        
        success, message = pool.alloc(
            buf_id, size, allow_spill, current_time, 
            free_times, self.buf_has_copy_in_dict, W1, W2
        )
        
        # 记录L0分配失败情况（但不重复警告）
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

# ==================== 贪心拓扑排序（使用内存池模拟L0） ====================
def greedy_topo_with_L0_pool(nodes, graph, capacities):
    """使用内存池模拟L0内存的贪心拓扑排序"""
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
    
    # 创建L0内存池模拟器（禁止spill）
    l0_pools = {
        'L0A': CompactPool('L0A', capacities['L0A']),
        'L0B': CompactPool('L0B', capacities['L0B']),
        'L0C': CompactPool('L0C', capacities['L0C']),
    }
    
    l0_alloc_failed = False  # 记录是否发生过L0分配失败
    
    while zero_degree_nodes:
        # 选择Cost最小的可行节点
        min_cost = float('inf')
        selected_node = None
        
        # 遍历所有零入度节点，找到cost最小且满足L0内存池限制的节点
        for node in list(zero_degree_nodes):
            attr = node_attrs[node]
            
            # 检查L0内存池限制
            can_allocate = True
            if attr['Op'] == "ALLOC" and attr['Type'] in l0_pools:
                buf_type = attr['Type']
                pool = l0_pools[buf_type]
                
                # 尝试在L0内存池中分配（禁止spill）
                success, _ = pool.alloc(attr['BufId'], attr['Size'], allow_spill=False)
                if not success:
                    can_allocate = False
                else:
                    # 分配成功，检查新增的约束条件
                    # 条件1：当前缓冲区段数不超过2个
                    current_buf_count = len(pool.used_blocks)
                    
                    # 条件2：或者即使超过2个，但占用空间不超过一半
                    current_usage_ratio = pool.total_allocated / pool.capacity
                    
                    # 新增约束：缓冲区段数不超过2个，除非占用空间不超过一半
                    if current_buf_count > 2 and current_usage_ratio > 0.5:
                        can_allocate = False
                    
                    # 释放刚刚分配的缓冲区（我们只是模拟检查）
                    pool.free(attr['BufId'])
            
            if not can_allocate:
                continue
            
            # 选择cost最小的节点
            if attr['Cost'] < min_cost:
                min_cost = attr['Cost']
                selected_node = node
        
        if selected_node is None:
            # 如果没有满足条件的节点，尝试放宽条件：只检查基本分配是否成功，不检查缓冲区段数限制
            min_cost = float('inf')
            for node in list(zero_degree_nodes):
                attr = node_attrs[node]
                
                # 只检查基本分配是否成功
                can_allocate = True
                if attr['Op'] == "ALLOC" and attr['Type'] in l0_pools:
                    buf_type = attr['Type']
                    pool = l0_pools[buf_type]
                    
                    success, _ = pool.alloc(attr['BufId'], attr['Size'], allow_spill=False)
                    if not success:
                        can_allocate = False
                    else:
                        pool.free(attr['BufId'])
                
                if not can_allocate:
                    continue
                
                if attr['Cost'] < min_cost:
                    min_cost = attr['Cost']
                    selected_node = node
            
            # 如果仍然没有找到，选择第一个节点（理论上不应该发生）
            if selected_node is None:
                selected_node = zero_degree_nodes[0]
                if not l0_alloc_failed:
                    l0_alloc_failed = True
        
        current_node = selected_node
        current_attr = node_attrs[current_node]
        
        # 实际执行L0内存分配/释放操作
        if current_attr['Op'] == "ALLOC" and current_attr['Type'] in l0_pools:
            pool = l0_pools[current_attr['Type']]
            success, _ = pool.alloc(current_attr['BufId'], current_attr['Size'], allow_spill=False)
            if not success and not l0_alloc_failed:
                l0_alloc_failed = True
        elif current_attr['Op'] == "FREE" and current_attr['Type'] in l0_pools:
            pool = l0_pools[current_attr['Type']]
            pool.free(current_attr['BufId'])
        
        # 更新后继节点入度
        for successor in graph.successors(current_node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                zero_degree_nodes.append(successor)
        
        execution_order.append(current_node)
        if current_node in zero_degree_nodes:
            zero_degree_nodes.remove(current_node)
    
    # 输出L0分配结果
    if l0_alloc_failed:
        print("警告：在拓扑排序过程中检测到L0内存分配失败")
    else:
        print("L0内存分配全部成功")
    
    return execution_order

# ==================== 其余代码保持不变 ====================
def read_csv_optimized(case):
    """读取节点和边数据"""

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

def schedule_with_memory_management(nodes, edges, capacities, W1=1.0, W2=1.0):
    """带内存管理的调度器"""
    # 构建DAG
    G = build_dag_fast(edges)
    
    # 生成初始拓扑序（使用新的L0内存池模拟）
    execution_order = greedy_topo_with_L0_pool(nodes, G, capacities)
    
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
            success, message = manager.alloc(
                row.BufId, row.Size, row.Type, i, free_time, W1, W2
            )
            
            if success:
                offset = manager.get_offset(row.BufId, row.Type)
                if offset != -1:
                    memory_alloc[row.BufId] = offset
            else:
                # 对于L0分配失败，我们已经记录过，这里不需要重复警告
                if row.Type not in ['L0A', 'L0B', 'L0C']:
                    print(f"警告: 节点 {node_id} 分配失败: {message}")
        
        elif row.Op == 'FREE':
            manager.free(row.BufId, row.Type)
    
    # 收集SPILL信息
    total_cost, spill_log = manager.collect_spill_info()
    
    # 输出L0分配总体情况
    if manager.l0_alloc_failed:
        print("警告：在内存管理过程中检测到L0内存分配失败")
    else:
        print("L0内存管理全部成功")
    
    return final_schedule, memory_alloc, spill_log, total_cost, []

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

def main():
    """主函数"""
    # 硬件缓存容量
    capacities = {
        'L1': 4096,
        'UB': 1024,
        'L0A': 256,
        'L0B': 256,
        'L0C': 512
    }
    
    cases = [
        "FlashAttention_Case0", "FlashAttention_Case1",
        "Matmul_Case0", "Matmul_Case1", 
        "Conv_Case0", "Conv_Case1"
    ]
    
    print("==== 内存调度优化算法（问题二） ====")
    start_time = time.time()
    
    for case in cases:
        print(f"\n=== 处理 {case} ===")
        
        try:
            # 数据读取和预处理
            data_start = time.time()
            nodes, edges = read_csv_optimized(case)
            data_time = time.time() - data_start
            print(f"数据预处理耗时: {data_time:.2f}s")
            print(f"节点数: {len(nodes)}, 边数: {len(edges)}")
            
            # 拓扑排序和内存管理
            topo_start = time.time()
            schedule, memory_alloc, spill_log, total_cost, spill_nodes = schedule_with_memory_management(
                nodes, edges, capacities, W1=1.0, W2=1.0
            )
            topo_time = time.time() - topo_start
            print(f"拓扑排序和内存管理耗时: {topo_time:.2f}s")
            
            # 输出结果
            write_outputs(case, schedule, memory_alloc, spill_log, total_cost)
            print(f"调度序列长度: {len(schedule)}")
            print(f"内存分配数: {len(memory_alloc)}")
            print(f"总额外搬运量: {total_cost}")
            print(f"SPILL操作次数: {len(spill_log)}")
            
        except Exception as e:
            print(f"处理 {case} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n总耗时：{total_time:.2f}s")

if __name__ == "__main__":
    main()
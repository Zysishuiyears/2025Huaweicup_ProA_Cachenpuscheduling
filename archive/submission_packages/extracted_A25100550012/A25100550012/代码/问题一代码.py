#本程序及代码是在人工智能工具辅助下完成的，使用了Deepseek模型
'''
模型版本：DeepSeek-V3.1-Terminus 
开发机构：杭州深度求索人工智能基础技术研究有限公司 
颁布日期：2025年9月22日 
'''

import pandas as pd
import os
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from collections import deque


def read_csv_optimized(case):
    """读取节点和边数据，向量化计算Cost"""
    nodes = pd.read_csv(f"{case}_Nodes.csv")
    edges = pd.read_csv(f"{case}_Edges.csv")
    
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
    
    return nodes, edges


def build_dag_fast(edges):
    """快速构建有向无环图"""
    edge_tuples = [(int(r.StartNodeId), int(r.EndNodeId)) for _, r in edges.iterrows()]
    return nx.from_edgelist(edge_tuples, create_using=nx.DiGraph)


def precompute_dominating_effects(dominating_relation, row_index, rev):
    """
    预处理支配关系效果
    dominating_effects[node] = 被node支配的节点列表
    """
    dominating_effects = {}
    n = len(row_index)
    
    for i in range(n):
        dominator_node = row_index[i]
        dominating_effects[dominator_node] = []
        
        for j in range(n):
            if dominating_relation[i][j] == 1:
                dominated_node = row_index[j]
                dominating_effects[dominator_node].append(dominated_node)
    
    return dominating_effects


def greedy_topo_optimized(nodes, graph, dominating_relation, row_index, rev, index):
    """优化版贪心拓扑排序"""
    # 预处理支配关系
    dominating_effects = precompute_dominating_effects(dominating_relation, row_index, rev)
    
    # 预先提取属性到字典（避免频繁pandas操作）
    node_attrs = {}
    for nid, row in nodes.iterrows():
        node_attrs[nid] = {
            'Op': row['Op'], 
            'Type': row['Type'], 
            'Cost': row['Cost']
        }
    
    # 初始化入度和零入度节点队列
    in_degree = {n: graph.in_degree(n) for n in graph.nodes}
    zero_degree_nodes = [n for n in graph.nodes if in_degree[n] == 0]
    
    execution_order = []
    l0a_count, l0b_count, l0c_count = 0, 0, 0
    
    while zero_degree_nodes:
        # 选择Cost最小的节点
        min_cost = float('inf')
        selected_node = None
        
        for node in zero_degree_nodes:
            attr = node_attrs[node]
            if attr['Cost'] < min_cost:
                # 检查L0类型分配限制
                if attr['Op'] == "ALLOC":
                    if (attr['Type'] == "L0A" and l0a_count >= 1) or \
                       (attr['Type'] == "L0B" and l0b_count >= 1) or \
                       (attr['Type'] == "L0C" and l0c_count >= 1):
                        continue
                min_cost = attr['Cost']
                selected_node = node
        
        current_node = selected_node
        current_attr = node_attrs[current_node]
        
        # 更新L0类型计数
        if current_attr['Op'] == "ALLOC":
            if current_attr['Type'] == "L0A": 
                l0a_count += 1
            elif current_attr['Type'] == "L0B": 
                l0b_count += 1
            elif current_attr['Type'] == "L0C": 
                l0c_count += 1
        elif current_attr['Op'] == "FREE":
            if current_attr['Type'] == "L0A": 
                l0a_count -= 1
            elif current_attr['Type'] == "L0B": 
                l0b_count -= 1
            elif current_attr['Type'] == "L0C": 
                l0c_count -= 1
        
        # 处理支配关系效果
        if (current_attr['Op'] == "ALLOC" and 
            current_attr['Type'] in {"L0A", "L0B", "L0C"} and
            current_node in dominating_effects):
            
            for affected_node in dominating_effects[current_node]:
                node_attrs[affected_node]['Cost'] = -1
        
        # 更新后继节点入度
        for successor in graph.successors(current_node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                zero_degree_nodes.append(successor)
        
        execution_order.append(current_node)
        zero_degree_nodes.remove(current_node)
    
    return execution_order


def calculate_memory_peak(nodes, execution_order):
    """计算内存使用峰值"""
    current_memory = peak_memory = 0
    l0a_count, l0b_count, l0c_count = 0, 0, 0
    allocation_match = True
    
    for node_id in execution_order:
        row = nodes.loc[node_id]
        
        # UB和L1类型的内存分配/释放
        if row.Op == "ALLOC" and row.Type in {"UB", "L1"}:
            current_memory += row.Size
            peak_memory = max(peak_memory, current_memory)
        elif row.Op == "FREE" and row.Type in {"UB", "L1"}:
            current_memory -= row.Size
        
        # L0类型的计数跟踪
        elif row.Op == "ALLOC":
            if row.Type == "L0A": l0a_count += 1
            elif row.Type == "L0B": l0b_count += 1
            elif row.Type == "L0C": l0c_count += 1
        elif row.Op == "FREE":
            if row.Type == "L0A": l0a_count -= 1
            elif row.Type == "L0B": l0b_count -= 1
            elif row.Type == "L0C": l0c_count -= 1
        
        # 检查L0分配是否匹配
        if l0a_count > 1 or l0b_count > 1 or l0c_count > 1:
            allocation_match = False
    
    allocation_status = "匹配" if allocation_match else "不匹配"
    print(f"L0A/B/C 分配与释放{allocation_status}")
    
    return peak_memory


def save_execution_schedule(case, execution_order, peak_memory):
    """保存执行调度结果"""
    print(f"{case:<20} 节点数：{len(execution_order):<6}  峰值(UB+L1)：{peak_memory:<6}")
    
    with open(f"{case}_schedule.txt", "w") as f:
        for node_id in execution_order:
            f.write(f"{node_id}\n")


def calculate_alloc_free_distances_optimized(nodes, edges, graph, case_name):
    """优化版计算ALLOC L0节点到FREE节点的距离"""
    # 快速构建节点信息
    node_info = {
        nid: {'Op': row['Op'], 'Type': row['Type']} 
        for nid, row in nodes.iterrows()
    }
    
    # 找出集合A（ALLOC L0节点）和集合B（相连的FREE节点）
    alloc_nodes = [
        nid for nid, row in nodes.iterrows() 
        if row['Op'] == 'ALLOC' and row['Type'] in {'L0A', 'L0B', 'L0C'}
    ]
    
    free_nodes = set()
    for _, edge in edges.iterrows():
        start_node = int(edge.StartNodeId)
        end_node = int(edge.EndNodeId)
        
        if (start_node in node_info and end_node in node_info and 
            start_node in alloc_nodes and node_info[end_node]['Op'] == 'FREE'):
            free_nodes.add(end_node)
    
    sorted_alloc_nodes = sorted(alloc_nodes)
    sorted_free_nodes = sorted(free_nodes)
    
    # 多源BFS计算距离
    distance_matrix = multi_source_bfs_distances(graph, sorted_alloc_nodes, sorted_free_nodes)
    
    return distance_matrix, sorted_alloc_nodes, sorted_free_nodes


def multi_source_bfs_distances(graph, sources, targets):
    target_to_index = {target: j for j, target in enumerate(targets)}
    source_to_index = {source: i for i, source in enumerate(sources)}
    
    distance_matrix = np.full((len(sources), len(targets)), -1, dtype=int)
    
    # 预先计算所有目标节点的可达性
    target_set = set(targets)
    
    for i, source in enumerate(sources):
        # 使用字典记录到各个目标节点的最短距离
        min_distances = {target: float('inf') for target in targets}
        
        visited = set()
        queue = deque([(source, 0)])
        visited.add(source)
        
        if source in target_set:
            min_distances[source] = 0
        
        while queue:
            current, dist = queue.popleft()
            
            # 如果已经找到所有目标节点，可以提前终止
            if all(d != float('inf') for d in min_distances.values()):
                break
                
            for neighbor in graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_dist = dist + 1
                    
                    if neighbor in target_set:
                        if new_dist < min_distances[neighbor]:
                            min_distances[neighbor] = new_dist
                    
                    queue.append((neighbor, new_dist))
        
        # 更新距离矩阵
        for target, min_dist in min_distances.items():
            if min_dist != float('inf'):
                j = target_to_index[target]
                distance_matrix[i, j] = min_dist
    
    return distance_matrix


def plot_alloc_size_histogram(nodes, case_name):
    """绘制ALLOC操作的Size分布直方图"""
    alloc_nodes = nodes[nodes['Op'] == 'ALLOC']
    
    if alloc_nodes.empty:
        print(f"{case_name}: 没有找到ALLOC操作")
        return
    
    sizes = alloc_nodes['Size'].values
    
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    plt.title(f'{case_name} - ALLOC Size Distribution', fontsize=14)
    plt.xlabel('Size', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    
    # 添加统计信息
    stats_text = f'Total ALLOC: {len(sizes)}\nMin Size: {np.min(sizes)}\nMax Size: {np.max(sizes)}\nMean Size: {np.mean(sizes):.2f}'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(f'{case_name}_alloc_size_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    """主函数"""
    cases = [
        "FlashAttention_Case0", "FlashAttention_Case1",
        "Matmul_Case0", "Matmul_Case1",
        "Conv_Case0", "Conv_Case1"
    ]
    
    print("==== 内存调度优化算法 ====")
    start_time = time.time()
    
    for case in cases:
        print(f"\n=== 处理 {case} ===")
        
        # 数据读取和预处理
        data_start = time.time()
        nodes, edges = read_csv_optimized(case)
        graph = build_dag_fast(edges)
        data_time = time.time() - data_start
        print(f"数据预处理耗时: {data_time:.2f}s")
        
        # 距离计算
        dist_start = time.time()
        distances, alloc_nodes, free_nodes = calculate_alloc_free_distances_optimized(
            nodes, edges, graph, case
        )
        print(distances)
        dist_time = time.time() - dist_start
        print(f"距离计算耗时: {dist_time:.2f}s")
        print(f"ALLOC节点数: {len(alloc_nodes)}, FREE节点数: {len(free_nodes)}")
        
        # 构建索引映射
        index_map = np.zeros(np.max(free_nodes) + 1, dtype=int)
        for i, node_id in enumerate(free_nodes):
            index_map[node_id] = alloc_nodes[i]
        
        # 处理距离关系并添加边
        for i, alloc_node in enumerate(alloc_nodes):
            for j, free_node in enumerate(free_nodes):
                if distances[i][j] > 1:
                    distances[i][j] = 1
                    alloc_type = nodes.loc[alloc_node]['Type']
                    free_type = nodes.loc[free_node]['Type']
                    
                    if alloc_type == free_type:
                        graph.add_edge(alloc_node, index_map[free_node])
                else:
                    distances[i][j] = 0
        
        # 构建反向索引
        reverse_index = np.zeros(np.max(alloc_nodes) + 1, dtype=int)
        for i, node_id in enumerate(alloc_nodes):
            reverse_index[node_id] = i
        
        # 拓扑排序
        topo_start = time.time()
        execution_order = greedy_topo_optimized(
            nodes, graph, distances, alloc_nodes, reverse_index, index_map
        )
        topo_time = time.time() - topo_start
        print(f"拓扑排序耗时: {topo_time:.2f}s")
        
        # 计算结果并保存
        peak_memory = calculate_memory_peak(nodes, execution_order)
        save_execution_schedule(case, execution_order, peak_memory)
        
        # 可选：绘制直方图
        # plot_alloc_size_histogram(nodes, case)
    
    total_time = time.time() - start_time
    print(f"\n总耗时：{total_time:.2f}s")


if __name__ == "__main__":
    main()
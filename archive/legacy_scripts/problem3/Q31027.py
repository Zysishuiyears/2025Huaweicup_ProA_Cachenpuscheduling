# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 10:27:50 2025

@author: JZX
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import deque, defaultdict
import time, os, ast
from copy import deepcopy
                
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
        self.spill_counter = 0  # 用于生成新 spill 节点 ID


    def alloc(self, buf_id, size, allow_spill=True, current_time=0,
              free_times=None, buf_has_copy_in_dict=None,
              W1=1.0, W2=1.0, nodes=None, schedule=None):
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
                spill_success, _, _ = self.spill_victims(
                    victim_bufs, current_time,
                    free_times or {}, buf_has_copy_in_dict or {},
                    W1, W2, nodes=nodes, schedule=schedule
                )
                if spill_success:
                    return self.alloc(buf_id, size, allow_spill,
                                      current_time, free_times,
                                      buf_has_copy_in_dict, W1, W2,
                                      nodes=nodes, schedule=schedule)
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
                  W1=1.0, W2=1.0, nodes=None, schedule=None, edges=None):
        new_spill_nodes = []
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
            
        # === 插入 SPILL 节点（可选：第三问也新增）===
            if nodes is not None and schedule is not None and edges is not None:

                out_id = f"SPILL_OUT_{self.type}_{self.spill_counter}"
                in_id  = f"SPILL_IN_{self.type}_{self.spill_counter}"
                self.spill_counter += 1
                
            # 构造节点
                out_node = deepcopy(nodes.loc[victim_buf])
                out_node.update({
                "Op": "SPILL_OUT",
                "Cycles": 0,
                "Pipe": "MTE3" if self.type in ['UB','L1'] else "MTE2"
                })
                in_node = deepcopy(nodes.loc[victim_buf])
                in_node.update({
                "Op": "SPILL_IN",
                "Cycles": 0,
                "Pipe": "MTE2" if self.type in ['UB','L1'] else "MTE3"
                })

                # 追加到 DataFrame 和 schedule
                nodes.loc[out_id] = out_node
                nodes.loc[in_id]  = in_node
                schedule.append(out_id)
                schedule.append(in_id)

                # 写回边
                edges = pd.concat([edges, pd.DataFrame([
                [out_id, in_id]
            ], columns=['StartNodeId', 'EndNodeId'])], ignore_index=True)

                new_spill_nodes.extend([out_id, in_id])

            self.free(victim_buf)
        return True, self.spill_cost, new_spill_nodes


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
              free_time, W1=1.0, W2=1.0, nodes=None, schedule=None):
        pool = self.pools[buf_type]
        allow_spill = buf_type not in ['L0A', 'L0B', 'L0C']
        success, message = pool.alloc(
            buf_id, size, allow_spill,
            current_time, {buf_id: free_time},
            self.buf_has_copy_in_dict, W1, W2,
            nodes=nodes, schedule=schedule
        )
        if not success and not allow_spill:
            self.l0_alloc_failed = True
        return success, message
    #     return self.pools[buf_type].alloc(
    #     buf_id, size, buf_type, current_time, {buf_id: free_time},
    #     self.buf_has_copy_in_dict, W1, W2,
    #     nodes=nodes, schedule=schedule, edges=edges
    # )
    
    def alloc(self, buf_id, size, buf_type, current_time,
          free_time, W1=1.0, W2=1.0, nodes=None, schedule=None, edges=None):
    """
    内存分配 + 可选插入 SPILL 节点（第三问也新增）
    nodes, schedule, edges 仅用于「插入 SPILL 节点」阶段，
    若为空则行为与原代码完全一致（零新增）。
    """
        pool = self.pools[buf_type]
        # 是否允许 spill（L0 池禁止）
        allow_spill = buf_type not in ['L0A', 'L0B', 'L0C']

        # 原分配逻辑
        success, message = pool.alloc(
        buf_id, size, allow_spill,
        current_time, {buf_id: free_time},
        self.buf_has_copy_in_dict, W1, W2,
        nodes=nodes, schedule=schedule, edges=edges
    )

    # L0 池失败标记（与原逻辑一致）
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


# ==================== 其余部分（拓扑排序 / 时钟计算 / 文件写出）====================
# （保持和你上次的版本一致，只在调用 alloc 时加了 nodes, schedule 参数）

# ==================== 时钟数赋值 ====================
def assign_cycles(nodes):
    def cycle_rule(r):
        op=str(r['Op']).upper()
        if op in ['ALLOC','FREE','COPY_IN','COPY_OUT']: return 0
        if 'Cycles' in r and pd.notna(r['Cycles']): return int(r['Cycles'])
        if op in ['MTE1','MTE2','MTE3','MMAD','CUBE']:
            return max(1,int(r.get('Size',1))//16)
        return 1
    nodes['Cycles']=nodes.apply(cycle_rule,axis=1); return nodes

# ==================== 图构建和调度 ====================
def build_dag_fast(edges):
    G=nx.DiGraph(); G.add_edges_from([(int(r.StartNodeId),int(r.EndNodeId)) for _,r in edges.iterrows()]); return G

def greedy_topo_with_L0_pool(nodes, graph, capacities):
    node_attrs={nid:row.to_dict() for nid,row in nodes.iterrows()}
    indeg={n:graph.in_degree(n) for n in graph.nodes}
    ready=deque([n for n in graph.nodes if indeg[n]==0])
    order=[]; l0_pools={t:CompactPool(t,capacities[t]) for t in ['L0A','L0B','L0C']}
    while ready:
        min_cost,sel=float('inf'),None
        for n in list(ready):
            a=node_attrs[n]
            if a['Op']=="ALLOC" and a['Type'] in l0_pools:
                ok,_=l0_pools[a['Type']].alloc(a['BufId'],a['Size'],allow_spill=False)
                if ok: l0_pools[a['Type']].free(a['BufId'])
                else: continue
            if a['Cost']<min_cost: min_cost,sel=a['Cost'],n
        if sel is None: sel=ready[0]
        cur=sel; a=node_attrs[cur]
        if a['Op']=="ALLOC" and a['Type'] in l0_pools:
            l0_pools[a['Type']].alloc(a['BufId'],a['Size'],allow_spill=False)
        elif a['Op']=="FREE" and a['Type'] in l0_pools:
            l0_pools[a['Type']].free(a['BufId'])
        for succ in graph.successors(cur):
            indeg[succ]-=1
            if indeg[succ]==0: ready.append(succ)
        order.append(cur)
        if cur in ready: ready.remove(cur)
    return order

# ==================== 时钟计算和左滑 ====================
def compute_clock(schedule,nodes,edges):
    pred_end,pipe_last=defaultdict(int),defaultdict(int)
    S,E={},{}; edge_map=defaultdict(list)
    for _,r in edges.iterrows(): edge_map[int(r.StartNodeId)].append(int(r.EndNodeId))
    for nid in schedule:
        row=nodes.loc[nid]; p=row.Pipe; cycles=int(row['Cycles'])
        early=max(pred_end[nid],pipe_last[p]); S[nid]=early; E[nid]=early+cycles; pipe_last[p]=E[nid]
        for succ in edge_map[nid]: pred_end[succ]=max(pred_end[succ],E[nid])
    return S,E,max(E.values(),default=0)

def conservative_left_slide(schedule,nodes,edges):
    pred_end=defaultdict(int)
    for _,r in edges.iterrows(): pred_end[int(r.EndNodeId)]=max(pred_end[int(r.EndNodeId)],0)
    addr_free={}
    for i,nid in enumerate(schedule):
        row=nodes.loc[nid]
        if row.Op=='FREE': addr_free[row.BufId]=i
    pipe_last=defaultdict(int); S_new,E_new={},{}
    for i,nid in enumerate(schedule):
        row=nodes.loc[nid]; p=row.Pipe; cyc=int(row.Cycles)
        early=max(pred_end[nid],pipe_last[p],addr_free.get(row.get('BufId'),0))
        S_new[nid]=early; E_new[nid]=early+cyc; pipe_last[p]=E_new[nid]
    return S_new,E_new,max(E_new.values(),default=0)

# ==================== 数据读取 ====================
def read_csv_optimized(case):
    nodes=pd.read_csv(f"{case}_Nodes.csv"); edges=pd.read_csv(f"{case}_Edges.csv")
    if 'Type' not in nodes: nodes['Type']=''
    if 'BufId' not in nodes: nodes['BufId']=-1
    if 'Size' not in nodes: nodes['Size']=0
    def cost(r):
        op=str(r['Op']).upper()
        if op=='ALLOC': return int(r['Size'])
        elif op=='FREE': return -int(r['Size'])
        else: return 0
    nodes['Cost']=nodes.apply(cost,axis=1)
    if "NodeId" in nodes: nodes.set_index("NodeId",inplace=True)
    elif "Id" in nodes: nodes.set_index("Id",inplace=True)
    nodes=assign_cycles(nodes); return nodes,edges

# ==================== 输出 ====================
def write_outputs(case, schedule, memory_alloc, spill_log, total_cost, folder):
    os.makedirs(folder,exist_ok=True); prefix=f"{folder}/{case}"
    with open(f"{prefix}_schedule.txt","w") as f:
        for nid in schedule: f.write(f"{nid}\n")
    with open(f"{prefix}_memory.txt","w") as f:
        for buf,off in memory_alloc.items(): f.write(f"{buf}:{off}\n")
    with open(f"{prefix}_spill.txt","w") as f:
        for rec in spill_log:
            if isinstance(rec,dict):
                if 'buf_id'in rec and 'new_offset'in rec: f.write(f"{rec['buf_id']}:{rec['new_offset']}\n")
            elif isinstance(rec,(list,tuple)) and len(rec)>=2: f.write(f"{rec[0]}:{rec[1]}\n")

# ==================== 主函数 ====================
def main():
    capacities={'L1':4096,'UB':1024,'L0A':256,'L0B':256,'L0C':512}
    cases=["FlashAttention_Case0","FlashAttention_Case1","Matmul_Case0","Matmul_Case1","Conv_Case0","Conv_Case1"]
    print("==== 二问+三问 ====")
    for case in cases:
        print(f"\n=== {case} ===")
        nodes,edges=read_csv_optimized(case); G=build_dag_fast(edges)
        schedule=greedy_topo_with_L0_pool(nodes,G,capacities)
        # 内存管理
        buf_free={}; [buf_free.setdefault(nodes.loc[nid].BufId,i) for i,nid in enumerate(schedule) if nodes.loc[nid].Op=='FREE']
        manager=MultiPoolManager(capacities,{})
        mem_alloc={}
        for i,nid in enumerate(schedule):
            row=nodes.loc[nid]
            if row.Op=='ALLOC':
                free_t=buf_free.get(row.BufId,i+1000)
                ok,_=manager.alloc(row.BufId,row.Size,row.Type,i,free_t)
                if ok:
                    off=manager.get_offset(row.BufId,row.Type)
                    if off!=-1: mem_alloc[row.BufId]=off
            elif row.Op=='FREE': manager.free(row.BufId,row.Type)
        total_cost,spill_log=manager.collect_spill_info()
        # 写 Problem2
        write_outputs(case,schedule,mem_alloc,spill_log,total_cost,"Problem2")
        # 时钟计算
        S,E,T=compute_clock(schedule,nodes,edges)
        S_new,E_new,T_new=conservative_left_slide(schedule,nodes,edges)
        # 写 Problem3（schedule 同二问，memory 同二问，但 spill 文件可不同）
        write_outputs(case,schedule,mem_alloc,spill_log,total_cost,"Problem3")
        print(f"搬运量={total_cost}, 原周期={T}, 左滑后周期={T_new}")

if __name__=="__main__": main()

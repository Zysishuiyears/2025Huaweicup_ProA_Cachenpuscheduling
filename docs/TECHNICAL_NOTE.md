# Technical Note

## 1. Introduction

This project studies DAG scheduling, contiguous memory allocation, spill selection, and lightweight pipeline optimization on SIMD/NPU compute graphs.

原始项目来自 2025 年“华为杯”中国研究生数学建模竞赛。当前仓库将赛后材料整理为可运行的研究代码项目，便于复现核心流程、检查输出格式，并为后续算法重构保留清晰边界。

从系统角度看，这类问题接近 LLM inference runtime / compiler 中的若干核心任务。Attention、GEMM-heavy projection、fused elementwise kernel 和数据搬运 kernel 通常会被 lower 成依赖受限的算子 DAG。调度器需要在有限片上存储层次中同时考虑算子 ready 状态、cache residency、memory pressure、DMA 搬运，以及 spill / recompute 类 tradeoff。当前实验仍基于竞赛给定的 SIMD/NPU 计算图，不声称覆盖生产级 LLM 推理 benchmark。

## 2. Problem abstraction

输入计算图为有向无环图 `G=(V,E)`。节点分为两类：

- 操作节点：在 `CUBE`、`VECTOR`、`MTE*`、`FIXP` 等执行单元上运行，包含 `Op`、`Pipe`、`Cycles`、`Bufs` 等属性。
- 缓存管理节点：执行 `ALLOC` 或 `FREE`，包含 `BufId`、`Size`、`Type` 等属性。

调度需要同时处理：

- 拓扑依赖约束：若 `(u,v) in E`，则 `u` 必须先于 `v` 执行。
- 多执行单元流水约束：同一 `Pipe` 上的节点不能重叠执行。
- L1、UB、L0A、L0B、L0C 等连续缓存空间限制。
- 缓存不足时的 SPILL 换出和换入代价。

输出分为三类：

- `schedule.txt`：节点执行序列。
- `memory.txt`：缓冲区到缓存地址 offset 的映射。
- `spill.txt`：被换出的缓冲区及其代价。

## 3. Workloads and data schema

官方示例包含六个 case：

| Case family | Cases | Interpretation |
| --- | --- | --- |
| FlashAttention | `FlashAttention_Case0`, `FlashAttention_Case1` | Attention-block style graphs with `MATMUL`, softmax-like vector operations, and frequent UB/L0 interaction. |
| Matmul | `Matmul_Case0`, `Matmul_Case1` | GEMM-heavy graphs resembling dense projection, QKV projection, and FFN layers. |
| Conv | `Conv_Case0`, `Conv_Case1` | Convolution-style graphs with dense interaction between movement and CUBE compute. |

每个 case 包含 `{Case}_Nodes.csv` 和 `{Case}_Edges.csv`。`Edges.csv` 给出 DAG 依赖；`Nodes.csv` 的关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `Id` | 节点编号 |
| `Op` | 操作类型，例如 `ALLOC`、`FREE`、`MATMUL`、`MUL`、`EXP`、`SUB`、`CONV`、`MOVE`、`COPY_IN`、`COPY_OUT` |
| `BufId`, `Size`, `Type` | 缓冲区编号、申请大小、缓存池类型 |
| `Pipe`, `Cycles` | 执行单元和估计执行周期 |
| `Bufs` | 操作节点读取或写入的缓冲区 |

`FlashAttention` case 最接近 LLM attention 中的实际内存压力：矩阵乘、softmax-like vector ops 和 UB/L0 中间结果生命周期交织在一起。`Matmul` case 更规则，适合观察大规模 GEMM tiling 和 CUBE pipeline 压力。`Conv` case 则作为 memory movement 与 compute pipeline 协调的对照负载。

## 4. Method

### 4.1 Problem 1: heuristic scheduling under L0 constraints

问题一生成一个包含原始 DAG 所有节点的拓扑序。主线启发式使用 UB/L1 缓存压力作为优先级：

```text
pressure(v) =
  +Size(v), if Op(v)=ALLOC and Type(v) in {UB,L1}
  -Size(v), if Op(v)=FREE  and Type(v) in {UB,L1}
   0,       otherwise
```

每一步维护 ready set `R={v | all predecessors of v are scheduled}`，并选择压力最小的可行节点。L0A/L0B/L0C 额外使用 live-buffer 约束：如果某个 L0 pool 已有 live allocation，则新的同类 L0 `ALLOC` 会被推迟，除非没有其他 ready 节点可选。

```text
while R is not empty:
    F = nodes in R that do not violate current L0 live constraints
    candidates = F if F is not empty else R
    choose v in candidates with minimal (pressure(v), stable_id(v))
    append v to schedule
    update L0 live state if v is ALLOC/FREE
    release successors whose indegree becomes zero
```

该策略的目标不是证明最优，而是用一个稳定、可复现的贪心准则降低早期缓存驻留压力。

### 4.2 Problem 2: Best-fit + WCB + MATCH

问题二在调度序列上执行多缓存池地址分配。每个缓存池维护两组区间：

- `free_blocks`: 当前可用的连续地址区间。
- `used_blocks`: 已分配的 `(start,end,buf_id)` 区间。

普通分配采用 Best-fit：

```text
choose block b in free_blocks
such that len(b) >= request_size
and len(b) - request_size is minimized
```

若 UB/L1 分配失败，则扫描候选起点 `pos`，找出会与 `[pos, pos+request_size)` 重叠的 live buffers，并计算 victim score：

```text
remaining_lifetime(buf) = max(1, free_time(buf) - current_time)

score(buf) = copy_coeff(buf) * w1 / size(buf)
           + w2 / remaining_lifetime(buf)

score(pos) = sum(score(buf) for buf overlapped by [pos, pos+request_size))
```

其中 `copy_coeff(buf)=1` 表示该 buffer 与 copy-in 相关，`copy_coeff(buf)=2` 表示换出代价更高。算法选择 `score(pos)` 最低的位置，将对应 victim 写入 `spill.txt`，并在 schedule 中插入 SPILL 标记节点。这个过程对应原项目中的 MATCH 风格虚拟区间滑动思想：通过移动或换出低代价区间，为当前申请腾出连续空间。

```text
for node in schedule:
    if node is ALLOC:
        try best-fit allocation
        if allocation fails in UB/L1:
            evaluate candidate positions
            spill victims at the minimum-score position
            retry allocation
    if node is FREE:
        release its interval and merge adjacent free blocks
```

### 4.3 Problem 3: ASAP-style pipeline compression

问题三在问题二的调度和内存分配基础上估计流水执行周期。核心是保守的 ASAP-style left slide：

```text
start(v) = max(
    max(end(u) for u in pred(v)),
    last_finish(pipe(v))
)
end(v) = start(v) + cycles(v)
last_finish(pipe(v)) = end(v)
```

该过程只在依赖约束和同一执行单元串行约束都满足时将节点提前，因此得到的是保守的流水压缩估计。当前版本仍导出与正式提交一致的 schedule / memory / spill 文件，并在运行日志中报告压缩前后的周期估计。

## 5. Experimental setup

仓库还提供 `Mini_Case0` 作为 5 节点线性 DAG，用于快速验证 CLI、输出路径和基本文件格式。完整六 case 运行会覆盖 FlashAttention、Matmul、Conv 三类图，其中 `Matmul_Case1` 和 `Conv_Case1` 节点数较大，适合作为后续性能和可扩展性测试对象。

正式提交结果保存在 `outputs/submission/`，这是论文和竞赛附件对应的 canonical baseline。当前代码重新运行的结果保存在 `outputs/reconstructed/`，用于开发验证。

两个结果区域不混用，原因是赛后归档目录中存在多个后期脚本和中间输出版本，部分结果与最终提交附件不完全一致。将 baseline 和 reconstructed outputs 分离，可以避免后续开发误覆盖正式提交结果。

需要生成竞赛规范附件时，使用 `scripts/runners/export_submission.py`。默认导出源为 canonical baseline，生成 `Attachment/Problem1`、`Attachment/Problem2`、`Attachment/Problem3` 以及 `Q1_`、`Q2_`、`Q3_` 前缀文件名，从而与最终提交附件结构对齐。

如果需要“先运行当前整理后的代码，再生成比赛附件结构”，使用 `scripts/runners/run_submission.py`。该命令先写出 `outputs/reconstructed/problem1|2|3/`，再导出为同样的 `Attachment/Problem*/Q*_...txt` 提交格式。

## 6. Limitations

- 方法是启发式算法，没有全局最优或近似保证。
- 实验主要覆盖竞赛提供的六个 case，不等同于生产级 LLM inference benchmark。
- 当前实现以可复现和可维护为主，不是生产级调度系统或完整编译器后端。
- 部分 legacy 代码存在版本分叉，已归档但未全部验证。
- 测试目前以 CLI smoke test 为主，后续需要补充拓扑合法性、地址不重叠、SPILL 语义和周期估计测试。

## 7. Provenance and authorship

本仓库来自数学建模竞赛项目的赛后重构。原始论文、赛题材料、提交附件和历史脚本均被保留在 `docs/` 或 `archive/` 中。

Parts of the original competition code and this repository reconstruction were developed with AI assistance under human direction. AI tools were used for drafting, refactoring support, documentation, and repository cleanup. The problem abstraction, algorithmic choices, experiment organization, result interpretation, and final repository decisions are maintained as human-directed work.

# Technical Note

## 1. Introduction

This project studies DAG scheduling, contiguous memory allocation, spill selection, and lightweight pipeline optimization on SIMD/NPU compute graphs.

原始项目来自 2025 年“华为杯”中国研究生数学建模竞赛。当前仓库将赛后材料整理为可运行的研究代码项目，便于复现核心流程、检查输出格式，并为后续算法重构保留清晰边界。

从系统角度看，这类问题也接近 LLM inference runtime / compiler 中的若干核心任务：在依赖受限的算子 DAG 上安排执行顺序，控制 cache residency 和 memory pressure，并在有限片上存储下处理 spill / recompute tradeoff。当前实验仍基于竞赛给定的 SIMD/NPU 计算图，不声称覆盖生产级 LLM 推理 benchmark。

## 2. Problem abstraction

输入计算图为有向无环图。节点分为两类：

- 操作节点：在 Cube、Vector、MTE、FIXP 等执行单元上运行，包含 `Op`、`Pipe`、`Cycles`、`Bufs` 等属性。
- 缓存管理节点：执行 `ALLOC` 或 `FREE`，包含 `BufId`、`Size`、`Type` 等属性。

调度需要同时处理：

- 拓扑依赖约束。
- 多执行单元的流水执行约束。
- L1、UB、L0A、L0B、L0C 等连续缓存空间限制。
- 缓存不足时的 SPILL 换出和换入代价。

## 3. Method

### 3.1 Problem 1: heuristic scheduling under L0 constraints

问题一生成一个包含原始 DAG 所有节点的拓扑序。主线启发式使用缓存压力作为优先级：`FREE` 节点降低 UB/L1 驻留压力，普通计算和搬运节点为中性，`ALLOC` 节点增加驻留压力。调度时额外跟踪 L0A/L0B/L0C 的活跃分配，避免同类 L0 缓冲区同时超过约束。

### 3.2 Problem 2: Best-fit + WCB + MATCH

问题二在调度序列上执行多缓存池地址分配。每个缓存池维护空闲区间和已用区间，并使用 Best-fit 放置缓冲区。当 UB/L1 空间不足时，算法评估候选换出缓冲区，结合缓冲区大小、未来释放时间和 copy-in 相关性做 WCB-style victim scoring。该设计对应原项目中的 MATCH 风格虚拟区间滑动思想：通过选择低代价 victim，为当前申请腾出连续区间。

### 3.3 Problem 3: ASAP-style pipeline compression

问题三在问题二的调度和内存分配基础上估计流水执行周期。当前实现保留最终提交代码的保守左滑组织方式：在不打破依赖和执行单元约束的前提下，尽量把节点提前到可执行的最早位置。当前版本输出仍与正式提交保持同类格式，并在运行日志中报告压缩前后的周期估计。

## 4. Experimental setup

官方示例包含六个 case：

- `FlashAttention_Case0`
- `FlashAttention_Case1`
- `Matmul_Case0`
- `Matmul_Case1`
- `Conv_Case0`
- `Conv_Case1`

每个 case 包含 `{Case}_Nodes.csv` 和 `{Case}_Edges.csv`。仓库还提供 `Mini_Case0` 作为 5 节点线性 DAG，用于快速验证 CLI、输出路径和基本文件格式。

## 5. Results

正式提交结果保存在 `outputs/submission/`，这是论文和竞赛附件对应的 canonical baseline。当前代码重新运行的结果保存在 `outputs/reconstructed/`，用于开发验证。

两个结果区域不混用，原因是赛后归档目录中存在多个后期脚本和中间输出版本，部分结果与最终提交附件不完全一致。将 baseline 和 reconstructed outputs 分离，可以避免后续开发误覆盖正式提交结果。

需要生成竞赛规范附件时，使用 `scripts/runners/export_submission.py`。默认导出源为 canonical baseline，生成 `Attachment/Problem1`、`Attachment/Problem2`、`Attachment/Problem3` 以及 `Q1_`、`Q2_`、`Q3_` 前缀文件名，从而与最终提交附件结构对齐。

如果需要“先运行当前整理后的代码，再生成比赛附件结构”，使用 `scripts/runners/run_submission.py`。该命令先写出 `outputs/reconstructed/problem1|2|3/`，再导出为同样的 `Attachment/Problem*/Q*_...txt` 提交格式。

## 6. Limitations

- 方法是启发式算法，没有全局最优或近似保证。
- 实验主要覆盖竞赛提供的六个 case。
- 当前实现以可复现和可维护为主，不是生产级调度系统。
- 部分 legacy 代码存在版本分叉，已归档但未全部验证。
- 测试目前以 CLI smoke test 为主，后续需要补充拓扑合法性、地址不重叠、SPILL 语义和周期估计测试。

## 7. Provenance and authorship

本仓库来自数学建模竞赛项目的赛后重构。原始论文、赛题材料、提交附件和历史脚本均被保留在 `docs/` 或 `archive/` 中。

Parts of the original competition code and this repository reconstruction were developed with AI assistance under human direction. AI tools were used for drafting, refactoring support, documentation, and repository cleanup. The problem abstraction, algorithmic choices, experiment organization, result interpretation, and final repository decisions are maintained as human-directed work.

# 2025-huaweicup-cache-npu-scheduling

We study cache-aware scheduling, memory allocation, spill selection, and pipeline compression on DAG-structured compute graphs for SIMD/NPU execution.

本仓库整理自 2025 年“华为杯”中国研究生数学建模竞赛项目。它不是原始提交包的简单解压，而是将赛后归档材料重构为一个可运行、可追溯、便于继续研究的 lightweight research artifact。

This repository contains a cleaned research artifact distilled from a competition project, including a heuristic scheduler, a spill-aware memory allocator, and an ASAP-style pipeline compression stage.

## Project overview

项目面向通用神经网络处理器中的核内调度问题。输入是表示算子细粒度执行过程的 DAG 计算图，节点包含计算、数据搬运和缓存管理操作；输出是满足依赖约束的调度序列、缓存地址分配和必要的 SPILL 换入换出记录。

当前版本的目标是工程化整理与复现，而不是重新发明算法。仓库保留正式竞赛提交结果作为 canonical baseline，同时提供清理后的主线代码和可执行 CLI，方便后续继续做算法、测试和性能重构。

## Problem abstraction

给定一个 DAG-structured compute graph，每个节点代表硬件执行单元上的操作或缓存管理操作，每条边代表执行依赖。调度系统需要解决三个相互关联的问题：

1. 在依赖约束下生成节点调度序列，并尽量降低 UB/L1 缓存驻留压力。
2. 在 L1、UB、L0A、L0B、L0C 等连续地址空间中分配缓冲区地址；当空间不足时选择 SPILL victim 并记录额外搬运。
3. 在已有调度和依赖约束下做 ASAP-style 左滑压缩，估计流水执行周期的改进空间。

## What is included

- 六个官方示例 case 的 CSV 输入。
- 最终竞赛提交附件结果，按 Problem1/Problem2/Problem3 分离保存。
- 从最终提交代码整理出的主线实现。
- mini case fixture，用于快速 smoke test。
- 原始赛题、最终论文、提交压缩包和历史脚本归档。
- 面向公开仓库的 README、technical note、reconstruction notes、MIT License 和 citation metadata。

## Repository structure

```text
data/raw/csv/                  六个官方示例计算图输入
data/fixtures/mini_case/       快速测试用 5 节点 DAG
src/cache_npu_scheduling/      清理后的主线调度代码
scripts/                       CLI 运行入口
outputs/submission/            正式提交附件结果
outputs/reconstructed/         重新运行脚本生成的结果
docs/                          赛题、论文、技术说明与重构记录
figures/                       论文和实验支撑图表
tests/                         CLI smoke tests
archive/                       原始压缩包、legacy 脚本和旧输出
```

## Installation

建议使用 Python 3.10 或更新版本。

```bash
python -m venv .venv
pip install -r requirements.txt
```

如果只运行脚本，核心依赖为 `pandas`、`numpy`、`networkx` 和 `matplotlib`。`pytest` 用于 smoke tests。

## Quick start

运行 mini case：

```bash
python scripts/run_problem1.py --data-dir data/fixtures/mini_case --case Mini_Case0
python scripts/run_problem2.py --data-dir data/fixtures/mini_case --case Mini_Case0
python scripts/run_problem3.py --data-dir data/fixtures/mini_case --case Mini_Case0
```

运行一个官方 case：

```bash
python scripts/run_problem1.py --case FlashAttention_Case0
python scripts/run_problem2.py --case FlashAttention_Case0
python scripts/run_problem3.py --case FlashAttention_Case0
```

默认输入目录为 `data/raw/csv/`，默认输出目录为 `outputs/reconstructed/`。完整六个官方 case 的批跑可能耗时较长，建议开发时先用 `--case` 指定单个 case。

## Reproducing key results

正式竞赛提交结果保存在：

```text
outputs/submission/problem1/
outputs/submission/problem2/
outputs/submission/problem3/
```

重新运行主线脚本会写入：

```text
outputs/reconstructed/problem1/
outputs/reconstructed/problem2/
outputs/reconstructed/problem3/
```

这两个区域刻意分离：`submission` 是原始提交基线，`reconstructed` 是当前代码重新生成的结果。后者用于验证和继续开发，不会覆盖正式提交附件。

## Exporting submission-ready results

如果目标是生成与竞赛附件规范对齐的最终文件树，有两条路径。

第一条路径用于生成与原正式提交附件对齐的 canonical submission package：

```bash
python scripts/export_submission.py
```

默认导出源是 `outputs/submission/` 中保留的正式提交基线，输出为：

```text
outputs/submission_ready/A25100550012/
outputs/submission_ready/A25100550012_submission_ready.zip
```

第二条路径会先运行当前整理后的三问代码，再把运行结果转换为同样的比赛附件结构：

```bash
python scripts/run_submission.py
```

如果只是验证结构，可以先跑 mini case：

```bash
python scripts/run_submission.py --data-dir data/fixtures/mini_case --case Mini_Case0 --package-name MiniSubmission --no-zip
```

导出的附件结构为：

```text
A25100550012/
├── Attachment/
│   ├── Problem1/
│   │   └── Q1_{Case}_schedule.txt
│   ├── Problem2/
│   │   ├── Q2_{Case}_schedule.txt
│   │   ├── Q2_{Case}_memory.txt
│   │   └── Q2_{Case}_spill.txt
│   └── Problem3/
│       ├── Q3_{Case}_schedule.txt
│       ├── Q3_{Case}_memory.txt
│       └── Q3_{Case}_spill.txt
└── code/
    ├── problem1_scheduler.py
    ├── problem2_allocator.py
    └── problem3_pipeline.py
```

如果确实要从当前重新运行结果导出，而不是使用正式提交基线，可以执行：

```bash
python scripts/export_submission.py --source reconstructed
```

注意：`--source reconstructed` 要求 `outputs/reconstructed/problem1|2|3/` 中已经存在六个官方 case 的完整结果文件。

## Output format

Problem 1:

```text
{Case}_schedule.txt
```

每行一个节点 ID，表示满足 DAG 依赖的调度顺序。

Problem 2:

```text
{Case}_schedule.txt
{Case}_memory.txt
{Case}_spill.txt
```

`schedule` 保存加入 SPILL 标记后的调度序列，`memory` 保存缓冲区到起始地址偏移的映射，`spill` 保存被换出的缓冲区及估计搬运代价。

Problem 3:

```text
{Case}_schedule.txt
{Case}_memory.txt
{Case}_spill.txt
```

当前 artifact 与最终提交保持一致：问题三复用问题二的调度和内存输出，同时在运行时计算保守左滑前后的周期估计。

## Method ownership

本仓库保留的主要 ownership 点包括：

- 问题一：面向缓存驻留压力的贪心拓扑调度，以及对 L0A/L0B/L0C 类型约束的处理。
- 问题二：Best-fit 地址分配、WCB-style spill victim 评分，以及 MATCH 风格的虚拟区间滑动思想。
- 问题三：ASAP-style 左滑压缩，用于在依赖约束和执行单元约束下压缩流水。
- 实验组织：六个官方 case 的输入、输出、正式提交基线与重构输出分离。

## Limitations

- 当前实现是 heuristic artifact，没有近似比或全局最优保证。
- 代码主要围绕竞赛给定 case 整理，尚未扩展为生产级调度器。
- 部分 legacy 脚本和后期输出已归档，但没有逐一验证可运行性。
- 问题一在较大 case 上可能运行较慢。
- 当前 smoke tests 只验证 CLI 和基础输出存在性，不覆盖完整调度正确性证明。

## AI-assisted coding disclosure

Parts of the original competition code and this repository reconstruction were developed with AI assistance under human direction. AI tools were used for code drafting, refactoring support, documentation, and repository cleanup. The problem interpretation, algorithmic choices, experiment organization, result selection, and final repository decisions are maintained as human-directed work.

## Citation

如果引用本仓库，请参考 `CITATION.cff`。当前 citation 使用项目贡献者占位名称，不包含个人隐私信息。

## License

本仓库采用 MIT License。原始竞赛题面、附件和论文材料保留其原始来源语境；公开复用时请同时尊重竞赛材料的来源说明。

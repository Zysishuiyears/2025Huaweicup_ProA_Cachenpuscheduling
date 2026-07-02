# Reconstruction Notes

本文档记录本仓库从赛后归档目录整理为工程仓库时的主要判断和文件归属。

## Canonical baseline

正式竞赛提交基线来自原目录中的：

- `A25100550012.pdf`
- `A25100550012/A25100550012/Attachment/Problem1`
- `A25100550012/A25100550012/Attachment/Problem2`
- `A25100550012/A25100550012/Attachment/Problem3`
- `A25100550012/A25100550012/代码/问题一代码.py`
- `A25100550012/A25100550012/代码/问题二三代码.py`

整理后对应位置：

- `docs/paper/submission_paper_A25100550012.pdf`
- `outputs/submission/problem1/`
- `outputs/submission/problem2/`
- `outputs/submission/problem3/`
- `scripts/core/cache_npu_scheduling/`

正式提交包本体移动到 `archive/submission_packages/`。

## Original problem materials

原始赛题说明与附件保留为：

- `docs/contest/problem_statement.docx`
- `archive/original_materials/problem_attachment_original.zip`

根目录 12 个官方 CSV 输入文件已移动到 `data/raw/csv/`。这些文件与原始附件 ZIP 中的 `CSV版本/` 内容一致。

## Mainline code selection

主线代码以最终提交包中的两份 Python 源码为来源：

- 问题一整理为 `scripts/core/cache_npu_scheduling/problem1_scheduler.py`
- 问题二整理为 `scripts/core/cache_npu_scheduling/problem2_allocator.py`
- 问题三包装为 `scripts/core/cache_npu_scheduling/problem3_pipeline.py`

整理时仅做工程性修改：

- 使用 `Path` 管理输入和输出目录。
- 增加 `--data-dir`、`--output-dir`、`--case` CLI 参数。
- 将重新生成结果写入 `outputs/reconstructed/`。
- 保留问题二和问题三的提交输出语义分离。

竞赛提交规范导出由 `scripts/runners/export_submission.py` 和 `scripts/runners/run_submission.py` 负责。`export_submission.py` 默认导出源为 `outputs/submission/`，因此生成的 `outputs/submission_ready/A25100550012/Attachment/` 与正式提交附件的目录、文件名和文件集合对齐。`run_submission.py` 会先运行当前整理后的三问代码，再把结果导出为同样的比赛附件结构。

二次整理时，原 `src/cache_npu_scheduling/` 被移动到 `scripts/core/cache_npu_scheduling/`，原顶层 `scripts/*.py` 入口被移动到 `scripts/runners/`。这样保留核心实现和运行入口的层次，同时避免顶层同时出现 `src/` 与 `scripts/` 造成理解负担。

## Legacy materials

以下内容归档，不作为默认运行主线：

- 根目录 `Q*.py`
- 根目录 `problem2_*.py`
- 根目录 `HuaweiAprob2*.py`
- 根目录 `python problem3_pipelin.py`
- 根目录 `建图.py`
- 根目录旧 schedule / memory / spill / timeline 输出
- 原 `Problem2/`
- 原 `Problem3/`
- 原 `Problem2_Output/`
- 原 `Problem3_Output/`
- 原 `Problem2.zip`

归档位置：

- `archive/legacy_scripts/`
- `archive/legacy_outputs/`

## Figures

图表文件整理到：

- `figures/problem1/`
- `figures/parameter_scan/`
- `figures/sensitivity_heatmaps/`

这些图表作为分析和论文支撑材料保留，不作为主运行流程的必需输入。

## Known risks

- 正式提交附件、根目录后期输出和部分 legacy 脚本之间存在结果差异。
- 问题二和问题三的正式提交 schedule 包含 SPILL 节点，行数可能超过原始节点数。
- 问题一在较大 case 上可能运行较慢。
- 当前 smoke tests 不等价于完整算法正确性验证。

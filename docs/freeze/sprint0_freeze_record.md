# Sprint0 冻结记录

## 最终冻结 tag

- Tag: `sprint0-freeze-v1.0`（轻量 tag）
- Commit: `bbe1a24`
- Commit message: `docs(sprint0): align case001 calibration with Sprint0 V1.3 (EVC statement, benchmark entry, case_class source)`
- 冻结日期: 2026-10-04

## 历史冻结点

- `68653d8`：原冻结点（文档一致性修正前）。
- `64ad792`：更早的 V1.3 基线。

## 验收证据

- pytest: 263 tests green
- ruff: 0 errors
- benchmark: `out/bench_ruleonly_sprint0.csv`
- demo: `out/demo_sprint0/` 产物齐全

## Sprint1 起点

Sprint1 从 `sprint0-freeze-v1.0` 拉分支。
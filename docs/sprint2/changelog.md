# Sprint 2 变更与阻塞记录

> 起始：2026-10-27
> 依据：`Sprint2_详细执行计划_V1.0.md`

---

## S2-A Day 1：基线核查（2026-10-27）

> 状态：🟡 进行中（第 1 项已完成，第 2~10 项待查）
> 依据：§7.0 + 附录 C（10 项唯一真值）

### 完成项

| # | 核查项 | 结论 |
|---|---|---|
| 1 | Git HEAD 与测试基线 | HEAD `ec47791`；工作区 clean；非 integration `539 passed / 1 skipped / 24 deselected`；bench `tp=16 / fn=0 / fp ∈ [0, 2]` |

### 发现的差异与处理

| # | 差异 | 处理动作 |
|---|---|---|
| 1 | 计划记录的 HEAD `141e9f4` 已过时，实际为 `ec47791` | 以 `ec47791` 为 Sprint 2 基线，记入 `baseline_audit.md` 第 1 项 |
| 2 | `backend/tests/test_llm_zhipu.py` 在工作区被删（`D` 状态），导致全量测试从 538 → 510 | 确认是未提交的误删（git log 无删除 commit），`git checkout --` 恢复；恢复后 `539 passed / 1 skipped / 24 deselected` |
| 3 | 恢复后 passed 比 Sprint 1 记录多 1（539 vs 538），`test_llm_zhipu.py` 内部测试数 29 vs Sprint 1 记录的 28 | 以 539 为新基线，记入 `baseline_audit.md` 第 1 项；不追究 +1 原因（不影响基线判定） |
| 4 | bench integration 3 次实测 fp = 0 / 2 / 1，**不稳定** | 归因为 glm-4-flash 随机性（Sprint 1 S1-B.5 已登记为遗留项，Sprint 2 不修复）；**基线改为 `tp=16 / fn=0 / fp ≤ 2`** |

### 计划修订

| # | 位置 | 修订内容 |
|---|---|---|
| 1 | §2.2 向后兼容最后一条 | `tp=16/fp=0/fn=0` → `tp=16 / fn=0 / fp ≤ 2` + 随机性说明 |
| 2 | §8.2 准出清单第 2 条 | 同上 |
| 3 | §14 S2-A 行「阶段验收」列 | 同上 |
| 4 | §14.4 DoD 第 5 条 | 同上 |

> 修订依据：附录 C 第 1 项实测结果 + Sprint 1 S1-B.5 遗留项「LLM 输出随机性」。
> 修订原则：不通过「重跑直到 fp=0」凑基线（自欺欺人），而是接受 `fp ≤ 2` 为业务可接受波动。

### 明日待办

| # | 核查项 | 优先级 |
|---|---|---|
| 2 | `task_status` 当前模型/迁移/历史数据情况 | P0 |
| 3 | `RuleCandidate` / `FeedbackItem` / `KnowledgeDoc/KnowledgeChunk` 实际字段 | P0 |
| 4 | 正式规则加载路径、DB/YAML 同步与引擎重载机制 | P0 |
| 5 | 规则候选现有状态机与 CLI 能力 | P1 |
| 6 | `metrics_definition.md` 当前版本、EAR/AI_NER 公式、门槛、Seed 过滤字段 | P0 |
| 7 | 已有 API/服务盘点 | P1 |
| 8 | 前后端类型一致性 | P1 |
| 9 | AI_NER 样本门槛是否已冻结（大概率未冻结，需补） | P0 |
| 10 | **KnowledgeGap 一对多业务场景核查**（决定 S2-A 建表形态） | **P0 最高** |

### 遗留项

| 项 | 影响 | 归属 |
|---|---|---|
| `test_llm_zhipu.py` 恢复后比 Sprint 1 记录多 1 个测试（539 vs 538） | 无（基线已更新为 539） | 不追究 |
| bench `fp` 在 [0, 2] 波动 | 准出项无法用「fp=0」判定，已改为「fp ≤ 2」 | Sprint 3（模型切换 / Prompt 收敛时修复） |

---

## S2-A Day 2：迁移脚本执行（2026-10-28）

> 状态：✅ 完成
> 依据：`baseline_audit.md` 第 2、3、10 项 + 附录 C 第 9 项

### 迁移内容

| # | 表 | 改动 | 来源 |
|---|---|---|---|
| 1 | `review_result` | 加 `task_status VARCHAR(32) NOT NULL DEFAULT 'pending'` + 索引 `idx_review_result_task_status` | `baseline_audit.md` 第 2 项 |
| 2 | `rule_candidates` | 加 `benchmark_result JSONB` / `benchmarked_at TIMESTAMPTZ` / `rejected_reason TEXT` | 第 3 项 |
| 3 | `knowledge_gap`（新建） | 12 列 + CHECK `status IN (...)` + 3 索引 | 第 10 项 |
| 4 | `gap_feedback_link`（新建） | 2 列复合主键 + 2 FK CASCADE + 1 索引 | 第 10 项 |

**迁移文件**：`backend/alembic/versions/s2a_schema_updates.py`

### ORM 改动

`backend/app/persistence/models.py`：
- `ReviewResult` 加 `task_status` + 索引
- `RuleCandidate` 加 3 字段
- 新增 `KnowledgeGap` / `GapFeedbackLink` 类

### 验收证据

| 项 | 结果 |
|---|---|
| `alembic current` | `s2a_schema_updates (head)` |
| Schema 验证 | `review_result` 18 列（含 `task_status`）；`rule_candidates` 16 列（含 3 新列）；`knowledge_gap` 12 列；`gap_feedback_link` 2 列 |
| 全量非 integration | **539 passed / 1 skipped / 24 deselected**（与基线一致） |
| bench integration | `tp=16 / fn=0 / fp=2`（在 `fp ≤ 2` 基线范围内） |
| `downgrade -1` + `upgrade head` | 待验证（见下方） |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| PowerShell here-string 写脚本 | `@'...'@` 被写进文件 | 改用记事本/VS Code 建 `scripts/_tmp_cols.py` |
| PowerShell 引号嵌套 | `python -c` 里 `''review_result''` 被吃成 `review_result` | 改用临时脚本 + SQL 参数化 `:t` |
| `review_result` 列数预期 | `baseline_audit.md` 第 2 项写「16 列」，实际迁移前 17 列（Sprint 1 加了 `schematic_case_id`），迁移后 18 列 | 文档笔误，不影响功能，下次整理时改 |

### 遗留项

| 项 | 影响 | 归属 |
|---|---|---|
| `baseline_audit.md` 第 2 项「16 列」应为「17 列」 | 文档笔误，不影响功能 | 下次整理时改 |
| `downgrade -1` 验证 | 待跑 | S2-A 收尾时补 |
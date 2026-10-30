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
| `downgrade -1` 验证 | **已完成**：`downgrade -1` → `c1_embedding_dim_1024` → `upgrade head` → `s2a_schema_updates`，双向无报错 | 已完成 |
---

## S2-A Day 2 补记：昨日遗留项处理 + 关键技术总结（2026-10-28）

### 一、昨日（Day 1）遗留项处理

| # | 遗留项 | 处理结果 | 证据 |
|---|---|---|---|
| 1 | `test_llm_zhipu.py` 恢复后比 Sprint 1 多 1 个测试（539 vs 538） | **不追究**。确认是 Sprint 1 收口后本地新增 1 个测试未 commit，恢复后自然带出。**以 539 为基线**，已记入 `baseline_audit.md` 第 1 项 | `git checkout -- backend/tests/test_llm_zhipu.py` 后 `pytest tests/ -m "not integration"` 输出 `539 passed / 1 skipped` |
| 2 | bench `fp` 在 [0, 2] 波动（3 次实测 0/2/1） | **基线改为 `tp=16 / fn=0 / fp ≤ 2`**，已同步到计划 §2.2 / §8.2 / §14 S2-A / §14.4 DoD 第 5 条 / 附录 D，共 5 处 | 今日 `test_benchmark_llm.py -m integration` 实测 `tp=16 / fn=0 / fp=2`，**在 `fp ≤ 2` 范围内，基线不退化** |
| 3 | 计划 §2.2 等 4 处 `fp=0` 未同步 | **今日已全部改为 `fp ≤ 2`**，并在附录 D 补「bench 基线 `tp=16/fn=0/fp ≤ 2`」 | 计划全文搜 `fp=0` 已 0 处命中（除历史描述） |
| 4 | `baseline_audit.md` 第 1 项「HEAD 141e9f4」过时 | **已改为 `ec47791`**，并注明「计划值已过时」 | `baseline_audit.md` 第 1 项第 1 行 |
| 5 | `baseline_audit.md` 第 6 项 3 个 `?`（EAR/AI_NER 公式、Seed 过滤字段） | **已填实**：EAR 公式指向 Sprint 1 §9.6，AI_NER 公式指向 §6.7 + Sprint 1 §23.2，Seed 过滤字段定为 `comment` 前缀 `[SEED]`（Sprint 1 E.2.4 约定） | `baseline_audit.md` 第 6 项 5 列对齐 |
| 6 | `baseline_audit.md` 第 8 项「`ExpertDecision` 需核对」 | **已确认**：`DefectsResponse` 为 `items: list[dict]` 透传，`schemas.py` 不定义字段，`service.get_task_defects` 已含 3 字段（Sprint 1 G1） | `Select-String "class DefectsResponse" -Context 0,15` 输出确认 |
| 7 | `baseline_audit.md` 第 8 项末尾「结论」与表格矛盾 | **已改**：「S2-D 前确认」→「S2-B 前跑一次 `GET /review/tasks/{id}/defects` 确认 3 字段」 | `baseline_audit.md` 第 8 项末尾 |
| 8 | `baseline_audit.md` 文档头「结论：待填」「核查日期：`<日期>`」 | **已填**：结论「通过（有差异但均已处理，不阻塞 S2-A 开工）」，日期 `2026-10-28` | `baseline_audit.md` 文档头 |
| 9 | `metrics_definition.md` 版本号从 v3.1 → v3.2 | **已完成**：新增 §6.7「AI_NER 样本门槛」，文档头改 v3.2，§10 版本记录追加 v3.2 行 | `metrics_definition.md` v3.2 全文 |
| 10 | `metrics_definition.md` §10 v3.1 重复两行 | **已删**，只留 1 行 | `metrics_definition.md` §10 |

### 二、关键技术总结（S2-A 全程）

#### 1. 「基线核查」不是「一次性动作」，是「持续对照」

**做法**：Day 1 填 `baseline_audit.md` 10 项 → 发现 5 项差异 → 修订计划 5 处 → Day 2 迁移后**回填**第 2、3、10 项「已处理」→ 全量测试 + bench 验证「基线未退化」。

**价值**：`baseline_audit.md` 是「活文档」，迁移后要回填「处理动作」，不是「填完就冻结」。**否则「差异」和「实际」会脱节**。

#### 2. 「bench 基线」不能用「单次 fp=0」判定，要用「区间」

**做法**：3 次实测 fp=0/2/1 → 归因 glm-4-flash 随机性 → 基线改为 `tp=16 / fn=0 / fp ≤ 2` → 今日实测 fp=2 仍判定「不退化」。

**价值**：**LLM 有随机性，基线必须是「可接受区间」，不是「单点值」**。否则每次跑 bench 都可能因随机性「假失败」，导致团队对基线失去信任。Sprint 1 S1-B.5 已登记「glm-4-flash 随机性」，Sprint 2 接受此现状，**不通过「重跑直到 fp=0」凑基线**（自欺欺人）。

#### 3. 「Alembic 迁移」必须「先 dry-run 看 SQL，再实际执行」

**做法**：`alembic upgrade head --sql` 打印 SQL → 人工核对「3 个 ADD COLUMN + 2 个 CREATE TABLE + 5 个 CREATE INDEX」→ 确认无遗漏、无多余 → `alembic upgrade head` 实际执行。

**价值**：**dry-run 是「迁移的安全网」**。如果直接 `upgrade head`，一旦 SQL 有错（如漏索引、FK 顺序错），回滚要 `downgrade -1`，可能影响数据。**dry-run 0 成本，能提前发现 90% 的问题**。

#### 4. 「ORM 改动」必须先「import 验证」，再「跑迁移」

**做法**：改 `models.py`（3 处 + 2 新类）→ `python -c "from app.persistence.models import KnowledgeGap, GapFeedbackLink, ...; print([c.name for c in ...])"` 验证 → 确认列名、关系、FK 都正确 → 再写迁移脚本。

**价值**：**ORM 的 `relationship` / `ForeignKey` 配置错，`import` 时就会报错**（如 `back_populates` 找不到对方、`ForeignKey` 指向不存在的表）。**先 import 验证，比跑迁移后报错再排查快 10 倍**。

#### 5. 「PowerShell 引号地狱」的 3 个解法

**问题**：`python -c "..."` 多行 + SQL 字符串 + PowerShell 转义，`''review_result''` 被吃成 `review_result`，`@'...'@` 被写进文件。

**解法**：
- **解法 A（推荐）**：写临时脚本 `scripts/_tmp_cols.py`，用 SQL 参数化 `:t`，**完全避开引号嵌套**。
- **解法 B**：用 `@"..."@`（双引号 here-string），不是 `@'...'@`（单引号 here-string）。
- **解法 C**：用记事本/VS Code 建文件，**不用 PowerShell here-string**。

**价值**：**Windows 开发环境下，PowerShell 引号是高频踩坑点**。Sprint 2 后续所有「多行 Python + SQL」场景，**统一用「临时脚本 + 参数化」**，不再在 `-c` 里折腾。

#### 6. 「测试通过数」是「迁移不破坏现有功能」的最硬证据

**做法**：迁移前 539 passed → 迁移后 539 passed → **passed 数不变，说明 ORM 改动 + 迁移没破坏任何现有测试**。

**价值**：**`test_review_persistence.py` 18 个测试全过**，说明没有测试硬编码 `review_result` 列数（16→18）。**如果 passed 数变了，第一时间看哪个测试失败**，是「改测试」还是「改迁移」，不要「改测试凑通过」。

#### 7. 「迁移可回滚」是「Schema 唯一来源」的底线

**做法**：`alembic downgrade -1` → `alembic current` 确认回到 `c1_embedding_dim_1024` → `alembic upgrade head` 确认回到 `s2a_schema_updates`。

**价值**：**`downgrade()` 写对，才能「迁移出错时回滚」**。如果 `downgrade()` 顺序错（如先删表后删索引），回滚会报错，**生产环境迁移失败就卡住**。Sprint 2 每个迁移都必须「upgrade + downgrade 双向验证」。

### 三、S2-A 全程时间线

| 日期 | 阶段 | 完成内容 |
|---|---|---|
| 2026-10-27 | Day 1 | 基线核查 10 项；发现 5 项差异；修订计划 5 处；恢复误删文件；确定 bench 基线 `fp ≤ 2` |
| 2026-10-28 上午 | Day 2 前半 | 补填 `baseline_audit.md` 第 4、5、7、8 项；填结论 + 处理动作；补冻结 AI_NER 门槛（`metrics_definition.md` v3.2） |
| 2026-10-28 下午 | Day 2 后半 | 改 `models.py`（3 处 + 2 新类）；写迁移脚本；dry-run 核对 SQL；`upgrade head` 执行；Schema 验证；全量测试 539 passed；bench `tp=16/fn=0/fp=2` |

**S2-A 核心工作（基线核查 + 迁移）已完成。剩余：`downgrade -1` 验证 + commit。**

### 四、S2-A 对 S2-B 的输入

| 输入 | 内容 | 用途 |
|---|---|---|
| `review_result.task_status` 列 | 已加，默认 `pending` | S2-C 的 `?status=` 筛选、S2-B 的 `get_task` 读真实状态 |
| `rule_candidates.benchmark_result` 列 | 已加，JSONB | S2-F 的候选 Benchmark 落库 |
| `rule_candidates.benchmarked_at` 列 | 已加，TIMESTAMPTZ | S2-F 的「刷新不重复跑」 |
| `rule_candidates.rejected_reason` 列 | 已加，TEXT | S2-F 的驳回理由记录 |
| `knowledge_gap` 表 | 已建，12 列 + 3 索引 | S2-G 的知识缺口 Backlog |
| `gap_feedback_link` 表 | 已建，2 列复合主键 | S2-G 的一对多关系 |
| `baseline_audit.md` | 10 项核查完成 | S2-B~H 的「基线事实」来源 |
| `business_logic.md` | 业务逻辑冻结 | S2-B~H 的「角色边界」来源 |
| `metrics_definition.md` v3.2 | AI_NER 门槛 ≥ 10 已冻结 | S2-C 的指标计算 |

---

## S2-B：专家判定 PATCH 接口（2026-10-28）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §4.3 + §14 S2-B 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/api/schemas.py` | 追加 `ExpertDecisionLiteral` / `UpdateExpertDecisionRequest` / `ExpertDecisionResponse` |
| 2 | `backend/app/services/task_service.py` | 顶部加 `timezone`；追加 `update_expert_decision(task_id, defect_id, expert_decision, updated_by)`；`__all__` 加 1 项 |
| 3 | `backend/app/api/review_tasks.py` | 顶部 import 补 3 处；追加 `PATCH /{task_id}/defects/{defect_id}` 端点 |
| 4 | `backend/tests/test_api.py` | 追加 3 条 Mock 测试（200 / 422 / 404） |
| 5 | `backend/tests/test_review_persistence.py` | 追加 1 条 Service 测试（正确更新 3 字段） |

### API 契约

```http
PATCH /api/v1/review/tasks/{task_id}/defects/{defect_id}
Content-Type: application/json

{
  "expert_decision": "confirmed",
  "expert_decision_by": null
}

# 200 响应
{
  "defect_id": "RUN-xxx-DEF-RULE-POWER_001",
  "expert_decision": "confirmed",
  "expert_decision_updated_at": "2026-10-28T12:00:00+00:00",
  "expert_decision_by": null
}
```

### 关键设计

1. **双校验**：task 存在 + defect 属于该 task（`review_defect.review_result_id == review_result.id`），防越权。
2. **不改 `review_status`**：只更新 `expert_decision` 3 字段，AI 质量状态独立。
3. **不隐式创建 `feedback_item`**：专家判定与过程反馈解耦。
4. **枚举兜底**：API 层 `Literal` 校验 + Service 层 `ValueError` 双保险。
5. **`expert_decision_by` 允许 `None`**：Sprint 2 无用户系统，前端传 `null`。

### 验收证据

| 项 | 结果 |
|---|---|
| import 验证 | `OK` |
| `test_api.py -k "expert_decision"` | **3 passed**（200 / 422 / 404） |
| `test_review_persistence.py -k "update_expert_decision"` | **1 passed** |
| 全量非 integration | **543 passed / 1 skipped / 24 deselected**（539 + 4，无回归） |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| Mock 测试打错位置 | 一开始想 patch `review_tasks.update_expert_decision`，但 `review_tasks.py` 是 `task_service.update_expert_decision(...)` 模块级调用 | **patch `task_service.update_expert_decision`**，不是 `review_tasks` 的属性 |
| `task_service.py` 顶部 `timezone` 缺失 | 顶部只有 `from datetime import datetime`，`update_expert_decision` 需要 `timezone` | 顶部改为 `from datetime import datetime, timezone`，删函数内局部 import |
| `__all__` 漏 `update_expert_decision` | 不影响功能（`review_tasks.py` 用模块级调用，不走 `__all__`） | 已补进 `__all__` |

### 对 S2-C / S2-D 的输入

| 输入 | 内容 | 用途 |
|---|---|---|
| PATCH 端点 | `PATCH /review/tasks/{task_id}/defects/{defect_id}` | S2-D 前端「确认 / 驳回 / 待核验」按钮调用 |
| `ExpertDecisionResponse` | 4 字段（`defect_id` / `expert_decision` / `expert_decision_updated_at` / `expert_decision_by`） | S2-D 前端乐观更新后覆盖徽标 |
| `expert_decision` 枚举 | `pending / confirmed / rejected / needs_verification` | S2-D 三按钮映射 |

---

## S2-C：任务筛选 + 真实指标聚合（2026-10-28）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §4.2 + §14 S2-C 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/services/benchmark_service.py` | 追加 `compute_real_ear` / `compute_real_ai_ner` / `compute_sample_info` + `AI_NER_SAMPLE_THRESHOLD=10` / `EAR_SAMPLE_THRESHOLD=20` |
| 2 | `backend/app/services/task_service.py` | 追加 `get_task_summary_metrics`；`list_tasks_filtered` 加 `status` 参数 + item 透出 `status`；`feedback_dicts` 修正 `review_defect_id → defect_id` 映射 |
| 3 | `backend/app/api/schemas.py` | 追加 `SampleInfo` / `SummaryMetrics`（前置定义）；`TaskDetailResponse.summary_metrics`；`TaskListItem.status` |
| 4 | `backend/app/api/review_tasks.py` | `GET /review/tasks` 加 `?status=`；`GET /review/tasks/{id}` 组装 `summary_metrics` |
| 5 | `backend/tests/test_benchmark.py` | 追加 7 条（EAR 3 + AI_NER 3 + sample_info 1） |
| 6 | `backend/tests/test_review_persistence.py` | 追加 3 条，修 1 条老测试 |
| 7 | `frontend/src/types/task.ts` | `TaskStatus` 加 `"pending"`；加 `SampleInfo` / `SummaryMetrics`；`TaskDetail.summary_metrics`；`TaskListItem.status` |

### 指标口径（严格对齐 metrics_definition.md v3.2）

| 指标 | 公式 | 样本门槛 | 无样本时 |
|---|---|---|---|
| EAR | `N_adopted / N_defect_with_feedback` | ≥ 20 | `None` |
| AI_NER | `N_ai_effective / (N_rule_effective + N_ai_effective)` | ≥ 10 条 `ai_discovered` | `None` |

**关键边界**：
- `knowledge_gap` / `new_rule_candidate` 不进 EAR 分母、不覆盖分子。
- `comment` 前缀 `[SEED]` 的反馈被排除（Sprint 1 E.2.4 约定，不加 `simulated` 列）。
- 实时聚合，不写 `review_result` 静态列。

### 验收证据

| 项 | 结果 |
|---|---|
| `test_benchmark.py -k "real_ear or real_ai_ner or sample_info"` | **7 passed** |
| 全量 `test_benchmark.py` | **24 passed**（17 现有 + 7 新增） |
| `test_review_persistence.py -k "summary_metrics or filtered_by_status or filtered_basic"` | **4 passed** |
| 全量非 integration | **553 passed / 1 skipped / 24 deselected**（543 + 10，无回归） |
| `npm run type-check` | **0 错误** |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| `review_defect_id` vs `defect_id` | `compute_real_ear` 期望业务 ID（str），`get_task_summary_metrics` 传 DB 外键 id（int），`ear_sample_size = 0` | `get_task_summary_metrics` 用 `defect_id_map = {d.id: d.defect_id}` 翻译 |
| 测试 FeedbackItem 未关联 defect | 3 条 feedback 的 `review_defect_id=None` | 先 `db.flush()` 拿 defect DB id，再关联 |
| Sprint 1 G2 老测试断言过时 | `test_list_tasks_filtered_basic` 断言「不含 status」，S2-C 加了 | 改为「含 status + 值合法」，改注释 2 处 |

### 对 S2-D / S2-E 的输入

| 输入 | 内容 | 用途 |
|---|---|---|
| `SummaryMetrics` 类型 | 5 字段 | S2-E 的 `TaskSummaryCard` 渲染 |
| `sample_info` | 4 字段（样本数 + 门槛） | S2-E 显示「N/A · 样本不足」 |
| `TaskStatus` 加 `pending` | 前端 `task.ts` | S2-D/S2-E 渲染 `task_status` 徽标 |
| `?status=` | `GET /review/tasks?status=done/failed/pending` | S2-D 前端任务列表筛选 |

---

## S2-D：前端专家判定交互（2026-10-29）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §5.1 + §14 S2-D 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `frontend/src/api/review.ts` | 加 `updateExpertDecision` 函数 |
| 2 | `frontend/src/types/defect.ts` | 加 `UpdateExpertDecisionRequest` / `UpdateExpertDecisionResponse` |
| 3 | `frontend/src/components/ExpertDecisionPanel.vue` | **新建**：三按钮 + 乐观更新 + 失败回滚 + 审计行 |
| 4 | `frontend/src/components/DefectCard.vue` | 展开区集成 `ExpertDecisionPanel`，`props` 加 `taskId` + `emit` |
| 5 | `frontend/src/components/ExpertDecisionBadge.vue` | **修复**：`const badge` → `computed(() => ...)`，解决 props 变化样式不更新 |
| 6 | `frontend/src/pages/TaskDetailPage.vue` | `<DefectCard>` 加 `:task-id` + `@expert-decision-updated`，加 `onExpertDecisionUpdated` 用 `splice` 触发响应式 |

### 验收证据

| 项 | 结果 |
|---|---|
| `npm run type-check` | **0 错误** |
| 展开缺陷卡片 | 显示「专家判定」三按钮 + 当前徽标 |
| 点「确认」 | 徽标变绿 + 审计行出现「判定于 … · 操作者 未记录」 |
| 折叠区徽标同步 | 右上角 `专家确认` 同步变绿 |
| 当前态按钮禁用 | `✓ 确认` 变灰 disabled |
| 刷新页面 | 徽标仍为「专家确认」，审计行仍在 |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| `ExpertDecisionBadge` 样式不更新 | 点击确认后，内部徽标颜色不变（依然灰色） | `const badge = DECISION_MAP[props.decision]` → `const badge = computed(() => DECISION_MAP[props.decision])`，使响应式生效 |
| 折叠区徽标不同步 | 内部变了，外部没变 | 父组件 `onExpertDecisionUpdated` 用 `defects.value.splice(idx, 1, updated)` 替换元素，强触发 Vue 数组响应式 |

### 对 S2-E 的输入

| 输入 | 内容 | 用途 |
|---|---|---|
| `ExpertDecisionPanel` 组件 | 已完成 | 后续无改动 |
| `onExpertDecisionUpdated` 父组件回调 | 已完成 | 后续无改动 |

---

## S2-E：前端摘要卡片与表单增强（2026-10-29）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §5.2 / §5.7 + §14 S2-E 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `frontend/src/types/feedback.ts` | 加 `SuggestionDiffItem` / `RuleCandidateInput` |
| 2 | `frontend/src/components/TaskSummaryCard.vue` | **新建**：EVC / AI_NER / EAR / 质量分布四格卡片（支持 N/A 语义） |
| 3 | `frontend/src/components/SuggestionDiffEditor.vue` | **新建**：旧建议 / 新建议双列对比表单 |
| 4 | `frontend/src/components/RuleCandidateForm.vue` | **新建**：标题 / 描述 / severity 结构化表单 |
| 5 | `frontend/src/components/FeedbackPanel.vue` | 改用 `SuggestionDiffEditor` / `RuleCandidateForm`，加前端必填校验 |
| 6 | `frontend/src/api/feedback.ts` | 适配后端 `RuleCandidate` 必填字段（`severity_hint` → `severity`，兜底 `rule_id` 等） |
| 7 | `frontend/src/pages/TaskDetailPage.vue` | 顶部集成 `TaskSummaryCard` |
| 8 | `frontend/src/api/http.ts` | 修复路径拼接（去掉双斜杠），加 DEV 环境 URL 日志打印 |
| 9 | `frontend/vite.config.ts` | 补全 `server.proxy` 代理配置（修复 API 请求被 Vite 路由拦截） |
| 10 | `frontend/src/components/ExpertDecisionPanel.vue` | 加 `watch` 同步 props，加请求链路 console 日志 |

### 验收证据

| 项 | 结果 |
|---|---|
| `npm run type-check` | **0 错误** |
| `TaskSummaryCard` 渲染 | ✅ 四格布局，`null` 值正确显示 `N/A · 样本不足` |
| `SuggestionDiffEditor` | ✅ 双列表单 + 增删按钮 |
| `RuleCandidateForm` | ✅ 三字段表单 |
| `FeedbackPanel` 提交 | ✅ 成功提交，无 422 报错，控制台显示 `[API 成功] POST /api/v1/feedbacks` |
| `ExpertDecisionPanel` | ✅ 点击确认后乐观更新、`[API 成功] PATCH` 日志正常 |

---

### 踩坑与修正（S2-D & S2-E 核心排障记录）

| 坑 | 现象 | 修正 |
|---|---|---|
| **Vite HMR 失效** | 修改代码或 `vite.config.ts` 后页面无变化，或请求被错误路由 | 终端 `Ctrl+C` 停 Vite，必要时 `taskkill /F /IM node.exe` 清理进程，重启 `npm run dev` + 浏览器 `Ctrl+F5` 强刷。 |
| **API 请求返回 HTML** | Network 里请求 Type 为 `document`，返回 `<html>` 而非 JSON | 未配代理导致被 Vite 路由拦截。在 `vite.config.ts` 补全 `server.proxy['/api']` 指向 `http://localhost:8000`，重启 Vite。 |
| **组件状态不响应** | 专家判定后，徽标颜色不更新 / 外部折叠区列表不同步 | `ExpertDecisionBadge` 改用 `computed` 响应 prop 变化；父组件 `TaskDetailPage` 更新数组改用 `splice(idx, 1, updated)` 强触发 Vue 响应式。 |
| **后端 Pydantic 422** | 提交「新规则候选」报 `Extra inputs are not permitted` | 前端表单是轻量级（title/desc），后端要求全字段（rule_id/severity）。在 `api/feedback.ts` 做适配层拦截并映射 `severity_hint`，填入默认占位。 |
| **`TaskSummaryCard` 消失** | 后端 `evidence_coverage` 为 `null` 时，整个卡片消失 | 去掉外壳的 `v-if="metrics"` 使其常驻；内部单格处理 `null` 显示为 `N/A`，严守口径不伪造 `0%`。 |
| **提交后“无反应”错觉** | 点击提交后，界面瞬间恢复原样（Network 其实 200 成功） | 父组件监听了 `@success` 瞬间折叠了面板，导致绿框提示被销毁。目前需看 Network 确认，后续可加 Toast 提示优化体验。 |

### 开发者调试利器（全链路 Console 日志）

为了快速定位前后端交互问题，S2-D/E 阶段在关键路径埋了 `import.meta.env.DEV` 的日志：
- **`http.ts`**：打印 `[API 请求开始]` / `[API 成功]`（含耗时、状态码）/ `[API 失败]`（含 error detail）。
- **`ExpertDecisionPanel.vue`**：打印 `[专家判定]` 按钮点击及接口回包。
- **`TaskDetailPage.vue`**：打印 `[页面状态更新]` 数组渲染。
- **排障建议**：联调时按 `F12` 打开 Console，过滤 `[API` 即可追踪完整的请求生命周期。


---

## S2-G：规则候选审核页（2026-10-29）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §4.4 + §14 S2-G 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/services/rule_evolution_service.py` | 补充 `get_all_candidates` / `get_candidate` / `update_candidate_status` 3 个 API 方法 |
| 2 | `backend/app/api/schemas.py` | 追加 4 个 Pydantic 模型；清理了末尾重复的 `SampleInfo` / `SummaryMetrics` 定义 |
| 3 | `backend/app/api/rule_candidates.py` | **新建**：3 个 API 端点（列表 / 详情 / 状态更新） |
| 4 | `backend/app/api/__init__.py` | 注册 `rule_candidates` 路由 |
| 5 | `frontend/src/types/rule_candidate.ts` | **新建**：类型定义 |
| 6 | `frontend/src/api/rule_candidate.ts` | **新建**：前端 API 封装 |
| 7 | `frontend/src/pages/RuleCandidatePage.vue` | **新建**：列表 + YAML 预览 + 批准/驳回交互 |
| 8 | `frontend/src/router/index.ts` | 新增 `/rule-candidates` 路由 |
| 9 | `frontend/src/App.vue` | 导航栏加「规则候选」入口 |

### 验收证据

| 项 | 结果 |
|---|---|
| `npm run type-check` | 0 错误 |
| Swagger `/api/v1/rule-candidates` | 3 个接口注册成功 |
| 前端页面加载列表 | 成功渲染 `proposed` 状态的候选 |
| YAML 预览 | 正确显示并高亮 |
| 批准/驳回 | 点击后调用后端成功，状态变更 |

### 关键约束（守住 S2 底线）

- **批准 ≠ 发布**：前端点击「批准」仅将状态改为 `accepted`，**不会自动写入 `data/rules/`**。正式规则的合入必须人工手动操作（Sprint 2 冻结）。
- **驳回必须有理由**：前端弹窗和后端 Schema 双重校验了 `rejected_reason` 不为空。

---

## S2-H（一）：Markdown 报告导出（2026-10-30）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §5.7 + §14 S2-H 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/services/report_service.py` | **新建**：`generate_markdown_report(task_id)`，负责查库并渲染 Markdown 报告 |
| 2 | `backend/app/api/review_tasks.py` | 新增 `GET /review/tasks/{task_id}/export` 接口，返回 `PlainTextResponse` 并带 `Content-Disposition` 触发下载 |
| 3 | `frontend/src/pages/TaskDetailPage.vue` | 头部增加「导出报告」按钮，通过 `window.location.href` 触发浏览器下载 |

### 验收证据

| 项 | 结果 |
|---|---|
| 浏览器测试 | ✅ 点击「导出报告」按钮，成功下载 `RUN-xxx_report.md` |
| 内容完整性 | ✅ Task ID、评审状态、缺陷总数、三态分布、缺陷明细、根因/建议、Evidence 完整透出 |
| 专家判定透出 | ✅ 已点确认的缺陷显示「专家确认」，未点的显示「待判定」 |

### 关键约束

1. **导出结构严格遵守证据链规范**：IR/RAG 等 evidence 必须完整带出（`source/section/reason` 或 `detail`）。
2. **只有 `done` 状态的任务允许导出**：`failed` 任务后端返回 400，前端也不显示按钮。
3. **文件编码**：设置 `media_type="text/markdown; charset=utf-8"`，避免 Windows 下中文乱码。

### 遗留项

| 项 | 影响 | 归属 |
|---|---|---|
| 旧任务头部显示 `评审状态: pending` | Sprint 1 遗留的历史数据无 `task_status` 列，使用默认值 | Sprint 3 做数据迁移或清洗 |

---

## S2-H（二）：批量评审（2026-10-30）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §14 S2-H 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/api/schemas.py` | 追加 `BatchReviewRequest` / `BatchReviewItem` / `BatchReviewResponse` |
| 2 | `backend/app/api/review_tasks.py` | 新增 `POST /review/tasks/batch` 接口，限制最大 5 个，异常隔离不中断 |
| 3 | `frontend/src/api/review.ts` | `createTask` 超时由 20s 调整为 30s（给 LLM 留足时间，避免批量时大面积 TIMEOUT） |
| 4 | `frontend/src/pages/CaseListPage.vue` | 增加卡片多选、底部悬浮操作栏、全屏批量进度遮罩 |

### 验收证据

| 项 | 结果 |
|---|---|
| 前端多选限制 | ✅ 超过 5 个弹窗拦截 |
| 批量执行 | ✅ 进度条实时显示（1/3 → 3/3） |
| 批量结果 | ✅ 3 个 Case 全部成功，无 TIMEOUT |
| 跳转 | ✅ 跑完自动跳转到任务列表 |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| 批量时部分 Case 报 `TIMEOUT` | 前端单次请求超时 20 秒，但实际 LLM 跑 20~25 秒，导致批量前半段被截断 | `review.ts` 的 `createTask` 超时调整为 30 秒；批量执行文案提示预计总耗时，避免用户焦虑 |
| 模板嵌套标签未闭合 | 替换代码时报 `Element is missing end tag` | 全量覆盖 `CaseListPage.vue`，修复标签嵌套 |

---

## S2-H（三）：RAG 知识库管理方案确认（2026-10-30）

> 状态：🟡 方案已冻结，待开发
> 依据：`Sprint2_详细执行计划_V1.0.md` §14 S2-H 行

### 1. 需求边界与范围（S2 冻结）

**本期交付（In Scope）**：
1. **知识文档列表查看**：Web 端展示 `knowledge_doc` 表的数据（ID、标题、来源、入库时间）。
2. **Markdown 上传与切片入库**：支持上传 `.md` 文件，后端自动走切片（Chunking）与向量化（Embedding），落库到 `knowledge_doc` 和 `knowledge_chunk`。

**明确不做（Out of Scope / 遗留 Sprint 3）**：
1. **知识文档启停（Enable/Disable）**：当前 `knowledge_doc` 表无 `enabled` 字段。为避免 S2 引入 DB 迁移，S2 不接受“禁用知识”需求，所有入库知识默认全局生效。启停功能留 Sprint 3。
2. **物理删除**：Web 端不提供一键删除按钮，防止误删导致 RAG 库瘫痪。高危清理交由 CLI。
3. **PDF / Word 解析**：仅支持标准 Markdown 格式（`.md`），不支持 PDF/OCR 解析。

### 2. 技术实现路径

**后端 API**：
- `GET /api/v1/knowledge-docs`：返回 `KnowledgeDoc` 列表。
- `POST /api/v1/knowledge-docs/upload`：接收 `multipart/form-data`，校验后缀后写入容器临时目录 `/tmp/upload_<uuid>.md`，调用 `app/rag/ingest.py::ingest_single_file()` 落库，最后删除临时文件并返回新文档 ID。

**前端页面 `KnowledgeBasePage.vue`**：
- 顶部「上传 Markdown」按钮。
- 下方表格展示已入库文档（标题、来源类型、入库时间）。
- 上传期间显示 Loading（因涉及真实 Embedding 调用，耗时约 3~5 秒）。

### 3. 踩坑预判
| 风险点 | 应对方案 |
|---|---|
| 临时文件残留 | `ingest_single_file` 必须包在 `try/finally` 中，确保 `os.remove()` 执行 |
| Embedding 失败 | 若 API Key 错误或欠费，接口返回 500，前端提示“向量化失败” |
| 大文件阻塞 | 前端限制 5MB 以内，避免大文件直接撑爆内存 |

---

## S2-H（三）：RAG 知识库管理（2026-10-30）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §14 S2-H 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/pyproject.toml` | 新增 `python-multipart` 依赖 |
| 2 | `backend/app/api/schemas.py` | 追加 `KnowledgeDocItem` / `KnowledgeDocListResponse` / `KnowledgeUploadResponse` |
| 3 | `backend/app/api/knowledge_docs.py` | **新建**：`GET /knowledge-docs` 列表、`POST /knowledge-docs/upload` 上传切片入库 |
| 4 | `backend/app/api/__init__.py` | 注册 `knowledge_docs` 路由 |
| 5 | `frontend/src/types/knowledge.ts` | **新建**：知识库类型定义 |
| 6 | `frontend/src/api/knowledge.ts` | **新建**：API 封装 |
| 7 | `frontend/src/pages/KnowledgeBasePage.vue` | **新建**：表格展示 + 上传交互 + 异常提示 |
| 8 | `frontend/src/router/index.ts` | 新增 `/knowledge-base` 路由 |
| 9 | `frontend/src/App.vue` | 导航栏加「知识库」入口 |
| 10 | `frontend/src/api/http.ts` | 修复 `FormData` 请求头，避免 JSON 解析报错；增加格式错误友好提示 |
| 11 | `scripts/bootstrap.ps1` | **优化**：启动成功后自动执行 `uv sync`，防依赖漂移 |
| 12 | `scripts/dev.ps1` | **优化**：启动/重启后自动执行 `uv sync`，防依赖漂移 |

### 验收证据

| 项 | 结果 |
|---|---|
| 上传合法 Markdown（`ams1117_startup.md`） | ✅ 成功提示，列表刷新 |
| 上传无元数据文件（`test.md`） | ✅ 拦截并友好提示「缺失 YAML frontmatter」 |
| 表格列表 | ✅ 正确展示 15 条 |
| `bootstrap.ps1` / `dev.ps1` | ✅ 均带 `uv sync` 自动同步且执行正常 |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| `python-multipart` 缺失 | 后端崩溃 `RuntimeError: Form data requires...` | 加入 `pyproject.toml`；修改 `dev.ps1` / `bootstrap.ps1` 自动跑 `uv lock` + `uv sync` |
| `FormData` 序列化失败 | 报错 `Unexpected token 'F'` | `http.ts` 动态判断 `options.body`，不打 JSON 日志，不强制设 `Content-Type` |
| 上传非规范文件 | 报错 `缺失 YAML frontmatter` | 后端保留原报错，前端转换为友好提示 |

## S2-H（四）指标看板

- 后端：新增 `GET /api/v1/dashboard/summary`，4 顶层对象契约（scope / metrics / rule_candidates / knowledge_gaps）
- 聚合口径：仅 task_status='done'；EVC 算术平均；EAR/AI_NER 全局池复用 benchmark_service；样本不足返 null
- Seed 检测：双路径（review_result_id 直接 + review_defect_id 间接），仅提示不参与计算
- 前端：DashboardPage.vue 5 块区域（指标卡 / AI 质量 / 专家判定 / 规则候选 / 知识缺口），N/A 语义，Seed 警告条
- 导航 + 路由：/dashboard
- Infra 修复：Dockerfile 加 UV_HTTP_TIMEOUT/RETRIES；final 拷含 dev 的 venv；构建期 pytest 校验
- Fix：bootstrap.ps1 删除 [2.5/5] uv sync（防 dev 依赖被卸）
- Fix：test_db_schema.py 触发器测试 t1 改从 DB 读（修既有时间采样缺陷）
- 已知问题：前端 4 格卡高度在空数据时略不齐（Sprint 3 视觉打磨）

## [S2-H（四）] 2026-10-30 - 指标看板与全局统计口径落地

### 交付内容
- **后端**：
  - 新增 `backend/app/services/dashboard_service.py`：全局指标聚合逻辑，内部使用 `get_db_session()` 独立管理事务。
  - 新增 `backend/app/api/dashboard.py`：提供 `GET /api/v1/dashboard/summary` 端点。
  - 追加 `backend/app/api/schemas.py`：定义 `DashboardSummaryResponse` 及相关子模型。
- **前端**：
  - 新增 `frontend/src/pages/DashboardPage.vue`：包含 5 块指标区域展示。
  - 新增 `frontend/src/api/dashboard.ts` 与 `types/dashboard.ts`。
  - 修改 `App.vue` 和 `router/index.ts`：挂载指标看板导航与路由。
- **测试**：
  - 新增 `backend/tests/test_dashboard.py`：覆盖 EVC 算术平均、Seed 检测、状态独立统计等 6 个用例。

### 核心契约与统计口径（冻结）
严格遵循 S2-H（四）编码前的四项最终确认：
1. **API 契约**：固定返回 `scope`、`metrics`、`rule_candidates`、`knowledge_gaps` 四个顶层对象。`sample_info` 复用现有 `SampleInfo` 模型。不引入 `metric_source` 字段，不设时间范围筛选。
2. **EVC 全局口径**：仅对 `task_status='done'` 且 `evidence_coverage IS NOT NULL` 的任务取**算术平均**。不按缺陷数加权，不重新计算 GT 缺陷级 EVC。未来如需加权，作为后续版本变更。
3. **has_seed_data 双路径检测**：仅检查当前 done 任务范围的反馈。既检查 `feedback_item.review_result_id` 直接关联，也检查 `review_defect_id → review_defect.review_result_id` 间接关联。该字段仅作警告提示，EAR/AI_NER 的 Seed 排除由底层计算函数自动处理，不在此层复制过滤逻辑。
4. **前端 5 块区域**：仅实现指标卡片、review_status 分布、expert_decision 分布、规则候选统计、知识缺口统计。保持纯数字表格，不引入图表、时间筛选、下钻或复杂筛选。
5. **独立统计原则**：规则候选（rule_candidates）和知识缺口（knowledge_gaps）完全独立统计，不套用 `task_status='done'` 过滤。
6. **空值语义**：所有未达到样本门槛的真实指标返回 `null`，前端必须显示 `N/A · 样本不足`，绝不渲染为 `0%`。

### 基础设施与修复项
- **Dockerfile 容灾**：新增 `UV_HTTP_TIMEOUT=120` / `UV_HTTP_RETRIES=5`，并将 final 阶段改为拷贝含 dev 的 `/opt/venv`，同时在构建期加入 `import pytest` 校验。
- **脚本修复**：移除 `scripts/bootstrap.ps1` 与 `scripts/dev.ps1` 中不带 `--extra dev` 的 `uv sync`，防止容器启动时卸载开发依赖（pytest 等）。
- **遗留测试修复**：修正 `tests/test_db_schema.py` 中关于 `updated_at` 触发器验证用例的采样逻辑，确保 `t1` 从 DB 真实重读，解决历史测试时间戳比对错位报错。

### 验收结果
- 后端全量非 integration 测试：`559 passed, 1 skipped, 24 deselected, 0 failed`。
- 前端 `npm run type-check`：0 错误。
- 前端 `npm run build`：构建成功（1988 modules transformed）。
- 浏览器人工验收：5 块区域正确渲染，Seed 提示逻辑正常，`/dashboard` 路由畅通。

## [S2-I 预研] 2026-10-30 - vLLM 兼容验证（占位）

### 现状与决策
- **决策**：Sprint 2 不实装 vLLM，仅保留占位。主用路径为智谱（`ZhipuProvider`），已通过全量测试验证。
- **占位基础**：`backend/app/llm/provider.py` 中 `VLLMProvider` 已存在，`get_provider("vllm")` 可返回实例，但 `_do_complete()` 抛 `NotImplementedError`（含实装指引）。
- **抽象层就绪**：`LLMProvider` 基类统一了计时（`latency_ms`）、重试（`LLM_MAX_RETRIES=1`，仅网络类）、异常分类（`_RetryableError` / `_NonRetryableError`），实装 vLLM 只需填 `_do_complete`。
- **工厂分发**：`get_provider()` 按 `settings.llm_provider` 支持 `mock` / `zhipu` / `openai` / `vllm` 四路。

### 契约测试
- 新增 `backend/tests/test_llm_vllm.py`，3 条用例：
  1. 工厂按 name 正确返回 `VLLMProvider` 实例
  2. 未实装时 `complete()` 抛 `NotImplementedError`，信息包含“尚未实装”
  3. 未传参时从 `settings` 兜底 `default_model`
- 不依赖任何外部服务，CI 稳定可跑。

### 实装路径（Sprint 3 计划）
1. 部署 vLLM 服务。**RTX 5090（Blackwell sm_120）需按 `torch_cuda_arch_list="12.0 12.1"` 单独编译**，官方预编译镜像默认不含该 kernel，直接 `docker run vllm/vllm-openai` 会报 `no kernel image`。
2. 将 `VLLMProvider._do_complete` 改为基于 `openai.OpenAI(base_url=..., api_key="EMPTY")` 的调用，复用 `_try_parse_json`、`finish_reason`、`request_id` 语义。
3. 设 `LLM_PROVIDER=vllm` + `LLM_BASE_URL` + `LLM_MODEL`，把 `test_llm_vllm.py` 的占位测试升级为 `@pytest.mark.integration` 真实调用。
4. CI 对 vLLM 集成测试默认 skip（需 GPU runner），仅在本地 GPU 机器手动跑。

### 环境备注
- 本机已具备 RTX 5090，硬件满足 vLLM 运行条件；Sprint 2 不投入时间做环境搭建，留待 Sprint 3。

## [S2-I] 2026-10-30 - 资产治理与 CI/CD 预研

### 交付内容
- **vLLM 占位契约测试**：新增 `test_llm_vllm.py`，3 条用例验证工厂分发、占位语义、settings 兜底。
- **Benchmark CSV 导出**：新增 `scripts/export_benchmark_csv.py`，支持 rule_only / full_review 模式导出，产物格式符合 S1-F 冻结契约。
- **data 资产治理**：新增 `data/README.md`，约定 Case / 知识库 / 规则文件的命名与新增流程。
- **诊断脚本隔离**：`.gitignore` 已忽略 `scripts/_diag_*.py`、`scripts/_cleanup_*.py`、`scripts/_tmp_*.py`，避免误提交。
- **CI 预研**：确认 CI 需跑全量非 integration 测试（562 passed）+ 前端 build；Benchmark CSV 可作为 artifact。

### 已知问题
- CI 平台尚未最终确认（待定）。
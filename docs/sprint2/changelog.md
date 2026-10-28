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





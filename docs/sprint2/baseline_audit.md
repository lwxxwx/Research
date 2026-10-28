# Sprint 2 基线核查报告

> 核查日期：2026-10-28
> 依据：Sprint2_详细执行计划_V1.0.md 附录 C（10 项）
> 结论：**通过（有差异但均已处理，不阻塞 S2-A 开工）**

## 1. Git HEAD 与测试基线

| 项 | 实际值 | 计划假设 | 差异 | 处理 |
|---|---|---|---|---|
| Git HEAD | ec47791 | 141e9f4 | 计划值已过时 | 以 ec47791 为 Sprint 2 基线 |
| 工作区状态 | clean（仅 2 个新文档 + docs/sprint2/） | clean | 无 | 正常 |
| 非 integration 测试 | **539 passed / 1 skipped / 24 deselected** | 538 / 1 / 24 | +1 passed | 以 539 为新基线（`test_llm_zhipu.py` 恢复后比 Sprint 1 记录多 1 个测试） |
| bench integration | **tp=16 / fn=0 / fp ∈ [0, 2]**（3 次实测 0/2/1） | tp=16 / fp=0 / fn=0 | fp 随机波动 | **基线改为 tp=16 / fn=0 / fp ≤ 2**；fp 波动来自 glm-4-flash 随机性，S1-B.5 已登记，Sprint 2 不修复 |

**命令**：
- `git log -1 --format="%H %s"`
- `git status --short`
- `docker compose ... exec backend pytest tests/ -m "not integration"`
- `docker compose ... exec backend pytest tests/test_benchmark_llm.py -m integration`

## 2. task_status 当前状态

| 项 | 实际值 | 计划假设 | 差异 | 处理 |
|---|---|---|---|---|
| models.py 是否有 task_status | **无** | 无 | 一致 | 无 |
| Alembic 迁移是否含 task_status | **无**（当前 head `c1_embedding_dim_1024`） | 无 | 一致 | 无 |
| review_result 表实际列 | 16 列，**无 `task_status`** | 无 | 一致 | 无 |
| 历史数据是否有 pending | **无**（无此列，不存在 pending 数据） | 无 | 一致 | 无 |

**结论**：S2-A 必须新增 `task_status` 列 + 索引。

**命令**：
- `Select-String "task_status" models.py` → 空
- `alembic current` → `c1_embedding_dim_1024 (head)`
- 列查询脚本 → 16 列，无 task_status

## 3. RuleCandidate / FeedbackItem / KnowledgeDoc/Chunk 实际字段

| 表 | 计划假设 | 实际字段 | 差异 | 处理 |
|---|---|---|---|---|
| `rule_candidates` | 需 `benchmark_result` / `benchmarked_at` / `rejected_reason` | 13 列，**缺这 3 列** | 缺 3 列 | S2-A 加 3 列 |
| `feedback_item` | `suggestion_diff_json` / `rule_candidate_ref` / `attached_refs` / `review_defect_id` 允许 null | 全部有 | 无 | 不改 |
| `knowledge_doc` | `title` / `source` / `source_type` / `content_md` / `metadata` / `version` | 全部有 | 无 | 不改 |
| `knowledge_chunk` | `chunk_text` / `embedding`（vector 1024）/ `metadata` | 全部有 | 无 | 不改 |

**结论**：S2-A 只改 `rule_candidates`（加 3 列），其余 3 表不动。

## 4. 规则加载路径与引擎重载机制

| 项 | 实际值 | 计划假设 | 差异 | 处理 |
|---|---|---|---|---|
| 规则加载路径 | 函数参数 `rules_dir: Path`，无模块级 `RULES_DIR` 常量；调用方传 `Path("/data/rules")`（注释确认） | `engine.py::RULES_DIR` 常量 | 无常量，是函数参数 | S2-F 发布时显式传 `Path("/data/rules")` |
| 引擎是否监听文件 | **否**（无 watchdog/inotify/reload） | 不监听 | 一致 | 无 |
| 重载触发方式 | **每次 `execute_all_rules` / `execute_defect_rules` 都重新 `load_rule_definitions(rules_dir)`**（第 212、280 行），进程内无缓存 | 进程重启 / 手动 | 无需触发，下次执行自动重读 | S2-F 发布后**显式跑一次 `load_rule_definitions(RULES_DIR)` 验证**，成功才标记 `published` |

**结论**：S2-F 的「引擎重载」= 显式调用 `load_rule_definitions(RULES_DIR)` 验证新 YAML 可加载，不是「通知监听器」。

## 5. 规则候选状态机与 CLI

| 项 | 实际值 | 计划假设 | 差异 | 处理 |
|---|---|---|---|---|
| 现有状态枚举 | **仅 `proposed`**（`generate_candidate_from_feedback` 固定写 `status="proposed"`；CLI `--status` 默认 `proposed`） | `proposed/designing/benchmarked/accepted/rejected` | **缺 4 个状态** | S2-F 扩展状态机 + S2-A 加 `rejected_reason` 列 |
| CLI `--list` | **可用**，输出 12 条 `proposed` 候选 | 可用 | 一致 | 复用 |
| 候选数据 | 12 条 `proposed`，`case_id` 全 `null`，`from_feedback_id` 有值 | `case_id` 非空 | 历史数据 `case_id` 未填 | S2-F 的来源 Case 追溯从 `from_feedback_id → feedback_item → review_result → schematic_case` 反查，容忍 `case_id=null` |
| 现有 CLI 参数 | `--list` / `--status` / `--export-yaml` / `--export-knowledge-gap` / `--bench-candidate` | 同上 | 一致 | 复用 |

**结论**：S2-F 要扩展状态机（4 个新状态）+ 来源 Case 反查（容忍 `case_id=null`）；CLI 骨架可复用。

## 6. metrics_definition.md 当前版本

| 项 | 实际值 | 计划假设 | 差异 | 处理 |
|---|---|---|---|---|
| 版本 | **v3.2**（2026-10-28，S2-A 补冻结 AI_NER 门槛） | v3.1 | 已升级 | 记录 |
| EAR 公式 | **真实公式见 Sprint 1 §9.6**；metrics 文档只写模拟公式 §6.3 | §9.6 | 真实公式在 Sprint 1 计划 | S2-C 开工时引用 §9.6 |
| EAR 门槛 | 20 | 20 | 一致 | 无 |
| AI_NER 公式 | **真实公式见 §6.7 + Sprint 1 §23.2**；模拟公式在 §6.2 | §23.2 | 真实公式已补 §6.7 | S2-C 开工时引用 §6.7 |
| AI_NER 门槛 | **已冻结 ≥ 10 条**（§6.7，v3.2 新增） | 未冻结 | 已补 | 记录 |
| Seed 过滤字段 | **`comment` 前缀 `[SEED]`**（Sprint 1 E.2.4 已用）；`feedback_item` 无 `simulated` 列 | `simulated=true` | 方案变更：不加列，用 `comment` 前缀 | S2-C 聚合时 `WHERE (comment IS NULL OR comment NOT LIKE '[SEED]%')` |

## 7. 已有 API/服务盘点

**已有 7 个路由**（`backend/app/api/`）：

| # | 路径 | 方法 | 文件 | 状态 |
|---|---|---|---|---|
| 1 | `/healthz` | GET | `healthz.py` | ✅ 已有，不改 |
| 2 | `/review/tasks` | POST | `review_tasks.py` | ✅ 已有，不改 |
| 3 | `/review/tasks` | GET | `review_tasks.py` | ⚠️ 已有，**S2-C 加 `?status=`** |
| 4 | `/review/tasks/{task_id}` | GET | `review_tasks.py` | ⚠️ 已有，**S2-C 加 `summary_metrics`** |
| 5 | `/review/tasks/{task_id}/defects` | GET | `review_tasks.py` | ✅ 已有，不改 |
| 6 | `/feedbacks` | POST | `feedbacks.py` | ✅ 已有，不改 |
| 7 | `/cases` | GET | `cases.py` | ⚠️ 已有，**S2-G 扩展**（加 `{case_id}` 详情、benchmark） |

**需新建的路由**：

| 计划路由 | 阶段 | 状态 |
|---|---|---|
| `PATCH /review/tasks/{id}/defects/{id}` | S2-B | **缺**，新建 |
| `GET /rule-candidates` + 5 个子端点 | S2-F | **缺**，新建 `rule_candidates.py` |
| `GET /knowledge-gaps` + `PATCH` | S2-G | **缺**，新建 `knowledge_gaps.py` |
| `GET /cases/{case_id}` + benchmark + benchmarks | S2-G | **缺**，扩展 `cases.py` |
| `GET /review/tasks/{id}/export` | S2-H | **缺**，新建 |
| `POST /cases/batch-review` | S2-H | **缺**，新建 |
| `GET /benchmark/batch` + `export` | S2-H | **缺**，新建 |
| `GET /dashboard/summary` | S2-H | **缺**，新建 |
| 知识库 Markdown 导入/启停 | S2-H | **缺**，新建 |

**结论**：S2-B 新建 1 个端点；S2-C 改 2 个端点；S2-F 新建 1 个文件 6 个端点；S2-G 新建 1 个文件 2 个端点 + 扩展 1 个文件；S2-H 新建 5 类端点。

## 8. 前后端类型一致性

**后端 11 个 Pydantic class，前端 23 个 interface/type**，关键对照：

| 语义 | 后端 | 前端 | 状态 | 处理 |
|---|---|---|---|---|
| `TaskStatus` | `TaskDetailResponse.status: str` | `"done" \| "failed"` | ⚠️ **前端缺 `pending`** | S2-C 前端加 `pending` |
| `ExpertDecision` | `DefectsResponse` 为 `items: list[dict]` 透传；`service.get_task_defects` 已含 `expert_decision` / `_updated_at` / `_by`（Sprint 1 G1 已补） | `defect.ts::ExpertDecision` | ✅ 一致（透传） | S2-B 前跑一次 `GET /review/tasks/{id}/defects` 确认 3 字段实际出现 |
| `ReviewStatus` | `TaskDetailResponse.review_status: str` | `defect.ts::ReviewStatus` | ✅ 一致 | 无 |
| `SummaryMetrics` | **无** | **无** | ❌ **两端都缺** | S2-C 两端加，字段名严格对齐 `metrics_definition.md` v3.2 `sample_info` |
| `SuggestionDiff` | `FeedbackCreateAPI.suggestion_diff` | `feedback.ts::SuggestionDiff` | ✅ 一致 | 无 |
| `RuleCandidate` | `FeedbackCreateAPI.rule_candidate` | `feedback.ts::RuleCandidate` | ✅ 一致 | 无 |
| `Evidence` | `DefectsResponse` 内嵌 | `evidence.ts::Evidence` | ✅ 一致 | 无 |
| `DefectFromAPI` vs `DefectFromReport` | `DefectsResponse` 一个 | 前端**两个** | ✅ 已分离（Sprint 1 H） | 无 |

**结论**：
- S2-C 两端加 `SummaryMetrics`（含 `sample_info`）；
- S2-C 前端 `TaskStatus` 加 `pending`；
- S2-B 前跑一次 `GET /review/tasks/{id}/defects` 确认 `expert_decision` 3 字段实际出现（`DefectsResponse` 透传，`schemas.py` 不定义字段，需看 service 层实际返回值）。

## 9. AI_NER 样本门槛

| 项 | 实际值 | 处理 |
|---|---|---|
| `metrics_definition.md` 当前版本 | **v3.2**（2026-10-28，S2-A 补冻结 AI_NER 门槛；原 v3.1 为 2026-10-24 S1-B 后版本） | 记录 |
| AI_NER 样本门槛是否已冻结 | **是，已冻结 ≥ 10 条 `ai_discovered`**（§6.7） | **已完成，无需再补** |
| 补冻结动作 | 已完成：新增 §6.7「AI_NER 样本门槛」；文档头改 v3.2；§10 版本记录追加 v3.2 行 | 见 changelog |

## 10. KnowledgeGap 业务场景核查

| 项 | 结论 | 处理 |
|---|---|---|
| 历史数据核查 | 0 条「同一 defect ≥2 条 knowledge_gap」 | 数据侧无一对多痕迹，但不构成「必须一对一」 |
| 业务场景判断 | **一对多**：B 类 Case 证据链天然多份资料（Datasheet + AppNote + 降额准则），专家一条反馈指出多个缺失点是常态 | **S2-A 按一对多建** |
| 建表形态 | `KnowledgeGap` **不加** `source_feedback_id` 唯一约束；建 `gap_feedback_link(gap_id, feedback_id)` 中间表；`source_feedback_id` 保留为「主来源」冗余字段，不唯一 | 迁移脚本按此写 |

**核查方法**：
- 查 `data/cases/*/expected_review.json` 是否有「一条 knowledge_gap 反馈派生多个缺口」的痕迹。
- 查历史 `feedback_item` 中 `feedback_type='knowledge_gap'` 的记录，是否有重复 `comment` / 同一 defect 多条。
- 若无法从数据判断，**问产品/业务负责人**，不要自己拍。

## 结论

- **与计划假设一致项**：5 项（第 2、4、6、9、10）
- **有差异需修订计划项**：5 项（第 1、3、5、7、8）
- **阻塞后续开发项**：0 项（所有差异都有明确处理动作，不阻塞 S2-A 开工）

### 差异清单

| # | 项 | 差异 | 是否阻塞 |
|---|---|---|---|
| 1 | Git HEAD / 测试基线 | HEAD 从 141e9f4 → ec47791；passed 538 → 539；bench fp=0 → fp ≤ 2 | 否，已更新基线 |
| 3 | `rule_candidates` 字段 | 缺 `benchmark_result` / `benchmarked_at` / `rejected_reason` | 否，S2-A 加 3 列 |
| 5 | 候选状态机 | 仅 `proposed`，缺 4 个状态；`case_id` 全 null | 否，S2-F 扩展 + 反查 |
| 7 | API 盘点 | 7 个已有，20+ 个要新建/扩展 | 否，按 S2-B/C/F/G/H 分阶段新建 |
| 8 | 类型一致性 | `TaskStatus` 前端缺 `pending`；`SummaryMetrics` 两端缺 | 否，S2-C 两端加 |

## 处理动作

| # | 差异 | 处理动作 | 阶段 | 负责人 | 截止 |
|---|---|---|---|---|---|
| 1 | HEAD / 测试基线 | 以 `ec47791` 为基线；bench 基线 `tp=16/fn=0/fp ≤ 2`；已同步计划 §2.2 / §8.2 / §14 / §14.4 / 附录 D | S2-A | Tech Lead | 已完成 |
| 2 | `task_status` 缺失 | S2-A 迁移加 `task_status VARCHAR(32) NOT NULL DEFAULT 'pending'` + 索引 | S2-A | Tech Lead | S2-A Day 2 |
| 3 | `rule_candidates` 缺 3 列 | S2-A 迁移加 `benchmark_result JSONB` / `benchmarked_at TIMESTAMPTZ` / `rejected_reason TEXT` | S2-A | Tech Lead | S2-A Day 2 |
| 5 | 候选状态机仅 `proposed` | S2-F 扩展 `designing/benchmarked/accepted/rejected`；来源 Case 从 `from_feedback_id` 反查，容忍 `case_id=null` | S2-F | Tech Lead B | S2-F |
| 6 | Seed 过滤字段待确认 | 代码中 `feedback_item` 无 `simulated` 列，**采用 `comment` 前缀 `[SEED]`**（与 Sprint 1 一致），不加列；S2-C 指标聚合按前缀排除 | S2-C | Tech Lead A | S2-C |
| 7 | 20+ 路由缺失 | 按 S2-B（1）/ S2-C（2 改）/ S2-F（6）/ S2-G（2+1 扩展）/ S2-H（5 类）分阶段新建 | S2-B~H | Tech Lead A/B | 各阶段 |
| 8 | `TaskStatus` 缺 `pending` | S2-C 前端 `task.ts::TaskStatus` 加 `"pending"`；后端 `TaskDetailResponse.status` 保持 `str` | S2-C | Tech Lead B | S2-C |
| 8 | `SummaryMetrics` 两端缺 | S2-C 后端 `TaskDetailResponse` 加 `summary_metrics`；前端 `task.ts::TaskDetail` 加 `SummaryMetrics` 类型；字段名严格对齐 `metrics_definition.md` v3.2 `sample_info` | S2-C | Tech Lead A/B | S2-C |
| 8 | `DefectsResponse` 3 字段 | `schemas.py::DefectsResponse` 为 `items: list[dict]` 透传，不定义字段；`service.get_task_defects` 已含 3 字段（Sprint 1 G1）；S2-B 前跑一次 API 确认实际出现 | S2-B | Tech Lead A | S2-B 初 |
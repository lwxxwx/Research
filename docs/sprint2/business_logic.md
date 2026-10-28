# Sprint 2 业务逻辑基线

> **文档定位**：Sprint 2 业务逻辑的起点与唯一事实来源。S2-B ~ S2-H 所有阶段开工前必须对齐本文档。
> **冻结日期**：2026-10-28
> **依据**：`Sprint2_详细执行计划_V1.0.md` + Sprint 1 已冻结设计（`sprint1_changelog.md`）+ `metrics_definition.md` v3.2
> **状态**：正式冻结，任何业务逻辑变更必须先改本文档，再改代码。

---

## 0. 一句话总结

> **真实评审 = 规则匹配（确定性）+ RAG 检索（证据）+ LLM 分析（根因/建议/AI 候选发现），三者协作，不是「只规则匹配」。**
> **规则是人工设计的，不是从 GT 生成；GT 是独立冻结的评分基准，用于验证规则和 AI 的发现质量，不产生规则。**
> **AI 输出预测，专家裁决 AI 预测；专家反馈 → 规则候选 → 人工审核 → 正式规则，这才是新规则的来源。**

---

## 1. 业务闭环主链路（Sprint 2 V1.0 冻结版）

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. 真实评审（Full Review）                                   │
│    IR → 规则匹配 → RAG 检索 → LLM 分析 → ReviewReport         │
│    产出：review_defect（origin=rule / ai_discovered）         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 专家裁决                                                  │
│    专家对每条 review_defect：确认 / 驳回 / 待核验             │
│    落库：review_defect.expert_decision（3 字段 + 审计）       │
│    同时可提交过程反馈：feedback_item（6 类）                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 反馈转规则候选                                            │
│    false_negative / new_rule_candidate → rule_candidates     │
│    knowledge_gap → knowledge_gap（Sprint 2 新建）            │
│    suggestion_update → 只存 suggestion_diff                  │
│    correct_defect → 只存反馈历史                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 人工审核候选                                              │
│    编辑 YAML → 跑候选 Benchmark → 批准 / 驳回                 │
│    approve → publishing → published（写入 data/rules/）      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 下次评审用新规则                                          │
│    引擎每次执行都重新 load_rule_definitions(rules_dir)       │
│    → 新规则自动生效（无需重启）                               │
└─────────────────────────────────────────────────────────────┘
```

> **GT 在这条链路里的位置**：只在第 4 步「跑候选 Benchmark」时用——候选规则跑 10 个 Case，和 GT 对比，看 Precision / Recall / FP / FN 是否退化。**GT 不参与规则生成，只参与规则验证。**

---

## 2. 三个角色 + GT 的准确边界

| 角色 | 输入 | 输出 | 是否冻结 |
|---|---|---|---|
| **规则引擎** | IR + 10 条 YAML | `rule_results`（确定性命中） | 规则 YAML 人工设计，**不自动生成** |
| **LLM** | IR + 规则命中 + RAG 证据 | `review_defect`（根因/建议/AI 候选发现） | 输出是「**预测**」，不是「金标」 |
| **专家** | `review_defect` | `expert_decision`（3 字段）+ `feedback_item`（6 类） | 专家裁决是「**最终判定**」 |
| **GT（金标）** | 专家在 Sprint 0/1 标定 | `expected_review.json` | **冻结，Sprint 2 不改** |

### 2.1 关键边界（不可违反）

1. **规则不来自 GT**：规则是人工设计的确定性检查（`data/rules/*.yaml`），GT 是评分基准，两者独立。
2. **AI 输出不修正 GT**：AI 的 `review_defect` 是「预测」，落库到 `review_defect` 表；GT 冻结在 `expected_review.json`，Sprint 2 不改。
3. **专家裁决不修正 GT**：专家对 AI 输出的 `expert_decision` 是「对预测的裁决」，不是「改 GT」。
4. **新规则来自专家反馈，不来自 GT**：反馈 → 规则候选 → 人工审核 → 正式规则，GT 只在「候选 Benchmark」时验证规则质量。
5. **规则引擎每次执行重读磁盘**：`execute_all_rules` / `execute_defect_rules` 每次调用都重新 `load_rule_definitions(rules_dir)`，进程内无缓存（`engine.py` 第 212、280 行）。**新规则自动生效，无需重启。**

---

## 3. 两种运行模式的边界

| 模式 | 常量 | 做什么 | 依赖 LLM | 用途 |
|---|---|---|---|---|
| **Rule Only** | `MODE_RULE_ONLY = "rule_only"` | 只跑规则引擎 | ❌ | Benchmark 基线，不是真实评审 |
| **Full Review** | `MODE_FULL_REVIEW = "full_review"` | 6 节点 Graph（规则 + RAG + LLM） | ✅ | **真实评审** |

**关键区分**：
- **Rule Only 不是真实评审**，是 Benchmark 的确定性基线，用于「规则本身准不准」。
- **Full Review 是真实评审**，是「规则 + RAG + LLM」三者协作，用于「AI 能不能帮专家发现问题」。

---

## 4. 6 节点 Graph（真实评审的执行链）

```text
load_ir → rule_check → rag_retrieve → expert_analysis → report_generate → feedback_save
```

| 节点 | 职责 | 失败处理 |
|---|---|---|
| `load_ir` | 从 `data/cases/<case_id>/schematic_ir.json` 加载 + 严格校验 | 写 `errors`，后续节点检测前置缺失时跳过 |
| `rule_check` | `execute_defect_rules` 执行 10 条规则 | 写 `errors` |
| `rag_retrieve` | `Retriever` 按 `rule_hits` + `rule_ids` 检索 | 写 `errors`，`rag_context=[]`，LLM 走「无 RAG 依据」降级 |
| `expert_analysis` | `render_prompt` + `get_provider().complete()` | 写 `errors`，`llm_raw_output` 缺失 |
| `report_generate` | `parse_report` 三级校验 + 三态判定 | 写 `errors` + `task_status=failed` |
| `feedback_save` | 落库 `review_result` + `review_defect` | 写 `errors` + `task_status=failed` |

**注意**：`feedback_save` 节点**只落库报告，不实现反馈提交**——反馈提交由 API 层调 `feedback_service`，两者解耦（Sprint 1 D.3.3 冻结）。

---

## 5. 状态分离（四者不可互相替代）

| 状态 | 枚举 | 语义 | 谁能改 |
|---|---|---|---|
| `task_status` | `pending / done / failed` | 任务执行结果 | Graph 落库时写 |
| `review_status` | `AI_CONFIRMED / NEED_EXPERT_REVIEW / LOW_CONFIDENCE` | AI 输出质量门槛 | `output_parser._classify_defect` |
| `expert_decision` | `pending / confirmed / rejected / needs_verification` | 专家最终判定 | 专家 PATCH 接口 |
| `feedback_item` | 6 类（`correct_defect` / `false_positive` / `false_negative` / `suggestion_update` / `new_rule_candidate` / `knowledge_gap`） | 专家反馈过程记录 | 专家提交 |

**硬约束**：
- 专家判定接口**不得**改写 `review_status`（AI 质量状态）。
- 专家判定接口**不得**隐式创建 `feedback_item`（过程记录）。
- `review_status=AI_CONFIRMED` **绝不**自动把 `expert_decision` 设为 `confirmed`。

---

## 6. 6 类反馈的闭环去向

| `feedback_type` | 触发规则候选？ | 去向 |
|---|---|---|
| `correct_defect` | ❌ | 只存反馈历史（正样本池） |
| `false_positive` | ❌ | 只存反馈历史（误报样本池） |
| `false_negative` | ✅ | 生成 `rule_candidates` 草稿 |
| `new_rule_candidate` | ✅ | 生成 `rule_candidates` 草稿 + 填 `title` / `description` |
| `suggestion_update` | ❌ | 只存 `suggestion_diff_json` |
| `knowledge_gap` | ❌ | 生成 `knowledge_gap`（Sprint 2 新建，一对多中间表） |

**关键**：**只有 `false_negative` 和 `new_rule_candidate` 触发规则候选**，其余 4 类只存反馈历史。

---

## 7. 规则发布的五阶段状态机（Sprint 2 S2-F）

```text
approved → publishing → published | publish_failed
```

| 阶段 | 做什么 | 失败处理 |
|---|---|---|
| `approved` | 人工批准候选（候选获批 ≠ 已发布） | — |
| `publishing` | 临时文件校验 → 备份 → 原子替换 → DB 事务同步 → 引擎重载 | 中途失败 → 补偿回滚 |
| `published` | 文件、DB、重载三者全成功 | — |
| `publish_failed` | 发布失败，保留证据 | 人工决定接受 / 回滚 |

**关键语义**：
- **文件原子替换后、DB 提交前，引擎不得重载**（引擎不监听文件，内存仍是旧规则）。这样「文件成功、DB 失败」时，引擎状态与 DB 一致，恢复备份文件即可，**不需要「引擎回滚」**。
- **发布后 Benchmark 回归失败不自动回滚**，保留变更和证据，要求人工选择接受当前版本或执行受控回滚。
- **Sprint 2 不自动 `git commit`**，由人工审查提交，CI 校验文件与 DB 一致。

---

## 8. 指标口径（严格对齐 `metrics_definition.md` v3.2）

| 指标 | 公式 | 样本门槛 | 无样本时 |
|---|---|---|---|
| **EVC** | GT 侧证据完整率 | — | `None` → `NaN` |
| **EAR** | `N_adopted / N_defect_with_feedback` | ≥ 20 条采纳相关反馈 | `None` → `NaN` |
| **AI_NER** | `N_ai_effective / (N_rule_effective + N_ai_effective)` | ≥ 10 条 `ai_discovered` 缺陷 | `None` → `NaN` |

**关键边界**：
- **Seed/simulated 数据不计入真实指标**。`feedback_item` 无 `simulated` 列，采用 `comment` 前缀 `[SEED]`（Sprint 1 E.2.4 约定），聚合时 `WHERE (comment IS NULL OR comment NOT LIKE '[SEED]%')`。
- **`AI_NER_SIMULATED` / `EAR_SIMULATED` 是模拟指标**，不是真实业务指标，只用于「无真实反馈阶段的过渡观察」。
- **真实业务指标无数据时标记 `N/A`，不能标记达标**。

---

## 9. 常见误解纠正

| 误解 | 事实 |
|---|---|
| ❌ 真实评审「只进行规则匹配」 | ✅ 真实评审是「规则 + RAG + LLM」6 节点 Graph；只跑规则的是 Rule Only 基线 |
| ❌ 规则来源是 GT | ✅ 规则是人工设计的确定性检查；GT 是独立冻结的评分基准 |
| ❌ AI 和专家评审「修正 GT」 | ✅ AI 输出预测，专家裁决预测；GT 冻结，Sprint 2 不改 |
| ❌ 新规则由 GT 产生 | ✅ 新规则来自专家反馈 → 规则候选 → 人工审核 → 正式规则；GT 只验证规则质量 |
| ❌ `review_status=AI_CONFIRMED` 等于专家确认 | ✅ 前者是 AI 输出质量门槛，后者是 `expert_decision` 字段，两者独立 |
| ❌ 规则发布 = 批准 | ✅ `approved → publishing → published`，批准不代表已发布 |
| ❌ 规则发布后需要重启服务 | ✅ 引擎每次执行都重读磁盘，新规则自动生效 |
| ❌ `EAR_SIMULATED=1.0` 表示专家采纳率 100% | ✅ 它只是 gold-label match precision，不是真实专家采纳率 |

---

## 10. 与 Sprint 1 冻结项的关系

| Sprint 1 冻结项 | Sprint 2 是否触碰 | 说明 |
|---|---|---|
| 10 条规则（Power 3 + Clock 2 + IO 3 + Control 2） | 否 | Sprint 2 只加规则候选审核，不改现有规则 |
| 10 个 Golden Case（6A + 4B） | 否 | GT 冻结，Sprint 2 不改 |
| `review_status` 三态 | 否 | 只加专家判定写接口 |
| `expert_decision` 4 枚举 | 否 | 只加 PATCH 接口 |
| 6 类 `feedback_type` | 否 | 只增强前端表单 |
| `EAR = N_adopted / N_defect_with_feedback` | 否 | 只落地计算 |
| bench 基线 `tp=16 / fn=0 / fp ≤ 2` | 否 | Sprint 2 结束时必须保持 |
| Alembic 唯一 Schema 来源 | 否 | Sprint 2 迁移继续走 Alembic |
| 上传配额 / 频率限制不做 | 否 | Sprint 2 仍不做 |

---

## 11. 变更管理

- **本文档是业务逻辑的唯一事实来源**。任何业务逻辑变更（角色边界、闭环链路、状态语义、指标口径）必须先改本文档，再改代码。
- **代码与文档冲突时，以本文档为准**，并同步修正代码。
- **本文档冻结后，S2-B ~ S2-H 所有阶段开工前必须对齐本文档**；发现不一致，先修订本文档，再继续实现。

---

## 附录 A：关键文件索引

| 模块 | 文件 |
|---|---|
| 真实评审 Graph | `backend/app/workflows/review_graph.py` |
| 规则引擎 | `backend/app/rules/engine.py` |
| LLM Provider | `backend/app/llm/provider.py` |
| 输出解析 | `backend/app/llm/output_parser.py` |
| 反馈服务 | `backend/app/services/feedback_service.py` |
| 规则演进 | `backend/app/services/rule_evolution_service.py` |
| Benchmark | `backend/app/services/benchmark_service.py` |
| 指标定义 | `docs/benchmark/metrics_definition.md`（v3.2） |
| 业务模型 | `backend/app/persistence/models.py` |

## 附录 B：版本记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-10-28 | 首次冻结：明确真实评审 = 规则 + RAG + LLM；规则人工设计不来自 GT；AI 预测 / 专家裁决 / GT 冻结三角色边界；6 类反馈闭环去向；规则发布五阶段状态机；指标口径对齐 v3.2；纠正 8 条常见误解。 |
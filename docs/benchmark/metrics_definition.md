# Benchmark Metrics Definition v3.2

> Business Freeze · Sprint0 业务冻结文档，Sprint1 阶段 F 升级为 v3.0，Sprint2 S2-A 升级为 v3.2
> 适用：
> - Sprint0：**Rule-Only Baseline 评测（A 类 Case，仅确定性规则引擎）**
> - Sprint1 阶段 F+：**Rule Only + Full Review 双模式评测**（Rule + RAG + LLM AI 评审，启用 EVC / EVC_PRED / 模拟 AI_NER / 模拟 EAR / 完整三态）
>
> Ground Truth 来源：每个 case 目录下 `expected_review.json`
> Prediction 来源：
> - **Rule Only 模式**：规则引擎输出 `List[RuleResult]`（执行 `execute_defect_rules()` 得到）
> - **Full Review 模式**：`review_defect` 表数据（`get_task_defects()` 返回，由 6 节点 Graph 落库）
>
> **本文档是指标口径的唯一事实来源**。代码（`benchmark_service.py`）与文档不得各自定义指标；若发现不一致，以本文档为准，并同步修正代码。

---

## 0. 运行模式（S1-F 新增）

Benchmark 支持两种运行模式，由 `run_all_benchmark(case_ids, mode)` 的 `mode` 参数控制。

| 模式 | 常量 | Pred 来源 | 依赖 LLM | 典型耗时 |
| :--- | :--- | :--- | :--- | :--- |
| Rule Only | `MODE_RULE_ONLY = "rule_only"` | `execute_defect_rules()` 的 `RuleResult` 序列化 | ❌ 不依赖 | 10 Case 约 3s |
| Full Review | `MODE_FULL_REVIEW = "full_review"` | `get_task_defects()` 的 `ReviewDefect` 数据 | ✅ 走 6 节点 Graph（含 RAG + LLM） | 10 Case 约 4 分钟（真实智谱） |

**关键区别**：

- **Rule Only** 只跑确定性规则引擎，不触网，是 Sprint 0 的基线。
- **Full Review** 走完整 6 节点 Graph（`load_ir` → `rule_check` → `rag_retrieve` → `expert_analysis` → `report_generate` → `feedback_save`），LLM 可解释规则命中、提出 AI 辅助发现候选。
- 两种模式的 Pred 结构不同（`RuleResult` vs `ReviewDefect`），匹配逻辑也不同（`match_defect` vs `match_defect_full_review`），详见 §2.1。

**`BenchmarkResult.mode` 字段**：记录该结果属于哪种模式，CSV 的 `RUN_MODE` 列输出此值。

---

## 1. 字段映射（代码落地事实）

> ⚠️ GT 与 Pred 的字段不一致，本映射为匹配逻辑依据。
> Sprint 1 阶段 F 起，Pred 有两种来源（Rule Only 的 `RuleResult` / Full Review 的 `ReviewDefect`），字段映射需分别说明。

### 1.1 GT（`expected_review.json`）字段

| 语义 | 字段 | 说明 |
| :--- | :--- | :--- |
| 缺陷 ID | `defect_id` | 全局唯一 |
| 分类 | `category` | 枚举见 `rule_format.md` §1 |
| 风险等级 | `risk` | low / medium / high / critical |
| 主元件 | `component` | 单值 |
| 主网络 | `net` | 单值或 `null` |
| 发现来源 | `origin` | `rule` / `ai_discovered` / `hybrid` |
| 规则编号 | ❌ **defect 顶层无**；埋在 `evidence[type=rule_hit].rule_id` | 匹配时用 `_extract_gt_rule_id()` 提取 |
| 证据 | `evidence[]` | 每项含 `type` / `source` / `section` / `reason`（`rag_ref`）或 `detail`（`ir_ref`） |

### 1.2 Pred · Rule Only 模式（`RuleResult`）

来源：`app/rules/engine.py::execute_defect_rules()`，`RuleResult.model_dump()`。

| 语义 | 字段 | 说明 |
| :--- | :--- | :--- |
| 缺陷 ID | ❌ **无** | Rule Only 的 Pred 无 `defect_id`，匹配时不依赖 |
| 分类 | `category` | 规则 YAML 的 `category` |
| 风险等级 | `severity` | 规则 YAML 的 `severity`（与 GT 的 `risk` 同枚举，语义等价） |
| 主元件 | `component` | `RuleHitItem.component_refs[0]`，可能为 `None` |
| 相关元件集合 | `component_refs` | `RuleHitItem.component_refs` 全量 |
| 主网络 | `net` | `RuleHitItem.net_refs[0]`，可能为 `None` |
| 相关网络集合 | `net_refs` | `RuleHitItem.net_refs` 全量 |
| 规则编号 | `rule_id` | 规则 YAML 的 `rule_id` |
| 发现来源 | `origin` | 固定 `"rule"` |
| 证据 | `evidence_ir_refs` | 字符串列表，非三要素结构 |

### 1.3 Pred · Full Review 模式（`ReviewDefect`）

来源：`app/persistence/models.py::ReviewDefect`，`get_task_defects(task_id)` 返回。

| 语义 | 字段 | 说明 |
| :--- | :--- | :--- |
| 缺陷 ID | `defect_id` | 落库时加 `{task_id}-` 前缀（S1-E 修正）。S1-B 后：`origin=rule` 的缺陷由 RuleResult 构造，格式为 `{task_id}-DEF-RULE-{rule_id}`；`origin=ai_discovered` 的缺陷格式为 `{task_id}-{LLM 原始 defect_id}` |
| 分类 | `category` | `origin=rule` 时由 `_resolve_rule_fields` 用 `RuleResult.category` 覆盖（S1-F）；`ai_discovered` 时保留 LLM 输出 |
| 风险等级 | `risk` | `origin=rule` 时由 `_resolve_rule_fields` 用 `RuleResult.severity` 覆盖（S1-F）；`ai_discovered` 时保留 LLM 输出 |
| 主元件 | `component` | LLM 输出；`"-"` / `""` 落库时归一化为 `None`（S1-F） |
| 主网络 | `net` | LLM 输出；`"-"` / `""` 落库时归一化为 `None`（S1-F） |
| 规则编号 | `rule_id` | `origin=rule` 时有值；`ai_discovered` 时为 `None` |
| 发现来源 | `origin` | `rule` / `ai_discovered` / `hybrid` |
| 质量状态 | `review_status` | `AI_CONFIRMED` / `NEED_EXPERT_REVIEW` / `LOW_CONFIDENCE` |
| 证据 | `evidence` | `list[dict]`，每项含 `type` + 三要素（`rag_ref`）或 `detail`（`ir_ref`） |

### 1.4 匹配用的字段对照

| 语义 | GT | Rule Only Pred | Full Review Pred |
| :--- | :--- | :--- | :--- |
| 风险等级 | `risk` | `severity` | `risk`（`severity` 为兜底） |
| 主元件 | `component` | `component` + `component_refs` | `component` |
| 主网络 | `net` | `net` + `net_refs` | `net` |
| 规则编号 | `evidence[rule_hit].rule_id`（可空） | `rule_id` | `rule_id`（可空） |
| 分类 | `category` | `category` | `category` |

**关键结论**：

- **Rule Only 匹配**：`match_defect()` 要求 `component` 交集 **且** `net` 交集 **且** `severity == risk`；`rule_id` 弱校验。
- **Full Review 匹配**：`match_defect_full_review()` 分严格路径与降级路径，详见 §2.1。
- **不以 `rule_id` 为主键**：GT 顶层取不到 `rule_id`，匹配以「元件 / 网络 / 风险」为主条件。
- **Full Review 的 `risk` 以规则引擎为准**：`origin=rule` 的缺陷，`risk` 由 `RuleResult.severity` 覆盖，不被 LLM 改写（S1-F 修正，详见 §1.3）。

---

## 2. 基础概念

- **GT**：`expected_review.json["defects"]`，专家标定的真实缺陷集合
- **Pred**：按运行模式不同：
  - **Rule Only**：`execute_defect_rules()` 返回的 `RuleResult` 序列化 dict 列表（只含 `severity ∈ {medium, high, critical}`，low 级提示不参与指标计算）
  - **Full Review**：`get_task_defects(task_id)` 返回的 `ReviewDefect` 数据（由 6 节点 Graph 落库）
- **TP**：预测缺陷与 GT 缺陷匹配命中真实问题
- **FP**：预测存在，GT 不存在（误报）
- **FN**：GT 存在，预测未检出（漏检）

### 2.1 缺陷匹配判定

两种模式使用不同的匹配函数，**不可混用**。

#### 2.1.1 Rule Only：`match_defect()`

主条件（**全部满足**）：
1. **元件集合交集**：pred = `{component} ∪ component_refs`；gt = `{component}`；非空交集
2. **网络集合交集**：pred = `{net} ∪ net_refs`；gt = `{net}`；非空交集
3. **风险等级等价**：`pred.severity == gt.risk`

弱校验（若双方 `rule_id` 都可取到）：
4. **`rule_id` 相等**：`pred.rule_id == gt.evidence[rule_hit].rule_id`

#### 2.1.2 Full Review：`match_defect_full_review()`

Full Review 的 Pred 来自 `review_defect` 表，字段比 `RuleResult` 少（无 `component_refs` / `net_refs`），且可能含 LLM 输出的占位符。匹配分**严格路径**与**降级路径**。

**严格路径**（`pred_net` 非空）：

1. **网络交集**：`pred_net ∩ gt_net` 非空
2. **风险严格一致**：`pred_risk == gt_risk`
3. **语义同一**：满足以下任一
   - `rule_id` 一致（双方都有 `rule_id` 且相等）
   - `category` 一致（双方都有 `category` 且相等）

**降级路径**（`pred_net` 为空，受限例外）：

1. **风险严格一致**：`pred_risk == gt_risk`
2. **语义同一**：满足以下任一
   - `rule_id` 一致
   - `category` 一致 **且** `component` 交集非空（双锚点）

**risk 匹配规则（S1-B 补充）**：

`risk` 采用"容忍匹配"而非"严格一致"，规则如下：

- `pred_risk` **不得低于** `gt_risk`：LLM 判轻了算漏检（如 GT=high，pred=medium → 不匹配）
- `pred_risk` **最多比 `gt_risk` 高 1 级**：LLM 判重了算"更保守的命中"（如 GT=high，pred=critical → 匹配）
- `GT=low` 时**要求严格一致**：`low` 不计入统计，任何高于 `low` 的预测都算误判（如 GT=low，pred=medium → 不匹配）
- **未知 risk 值**（不在 `low/medium/high/critical` 枚举内）→ 不匹配

**为什么允许"高一级"**：
- GT 的 `risk` 是专家标定的严重程度，LLM 可能基于更保守的判断填更高一级（如 GT=high，LLM 认为"结温超 125℃ 应判 critical"）
- 这是"更保守的命中"，不是"误判"
- 但只允许高 1 级（critical vs medium 差 2 级仍算不匹配），避免"过度报警"

**实现**：`benchmark_service._risk_matches(pred_risk, gt_risk) -> bool`，严格路径和降级路径统一调用。

**severity 独立诊断**（补充）：
- `risk` 容忍匹配后，`tp` 会包含"risk 高一级"的匹配
- 为了诊断"LLM 的 severity 判断准确率"，建议在 CSV 的 `notes` 列追加 `risk_exact_match=N/M`（N=严格一致的 tp 数，M=全部 tp 数）
- 这是独立诊断字段，不影响 Precision/Recall 计算
- **当前 S1-B 暂不实现，记入阶段 G 或后续 Sprint 的 Benchmark 增强**

**禁止反向放宽**：若 `pred_net` 非空但与 `gt_net` 不一致，直接返回 `False`，不进入降级路径。

**Full Review 的 rule 缺陷来源（S1-B 起）**：
- `origin=rule` 的缺陷**完全由 `RuleResult` 构造**（`_build_rule_defect`），不依赖 LLM 是否输出。
- LLM 输出的 `origin=rule` 缺陷全部被丢弃（Merge 层集合运算 v14），只保留 `origin=ai_discovered`。
- 因此 rule 缺陷的 `component` / `net` / `risk` / `category` 与规则引擎输出严格一致，匹配稳定性不再受 LLM 随机性影响。

**为什么 Full Review 放宽 `component`**：
- GT 金标可能选“更精确的定位”（如 `C3`），LLM 选“更宏观的定位”（如 `U1`），两者在同一网络上，语义等价。
- 严格路径不要求 `component` 交集，降级路径要求 `component` 交集（双锚点，避免“同 category 不同器件”误判）。

**为什么 `pred_net` 为空时用 `category + component` 双锚点**：
- `category` 是宽泛大类（如 `power`），单独不足以证明语义等价。
- 加 `component` 交集作为辅助锚点，保守处理。

**`net` / `component` 的归一化**（S1-F）：
- LLM 可能输出 `"-"` / `""` 作为占位符，视为“无值”。
- `_set_of()` 内部把 `"-"` / `""` 视为空集；`_persist_review_report` 落库时把 `"-"` / `""` 归一化为 `None`。
- 两层归一化确保“LLM 输出 `net='-'`”不会导致降级路径失效。

#### 2.1.3 两种匹配的对照

| 条件 | `match_defect`（Rule Only） | `match_defect_full_review` 严格路径 | `match_defect_full_review` 降级路径 |
| :--- | :--- | :--- | :--- |
| `net` 交集 | ✅ 必须 | ✅ 必须 | ❌ 不要求（`pred_net` 为空） |
| `component` 交集 | ✅ 必须 | ❌ 不要求 | ✅ 必须（与 `category` 组合） |
| `risk` 一致 | ✅ 必须（`severity == risk`） | ✅ 必须 | ✅ 必须 |
| `rule_id` 一致 | 弱校验（双方都有时） | 语义同一的锚点之一 | 语义同一的锚点之一 |
| `category` 一致 | 不检查 | 语义同一的锚点之一 | 语义同一的锚点之一 |
| `pred_net` 非空但不等 | 直接 `False` | 直接 `False` | 不适用（`pred_net` 为空才进降级） |

### 2.2 贪心匹配策略（两种模式共用）

- 外层遍历 GT，内层遍历未占用的 pred
- 一个 GT 命中一个 pred 后立即 `break`：**GT 唯一占用**
- pred 通过 `tp_set_pred` / `tp_set_pred_idx` 保证**唯一占用**
- 若多条 pred 命中同一 GT，**仅第一条计 TP，其余计入 FP**
  - 该 FP 反映「重复命中」，是有意义的误报信号
  - 分析规则 / LLM 质量时，此类 FP 应作为「冗余」重点排查

**一对一匹配的边界**：
- 同一 Pred 不会计入多个 TP（`tp_set_pred` 保证）
- 同一 GT 不会匹配多个 Pred（`tp_set_gt` 保证，内层 `break` 后不再遍历）
- 跨 Case 的相同类型问题按各 Case 独立计数（每个 Case 的 `tp_set_*` 是独立的）

---

## 3. Rule Only 核心指标

> 适用于 Rule Only 模式；Full Review 模式下同样计算，但 Pred 来源不同（见 §2）。

| 指标 | 公式 | 值域 | 分母为 0 时 |
| :--- | :--- | :--- | :--- |
| Rule Precision | TP / (TP + FP) | [0,1] | `None` → CSV 写 `NaN` |
| Rule Recall | TP / (TP + FN) | [0,1] | `None` → CSV 写 `NaN` |
| Rule FP Rate | FP / (TP + FP) | [0,1] | `None` → CSV 写 `NaN` |
| Rule FN Rate | FN / (TP + FN) | [0,1] | `None` → CSV 写 `NaN` |

**说明**：
- 四个指标在两种模式下口径一致，只是 Pred 来源不同。
- 全局 Precision / Recall 由 `summarize_benchmark()` 按 `sum(tp) / (sum(tp) + sum(fp))` 等汇总，不是各 Case 的算术平均。
- 分母为 0 时返回 `None`，CSV 写 `NaN`，不写 `0.0`（避免误读为“精度为零”）。

---

## 4. EVC · Evidence 完整率

Sprint 1 阶段 F 起，EVC 分为两个维度：

| 指标 | 计算对象 | 分母 | 用途 |
| :--- | :--- | :--- | :--- |
| `EVC` | GT 的 `defects[].evidence` | `total_gt_defects` | 评估**专家金标**的证据完整性（口径不变，Sprint 0 兼容） |
| `EVC_PRED` | Pred 的 `defects[].evidence` | `total_pred_defects` | 评估**系统输出**的证据完整性（S1-F 新增） |
| `EVC_COMPLETE_RATE` | 与 `EVC` 同值 | — | Sprint 1 §9.4 的 `evd_complete_rate` 别名列 |

### 4.1 判定规则

#### 4.1.1 GT 侧（`EVC`）：V1.2 §3.2 严格对齐

单条 evidence 计入「完整」当且仅当：
- `source` 非空且长度 ≥ 10
- `section` 非空且长度 ≥ 10
- `reason` 非空且长度 ≥ 10

**特例开关**：`benchmark_service.IR_REF_EXEMPT`
- 默认 `False`：`ir_ref` 亦须三要素（V1.2 §3.2 严格口径）
- 若架构组澄清 `ir_ref` 应豁免，改 `True`：`ir_ref` 只校验 `detail ≥ 10`

#### 4.1.2 Pred 侧（`EVC_PRED`）：按 `type` 分流

单条 evidence 计入「完整」的规则：

| `type` | 判定条件 |
| :--- | :--- |
| `rag_ref` | `source` / `section` / `reason` 均非空且长度 ≥ 10（同 GT 侧） |
| `ir_ref` | `detail` 非空且长度 ≥ 5（**不要求**三要素） |

**为什么 Pred 侧 `ir_ref` 不要求三要素**：
- Pred 的 `ir_ref` 是“IR 结构引用”，只提供 `detail`（如 `Net[VCC].connected_pins=[U1.VCC]`），不提供 `source` / `section` / `reason`。
- 要求 `ir_ref` 有三要素会误判所有 `ir_ref` 为不完整，导致 `EVC_PRED=0.0`（S1-F 修正前的问题）。
- 用 `_evidence_is_complete_pred()` 实现，与 GT 侧的 `_evidence_is_complete()` 分开。

### 4.2 计算层次

| 层次 | GT 侧（`EVC`） | Pred 侧（`EVC_PRED`） |
| :--- | :--- | :--- |
| Defect EVC | 该 defect 完整 evidence 条数 / evidence 总条数 | 同左，但用 `_evidence_is_complete_pred()` |
| Case EVC | 该 case 内全部 defect EVC 的算术平均 | 同左 |
| 全局 EVC | 全部 Case EVC 的算术平均（`summarize_benchmark()`） | 同左 |

**Defect 级“完整”的判定**：
- 至少 1 条 evidence 完整 → 该 defect 计入“完整”
- 无 evidence 或全部不完整 → 该 defect 不计入“完整”

### 4.3 Rule Only 与 Full Review 的差异

| 项 | Rule Only | Full Review |
| :--- | :--- | :--- |
| `EVC` | 有值（GT 侧） | 有值（GT 侧，口径不变） |
| `EVC_PRED` | `None` → CSV `NaN` | 有值（Pred 侧，`ReviewDefect.evidence`） |
| `EVC_COMPLETE_RATE` | 与 `EVC` 同值 | 与 `EVC` 同值 |

**为什么 Rule Only 的 `EVC_PRED` 为 `None`**：
- Rule Only 的 Pred 是 `RuleResult`，`evidence` 字段是 `evidence_ir_refs`（字符串列表），无 `source` / `section` / `reason` 三要素结构。
- 强行计算会误导，故返回 `None`。

### 4.4 Sprint 0 现状与 Sprint 1 基线

**case001（Rule Only）**：
- `EVC = 0.6667`：两条 defect 各 3 条 evidence，其中 `ir_ref` 缺 `source` / `section` → 每条 defect EVC = 2/3。
- `EVC_PRED = NaN`：Rule Only 无 Pred 证据三要素。
- `EVC_COMPLETE_RATE = 0.6667`（与 `EVC` 同值）。

**真实轨 10 Case（Full Review）**：
- 全局 `EVC = 0.5542`（GT 侧，数据资产现状）
- 全局 `EVC_PRED = 1.0`（Pred 侧，LLM 输出的 evidence 全部完整）
- 全局 `EVC_COMPLETE_RATE = 0.5542`（与 `EVC` 同值）

**EVC 值不作为阻断条件**：
- `EVC < 1.0` 属数据资产现状（GT 的 `ir_ref` 缺三要素），不是代码缺陷。
- 后续补齐 GT 的 `ir_ref` 结构化字段或启用 `IR_REF_EXEMPT` 后，`EVC` 可提升至 1.0。

---

## 5. review_status 三态计数

对齐 V1.2 §8.2 输出字段 `AI_CONFIRMED_N / NEED_EXPERT_REVIEW_N / LOW_CONFIDENCE_N`。

**重要边界**：`review_status` 表示**系统对输出质量的门槛判定**，不是专家的业务确认。专家确认由独立的 `expert_decision` 字段表达（`pending` / `confirmed` / `rejected` / `needs_verification`）。两者不可互相映射。

### 5.1 Rule Only 模式（Sprint 0 实现，口径不变）

对每条 GT defect：

| 条件 | status |
| :--- | :--- |
| 无 evidence | `LOW_CONFIDENCE` |
| 有 evidence 但无一条完整 | `NEED_EXPERT_REVIEW` |
| 至少一条 evidence 完整 | `AI_CONFIRMED` |

**说明**：Rule Only 的 Pred 无 `review_status` 字段（`RuleResult` 不包含），三态计数对象是 **GT defect**，反映“GT 金标自身的证据质量分布”。

### 5.2 Full Review 模式（S1-F 实现）

对每条 Pred defect，按 `ReviewDefect.review_status` 字段直接计数。

该字段由 `output_parser.py::_classify_defect()` 按 V1.2 §5.2 五条强规则判定：

1. 10 字段完整性（缺字段 → `LOW_CONFIDENCE`）
2. Evidence 三要素完整率（有效 evidence < 1 条 → `NEED_EXPERT_REVIEW`）
3. `ai_discovered` 的 `ir_ref` + `rag_ref` 双引用（缺任一项 → `NEED_EXPERT_REVIEW`）
4. 模糊词黑名单（`risk ≥ medium` 且 `root_cause` / `suggestion` 中模糊词 ≥ 2 → `LOW_CONFIDENCE`）
5. 以上全通过 → `AI_CONFIRMED`

**硬约束**：`origin=ai_discovered` 的缺陷，最终 `review_status` 不得为 `AI_CONFIRMED`（即使规则 1–5 全通过，也强制降级为 `NEED_EXPERT_REVIEW`）。

### 5.3 两种模式的差异

| 项 | Rule Only | Full Review |
| :--- | :--- | :--- |
| 计数对象 | GT defect | Pred defect（`ReviewDefect`） |
| 判定依据 | GT 的 evidence 三要素完整率 | Pred 的 `review_status` 字段（Parser 判定） |
| 语义 | “GT 金标自身的证据质量” | “LLM 输出的质量门槛” |
| 专家确认 | 不涉及 | 不涉及（`expert_decision` 独立） |

**注意**：两种模式的 `AI_CONFIRMED_N` 语义不同，跨模式对比时需说明。

---

## 6. 模拟指标（S1-F 新增）

### 6.1 为什么需要“模拟指标”

Sprint 1 阶段 F 的 Benchmark 跑完后，**没有真实专家反馈数据**：
- `feedback_item` 表可能为空，或只有测试数据。
- 真实 `EAR` / `AI_NER` 需要专家对每条缺陷提交 `correct_defect` / `suggestion_update` 等反馈后才能计算。

为在“无真实反馈”阶段仍能观察系统表现，阶段 F 引入两个**模拟指标**，用 GT 金标匹配模拟“专家确认 / 采纳”。

### 6.2 `AI_NER_SIMULATED` · 模拟 AI 新增有效发现率

**公式**：

```text
AI_NER_SIMULATED = N_ai_effective_sim / (N_rule_effective_sim + N_ai_effective_sim)
```

**计数口径**：

| 变量 | 定义 |
| :--- | :--- |
| `N_ai_effective_sim` | Pred 中 `origin=ai_discovered` 且与 GT 匹配的缺陷数 |
| `N_rule_effective_sim` | Pred 中 `origin=rule` 且与 GT 匹配的缺陷数 |
| 分母为 0 | 返回 `None` → CSV `NaN` |

**计算位置**：`compute_ai_ner_simulated(pred_defects, tp_set_pred)`，接收已匹配的 `tp_set_pred`，不重新匹配。

**实现边界**：
- `hybrid` 暂不计入（Sprint 1 的 GT 金标无 `hybrid`）。
- 匹配由调用方传入（`tp_set_pred`），保证与 `tp` / `fp` / `fn` 同一次匹配结果。

### 6.3 `EAR_SIMULATED` · 模拟专家采纳率

**公式**：

```text
EAR_SIMULATED = len(tp_set_pred) / len(pred_defects)
```

**语义**：

- **`EAR_SIMULATED` 的实际语义是 gold-label match precision**，即“Pred 中有多少比例被 GT 金标认可”。
- **不是真实专家采纳率**。真实 EAR 需要专家反馈，列在 `EAR` 旧列。
- 字段名 `ear_simulated` 暂保留以兼容已冻结的阶段 F 设计；若后续允许改名，`gold_match_precision` 更准确。

**计算位置**：`compute_ear_simulated(pred_defects, tp_set_pred)`。

**分母为 0**：Pred 为空时返回 `None` → CSV `NaN`。

### 6.4 模拟指标的边界（必读）

**⚠️ 以下四条是模拟指标的核心边界，不得误解：**

1. **模拟指标只说明“Pred 与 GT 金标的一致性”**，不是真实 AI 确认率，也不是真实专家采纳率。
2. **不写 `review_defect.expert_decision`**，不写 `review_result.ai_new_effective_rate` / `expert_adoption_rate`。模拟值只出现在 `BenchmarkResult.ai_ner_simulated` / `ear_simulated` 字段和 CSV 的 `AI_NER_SIMULATED` / `EAR_SIMULATED` 列。
3. **`AI_NER_SIMULATED = 0.0` 不代表 LLM 无 AI 能力**。它只说明“在当前定义下，Pred 中没有与 GT 的 `ai_discovered` 匹配的缺陷”。可能原因：
   - LLM 确实没有新增 AI 发现；
   - LLM 有 AI 发现但未匹配上 GT（如字段不一致）；
   - GT 的 `ai_discovered` 缺陷未被 LLM 覆盖。
   需结合 `origin` 分布和 `tp` / `fn` 具体分析，不能单凭 `0.0` 下结论。
4. **`EAR_SIMULATED = 1.0` 不代表专家采纳率 100%**。它只说明“Pred 全部匹配了 GT”，是 gold-label match precision 为 1.0。

### 6.5 真实指标 vs 模拟指标

| 指标 | 列名 | 数据来源 | Full Review 无反馈时 |
| :--- | :--- | :--- | :--- |
| 真实 AI_NER | `AI_NER` | 专家对 `ai_discovered` 的 `confirmed` 反馈 | `None` → CSV `NaN` |
| 真实 EAR | `EAR` | 专家对缺陷的 `correct_defect` / `suggestion_update(adopted)` 反馈 | `None` → CSV `NaN` |
| 模拟 AI_NER | `AI_NER_SIMULATED` | GT 金标匹配 | 有值（金标模拟） |
| 模拟 EAR | `EAR_SIMULATED` | GT 金标匹配（gold-label match precision） | 有值（金标模拟） |

**分口径管理**：
- 旧列 `AI_NER` / `EAR` 保持“真实数据口径”，Full Review 无反馈时留空。
- 新列 `AI_NER_SIMULATED` / `EAR_SIMULATED` 只填模拟值。
- CSV 的 `METRIC_SOURCE` 列标记本次运行的口径：`rule_only_baseline` / `golden_label_simulated`。

### 6.6 后续 Sprint 的演进

- **真实 AI_NER / EAR**：等真实专家反馈数据积累后，用 `feedback_item` 表计算，替代模拟指标作为验收依据。
- **模拟指标的作用**：作为“无真实反馈阶段的过渡观察指标”，以及“Prompt / RAG 优化的基线对比”。同一套 GT / 匹配 / 指标下，模拟指标的变化能反映系统改进方向。
- **命名调整**：若后续允许改字段名，`ear_simulated` 建议改为 `gold_match_precision`，更准确表达语义。

### 6.7 AI_NER 样本门槛（Sprint 2 S2-A 补冻结）

> **背景**：v3.1 只冻结了 EAR 的样本门槛（≥ 20 条采纳相关反馈，见 Sprint 1 §9.6），AI_NER 只有公式、模拟值、边界，**无真实样本门槛**。Sprint 2 S2-A 补齐。

**AI_NER 样本门槛：≥ 10 条 `ai_discovered` 缺陷**（以去重后的唯一缺陷实体计）。

**判定规则**：

- `N_ai_candidate`（去重后 `origin=ai_discovered` 的缺陷数）**< 10** 时：`AI_NER` 返回 `None`，CSV 输出 `NaN`，UI 显示 `N/A · 样本不足`。
- `N_ai_candidate` **≥ 10** 时：按 §2 匹配 + §6.2 公式计算真实 `AI_NER`。

**与 EAR 门槛的区别**：

| 指标 | 样本门槛 | 统计单位 | 门槛依据 |
| :--- | :--- | :--- | :--- |
| EAR | ≥ 20 条采纳相关反馈 | 唯一 `defect_id` | 采纳率统计稳定性 |
| AI_NER | ≥ 10 条 `ai_discovered` 缺陷 | 去重后唯一缺陷实体 | AI 发现能力统计稳定性（分母是 AI 候选，基数比 EAR 小，门槛相应降低） |

**Seed / simulated 数据不计入门槛**：只有真实 `expert_decision=confirmed` 的 `ai_discovered` 缺陷才计入 `N_ai_candidate`。`AI_NER_SIMULATED` 的计算不受本门槛影响（模拟指标口径不变）。

**字段落地**：

- `summary_metrics.sample_info.ai_ner_sample_size`：真实 `ai_discovered` 缺陷数（去重后）。
- `summary_metrics.sample_info.ai_ner_sample_threshold`：固定 `10`，由后端指标定义输出，前端不硬编码。
- 两者均写入 `docs/benchmark/metrics_definition.md` 与 `app/api/schemas.py` 的 `SummaryMetrics` schema。

**实现位置**：

- `benchmark_service.compute_real_ai_ner(pred_defects, expert_decisions)`：加门槛判断，`N_ai_candidate < 10` 返回 `None`。
- `task_service.get_task_summary_metrics(task_id)`：`sample_info` 填真实值。
- `tests/test_benchmark.py`：加 `test_ai_ner_threshold_below_10_returns_none` / `test_ai_ner_threshold_at_10_returns_value`。

---

## 7. CSV 输出字段（24 列，v3.0 起沿用）

输出路径：
- Rule Only：`out/bench_ruleonly_sprint0.csv`（Sprint 0 兼容路径）
- Full Review：`out/bench_full_review_sprint1.csv`（S1-F 新增路径）

**列布局原则**：
- **旧列在前（18 列）**：位置与 Sprint 0 完全一致，旧消费者按列名或固定索引读都不受影响。
- **新列在后（6 列）**：S1-F 新增，追加在末尾，不穿插。
- **汇总行**：CSV 末尾追加一行 `case_id=__SUMMARY__`，跨 Case 汇总。

### 7.1 旧列（Sprint 0 兼容，位置不变）

| # | 列名 | 语义 | 对应 V1.2 §8.2 |
| :--- | :--- | :--- | :--- |
| 1 | `case_id` | Case 标识；汇总行为 `__SUMMARY__` | `case_id` |
| 2 | `case_class` | `A` / `B` | `case_class` |
| 3 | `total_gt_defects` | GT 缺陷数 | `N_rule_only`（语义等价） |
| 4 | `total_pred_defects` | Pred 缺陷数 | `N_ai_new_total`（同义，Sprint 0 仅规则） |
| 5 | `tp` | 真正例数 | —（实现必需） |
| 6 | `fp` | 假正例数 | —（实现必需） |
| 7 | `fn` | 假负例数 | —（实现必需） |
| 8 | `Rule_Precision` | TP / (TP + FP) | `Rule_Precision` |
| 9 | `Rule_Recall` | TP / (TP + FN) | `Rule_Recall` |
| 10 | `Rule_FP_Rate` | FP / (TP + FP) | `Rule_FP_Rate` |
| 11 | `Rule_FN_Rate` | FN / (TP + FN) | `Rule_FN_Rate` |
| 12 | `EVC` | GT 侧 EVC（口径不变） | `EVC` |
| 13 | `EAR` | 真实 EAR；Full Review 无反馈时 `NaN` | `EAR` |
| 14 | `AI_NER` | 真实 AI_NER；Full Review 无反馈时 `NaN` | `AI_NER` |
| 15 | `AI_CONFIRMED_N` | `review_status=AI_CONFIRMED` 计数 | `AI_CONFIRMED_N` |
| 16 | `NEED_EXPERT_REVIEW_N` | `review_status=NEED_EXPERT_REVIEW` 计数 | `NEED_EXPERT_REVIEW_N` |
| 17 | `LOW_CONFIDENCE_N` | `review_status=LOW_CONFIDENCE` 计数 | `LOW_CONFIDENCE_N` |
| 18 | `notes` | 备注（如 `sprint0_baseline,rule_only`） | —（S3 建议升格） |

**旧列口径完全不变**：`EVC` 仍是 GT 侧、`EAR` / `AI_NER` 仍是真实数据口径（无反馈时 `NaN`），不因新增列而改变。

### 7.2 新列（S1-F 新增，追加在末尾）

| # | 列名 | 语义 | 无值时 |
| :--- | :--- | :--- | :--- |
| 19 | `EVC_COMPLETE_RATE` | 与 `EVC` 同值（Sprint 1 §9.4 的 `evd_complete_rate` 别名） | `NaN`（仅当 `EVC` 为 `None`） |
| 20 | `EVC_PRED` | Pred 侧 EVC（S1-F 新增） | Rule Only 时 `NaN` |
| 21 | `AI_NER_SIMULATED` | 模拟 AI_NER（金标匹配） | Rule Only 时 `NaN`；分母 0 时 `NaN` |
| 22 | `EAR_SIMULATED` | 模拟 EAR（gold-label match precision） | Rule Only 时 `NaN`；Pred 为空时 `NaN` |
| 23 | `RUN_MODE` | `rule_only` / `full_review` | 汇总行留空 |
| 24 | `METRIC_SOURCE` | `rule_only_baseline` / `golden_label_simulated` | 汇总行留空 |

**新列口径**：
- `EVC_COMPLETE_RATE` 是 `EVC` 的别名，两者永远同值。
- `EVC_PRED` / `AI_NER_SIMULATED` / `EAR_SIMULATED` 只在 Full Review 模式有值，Rule Only 模式为 `NaN`。
- `RUN_MODE` / `METRIC_SOURCE` 标记本次运行的口径，便于解析时区分。

### 7.3 已删除的列（与 Sprint 0 的差异）

| 列名 | Sprint 0 | Sprint 1 阶段 F |
| :--- | :--- | :--- |
| `Overall_Precision` | 有（始终 `NaN`） | **已删除** |
| `Overall_FP_Rate` | 有（始终 `NaN`） | **已删除** |
| `Overall_FN_Rate` | 有（始终 `NaN`） | **已删除** |

**删除理由**：
- Sprint 0 的这三列本来就是占位（始终 `None` → `NaN`），无实际数据。
- Sprint 1 的全局指标由 `summarize_benchmark()` 统一算，通过独立的汇总行输出，不混入 per-case 行的列。
- 删除后，旧消费者的列索引可能偏移（如果按固定索引读）；按列名读的消费者不受影响。

### 7.4 汇总行 `__SUMMARY__` 口径

CSV 末尾追加一行 `case_id=__SUMMARY__`，由 `summarize_benchmark()` 计算。

| 列 | 汇总行取值 |
| :--- | :--- |
| `case_id` | `__SUMMARY__` |
| `case_class` | 空 |
| `total_gt_defects` | 空（跨 Case 合计无意义） |
| `total_pred_defects` | 空（同上） |
| `tp` / `fp` / `fn` | `sum(tp)` / `sum(fp)` / `sum(fn)`（仅计 `status=done` 的 Case） |
| `Rule_Precision` | `sum(tp) / (sum(tp) + sum(fp))` |
| `Rule_Recall` | `sum(tp) / (sum(tp) + sum(fn))` |
| `Rule_FP_Rate` | `sum(fp) / (sum(tp) + sum(fp))` |
| `Rule_FN_Rate` | `sum(fn) / (sum(tp) + sum(fn))` |
| `EVC` | 各 Case `EVC` 的算术平均（仅计非 `None`） |
| `EAR` | 各 Case `EAR` 的算术平均（仅计非 `None`） |
| `AI_NER` | 各 Case `AI_NER` 的算术平均（仅计非 `None`） |
| `AI_CONFIRMED_N` / `NEED_EXPERT_REVIEW_N` / `LOW_CONFIDENCE_N` | 各 Case 计数之和 |
| `notes` | `summary, done=X/Y`（X=成功数，Y=总数） |
| `EVC_COMPLETE_RATE` | 与 `EVC` 同值 |
| `EVC_PRED` | 各 Case `EVC_PRED` 的算术平均（仅计非 `None`） |
| `AI_NER_SIMULATED` | 各 Case `AI_NER_SIMULATED` 的算术平均（仅计非 `None`） |
| `EAR_SIMULATED` | 各 Case `EAR_SIMULATED` 的算术平均（仅计非 `None`） |
| `RUN_MODE` | 空（跨 mode 时无意义） |
| `METRIC_SOURCE` | 空（同上） |

**汇总原则**：
- **只统计 `status=done` 的 Case**：`failed` 的 Case 不计入分母。
- **规则指标用 sum 汇总**：`sum(tp) / (sum(tp) + sum(fp))`，不是各 Case 的算术平均。
- **EVC / EAR / AI_NER 用算术平均**：仅计非 `None` 的 Case，避免 `NaN` 污染。
- **跨 mode 时 `RUN_MODE` / `METRIC_SOURCE` 留空**：汇总行可能跨 `rule_only` + `full_review`，留空避免误导。

### 7.5 无样本 / 分母为 0 的处理

统一规则：**分母为 0 时返回 `None`，CSV 写 `NaN`，不写 `0.0`**。

| 场景 | 处理 |
| :--- | :--- |
| `tp + fp = 0` | `Rule_Precision = None`，`Rule_FP_Rate = None` |
| `tp + fn = 0` | `Rule_Recall = None`，`Rule_FN_Rate = None` |
| `total_gt_defects = 0` | `EVC = None` |
| `total_pred_defects = 0` | `EVC_PRED = None`，`EAR_SIMULATED = None` |
| `N_rule_effective + N_ai_effective = 0` | `AI_NER_SIMULATED = None` |
| 无真实反馈 | `EAR = None`，`AI_NER = None` |
| 所有 Case 的某指标均为 `None` | 汇总行该指标为 `None` → `NaN` |

**为什么不用 `0.0`**：
- `0.0` 会被误读为“精度为零”或“完全失败”，而实际上是“无样本，不可计算”。
- `NaN` 明确表示“无值”，与真正的 0 值区分。

**`_fmt()` 的实现**：
```python
def _fmt(v: Optional[float], ndigits: int = 4) -> Any:
    if v is None:
        return "NaN"
    if isinstance(v, float):
        return round(v, ndigits)
    return v
```

### 7.6 CSV 列名与 V1.2 §8.2 的映射（补充）

| 本 CSV 新列 | V1.2 §8.2 对应 | 说明 |
| :--- | :--- | :--- |
| `EVC_COMPLETE_RATE` | `evd_complete_rate` | 语义等价，`EVC` 的别名 |
| `EVC_PRED` | —（V1.2 未列） | S1-F 新增，Pred 侧 EVC |
| `AI_NER_SIMULATED` | `AI_NER`（模拟口径） | 与真实 `AI_NER` 分列 |
| `EAR_SIMULATED` | `EAR`（模拟口径） | 与真实 `EAR` 分列 |
| `RUN_MODE` | —（V1.2 未列） | S1-F 新增，标记运行模式 |
| `METRIC_SOURCE` | —（V1.2 未列） | S1-F 新增，标记口径来源 |

**未输出字段**：`N_ai_effective` / `N_ai_fp` / `N_ai_fn` / `N_ai_adopted`（Sprint 1 无真实反馈数据，后续 Sprint 补）。

---

## 8. 基线预期值

### 8.1 case001 · Rule Only 基线（Sprint 0 Phase G）

| 项 | 值 |
| :--- | :--- |
| `case_class` | A |
| `mode` | `rule_only` |
| `total_gt_defects` | 2（POWER_001、POWER_002） |
| `total_pred_defects` | 2 |
| `tp` / `fp` / `fn` | 2 / 0 / 0 |
| `Rule_Precision` | 1.0 |
| `Rule_Recall` | 1.0 |
| `Rule_FP_Rate` | 0.0 |
| `Rule_FN_Rate` | 0.0 |
| `EVC` | 0.6667（GT 证据现状，待数据资产优化） |
| `EVC_PRED` | `NaN`（Rule Only 无 Pred 证据三要素） |
| `EVC_COMPLETE_RATE` | 0.6667（与 `EVC` 同值） |
| `EAR` | `NaN`（无真实反馈） |
| `AI_NER` | 0.0（A 类 case `ai_expected_count=0`） |
| `AI_NER_SIMULATED` | `NaN`（Rule Only 不适用） |
| `EAR_SIMULATED` | `NaN`（Rule Only 不适用） |
| `AI_CONFIRMED_N` / `NEED_EXPERT_REVIEW_N` / `LOW_CONFIDENCE_N` | 2 / 0 / 0 |
| `RUN_MODE` | `rule_only` |
| `METRIC_SOURCE` | `rule_only_baseline` |
| `notes` | `sprint0_baseline,rule_only` |

> `EVC = 0.6667` 属数据资产现状（GT 的 `ir_ref` 缺 `source` / `section`），**不作为 Sprint 0 阻断条件**。

### 8.2 case001 · Full Review 基线（S1-F）

| 项 | 值 |
| :--- | :--- |
| `case_class` | A |
| `mode` | `full_review` |
| `total_gt_defects` | 2 |
| `total_pred_defects` | 2 |
| `tp` / `fp` / `fn` | 2 / 0 / 0 |
| `Rule_Precision` / `Rule_Recall` | 1.0 / 1.0 |
| `EVC` | 0.6667（GT 侧，口径不变） |
| `EVC_PRED` | 1.0（Pred 侧，LLM 输出的 `ir_ref` 有 `detail`） |
| `EVC_COMPLETE_RATE` | 0.6667（与 `EVC` 同值） |
| `EAR` | `NaN`（无真实反馈） |
| `AI_NER` | `NaN`（无真实反馈） |
| `AI_NER_SIMULATED` | 0.0（Pred 全是 `origin=rule`，无 `ai_discovered`） |
| `EAR_SIMULATED` | 1.0（Pred 全部匹配 GT） |
| `RUN_MODE` | `full_review` |
| `METRIC_SOURCE` | `golden_label_simulated` |

### 8.3 真实轨 10 Case · Full Review 基线（S1-F）

阶段 F 真实智谱 Integration 测试（`test_benchmark_llm.py`）的 10 Case 结果：

| 指标 | 值 | 说明 |
| :--- | :--- | :--- |
| 全局 `tp` / `fp` / `fn` | 12 / 0 / 4 | 12 个 TP，4 个 FN |
| 全局 `Rule_Precision` | 1.0 | 无 FP |
| 全局 `Rule_Recall` | 0.75 | 4 个 FN 全是 B 类 Case 的 `ai_discovered` |
| 全局 `EVC` | 0.5542 | GT 侧，数据资产现状 |
| 全局 `EVC_PRED` | 1.0 | Pred 侧，LLM 输出证据完整 |
| 全局 `AI_NER_SIMULATED` | 0.0 | LLM 无 AI 辅助发现能力（见 §6.4 边界） |
| 全局 `EAR_SIMULATED` | 1.0 | Pred 全部匹配 GT |

**4 个 FN 的分布**：

| Case | FN | 原因 |
| :--- | :--- | :--- |
| `case_b01_buck_thermal_derating_B` | 1 | GT 的 `ai_discovered`（热降额）未被 LLM 发现 |
| `case_b02_reset_rc_B` | 1 | GT 的 `ai_discovered`（reset_rc 容差）未被 LLM 发现（LLM 无输出） |
| `case_b03_io_direction_B` | 1 | GT 的 `ai_discovered`（SDA 方向）未被 LLM 发现 |
| `case_b04_power_sequence_B` | 1 | GT 的 `ai_discovered`（U3 时序）未被 LLM 发现 |

**这 4 个 FN 是阶段 F 的已知边界**：LLM 无 AI 辅助发现能力，属后续 Prompt / RAG 优化范围，不是 Benchmark 逻辑缺陷。阶段 F 保留此基线，后续改进后用同一套 GT / 匹配 / 指标重评对比。

#### 8.3.1 S1-B 后基线（2026-10-24）

> S1-B（B 类 Case 跑通尝试）后的真实轨 10 Case 结果。改动：RAG 四路检索 + Prompt 专题线索注入 + Merge 层集合运算（v14）+ risk 容忍匹配 + Prompt 强制 category/net。

| 指标 | 阶段 F 基线 | S1-B 最终 | 变化 |
|---|---|---|---|
| 全局 tp / fp / fn | 12 / 0 / 4 | **16 / 0 / 0** | tp +4, fp 0, fn -4 |
| Rule_Precision | 1.0 | **1.0** | 0 |
| Rule_Recall | 0.75 | **1.0** | +0.25 |
| EVC | 0.5542 | 0.5542 | 持平（GT 侧未变） |
| EVC_PRED | 1.0 | 1.0 | 持平 |
| AI_NER_SIMULATED | 0.0 | **0.25** | +0.25 |
| EAR_SIMULATED | 1.0 | **1.0** | 持平 |

逐 B 类 Case：

| Case | 阶段 F | S1-B 最终 | 判断 |
|---|---|---|---|
| case_b01 | tp=1, fn=1 | **tp=2, fn=0** | 完全跑通（热降额） |
| case_b02 | tp=0, fn=1 | **tp=1, fn=0** | 完全跑通（RC 容差） |
| case_b03 | tp=1, fn=1 | **tp=2, fn=0** | 完全跑通（rule IO_002 + ai SDA） |
| case_b04 | tp=1, fn=1 | **tp=2, fn=0** | 完全跑通（rule POWER_001 + ai 电源时序） |

**B 类 4/4 全部跑通，A 类 6/6 保持完美**。全局 Precision=Recall=1.0。

**risk 容忍匹配（S1-B 规则 B）**：
- pred_risk 不得低于 gt_risk
- pred_risk 最多比 gt_risk 高 1 级（如 GT=high, pred=critical → 匹配）
- GT=low 时严格一致
- 未知 risk 值不匹配
- 实现：`benchmark_service._risk_matches(pred_risk, gt_risk)`

---

## 9. 验收标准

### 9.1 Rule Only 模式（Sprint 0 延续）

1. ✅ case001 完整跑通，输出 CSV
2. ✅ 控制台打印全部指标，可被 CI 解析
3. ✅ 匹配逻辑严格执行：元件集合交集 + 网络集合交集 + `severity`/`risk` 等价；**不以 `rule_id` 为主键**
4. ✅ CSV 旧列 18 列与 Sprint 0 定义一致
5. ✅ `EVC` / `EAR` / `AI_NER` 字段存在，Rule Only 按现状输出占位值
6. ✅ case001 满足 K7：Case001 Rule Hit ≥ 2
7. ✅ `EVC_PRED` / `AI_NER_SIMULATED` / `EAR_SIMULATED` 在 Rule Only 下为 `NaN`

### 9.2 Full Review 模式（S1-F 新增）

1. ✅ 10 个 Case 全部可跑通（`run_all_benchmark(MODE_FULL_REVIEW)`）
2. ✅ 每个 Case 输出 `task_id` 非空、`metric_source=golden_label_simulated`
3. ✅ 真实指标 `EAR` / `AI_NER` 在无真实反馈时为 `None` → CSV `NaN`
4. ✅ 模拟指标 `AI_NER_SIMULATED` / `EAR_SIMULATED` 按定义计算（允许合法 `None`）
5. ✅ `EVC_PRED` 按 Pred 的 `evidence` 结构计算（`ir_ref` 只需 `detail`，`rag_ref` 需三要素）
6. ✅ 匹配逻辑按 `match_defect_full_review` 严格 / 降级路径执行
7. ✅ `origin=rule` 的缺陷，`category` / `risk` 以 `RuleResult` 为权威来源
8. ✅ `net='-'` / `component='-'` 归一化为 `None`，不影响匹配
9. ✅ CSV 24 列齐全，汇总行 `__SUMMARY__` 存在
10. ✅ 10 Case 真实轨（`test_benchmark_llm.py -m integration`）至少 8 个 Case 成功

### 9.3 不阻断条件

以下情况**不阻断** Sprint 1 验收，但需记录：

- `EVC < 1.0`：GT 数据资产现状（`ir_ref` 缺三要素）
- `AI_NER_SIMULATED = 0.0`：LLM 无 AI 辅助发现能力（后续 Prompt / RAG 优化）
- B 类 Case 的 `fn > 0`：GT 的 `ai_discovered` 未被 LLM 发现
- 真实轨部分 Case 失败（≤ 2 个）：容忍偶发限流 / 网络抖动

---

## 10. 版本记录

| 版本 | 日期 | 变更 |
| :--- | :--- | :--- |
| v1.0 | 2026-09-15 | Sprint 0 首次冻结：Rule-Only 四项指标 + EVC + 占位 EAR/AI_NER |
| v2.0 | 2026-09-15 | M1 `rule_id` 弱校验；M2 贪心策略显式化；M3 `IR_REF_EXEMPT` 开关；M4 CSV ↔ V1.2 §8.2 映射表；S1 三态列；S2 Overall 预留列；S3 `notes` 字段 |
| v3.0 | 2026-10-23 | Sprint 1 阶段 F 升级：<br>1. 新增 `full_review` 运行模式<br>2. 新增 `BenchmarkResult` 统一结构<br>3. 新增 `EVC_PRED` / `EVC_COMPLETE_RATE`<br>4. 新增 `AI_NER_SIMULATED` / `EAR_SIMULATED`（金标模拟）<br>5. 新增 `match_defect_full_review`（严格 + 降级路径）<br>6. 新增 `_resolve_rule_fields`（`origin=rule` 的 `category` / `risk` 以 `RuleResult` 为准）<br>7. 新增 `net='-'` / `component='-'` 归一化<br>8. CSV 从 18 列扩展到 24 列，删除 `Overall_*` 三列<br>9. 新增汇总行 `__SUMMARY__`<br>10. 新增“模拟指标边界”章节（§6.4） |
| v3.1 | 2026-10-24 | S1-B 后升级：<br>1. §1.3 补 `defect_id` 格式说明（`DEF-RULE-*` vs LLM 原始）<br>2. §2.1.2 补 rule 缺陷来源（RuleResult 构造，不依赖 LLM）<br>3. §2.1.2 补 risk 容忍匹配规则（规则 B：pred_risk 不低于 gt_risk，最多高 1 级，GT=low 严格一致）<br>4. §8.3.1 追加 S1-B 最终基线（tp=16 fp=0 fn=0, Precision=1.0, Recall=1.0）<br>5. 新增 §8.3.1 逐 B 类 Case 对比（B 类 4/4 全部跑通） |
| v3.2 | 2026-10-28 | Sprint 2 S2-A 升级：<br>1. 新增 §6.7「AI_NER 样本门槛」，冻结 ≥ 10 条 `ai_discovered` 缺陷<br>2. 补 `summary_metrics.sample_info.ai_ner_sample_size` / `ai_ner_sample_threshold` 字段定义<br>3. 明确 Seed / simulated 不计入 AI_NER 门槛<br>4. 与 v3.1 的 EAR 门槛（≥ 20 条）形成对照表 |

---

## 11. 引用关联文件

1. `Sprint0_详细执行计划_V1.2_最终版_plan.md` §12 Phase G
2. `Sprint1_融合版开发落地规划_v1.0.md` §9 Benchmark
3. `第一阶段工程实施方案_V1.2_增强版_plan.md` §8 Benchmark
4. `backend/app/services/benchmark_service.py`（匹配、指标、CSV）
5. `backend/app/workflows/review_graph.py`（`_resolve_rule_fields` / `_persist_review_report`）
6. `backend/app/schemas/report.py`（`Defect` / `Evidence` / `ReviewReport`）
7. `backend/app/llm/output_parser.py`（`review_status` 三态判定）
8. `data/cases/case001/expected_review.json`（GT 真值）
9. `data/rules/POWER_002.yaml`（category 从 `timing` 改为 `power`）
10. `backend/app/rules/engine.py`（`RuleResult` 输出模型，`execute_defect_rules` 为 Benchmark 入口）
11. `data/cases/README_CASE_B_DESIGN.md`（Golden Case 规范）
12. `docs/sprint1/sprint1_changelog.md`（阶段 F 变更记录）
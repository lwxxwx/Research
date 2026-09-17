# Benchmark Metrics Definition v1.0
> Business Freeze · Sprint0 业务冻结文档（v2 优化版）
> 适用：
> - Sprint0：**Rule-Only Baseline 评测（A 类 Case，仅确定性规则引擎）**
> - Sprint1+：完整评测（Rule + RAG + LLM AI 评审，启用 EAR、AI_NER、完整三态）
>
> Ground Truth 来源：每个 case 目录下 `expected_review.json`
> Prediction 来源：规则引擎输出 `List[RuleResult]`（执行 `execute_all_rules()` 得到）

---

## 1. 字段映射（代码落地事实）

> ⚠️ GT 与 RuleResult 字段不一致，本映射为匹配逻辑依据。

| 语义 | Ground Truth (`expected_review.json`) | RuleResult (`app/rules/engine.py`) |
| :--- | :--- | :--- |
| 风险等级 | `risk`（low/medium/high/critical） | `severity`（同枚举，语义等价） |
| 主元件 | `component` | `component` |
| 相关元件集合 | —（无） | `component_refs` |
| 主网络 | `net` | `net` |
| 相关网络集合 | —（无） | `net_refs` |
| 规则编号 | ❌ **defect 顶层无**；埋在 `evidence[type=rule_hit].rule_id` | `rule_id` |

**关键结论**：
- 匹配逻辑**不以 rule_id 为主键**（GT 顶层取不到）
- 匹配以「元件集合 ∩ + 网络集合 ∩ + 风险等级等价」为**主条件**
- 若双方都能取到 rule_id，作为**弱校验**（必须相等）

---

## 2. 基础概念

- **GT**：`expected_review.json["defects"]`，专家标定的真实缺陷集合
- **Pred**：`execute_all_rules()` 返回的 RuleResult 序列化 dict 列表
- **TP**：预测缺陷与 GT 缺陷匹配命中真实问题
- **FP**：预测存在，GT 不存在（误报）
- **FN**：GT 存在，预测未检出（漏检）

### 2.1 缺陷匹配判定（`match_defect()` in `benchmark_service.py`）

主条件（**全部满足**）：
1. **元件集合交集**：pred = `{component} ∪ component_refs`；gt = `{component}`；非空交集
2. **网络集合交集**：pred = `{net} ∪ net_refs`；gt = `{net}`；非空交集
3. **风险等级等价**：`pred.severity == gt.risk`

弱校验（若双方 rule_id 都可取到）：
4. **rule_id 相等**：pred.rule_id == gt.evidence[rule_hit].rule_id

### 2.2 贪心匹配策略（显式说明）

- 外层遍历 GT，内层遍历未占用的 pred
- 一个 GT 命中一个 pred 后立即 `break`：**GT 唯一占用**
- pred 通过 `tp_set_pred_idx` 保证**唯一占用**
- 若多条 pred 命中同一 GT，**仅第一条计 TP，其余计入 FP**
  - 该 FP 反映「规则重复命中」，是有意义的误报信号
  - Sprint1 分析规则质量时，此类 FP 应作为「规则冗余」重点排查

---

## 3. Rule-Only 核心指标（Sprint0 Phase G 必算）

| 指标 | 公式 | 值域 | 分母为 0 时 |
| :--- | :--- | :--- | :--- |
| Rule Precision | TP / (TP + FP) | [0,1] | `None` → CSV 写 `NaN` |
| Rule Recall | TP / (TP + FN) | [0,1] | `None` → CSV 写 `NaN` |
| Rule FP Rate | FP / (TP + FP) | [0,1] | `None` → CSV 写 `NaN` |
| Rule FN Rate | FN / (TP + FN) | [0,1] | `None` → CSV 写 `NaN` |

---

## 4. EVC · Evidence 完整率

### 4.1 判定规则（V1.2 §3.2 严格对齐）

单条 evidence 计入「完整」当且仅当：
- `source` 非空且长度 ≥ 10
- `section` 非空且长度 ≥ 10
- `reason` 非空且长度 ≥ 10

**特例开关**：`benchmark_service.IR_REF_EXEMPT`
- 默认 `False`：`ir_ref` 亦须三要素（V1.2 §3.2 严格口径）
- 若架构组澄清 `ir_ref` 应豁免，改 `True`：`ir_ref` 只校验 `detail ≥ 10`

### 4.2 计算层次

| 层次 | 公式 |
| :--- | :--- |
| Defect EVC | 该 defect 完整 evidence 条数 / evidence 总条数 |
| Case EVC | 该 case 内全部 defect EVC 的算术平均 |
| 全局 EVC | 全部 case EVC 的算术平均（Sprint1 计算） |

### 4.3 Sprint0 现状

- 计算对象：**GT `expected_review.json` 的 defects**
- case001 EVC = **0.6667**（数据资产现状，非代码缺陷）
  - 两条 defect 各 3 条 evidence，其中 `ir_ref` 缺 source/section → 每条 defect EVC = 2/3
  - 后续补齐 `ir_ref` 结构化字段或启用豁免开关后可升至 1.0

---

## 5. review_status 三态计数

对齐 V1.2 §8.2 输出字段 `AI_CONFIRMED_N / NEED_EXPERT_REVIEW_N / LOW_CONFIDENCE_N`。

### 5.1 Rule-Only 简化版判定（Sprint0 实现）

对每条 GT defect：
| 条件 | status |
| :--- | :--- |
| 无 evidence | `LOW_CONFIDENCE` |
| 有 evidence 但无一条完整 | `NEED_EXPERT_REVIEW` |
| 至少一条 evidence 完整 | `AI_CONFIRMED` |

### 5.2 Sprint1 升级路径

接入 AI 输出后，按 V1.2 §5.2 五条强规则判定（10 字段完整性 → 证据完整率 → ai_discovered 的 rag/ir 双引用 → 模糊词黑名单 → 通过）。

---

## 6. 预留扩展指标（Sprint0 占位，Sprint1 实装）

### 6.1 EAR · Expert Adoption Rate
EAR = N_adopted / N_defect_with_feedback

- Sprint0：无 feedback 数据表，CSV 写 `NaN`
- Sprint1：接入 feedbacks 表后，统计 `correct_defect` + `suggestion_update(adopted)` 占比

### 6.2 AI_NER · AI New Effective Rate
AI_NER = N_ai_effective / (N_rule_effective + N_ai_effective)


- A 类 case（如 case001）`ai_expected_count=0`，固定输出 `0.0`
- B 类 case Sprint1 才会产生有效值

---

## 7. CSV 输出字段与 V1.2 §8.2 映射

输出路径：`out/bench_ruleonly_sprint0.csv`

| 本 CSV 字段 | V1.2 §8.2 对应 | 说明 |
| :--- | :--- | :--- |
| `case_id` | `case_id` | 一致 |
| `case_class` | `case_class` | 一致 |
| `total_gt_defects` | `N_rule_only`（语义等价） | Sprint0 仅 Rule-Only，GT 数即规则数 |
| `total_pred_defects` | `N_ai_new_total`（同义，Sprint0 仅规则） | Sprint0 = 预测数 |
| `tp / fp / fn` | —（V1.2 未列，实现必需） | 保留 |
| `Rule_Precision` | `Rule_Precision` | 一致 |
| `Rule_Recall` | `Rule_Recall` | 一致 |
| `Rule_FP_Rate` | `Rule_FP_Rate` | 一致 |
| `Rule_FN_Rate` | `Rule_FN_Rate` | 一致 |
| `EVC` | `EVC` | 一致 |
| `EAR` | `EAR` | 一致 |
| `AI_NER` | `AI_NER` | 一致 |
| `AI_CONFIRMED_N` | `AI_CONFIRMED_N` | ✅ S1 补齐 |
| `NEED_EXPERT_REVIEW_N` | `NEED_EXPERT_REVIEW_N` | ✅ S1 补齐 |
| `LOW_CONFIDENCE_N` | `LOW_CONFIDENCE_N` | ✅ S1 补齐 |
| `Overall_Precision` | `Overall_Precision` | Sprint0 单 case = NaN |
| `Overall_FP_Rate` | `Overall_FP_Rate` | Sprint0 单 case = NaN |
| `Overall_FN_Rate` | `Overall_FN_Rate` | Sprint0 单 case = NaN |
| `notes` | —（V1.2 未列，S3 建议升格） | `sprint0_baseline,rule_only` |

> **未输出字段**：`N_ai_effective / N_ai_fp / N_ai_fn / N_ai_adopted`（Sprint0 无 AI 数据，Sprint1 补）

---

## 8. case001 基线预期值（Sprint0 Phase G）

- case_class: A
- total_gt_defects: 2（POWER_001、POWER_002）
- total_pred_defects: 2
- tp / fp / fn: **2 / 0 / 0**
- Rule_Precision: **1.0**
- Rule_Recall: **1.0**
- Rule_FP_Rate: **0.0**
- Rule_FN_Rate: **0.0**
- EVC: **0.6667**（GT 证据现状，待数据资产优化）
- EAR: **NaN**
- AI_NER: **0.0**
- AI_CONFIRMED_N / NEED_EXPERT_REVIEW_N / LOW_CONFIDENCE_N: **2 / 0 / 0**
- Overall_*: NaN（单 case）
- notes: `sprint0_baseline,rule_only`

---

## 9. Sprint0 验收标准

1. ✅ case001 完整跑通，输出 CSV
2. ✅ 控制台打印全部指标，可被 CI 解析
3. ✅ 匹配逻辑严格执行：元件集合交集 + 网络集合交集 + severity/risk 等价；**不以 rule_id 为主键**
4. ✅ CSV 字段与本文档 §7 定义一致，并与 V1.2 §8.2 建立显式映射
5. ✅ EVC/EAR/AI_NER 字段存在，Sprint0 按现状输出占位值
6. ✅ case001 满足 K7：Case001 Rule Hit ≥ 2

> ⚠️ EVC = 0.6667 属数据资产现状，**不作为 Sprint0 阻断条件**。

---

## 10. 版本记录

| 版本 | 日期 | 变更 |
| :--- | :--- | :--- |
| v1.0 | 2026-09-15 | Sprint0 首次冻结：Rule-Only 四项指标 + EVC + 占位 EAR/AI_NER |
| v2.0 | 2026-09-15 | M1 rule_id 弱校验；M2 贪心策略显式化；M3 `IR_REF_EXEMPT` 开关；M4 CSV ↔ V1.2 §8.2 映射表；S1 三态列；S2 Overall 预留列；S3 notes 字段 |

## 11. 引用关联文件

1. `Sprint0_详细执行计划_V1.2_最终版_plan.md` §12 Phase G
2. `第一阶段工程实施方案_V1.2_增强版_plan.md` §8 Benchmark
3. `backend/app/services/benchmark_service.py`（匹配、指标、CSV）
4. `data/cases/case001/expected_review.json`（GT 真值）
5. `backend/app/rules/engine.py`（RuleResult 输出模型）
6. `data/cases/README_CASE_B_DESIGN.md`（Golden Case 规范）

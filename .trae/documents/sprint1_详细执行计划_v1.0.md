# Sprint 1 融合版开发落地规划 V1.0

> **定位**：在 Sprint 0 已有工程基础上，完成核心评审链路、AI 辅助发现与证据链、专家确认、Benchmark V1.2、反馈闭环、最小 API/Web 联调。
>
> **融合原则**：以 DeepSeek 方案的架构、模块边界、质量控制、环境冻结和排障细节为主干；吸收豆包方案的阶段速览、Golden Case 资产清单、CLI 验收、Demo 脚本和分阶段交付方式。
>
> **本版已吸收的项目决策**：
> 1. “规则发现”和“AI 辅助发现”并不矛盾：规则引擎负责确定性规则命中；LLM 也可以提出规则未覆盖的疑似缺陷。AI 提出的发现必须有证据并明确标记，最终由评审专家人工确认。
> 2. Sprint 1 规则总数为 **10 条**，包含 Sprint 0 已有的 5 条；本 Sprint 新增 5 条。
> 3. 当前处于开发阶段，数据库允许完全重置；无需为开发数据保留强兼容迁移路径，但数据库初始化必须保持单一事实来源。
> 4. 当前方案没有实现上传数量/次数限制；Sprint 1 不把上传配额、频率限制或上传计数作为交付内容。
> 5. Sprint 1 继承 Sprint 0 的 Docker + uv、目录挂载、运行时与工程规范，不重做环境底座。
> 6. ⚠️工作量提示：本方案各阶段合计预估约16人工作日，超出原始2周Sprint容量；Sprint启动前必须完成P0/P1/P2优先级裁剪，划分必须交付/可延后任务，不可直接按两周排期。

---

## 0. 文档使用说明

### 0.1 本文作为执行基线

本文用于 Sprint 1 开发、联调、测试和验收。开发中若发现代码现状与本文不一致，先登记差异，再由负责人确认是否调整本文；不要通过临时修改验收口径来绕过失败用例。

### 0.2 基线与继承关系

- Sprint 0 工程基线：`Sprint0_详细执行计划_V1.2_最终版_plan.md`（内容版本 V1.3）。
- 上游业务基线：项目 MVP 闭环方案、第一阶段工程实施方案 V1.2。
- Sprint 1 主要补齐：LLM Provider、LangGraph 六节点、10 个 Golden Case、10 条规则、AI 辅助发现、ReviewReport 证据链与状态、Benchmark V1.2、反馈与规则候选、API/Web MVP 和端到端 Demo。

### 0.3 Sprint 0 已有基础

Sprint 1 默认继承以下能力，不重复造轮子：

- Docker Compose + uv 可复现开发环境；Dev/Prod Compose 分离。
- Python 3.11.x、uv 0.12.6、PostgreSQL 15 + pgvector。
- 并列挂载方案 A；容器内数据、脚本、基础设施与输出目录挂载规范。
- Schematic IR Schema、规则 YAML 格式、规则注册白名单、RAG Seed Knowledge 基础格式。
- 已有 5 条规则：`POWER_001`、`POWER_002`、`POWER_003`、`IO_001`、`IO_002`。
- 基础反馈入库和 RuleCandidate 草稿能力。
- `case001` 与 Sprint 0 Rule Only Baseline Demo。

Sprint 1 需要回归验证这些基线，避免在新链路接入时破坏旧能力。

---

## 1. Sprint 1 目标与范围

### 1.1 目标

1. 打通 `Schematic IR → 规则检测 → RAG 检索 → LLM 分析/辅助发现 → 质量控制 → ReviewReport → 专家反馈 → 规则候选/知识缺口`。
2. 规则引擎与 AI 辅助发现并行协作，保留发现来源和证据，不把未经专家确认的 AI 输出当作已确认缺陷。
3. 输出标准化 ReviewReport：Defect 10 个字段、Evidence 三要素、`review_status` 三态。
4. 规则总数达到 10 条，Golden Case 总数达到 10 个（6A + 4B）。
5. Benchmark V1.2 支持 Rule Only 与 Full Review 两种模式，输出 EVC、AI_NER、EAR 等指标及 CSV。
6. 6 类反馈可提交，反馈可生成规则候选草稿或知识缺口 Backlog。
7. 提供可运行的 API 路由、最小 Web MVP 和 `demo_sprint1.ps1` 端到端演示。
8. 所有核心路径在容器内运行，CI 默认使用 Mock LLM 与 Fake Embedding，不依赖外网。

### 1.2 Sprint 1 范围内

- LLM Provider 抽象：`mock`、`openai`、`vllm` 兼容层；vLLM 兼容层实现但不要求本 Sprint 实际部署。
- 固定 LangGraph 6 Node DAG，不引入 Planner、ReAct 或多 Agent 自主规划。
- Prompt、OutputParser、Evidence 校验与低置信度降级。
- 10 条规则和 10 个 Golden Case。
- RAG 检索器与 LLM 分析节点联通。
- Benchmark V1.2、反馈闭环、候选规则草稿与候选规则临时 Benchmark 骨架。
- API：评审任务、反馈、案例列表、报告查询等必要路由。
- Web MVP：选择已有 Case 发起评审、查看报告/证据/状态/指标、提交反馈。
- 数据库结构重建和 Alembic 迁移验证。

### 1.3 明确不做

- 真实 EDA 原理图解析；仅保留 IR 与 Adapter 边界。
- 真实 PDF 解析、OCR、自动 Datasheet 抓取（后续 Sprint）。
- MCP Server 抽离、知识图谱、Planner/ReAct、多 Agent 自主规划。
- 企业级 RBAC、审计、规则管理后台。
- 复杂的文件上传系统。
- **上传数量限制、上传次数限制、上传频率限制及其配额管理**。
- Sprint 2/3 才规划的完整 Vue 产品化页面、Prompt 治理体系和真实文档解析能力。

> Web MVP 当前使用已有案例选择方式；不要为了页面上出现“上传”概念而提前实现文件接收、配额、限流或存储生命周期管理。

---

## 2. 核心业务规则与 AI 辅助发现边界

### 2.1 两类发现来源

| 发现来源 | 标识建议 | 责任 | 输出要求 | 最终状态 |
|---|---|---|---|---|
| 规则发现 | `rule_detected` / `source=rule` | 规则引擎按确定性规则匹配 IR | 关联 `rule_id`、IR 定位信息、规则依据和建议 | 由系统质量控制与专家流程处理 |
| AI 辅助发现 | `ai_discovered` / `source=ai` | LLM 基于 IR、RAG 资料和规则检测结果提出规则未覆盖的疑似缺陷 | 必须给出可追溯的 `ir_ref`，适用时给出 `rag_ref`，说明推理依据和不确定性 | **不得自动视为专家确认；由评审专家人工确认/驳回/补充** |

二者不是互斥关系。AI 可以解释规则已发现的问题，也可以提出规则引擎未命中的候选问题。报告必须能区分：

- 规则已发现、AI 补充解释；
- AI 新提出、规则未命中；
- 专家确认有效；
- 专家驳回或要求进一步核验。

### 2.2 AI 辅助发现的约束

1. AI 新发现必须是“候选缺陷”，不得仅凭模型自述就写成已确认事实。
2. 每条缺陷必须有至少一条 Evidence；Evidence 的 `source`、`section`、`reason` 三要素必须齐全。
3. AI 新发现至少提供一个 `ir_ref`；若引用了知识库/数据手册内容，则提供 `rag_ref`。无法提供可核验证据时，必须降级为 `NEED_EXPERT_REVIEW` 或 `LOW_CONFIDENCE`，不得伪造引用。
4. 规则命中结果与 AI 新发现应在内部结构中分别保存，合并展示时保留 `discovery_source` 或等价字段。
5. 专家反馈是 AI_NER、EAR 及规则演进质量评估的重要数据来源；不能把“模型生成”直接计为“有效发现”。
6. 不满足质量门槛的 Defect 保留在报告中并降级，不静默丢弃，以便后续分析误报、漏检和证据缺失。

### 2.3 `review_status` 三态

- `AI_CONFIRMED`：结构与证据满足自动质量门槛。该状态表示系统质量校验通过，不替代对 AI 新发现进行的专家业务确认。
- `NEED_EXPERT_REVIEW`：需要专家进一步核验，或证据/结论存在不确定性。
- `LOW_CONFIDENCE`：关键字段、证据或推理质量不足；保留结果并明确标记。

> 需要在 UI 与 API 文案中区分“AI 输出质量状态”和“专家确认结果”。如果现有 Schema 中 `review_status` 不能表达专家的确认/驳回状态，应使用反馈记录或新增独立的 `expert_decision` 字段表达，不能把两种状态混为一谈。

> 完整的专家判定字段`expert_decision`枚举、语义、存储约束详见 §23.1 缺陷状态与专家决定分离。

### 2.4 专家判定流程

1. 规则引擎产生规则命中集合。
2. RAG 检索相关知识证据。
3. LLM 分析规则命中，并可提出规则未覆盖的 AI 候选发现。
4. OutputParser 校验字段、Evidence、引用与状态。
5. ReviewReport 展示发现来源、证据和建议。
6. 专家通过反馈确认、驳回或补充发现。
7. 只有专家确认的 AI 新发现，才计入 AI_NER 的有效发现分子；专家采纳相关数据用于 EAR。

### 2.5 AI 辅助发现决策依据（Q36）与上游更正

**Q36：Sprint 1 是否允许 LLM 提出规则未覆盖的候选缺陷？**

**结论**：**允许**，但必须满足以下约束：

1. AI 新发现必须标记 `discovery_source=ai_discovered`（或既有等价字段）；
2. 每条 AI 新发现必须有至少 1 条 `ir_ref`；引用知识库时至少有 1 条 `rag_ref`；
3. AI 新发现**不得自动视为专家确认**，必须经专家确认 / 驳回 / 要求进一步核验；
4. AI 新发现与规则命中应在内部结构中分别保存，合并展示时保留 `discovery_source`；
5. 无证据或证据不满足质量门槛的 AI 输出，降级为 `NEED_EXPERT_REVIEW` 或 `LOW_CONFIDENCE`，不得伪造引用；
6. 只有经专家确认的 AI 新发现，才计入 AI_NER 的有效发现分子。

**对上游《项目实施方案》§5.2 的更正说明**：

- 原表述「禁止 LLM 独立『发现电路问题』」**修订为**：
  > 「禁止 LLM 无依据自由发挥；允许 LLM 在明确证据链约束下提出候选缺陷，但候选缺陷必须标记 `ai_discovered` 并经专家确认。」

- 原表述「所有候选问题必须来源于确定性规则检查输出（`rule_results`）」**修订为**：
  > 「所有候选问题必须来源可追溯（规则命中 或 AI 辅助发现）；AI 辅助发现必须满足证据链约束（`ir_ref` 必需，`rag_ref` 视引用情况），并经专家确认后方可视为有效发现。」

- **更正依据**：
  - 上游《第一阶段工程实施方案》§2.1 明确 B 类 Case 的核心是「AI 发现规则未覆盖的问题」；
  - 上游《第一阶段工程实施方案》§4.2 明确 AI 发现必须有证据链；
  - 上游《第一阶段工程实施方案》§8.1 明确 AI_NER 指标定义依赖 AI 新发现。

- **生效版本**：Sprint 1。

---

## 3. 规则清单：10 条（含 Sprint 0 原有 5 条）

### 3.1 规则总数与增量

Sprint 1 的规则目标为 **10 条总计**，不是在现有 5 条之外再增加 10 条。

| 分类 | Rule ID | 来源 | Sprint 1 要求 |
|---|---|---|---|
| Power | `POWER_001` | Sprint 0 已有 | 回归验证，不破坏原行为 |
| Power | `POWER_002` | Sprint 0 已有 | 回归验证，不破坏原行为 |
| Power | `POWER_003` | Sprint 0 已有 | Sprint 1 落实降额算法并补齐测试 |
| IO | `IO_001` | Sprint 0 已有 | 回归验证 |
| IO | `IO_002` | Sprint 0 已有 | 回归验证 |
| Clock | `CLOCK_001` | Sprint 1 新增 | 定义规则 YAML、检查逻辑、证据与测试 |
| Clock | `CLOCK_002` | Sprint 1 新增 | 定义规则 YAML、检查逻辑、证据与测试 |
| IO | `IO_003` | Sprint 1 新增 | 定义规则 YAML、检查逻辑、证据与测试 |
| Control | `CTRL_001` | Sprint 1 新增 | 定义规则 YAML、检查逻辑、证据与测试 |
| Control | `CTRL_002` | Sprint 1 新增 | 定义规则 YAML、检查逻辑、证据与测试 |

> 规则 ID 与类别清单是本融合版的统一口径。每条新增规则的具体工程条件、适用范围和判定阈值，应根据项目已有规则设计资料与 IR 能表达的信息落地，不应凭空增加无法由 IR 验证的规则。

### 3.2 规则工程要求

- 规则定义采用既有 YAML 格式；保留 `rule_id`、`rule_name`、`category`、`severity`、`rule_basis`、`applicable_condition`、`check_logic`、`suggestion` 等字段约定。
- `check_logic.function` 只能引用显式白名单注册表中的函数；禁止任意函数调用。
- 保留 `execute_all_rules()` 与 `execute_defect_rules()` 双入口；Benchmark 与正式 ReviewReport 使用 `execute_defect_rules()`。
- 每条规则在启用前必须形成可评审的规则规格：触发条件、必需 IR 字段、判定逻辑/公式、单位约定、严重级别、缺参处理、边界条件、正例、反例、预期输出及 Evidence 定位方式；缺少项目依据的阈值不得自行臆造，须标记为待业务/专家确认。
- 新增规则至少有单元测试、对应 Case 覆盖或明确说明其当前无正例/负例资产的原因。
- 对规则输出保存 `rule_id` 与可定位的 IR 引用，供报告、证据链和 Benchmark 使用。

**`POWER_003` 实现前必须冻结的规格**：明确额定参数、环境/工作温度或其他适用工况、降额曲线/系数的来源、输入输出单位及比较边界。项目资料未给出数值阈值时，不得自行假设；将待确认项列为阻塞决策。参数缺失、单位不兼容、超出曲线范围时应返回可解释的 `NEED_EXPERT_REVIEW`/规则不可判定结果，不得默认为通过。测试至少覆盖正常通过、明确违反、缺参、单位不兼容、边界值与曲线/范围外场景。

### 3.3 规则验收

- 规则目录共 10 条启用规则，ID 唯一，YAML Schema 校验通过。
- Sprint 0 已有 5 条规则回归测试通过。
- 5 条新增规则均有测试覆盖。
- `POWER_003` 不再只是空实现/预留接口。
- 未注册的规则函数必须安全失败并记录日志，不得执行任意代码。

### 3.4 规则数量决策依据（Q3'）与上游更正

**Q3'：Sprint 1 规则数量与命名**

**结论**：**10 条总计**（含 Sprint 0 已有的 5 条 + 本 Sprint 新增 5 条）。

**规则清单**（Sprint 1 冻结版）：

| 分类 | Rule ID | 来源 | Sprint 1 要求 |
| :--- | :--- | :--- | :--- |
| Power | `POWER_001` | Sprint 0 已有 | 回归验证，不破坏原行为 |
| Power | `POWER_002` | Sprint 0 已有 | 回归验证，不破坏原行为 |
| Power | `POWER_003` | Sprint 0 已有 | Sprint 1 落实降额算法并补齐测试 |
| IO | `IO_001` | Sprint 0 已有 | 回归验证 |
| IO | `IO_002` | Sprint 0 已有 | 回归验证 |
| Clock | `CLOCK_001` | Sprint 1 新增 | 定义 YAML、检查逻辑、证据与测试 |
| Clock | `CLOCK_002` | Sprint 1 新增 | 定义 YAML、检查逻辑、证据与测试 |
| IO | `IO_003` | Sprint 1 新增 | 定义 YAML、检查逻辑、证据与测试 |
| Control | `CTRL_001` | Sprint 1 新增 | 定义 YAML、检查逻辑、证据与测试 |
| Control | `CTRL_002` | Sprint 1 新增 | 定义 YAML、检查逻辑、证据与测试 |

**数字口径**：`Power 3 + Clock 2 + IO 3 + Control 2 = 10`。

**对上游《第一阶段工程实施方案》§0.1 的更正说明**：

- 原表述「Power 5 + Clock 2 共 7 条规则」**修订为**：
  > 「Sprint 1 规则总数为 10 条：Power 3 + Clock 2 + IO 3 + Control 2。」

- **更正依据**：
  - Q3 用户逐项确认结论：「Sprint 1 补齐 7 条规则（Power 3 + Clock 2 + IO 3 + Ctrl 2）」；
  - 该表述的**数字和 = 3 + 2 + 3 + 2 = 10**，与「7 条」矛盾；
  - Sprint 0 已冻结 5 条规则，Sprint 1 在此基础上新增 5 条，**合计 10 条**；
  - 10 条为唯一能合理解释 Q3 原始表述的数量。

- **生效版本**：Sprint 1。

- **补充说明**：
  - 新增规则的具体工程条件、适用范围和判定阈值，应根据项目已有规则设计资料与 IR 能表达的信息落地；
  - 不应凭空增加无法由 IR 验证的规则；
  - 若 `CTRL_001` / `CTRL_002` 在仓库既有设计中并非「Control」语义，应以仓库现状为准调整命名，但**总数 10 条不变**。

---

## 4. Golden Case 资产：10 个（6A + 4B）

### 4.1 Case 目标

- A 类：用于规则基线、Precision/Recall 和确定性回归。
- B 类：用于 AI 辅助发现、证据链、专家推理和 AI_NER/EAR 验证。
- 复用 Sprint 0 的 `case001`，其余案例按已有命名与数据约定补齐。

### 4.2 目录清单

| Case | 类别 | 用途/资产要求 |
|---|---|---|
| `case001_buck_converter` | A | Sprint 0 既有案例；补齐标准资产并回归 |
| `case002` | A | IR、期望规则结果、评价数据 |
| `case003` | A | IR、期望规则结果、评价数据 |
| `case004` | A | IR、期望规则结果、评价数据 |
| `case005` | A | IR、期望规则结果、评价数据 |
| `case006` | A | IR、期望规则结果、评价数据 |
| `case_b01_buck_thermal_derating_B` | B | 热降额/电源相关 AI 辅助分析 |
| `case_b02_reset_rc_B` | B | Reset/RC 相关分析 |
| `case_b03_io_direction_B` | B | IO 方向/连接关系相关分析 |
| `case_b04_power_sequence_B` | B | 电源时序相关分析 |

如果仓库已有 `case002` 至 `case006` 的正式目录名，以仓库现状为准，但必须在 `data/cases/`、测试配置和文档中保持同一组 `case_id`，禁止同一案例在不同文件里使用不同 ID。

### 4.3 每个 Case 的交付资产

所有 Case 至少应包含：

- `schematic_ir.json`：符合 IR Schema；
- `info.yaml`：case_id、case_class、描述等元数据；
- `evaluation.yaml`：预期结果、期望规则命中和可评估标签；
- A 类应有明确的专家标注/期望规则结果；B 类按既有方案补齐三件套，包括 `evaluation.yaml` 与专家推理/校准材料（例如 `expert_reasoning.md`）；
- 涉及 RAG 的 B 类案例应提供可追溯知识引用或知识来源说明。

### 4.4 Case 校验

落地三个可重复运行的校验入口，名称可按仓库实际脚本调整，但功能必须齐全：

1. IR Schema 校验；
2. `expected_review` 10 字段与 Evidence 三要素校验；
3. B 类 Case 三件套与专家标注完整性校验。
4. 维护 `case_coverage_matrix.md`（或等价机器可读清单），逐 Case 记录覆盖规则、预期缺陷、预期命中/不命中、专家金标、允许出现的 AI-only 候选、必需证据引用和验收判据。

所有 10 个 Case 必须在 CI 中通过校验。B 类 Case 的专家标注需由评审专家确认后才能作为 AI_NER/EAR 统计基准。10 个 Case 的数量本身不等于覆盖充分；覆盖矩阵中的空白必须说明原因并登记风险。

> Golden Case覆盖矩阵的强制要求、A/B类Case金标约束详见 §23.5 Golden Case覆盖矩阵要求。

---

## 5. 系统架构与模块边界

### 5.1 主链路

```text
Case / Schematic IR
        │
        ▼
Rule Engine ───────────────┐
        │                   │
        ▼                   │
RAG Retriever               │
        │                   │
        ▼                   │
LLM Expert Analysis ◄───────┘
  ├─ 解释规则命中
  └─ 提出 AI 辅助发现（候选）
        │
        ▼
OutputParser / Evidence Validator
        │
        ▼
ReviewReport + review_status + discovery_source
        │
        ▼
Expert Feedback / Decision
        ├─ Rule Candidate Draft
        └─ Knowledge Gap Backlog
        │
        ▼
Benchmark V1.2 / CSV / Demo
```

### 5.2 关键模块职责

| 模块 | 职责 | 不应承担的职责 |
|---|---|---|
| IR Loader | 按 `case_id` 从 `data/cases/<case_id>/schematic_ir.json` 加载并校验 IR | 不解析真实 EDA 文件 |
| Rule Engine | 确定性执行 10 条规则 | 不调用 LLM；不执行非白名单函数 |
| RAG Retriever | 检索种子知识/可用资料并返回可追溯引用 | 不伪造来源；不把检索相关性当作缺陷成立证明 |
| LLM Provider | 统一 Mock/OpenAI/vLLM 兼容调用接口 | 不在业务层硬编码某一家模型 SDK |
| Prompt | 给出 IR、规则结果、RAG 证据与输出约束 | 不要求模型输出无法验证的确定性结论 |
| OutputParser | 结构、字段、Evidence 和状态校验；不合格时降级 | 不静默丢弃质量不佳的 Defect |
| LangGraph | 固定 6 节点串联并记录错误状态 | 不引入 Planner/ReAct 或动态自主规划 |
| Feedback Service | 保存 6 类反馈并分派到规则演进/知识缺口 | 不自动把候选规则发布到生产规则集 |
| RuleEvolutionService | 候选草稿、YAML 导出、临时 Benchmark、知识缺口导出 | 不自动批准、合并或启用规则 |
| Benchmark Service | 计算规则/AI/证据/专家采纳指标并导出 CSV | 不改变指标分母来掩盖缺失标注 |
| Task Service/API | 创建评审任务、追踪状态、返回报告、接收反馈 | 不把 UI 状态当作后端任务事实来源 |

### 5.3 LangGraph 六节点

按现有方案固定六节点 DAG；节点名称以仓库实现为准，职责如下：

1. `load_ir`：加载 IR、校验 Case。
2. `rule_check`：执行规则并保存命中结果。
3. `rag_retrieve`：检索相关知识证据。
4. `expert_analysis`：调用 LLM 解释规则命中，并提出有证据的 AI 辅助发现候选。
5. `report_generate`：解析与校验输出、计算状态、组装 ReviewReport 并落库。
6. `feedback_save`：提供反馈闭环入口；评审任务执行时可作为后续反馈保存节点/入口，避免把用户尚未提交的反馈伪装成已发生事件。

Graph State 至少包含 `case_id`、`ir`、`rule_hits`、`rag_context`、`llm_raw_output`、`parsed_defects`、`review_report`、`errors` 和任务状态等信息。异常需落日志并可追踪，不能只返回通用 500 而丢失节点信息。

> 异常容错约束：单个节点发生异常，不允许整个评审任务直接抛出500错误终止；异常信息写入Graph State的`errors`字段，组装带错误标记的ReviewReport落库并返回API，前端可以展示故障原因。

---

## 6. 数据模型、数据库初始化与迁移

### 6.1 本 Sprint 数据库策略

当前为开发阶段，允许完全重置数据库。为避免初始化 SQL 与 Alembic 形成两个事实来源，本融合版规定：

1. **Sprint 1 的完整数据库 Schema 以 Alembic migration 为唯一事实来源。**
2. 新开发环境可删除开发用 PostgreSQL 数据卷后重建，再执行 `alembic upgrade head`。
3. `infra/docker/initdb/001_pgvector.sql` 可保留用于启用 pgvector 等扩展；`002_schema.sql` 不再与 Alembic 同时创建同一批业务表。若保留该文件，应明确标注为历史基线/不自动执行，避免重复建表。
4. 不要求保留 Sprint 0 的开发数据；重置前如需复现问题，开发者自行导出必要样例数据。禁止在生产数据环境执行开发重置命令。
5. `alembic upgrade head` 与适用的 `downgrade -1` 必须有测试；若采用单一初始迁移建立完整 Sprint 1 Schema：
   - 开发环境优先使用**完整重建数据卷**方式（`down -v` + `up -d --build`）；
   - `downgrade -1` 测试**仅验证 Schema 可回滚**，**不保证兼容 Sprint 0 存量业务数据**；
   - 回滚测试按该迁移的实际可逆性设计。

### 6.2 Schema 改动

以现有 ORM、Sprint 0 Schema 和业务冻结清单核对后，至少覆盖：

- `review_result` 与 `schematic_case` 的关联字段/外键；
- `rule_candidates` 增加 `schematic_case_id`，按已确认设计移除过渡性的 `task_id`；
- `feedback_item` 反馈字段扩展，包括 `suggestion_diff`、`rule_candidate`、`suggestion_adopted` 等；
- `ir_document` 保留为 IR 快照缓存（如当前实现需要），但 Case 文件仍是评审输入的主要来源；
- 报告缺陷、反馈、规则候选之间的外键与索引；
- `review_status`、任务状态与专家判定字段的职责分离。

实际表名与字段名以 Sprint 0 已冻结的 `feedback_item` 等约定为准；不得未经确认将其改成 `feedbacks` 或其他新表名。

> `expert_decision`字段完整定义详见 §23.1。

### 6.3 IR 文件加载

- Case IR 路径：`data/cases/<case_id>/schematic_ir.json`；容器内对应 `/data/cases/<case_id>/schematic_ir.json`。
- `load_schematic_ir_by_case(case_id)` 走文件加载与 Schema 校验。
- `store_schematic_ir` 如保留，应支持 `project_id` 并明确用途；数据库中的 `ir_document` 作为快照缓存，而不是与 Case 文件并列的第二事实来源。
- 对不存在的 Case、IR 格式错误和字段缺失返回可诊断错误。
> 【ir_document快照缓存强制约束】
> 1. **业务评审执行链路：永远只读取case目录下`schematic_ir.json`，禁止从ir_document表读取评审输入数据。**
> 2. `ir_document`写入为**可选审计快照行为**：评审执行完成之后，可将本次评审使用的IR副本写入本表，仅用于事后审计回溯。
> 3. 评审任务执行阶段，禁止读取本表作为业务输入。

### 6.4 数据库验收

- 从空开发数据库执行迁移可得到完整 Sprint 1 Schema。
- ORM 模型与数据库结构一致。
- 所需外键、唯一约束、索引存在。
- 不依赖 `initdb` SQL 和 Alembic 重复创建业务表。
- 迁移脚本和测试可在 CI/开发环境重复运行。

---

## 7. LLM Provider、Prompt 与 OutputParser

### 7.1 Provider 抽象

统一接口应支持：

- `LLM_PROVIDER=mock`：CI 与确定性测试默认值；不访问外网。
- `LLM_PROVIDER=openai`：开发环境通过配置使用 OpenAI 兼容接口。
- `LLM_PROVIDER=vllm`：实现 OpenAI-compatible `base_url` 切换能力，本 Sprint 不要求实际部署 vLLM。

Provider 选择应通过配置完成，不应要求修改业务逻辑。密钥从环境变量读取，不写入代码、日志或提交到仓库。

### 7.2 Prompt 输入

`expert_analysis` 至少接收：

1. 经校验的 Schematic IR；
2. 规则命中结果；
3. RAG 检索到的证据片段及来源；
4. 分析任务与输出 Schema 约束。

Prompt 应明确区分“规则已命中问题”与“AI 辅助发现候选”，要求模型说明证据、推理、风险、修改建议和不确定性。没有证据时不得补造 Datasheet 条款、章节号或 IR 元件/网络引用。

### 7.3 ReviewReport / Defect 结构

Defect 保持既有 10 字段定义，不在本计划中另造一套字段口径；同步支持：

- `category`、`risk`（与既有 `severity`/`risk` 映射保持一致）；
- `review_status` 三态；
- `evidence[]`，每项包含 `source`、`section`、`reason`；
- 发现来源字段，如 `discovery_source` 或既有等价字段；
- 适用的 `rule_id`、`ir_ref`、`rag_ref`。

Schema 的最终字段名以仓库中已冻结的 Pydantic Schema 为准，API、数据库、前端 TypeScript 类型与测试必须统一。

### 7.4 OutputParser 校验与降级

- JSON/结构解析失败：记录原始错误摘要，按既有失败策略返回可诊断错误或空的低置信度结果，不伪装成成功报告。
- 必填字段缺失：校验失败并降级。
- Evidence 缺失或三要素不完整：不得标记为 `AI_CONFIRMED`。
- AI 新发现缺少必要 IR 证据引用：降级并标记需要专家核验。
- `review_status` 由可测试的规则决定，不允许模型自行任意指定后绕过校验器。
- 质量不合格的缺陷保留并带状态，便于专家反馈和后续 Benchmark。

### 7.5 测试要求

- Provider 的 Mock 路径覆盖成功、超时、格式错误和异常。
- OutputParser 覆盖完整输出、缺字段、无 Evidence、Evidence 不完整、引用无效、状态冲突等情况。
- CI 强制 Mock Provider，Fake Embedding；不得要求网络访问或真实 API Key。

---

## 8. RAG 与知识缺口

### 8.1 Sprint 1 目标

- 复用 Sprint 0 Seed Knowledge 与现有向量存储方案。
- 提供可切换的 Embedding 后端：CI 使用 `fake`，开发环境可按现有配置使用真实 Embedding。
- 将 Retriever 结果传入 `expert_analysis` Prompt。
- RAG 引用可回溯到来源、章节/片段或知识条目 ID。

### 8.2 知识缺口 Backlog

- `feedback_type=knowledge_gap` 时记录知识缺口。
- `RuleEvolutionService.export_knowledge_gap_backlog()` 可导出待补知识清单。
- 导出的 Backlog 至少包括关联 Case/Defect、缺失知识描述、已尝试检索内容（如有）和来源反馈。
- Sprint 1 不实现自动抓取 PDF、OCR 或自动把外部资料写入可信知识库。

### 8.3 验收

- CI Fake Embedding 下测试可重复。
- Mock LLM 能消费固定 RAG 结果并输出结构化报告。
- RAG 无结果时流程仍可继续，但必须明确标识证据不足，不得伪造引用。

---

## 9. Benchmark V1.2

### 9.1 运行模式

- **Rule Only**：只运行规则引擎，用于确定性规则基线。
- **Full Review**：运行规则、RAG、LLM、OutputParser 和报告生成，用于评估证据、AI 新发现和专家反馈。

两种模式必须在 CSV 中可区分，不能把 Full Review 的结果覆盖成 Rule Only 基线。

### 9.2 指标口径

指标公式统一写入 `docs/benchmark/metrics_definition_v1.0.md`，代码和文档不得各自定义。当前计划沿用上游 V1.2 口径：

- **EVC（Evidence Coverage）**：满足 Evidence 三要素的有效证据占比；按 Defect、Case、全局分层输出。
- **AI_NER（AI New Effective Rate）**：`N_ai_effective / (N_rule_effective + N_ai_effective)`。`N_ai_effective` 只统计 AI 提出且经专家确认有效的新发现；规则已命中的同一问题不能重复计为 AI 新发现。具体去重与状态口径按 §9.7 执行。
- **EAR（Expert Adoption Rate）**：按上游 V1.2 冻结口径计算专家采纳率；在 `metrics_definition_v1.0.md` 明确分子、分母、无反馈样本处理方式，避免与 Sprint 0 旧定义混用。
- **Rule Precision / Recall**：基于 A 类 Case 的专家标注/`evaluation.yaml` 计算。
- **状态分布**：统计 `AI_CONFIRMED`、`NEED_EXPERT_REVIEW`、`LOW_CONFIDENCE` 数量与占比。

> EAR 在不同历史方案中存在过定义差异，融合版要求开发前将最终公式写入指标定义文档并以测试锁定。若尚未完成公式确认，不得把 EAR 数值作为验收通过依据。

> AI_NER去重、重复命中处理、零分母边界，EAR多次反馈统计规则完整实现口径详见 §23.2、§23.3。

### 9.3 目标值

按照当前 Sprint 1 计划中的目标值执行：

- A 类 Rule Precision ≥ 90%；
- A 类 Rule Recall ≥ 90%；
- 全局 EVC ≥ 70%；
- 全局 AI_NER ≥ 10%；
- 全局 EAR ≥ 60%（前提是 EAR 公式、有效反馈样本口径已冻结）。

这些目标用于 Sprint 1 评估，不应通过减少分母、排除失败 Case 或把未确认 AI 输出计为有效发现来人为达标。

### 9.4 CSV 输出

输出 V1.2 标准 CSV，至少包含：

- Case 标识、`case_class`、运行模式；
- Rule Precision / Recall 及既有基线字段；
- `evd_complete_rate` / EVC；
- `ai_new_effective_rate` / AI_NER；
- `expert_adopt_rate` / EAR；
- 三态 `review_status` 计数；
- 规则命中数、AI 候选数、专家确认数、专家驳回数等可解释统计。

最终列名以 `metrics_definition_v1.0.md` 和现有 CSV 消费端统一为准；所有字段应有测试验证。

### 9.5 验收

- 10 个 Case 全部可执行；
- Rule Only 和 Full Review 可分别运行；
- CSV 列齐全、数值可计算；
- 指标的分子、分母与 Case 级结果可追溯；
- 缺少专家反馈时 EAR 应按文档规则标记为 N/A/不可计算，而不是默认 0 或 100%。

### 9.6 EAR 公式冻结（Sprint 1 最终口径）

**EAR 公式（Sprint 1 冻结版）**：
- EAR = N_adopted / N_defect_with_feedback

**采纳相关反馈定义**：

- **采纳相关反馈**（计入分母）：`correct_defect`、`suggestion_update`、`false_positive`、`false_negative`；
- **无关反馈**（不计入分母）：`knowledge_gap`、`new_rule_candidate` 等与缺陷正确性/建议采纳判断无关的反馈类型。

**分子 `N_adopted`**：

- 在**分母范围**内，该缺陷的**最新一条采纳相关反馈**明确表明：
  - `feedback_type = correct_defect`（明确确认缺陷成立），或
  - `feedback_type = suggestion_update` 且 `suggestion_adopted ∈ {yes, partial}`；
- 一条 defect 有多次采纳相关反馈时，按**最后一次采纳相关反馈**的判定为准。

**分母 `N_defect_with_feedback`**：

- 至少存在 1 条**采纳相关反馈**的 defect 数；
- **无关反馈**（如 `knowledge_gap`）**不进入分母**。

**无关反馈不覆盖原则**：

- 若某 defect 先提交 `suggestion_update`（`adopted=yes`），后提交 `knowledge_gap`：
  - 该 defect **仍在分母**；
  - 分子判定仍为"已采纳"；
  - `knowledge_gap` **不改变**分子/分母。
- 若某 defect **仅有 `knowledge_gap`** 反馈，无采纳相关反馈：
  - **不进入分母**；
  - 不参与 EAR 计算。

**无反馈样本处理**：

- 若某 Case 无任何**采纳相关反馈**，该 Case 的 EAR 标记为 `N/A`，不计入全局分子/分母；
- 若全局无任何采纳相关反馈，全局 EAR 标记为 `N/A`，不用于验收。

**与 Sprint 0 §12 的差异**：

| 维度 | Sprint 0 §12 | Sprint 1 §9.6 |
| :--- | :--- | :--- |
| 覆盖范围 | 仅 Rule Only | Rule Only + AI 新发现 |
| 分母范围 | 全部 feedback | 仅**采纳相关反馈**（排除 `knowledge_gap` 等） |
| `N_adopted` | 仅 `correct_defect` | `correct_defect` + `suggestion_update` 且 adopted |
| 无关反馈处理 | 未明确 | **不进入分母、不覆盖分子** |
| 无反馈处理 | 未明确 | 标记 `N/A`，不计入分母 |

- **以 Sprint 1 §9.6（即上游 V1.2 §8.1 口径）为准**；
- Sprint 0 §12 的 EAR 定义**已废弃**。

**目标值**：

- 全局 EAR ≥ 60%（前提是有效**采纳相关反馈**样本 ≥ 20 条）；
- 若有效采纳相关反馈样本 < 20 条，EAR 不用于验收，仅作为观察指标。

**指标文档落地**：

- 公式写入 `docs/benchmark/metrics_definition_v1.0.md`；
- 代码与文档不得各自定义 EAR；
- 增加单元测试覆盖：
  - 分子/分母计算正确性；
  - 无反馈样本 → `N/A`；
  - `suggestion_adopted=partial` 是否计入分子；
  - 多次采纳相关反馈取最后一次的判定；
  - **"先采纳建议、后提交 knowledge_gap" 场景**：分子/分母不变；
  - **"仅有 knowledge_gap 反馈" 场景**：不进入分母。

---

## 10. 反馈闭环与规则演进

### 10.1 六类反馈

沿用 Sprint 0 已冻结的六类枚举：

| `feedback_type` | 处理方向 |
|---|---|
| `correct_defect` | 确认/补充缺陷判断，更新专家决策记录 |
| `false_positive` | 记录误报，用于规则/Prompt 后续分析 |
| `false_negative` | 记录漏检，生成待评估的规则候选草稿 |
| `suggestion_update` | 保存修改建议的新旧对比 `suggestion_diff` |
| `new_rule_candidate` | 生成 `rule_candidates` 中的 proposed 草稿 |
| `knowledge_gap` | 记录并导出知识缺口 Backlog |

### 10.2 RuleEvolutionService 接口

至少实现：

- `ingest_feedback(db, fb)`：保存反馈并触发相应闭环；
- `list_candidates(db, status)`：按状态查询规则候选；
- `to_yaml_draft(candidate_id)`：导出 YAML 草稿，显著标注“自动生成/待专家审核”；
- `run_benchmark_for_candidate(candidate_yaml, case_ids)`：临时评估候选规则并输出与当前基线的 FP/Recall 变化；
- `export_knowledge_gap_backlog()`：导出知识缺口清单。

### 10.3 规则发布边界

- AI 或反馈服务只生成 `proposed` 候选草稿。
- 候选规则必须经过专家审核、测试和 Benchmark 后，才可由人工合并到正式规则集。
- 不允许反馈提交后自动启用规则。
- 候选 Benchmark 应使用临时加载机制，不污染正式规则目录或正式 Benchmark 基线。
- 候选规则临时Benchmark：候选YAML仅内存加载运行，**禁止写入正式rules业务目录**，进程退出自动丢弃，不得污染正式规则基线资产。

### 10.4 CLI 交付

沿用豆包方案的 CLI 形式，模块路径以仓库实现为准：

```bash
python -m app.services.rule_evolution_service --list
python -m app.services.rule_evolution_service --export-yaml RC-XXXX
python -m app.services.rule_evolution_service --export-knowledge-gap
python -m app.services.rule_evolution_service --bench-candidate RC-XXXX --cases case001,case_b01_buck_thermal_derating_B
```

### 10.5 验收

- 六类反馈均可校验、入库并返回可追踪 ID。
- `new_rule_candidate` 与 `false_negative` 按既定业务规则生成候选草稿。
- `knowledge_gap` 可导出 Backlog。
- YAML 草稿带待审核警告。
- 候选 Benchmark 不修改正式规则集。
- 专家对 AI 新发现的确认/驳回可用于 AI_NER/EAR 计算。

---

## 11. API 与 Web MVP

### 11.1 API 路由

在既有 FastAPI 结构下落地必要路由，路由前缀以仓库现状为准。至少覆盖：

- `POST /api/v1/review/tasks`：按 `case_id` 创建评审任务；
- 任务状态/报告查询接口；
- `POST /api/v1/feedbacks` 或项目既有 `/feedback` 等价路由：提交六类反馈；
- `GET /api/v1/cases`：列出 Case、`case_class` 与必要元数据；
- 健康检查 `healthz` 保持可用。

不要在融合过程中同时保留两套语义相同、行为不同的反馈路由。以当前代码和已冻结 API 设计确定最终路径，并在 OpenAPI、测试、前端类型中统一。

### 11.2 报告响应

报告至少可展示：

- Defect 10 字段；
- `category`、`risk`、`review_status`；
- `discovery_source` 或等价发现来源；
- Evidence 的 `source`、`section`、`reason`；
- 可用的 `ir_ref`、`rag_ref`、`rule_id`；
- 摘要指标：EVC、AI_NER、EAR 与三态分布（无法计算时明确返回 N/A/空值语义）。

### 11.3 反馈请求体

支持六类 `feedback_type`，并按类型校验相关字段：

- `suggestion_diff`：修改建议前后对比；
- `rule_candidate`：规则候选标题/描述等；
- `suggestion_adopted`：专家是否采纳建议；
- 专家对 AI 新发现的确认、驳回或需补充核验结果。

如果现有 Schema 未定义最后一项字段，开发时应补齐 Schema/API/DB/测试，并在变更记录中说明；不要仅在前端本地保存专家结论。

### 11.4 Web MVP 页面

最小可运行页面与组件：

- Case 选择页：选择已有案例并发起评审；**不做真实文件上传与上传配额**。
- 任务列表/任务状态：查看任务进度与失败原因。
- 任务详情/报告页：展示摘要指标、Defect 卡片、发现来源、Evidence Chain、三态状态。
- 专家反馈面板：提交六类反馈，并对 AI 辅助发现进行确认/驳回/需进一步核验。
- TypeScript 类型与后端 Schema 同步。

### 11.5 验收

- 能从 Web 选择 Case 并发起评审；
- 页面可查看规则发现与 AI 辅助发现，二者来源清晰；
- Evidence 三要素可见，引用可追溯；
- 专家确认/驳回结果能写入后端；
- 六类反馈提交后可在后端查询；
- 页面不出现未实现的上传数量/次数限制承诺。

> Web MVP明确不实现UI功能清单：登录权限系统、完整规则管理后台、文件上传UI界面、配额/限流页面组件。

---

## 12. 环境与依赖冻结

### 12.1 继承 Sprint 0，不改变底座

- Python 3.11.x；
- uv 0.12.6；
- PostgreSQL 15 + pgvector；psycopg v3；连接串使用 `postgresql+psycopg://`；
- venv 容器路径 `/opt/venv`；
- `PYTHONPATH=/app:/`，`DATA_DIR=/data`；
- 并列挂载方案 A：数据、脚本、infra、out 作为独立挂载点；
- `.env` 由宿主机 Compose 读取，不挂载进容器；
- PowerShell 脚本使用项目根绝对路径，`docker compose` 调用按 Sprint 0 约定使用 `cmd /c` 包装；healthcheck 使用 `127.0.0.1`，避免 `localhost` 差异。

### 12.2 新增依赖

按当前 `pyproject.toml` 实际锁文件核对 `langgraph`、`openai`、`alembic` 等依赖。方案中给出的版本范围是候选范围，最终版本应与现有依赖和兼容测试结果一致，并提交锁文件；不得只修改文档版本号而不跑测试。

### 12.3 环境变量

- `LLM_PROVIDER=mock|openai|vllm`；CI 默认 `mock`。
- `RAG_EMBEDDING_BACKEND=fake`：CI 默认。
- 模型名称、Base URL、API Key 通过环境配置提供。
- CI 禁止外网调用；未配置真实模型凭据时，测试必须可用 Mock 完成。

### 12.4 数据重置保护

开发数据库允许完全重置，但重置命令必须：

- 明确标注为开发环境专用；
- 显示会删除数据库持久化卷/数据；
- 不与生产 Compose 配置混用；
- 在 README 或脚本帮助中给出醒目警告。

---

## 13. Repository 交付清单

> 以 Sprint 0 现有目录为基础增量修改。以下为功能清单，不要求机械创建重复文件；若仓库中已有同职责文件，应复用并更新。

### 13.1 后端

- `app/llm/provider.py`：Provider 抽象及 Mock/OpenAI/vLLM 兼容实现。
- `app/llm/prompts/expert_analysis.md`：专家分析 Prompt。
- `app/llm/output_parser.py`：Defect、Evidence、状态校验与降级。
- `app/workflows/state.py`：Graph State。
- `app/workflows/review_graph.py`：固定 6 节点工作流。
- `app/rag/`：Retriever 与 Embedding 后端适配。
- `app/rules/`：10 条规则与白名单检查函数。
- `app/services/task_service.py`：任务创建、状态流转与 Graph 调用。
- `app/services/feedback_service.py`：六类反馈与入库。
- `app/services/rule_evolution_service.py`：候选规则、YAML、候选 Benchmark、知识缺口导出。
- `app/services/benchmark_service.py`：Rule Only/Full Review、EVC/AI_NER/EAR、CSV。
- `app/api/`：评审、反馈、Case、报告路由。
- `app/domain/schemas/` 或当前既有 Schema 目录：IR、Rule、Report、Feedback、Knowledge 等模型。

### 13.2 数据与文档

- `data/rules/`：共 10 条规则 YAML。
- `data/cases/`：10 个 Case（6A + 4B）。
- `docs/ai_validation/ai_reasoning_constraints.md`。
- `docs/ai_validation/ai_quality_control.md`。
- `docs/feedback/feedback_loop_spec.md`。
- `docs/benchmark/metrics_definition_v1.0.md`：冻结 EVC、AI_NER、EAR 公式与无样本处理规则。
- `docs/demo/demo_sprint1_scope.md`。
- `docs/` 中补充本 Sprint 的 AI 辅助发现与专家决策边界说明。

### 13.3 测试与脚本

- Alembic migration 与迁移测试；
- `tests/unit/test_output_parser.py`；
- 规则单元测试，包含 10 条规则与 `POWER_003`；
- `tests/unit/test_rule_evolution.py`；
- API 测试，覆盖字段和请求校验；
- Case 资产校验脚本；
- Full Review E2E 测试；
- Benchmark CSV 测试；
- `scripts/demo_sprint1.ps1`。

---

## 14. 分阶段执行计划

> 下表为执行顺序和交付物。**具体工时见 Sprint 启动后的拆分文档**；本表格不承诺固定工时。已知**初步估算约 16 人工作日**，超过原始 2 周 Sprint 容量；Sprint 启动前必须完成 P0/P1/P2 优先级裁剪，划分必须交付 / 可延后任务。

| 阶段 | 重点工作 | 交付物 | 阶段验收 | 优先级(P0/P1/P2) |
| :--- | :--- | :--- | :--- | :--- |
| A. 基线回归与数据库 | 回归 Sprint 0；统一 Alembic 为 Schema 唯一来源；更新 ORM、外键与字段 | migration、模型、迁移测试 | 空库迁移成功，Schema 一致 | P0 |
| B. Golden Case 与规则 | 补齐 10 个 Case；完成规则总数 10 条；落地 `POWER_003` | 10 套 Case 资产、10 条规则、校验脚本 | Case 校验通过，规则单测通过 | P0 |
| C. LLM 层 | Provider 抽象、Prompt、OutputParser、Evidence 与状态降级 | Provider、Prompt、Parser、单测 | Mock 测试覆盖成功/失败/证据缺失 | P0 |
| D. LangGraph | 固定 6 节点；打通 IR、规则、RAG、LLM、报告 | Graph、State、E2E 测试 | 至少一个 A 类与一个 B 类 Case 跑通 | P0 |
| E. 反馈与规则演进 | 六类反馈、候选规则草稿、知识缺口导出、候选 Benchmark | Service、CLI、单测 | 反馈闭环端到端通过 | P0 |
| F. Benchmark V1.2 | Rule Only/Full Review、指标公式、CSV | Benchmark Service、指标文档、CSV | 10 Case 输出完整指标和状态分布 | P1 |
| G. API | 评审任务、报告、反馈、Case 路由 | API、OpenAPI、API 测试 | 请求/响应字段统一且通过测试 | P1 |
| H. Web MVP | Case 选择、任务/报告、证据、专家反馈 | 页面/组件、TS 类型 | 浏览器联调完成，无虚构上传功能 | P1 |
| I. Demo 与验收 | 一键 Demo、回归全量测试、修复阻塞 | `demo_sprint1.ps1`、验收记录 | 从评审到专家反馈/候选草稿闭环成功 | P1 |
| J. 文档与风险收尾 | 更新文档、风险清单、变更记录 | 正式计划、已知问题清单 | 交付物齐全，遗留项明确 | P2 |

### 14.1 建议执行顺序

1. **先完成 §21 五项待锁定项确认**（Defect 字段 / `expert_decision` 存储 / Case 目录名 / Alembic 初始迁移 / 依赖版本），再启动 A（数据库策略和模型），并行推进 B（Case/规则资产）。
2. C 完成后先做 OutputParser 单测，再进入 D 的端到端联调。
3. E、F 应共用已冻结的专家标注与指标定义；避免指标公式尚未确定就批量跑 Benchmark。
4. G 完成后再进行 H 的 UI 联调。
5. I 只在核心测试通过后进行；Demo 失败项登记为阻塞，不以手工绕过代替验收。

### 14.2 阻塞规则

- P0：数据库不可启动、核心 IR 无法加载、规则引擎回归失败、报告无法生成；立即处理。
- P1：AI 证据链缺失、专家决策不能持久化、Benchmark 指标不可复算；不得宣布 Sprint 验收完成。
- P2：页面视觉、非关键展示优化；不影响核心链路时可记录到后续 Sprint。
- 需求边界或指标口径存在歧义时，先更新决策记录，再继续相关实现。

---

## 15. 测试策略

### 15.1 分层

1. 静态检查与格式检查；
2. Schema / IR / Case 资产校验；
3. 单元测试：规则、Provider、Parser、RAG、反馈、规则演进、Benchmark；
4. 集成测试：数据库迁移、API、Graph 节点连接；
5. E2E：Case → 规则/RAG/LLM → ReviewReport → 专家反馈 → 候选/知识缺口；
6. Demo：在干净开发数据库上运行一键脚本。

### 15.2 必测场景

- 10 条规则加载、执行与规则白名单；
- 10 个 Case 的 IR 与资产校验；
- Mock LLM 成功、超时、无效 JSON、缺字段、无证据、引用错误；
- AI 新发现与规则命中区分，重复缺陷去重或标记关联；
- 专家确认、驳回、待核验结果持久化；
- 六类反馈和候选规则/知识缺口触发；
- 空数据库迁移、重复执行迁移、ORM 与 Schema 一致性；
- Rule Only 与 Full Review 指标可复算；
- Web API 与 TypeScript 类型一致。

### 15.3 CI 约束

- `LLM_PROVIDER=mock`；
- `RAG_EMBEDDING_BACKEND=fake`；
- 不调用外网；
- 不依赖真实模型凭据；
- 测试数据与期望结果版本化；
- 不在 CI 中执行任何可能触及生产数据库的重置命令。

### 15.4 准出标准

- 所有 P0 测试通过；
- 核心单测、集成测试、E2E 测试通过；
- 10 条规则、10 个 Case 的资产清单完整；
- AI 新发现有来源标记、证据和专家决策路径；
- Benchmark 公式已冻结，CSV 字段齐全；
- Demo 脚本可从干净开发环境执行；
- 未完成项有明确责任人、影响和后续计划。

---

## 16. Demo 脚本与验收场景

### 16.1 `scripts/demo_sprint1.ps1` 流程

1. 检查 Docker Compose 服务状态和 `healthz`。
2. 确认开发数据库处于 Sprint 1 Schema 最新版本。
3. 选择一个 A 类 Case 展示规则基线。
4. 选择一个 B 类 Case 运行 Full Review。
5. 展示报告中的规则发现、AI 辅助发现、Evidence 三要素和 `review_status`。
6. 由演示人员对至少一个 AI 新发现执行专家确认/驳回操作。
7. 提交 `new_rule_candidate` 或 `false_negative` 反馈。
8. 通过 CLI 查询候选规则并导出带警告的 YAML 草稿。
9. 运行候选规则临时 Benchmark（如该候选满足可执行条件）。
10. 导出 `bench_v12.csv`，展示 EVC、AI_NER、EAR 与状态分布。

### 16.2 验收清单

- [ ] Docker Compose 服务健康，`healthz` 正常。
- [ ] Alembic 可从空数据库初始化到 `head`。
- [ ] 10 条规则加载成功，Sprint 0 既有规则回归通过。
- [ ] 10 个 Golden Case 的 IR 与资产校验全部通过。
- [ ] LangGraph 六节点完整运行。
- [ ] LLM Provider Mock/OpenAI/vLLM 兼容层按配置切换；CI 使用 Mock。
- [ ] ReviewReport 包含 10 字段、Evidence 三要素和三态质量状态。
- [ ] AI 辅助发现与规则发现能区分，AI 发现可由专家确认/驳回。
- [ ] 六类反馈全部可提交并持久化。
- [ ] 规则候选与知识缺口导出正常，候选不会自动发布。
- [ ] Benchmark V1.2 CSV 完整，指标口径与文档一致。
- [ ] API 测试通过，Web MVP 可完成核心操作。
- [ ] Demo 脚本从干净开发数据库执行成功。
- [ ] 没有把上传数量/次数限制作为已实现功能展示或验收。

---

## 17. 风险管理

| 风险 | 影响 | 缓解方式 | 验收/监控 |
|---|---|---|---|
| AI 把猜测当成缺陷 | 误报、误导专家 | 候选标记、Evidence 校验、专家决策持久化 | 无证据 AI 输出不能直接成为专家确认结果 |
| 规则发现与 AI 新发现重复 | AI_NER 失真 | 以 IR/缺陷类型/关联规则等去重或标记重叠 | Benchmark 可追溯重复处理结果 |
| B 类 Case 标注不充分 | AI_NER/EAR 不可信 | 专家审核四个 B 类 Case 的推理和评价资产 | Case 校验与专家确认记录 |
| LLM 输出不稳定 | Parser 降级比例高 | Prompt 约束、RAG 质量、Mock 回归、状态分布监控 | 输出状态 CSV 可见 |
| Schema 初始化重复 | 启动失败或表结构漂移 | Alembic 作为唯一业务 Schema 来源，避免 init SQL 重复建表 | 空库初始化测试通过 |
| 指标公式不统一 | 指标无法比较 | 指标公式文档+单测冻结 | 文档公式与代码测试一致 |
| Web 与 API 类型漂移 | 展示缺字段/误解状态 | API 合约测试与 TS 类型同步 | API/UI 联调通过 |
| 开发库重置误删数据 | 本地测试数据丢失 | 仅开发环境可用的显式重置命令、警告与备份提示 | 验证命令目标 Compose 文件 |
| 进度估算不实 | 两周内无法完成 | 启动时按团队容量重新排期，阶段交付与阻塞优先级明确 | 里程碑有负责人和日期 |

---

## 18. 常用命令与运行约定

> 以下命令延续 Sprint 0 的 Compose 文件与容器执行约定。实际执行前请确认项目根目录、服务名和脚本路径与仓库一致。

### 18.1 启动开发环境

```powershell
$f1 = Join-Path $PWD "infra/docker/docker-compose.yml"
$f2 = Join-Path $PWD "infra/docker/docker-compose.dev.yml"
cmd /c "docker compose -f `"$f1`" -f `"$f2`" up -d --build"
cmd /c "docker compose -f `"$f1`" -f `"$f2`" ps"
```

### 18.2 数据库迁移

```powershell
cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec backend uv run alembic upgrade head"
cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec backend uv run alembic current"
```

### 18.3 测试

```powershell
cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec backend uv run pytest -q"
```

根据仓库中实际的测试目录，可以将全量测试拆成单元、集成和 E2E 分阶段执行。测试命令应在 README 与 CI 配置中保持一致。

### 18.4 Benchmark 与 Demo

```powershell
cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec backend uv run python -m app.services.benchmark_service --csv /out/bench_v12.csv --metrics"
.\scripts\demo_sprint1.ps1
```

### 18.5 开发数据库完全重置

允许重置开发数据库，但本操作会删除开发数据库持久化数据。执行前确认没有指向生产环境，且 Compose 文件只引用开发数据库。

```powershell
# 仅在确认是开发环境后执行；会删除 pgdata 中的开发数据。
cmd /c "docker compose -f `"$f1`" -f `"$f2`" down -v"
cmd /c "docker compose -f `"$f1`" -f `"$f2`" up -d --build"
cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec backend uv run alembic upgrade head"
```

> 若项目还有其他命名卷，先检查 Compose 配置并确认影响范围。不得在生产环境使用 `down -v`。

---

## 19. 交付物清单

- [ ] Sprint 1 Alembic migration 与空库初始化说明。
- [ ] 10 条规则 YAML、规则测试及 `POWER_003` 降额算法。
- [ ] 10 个 Golden Case（6A + 4B）及校验脚本。
- [ ] LLM Provider 抽象、Prompt、OutputParser 与测试。
- [ ] LangGraph 六节点工作流、State 与 E2E 测试。
- [ ] RAG Retriever/Embedding 配置与知识引用。
- [ ] ReviewReport/Feedback Schema 与 API 路由。
- [ ] 专家对 AI 新发现的确认/驳回/待核验记录机制。
- [ ] RuleEvolutionService、CLI、候选 YAML、候选 Benchmark 骨架、知识缺口 Backlog。
- [ ] Benchmark V1.2 指标定义文档与 CSV。
- [ ] Web MVP 页面与 TypeScript 类型。
- [ ] `scripts/demo_sprint1.ps1`。
- [ ] 单元/集成/E2E 测试报告、风险清单和已知问题。
- [ ] 文档变更记录与 Sprint 1 验收结果。

---

## 20. 最终业务与工程冻结清单

| 冻结项 | 最终口径 |
|---|---|
| Sprint 基线 | Sprint 1 在 Sprint 0 工程底座上增量开发 |
| **规则数量** | **10 条总计**（含 Sprint 0 已有的 5 条 + Sprint 1 新增的 5 条）；Power 3 + Clock 2 + IO 3 + Control 2 |
| Golden Case | 10 个，6A + 4B |
| AI 角色 | 可解释规则命中，也可提出 AI 辅助发现候选；不替代专家最终判断 |
| 专家决策 | AI 新发现必须能由专家确认、驳回或要求进一步核验，并持久化 |
| Evidence | `source` / `section` / `reason` 三要素齐全；AI 新发现有可追溯 IR 引用，引用知识时有 RAG 来源 |
| 状态 | `AI_CONFIRMED` / `NEED_EXPERT_REVIEW` / `LOW_CONFIDENCE` 表示输出质量状态；专家业务决定独立表达 |
| Provider | `mock` / `openai` / `vllm` 兼容层；CI 默认 Mock |
| Embedding | CI 默认 Fake Embedding |
| Workflow | 固定 6 Node DAG，不引入 Planner/ReAct |
| Benchmark | Rule Only + Full Review；EVC、AI_NER、EAR 公式必须统一文档与代码 |
| 规则演进 | 只生成候选草稿；经专家审核和测试后人工发布 |
| 数据库 | 开发库可重置；Sprint 1 Alembic 是业务 Schema 唯一事实来源 |
| 前端 | 选择已有 Case、发起评审、看报告/证据、提交专家反馈 |
| 上传限制 | **上传数量、次数、频率限制不属于 Sprint 1 已实现范围** |
| 环境 | 继承 Sprint 0 Docker + uv、PostgreSQL 15 + pgvector、并列挂载方案 A |
| **AI 辅助发现** | 允许 LLM 提出规则未覆盖的候选缺陷；必须满足证据链约束；必须标记 `ai_discovered`；必须经专家确认；无证据输出降级 |
| **EAR 公式** | `EAR = N_adopted / N_defect_with_feedback`；分母限定**采纳相关反馈**（`correct_defect` / `suggestion_update` / `false_positive` / `false_negative`）；`knowledge_gap` 等无关反馈**不进入分母、不覆盖分子**；`N_adopted` 含 `correct_defect` 与 `suggestion_update` 且 adopted；无有效采纳相关反馈样本标记 `N/A`；目标值 ≥ 60% 需有效采纳相关反馈样本 ≥ 20 条 |

---

## 21. 尚需在实现前锁定的细节

这些不是重新讨论已确认的方向，而是避免实现时出现多套口径：

1. **Defect 10 字段清单**：从仓库当前冻结的 Pydantic Schema 直接引用到 `docs/ai_validation/ai_quality_control.md`，确保 Prompt、Parser、API、前端一致。【备注：该条目需要对照仓库现有代码，开发启动前确认完成，确认后从本清单移除】
2. **专家决策字段**：确认现有反馈模型能否清楚表达 AI 新发现的确认 / 驳回 / 待核验；不足则补齐 Schema、数据库和 API。倾向性建议：优先在`review_defect`增加`expert_decision`枚举字段，每条缺陷保存最终专家判定结果；`feedback_item`保存每一条历史反馈记录。避免每次计算专家判定都要遍历全部反馈记录。【备注：该条目需要对照仓库现有代码，开发启动前确认完成，确认后从本清单移除】
3. **10 个 Case 的真实目录名**：以仓库已有文件为准统一，不同时保留占位 ID 与正式 ID。【备注：该条目需要对照仓库现有代码，开发启动前确认完成，确认后从本清单移除】
4. **Alembic 初始迁移**：确定是新建一份完整 Sprint 1 初始 Schema，还是沿用已有迁移链；无论选择哪种，必须确保业务表只有一个 Schema 来源。【备注：该条目需要对照仓库现有代码，开发启动前确认完成，确认后从本清单移除】
5. **依赖版本**：对照实际 `pyproject.toml` / 锁文件和容器执行测试冻结，不能只复制文档里的候选版本范围。【备注：该条目需要对照仓库现有代码，开发启动前确认完成，确认后从本清单移除】

**确认责任与截止时间**：

| # | 确认项 | 责任人 | 截止时间 | 确认方式 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Defect 10 字段清单 | 技术负责人 | Sprint 启动后 1 个工作日内 | 对照 `app/domain/schemas/report.py` 冻结 Schema |
| 2 | 专家决策字段（`expert_decision` 存储位置） | 技术负责人 | Sprint 启动后 1 个工作日内 | 确认 §23.1 的 `review_defect.expert_decision` 方案；不足则补 Schema |
| 3 | 10 个 Case 真实目录名 | 数据负责人 | Sprint 启动后 1 个工作日内 | 对照 `data/cases/` 实际目录 |
| 4 | Alembic 初始迁移方案 | 技术负责人 | Sprint 启动后 2 个工作日内 | 确定"新建完整初始迁移"或"沿用已有链" |
| 5 | 依赖版本 | 技术负责人 | Sprint 启动后 2 个工作日内 | 对照 `pyproject.toml` / `uv.lock` + 容器测试 |

> **执行约束**：以上 5 项**必须在 §14 阶段 A 启动前完成**；未完成项不得进入下一阶段。确认完成后，各项从本清单移除，并将结论回写至对应正文。

**已完成锁定的决策（不再重复讨论）**：

| 决策项 | 结论 | 落地位置 |
| :--- | :--- | :--- |
| **Q36：AI 辅助发现是否允许** | ✅ 允许，但必须满足证据链约束并经专家确认 | §2.5 |
| **Q3'：Sprint 1 规则数量** | ✅ 10 条总计（含 Sprint 0 已有 5 条） | §3.1 / §3.4 |
| **EAR 公式** | ✅ 以上游 V1.2 §8.1 口径为准 | §9.6 |

以上事项完成后，Sprint 1 可进入按阶段执行与验收，不再保留「7 条还是 10 条规则」「AI 能否提出新发现」「EAR 公式以哪个版本为准」等方向性悬而未决问题。

---

## 22. 变更记录

| 版本 | 内容 |
|---|---|
| V1.0 | 融合 DeepSeek 与豆包两份 Sprint 1 方案；吸收用户确认的五项项目事实；统一规则数量为 10 条（含 Sprint 0 的 5 条）；明确 AI 辅助发现与专家最终决策的边界；将 Alembic 设为 Sprint 1 业务 Schema 唯一来源；明确上传数量/次数限制不在当前交付范围；整合阶段交付、测试、CLI、Web MVP、Demo 与风险管理；补充评审意见落实条款（状态分离、指标去重与样本量、Evidence 校验、Case 覆盖矩阵、容量优先级及验收测试）。**同时按用户逐项确认补充三项关键决策**：Q36（AI 辅助发现允许，追加 §2.5 及对上游《项目实施方案》§5.2 的更正）；Q3'（10 条规则清单冻结，追加 §3.4 及对上游《第一阶段工程实施方案》§0.1 的更正）；EAR 公式（以上游 V1.2 §8.1 口径为准，追加 §9.6）；§21待锁定项整理为5项，已确认决策移入新增的决策记录表。|

## 23. 评审意见落实与实现口径补充（V1.0 修订）

本节将评审意见转为可实现、可测试的约束。若本节与前文存在口径差异，以本节作为 Sprint 1 实现和验收补充；不改变 Sprint 1 的既定范围：基于 Sprint 0 增量开发、10 条规则总计、10 个 Golden Case、无真实上传及无上传数量/次数/频率限制。

### 23.1 缺陷状态与专家决定分离

`review_status` 表示系统对输出结构、证据完整性和自动质量门槛的判断；`expert_decision` 表示专家对业务结论的最终决定。两者是不同维度，禁止互相映射或混用。

| 字段 | 允许值 | 含义 | 是否等于业务确认 |
|---|---|---|---|
| `review_status` | `AI_CONFIRMED` | 自动质量门槛通过，字段/证据/引用满足校验 | 否 |
| `review_status` | `NEED_EXPERT_REVIEW` | 需要人工核验或结论仍不确定 | 否 |
| `review_status` | `LOW_CONFIDENCE` | 关键字段、证据或推理质量不足 | 否 |
| `expert_decision` | `pending` | 尚无专家最终判定 | 否 |
| `expert_decision` | `confirmed` | 专家确认该候选缺陷成立 | 是 |
| `expert_decision` | `rejected` | 专家判定该候选缺陷不成立 | 否 |
| `expert_decision` | `needs_verification` | 需要补充资料或进一步核验 | 否 |

实现建议：`review_defect` 保存当前 `expert_decision` 与最后判定时间/操作者；`feedback_item` 保留每次反馈的不可覆盖历史。专家确认/驳回/待核验必须通过后端持久化并可审计。`AI_CONFIRMED` 绝不能自动把 `expert_decision` 设置为 `confirmed`。

> ⚠️边界提示：`review_defect`归属于单次评审任务实例；如果同一case重复评审会生成全新`review_defect`记录，则本记录上的`expert_decision`仅针对本次评审输出的缺陷。业务如果需要跨多次评审持久化专家对业务问题的判定，则需要独立的业务缺陷唯一ID，本Sprint1不在范围。

### 23.2 AI_NER 去重、分母和未决样本

沿用 §9.2 的公式：

`AI_NER = N_ai_effective / (N_rule_effective + N_ai_effective)`

在 `metrics_definition_v1.0.md` 与代码测试中同时冻结以下规则：

1. **统计单位**：以同一 Case 中的“唯一缺陷实体/问题”作为计数单位；同一缺陷被重复生成、拆成多条近义描述或跨节点重复输出，只计一次。跨 Case 的相同类型问题按各 Case 的独立评审实例计数，并保留 Case ID。
2. **规则与 AI 重复**：若规则命中与 AI 候选指向同一问题，归入规则发现，不计入 AI 新发现；必须用可测试的归并键/去重逻辑，不得仅依赖字符串完全相等。
3. **AI 有效发现**：仅 `discovery_source=ai_discovered` 且 `expert_decision=confirmed`，并通过缺陷去重后，计入 `N_ai_effective`。`pending`、`needs_verification`、`rejected` 均不计入有效发现分子。
4. **规则有效发现**：按冻结的规则基线与金标匹配口径计数，规则重复命中同一问题只计一次；规则结果与 AI 结果必须采用同一缺陷归并口径。
5. **驳回候选**：作为误报/质量诊断单独统计，不进入 AI_NER 的有效发现分子；不可因被驳回而从候选总量或审计日志中删除。
6. **无有效分母**：若 `N_rule_effective + N_ai_effective = 0`，AI_NER 返回 `N/A`，不可返回 0% 或 100%。未决候选数应单独输出，不能隐式当作有效发现。
7. **可复算**：CSV/报告必须输出统计窗口、Case 集合、规则发现数、专家确认的 AI 新发现数、重复合并数、驳回数、待核验数和 N/A 原因；同一输入重复运行应得到一致结果。

### 23.3 EAR 多次反馈与样本量口径

EAR 的正式公式仍为 §9.6：`EAR = N_adopted / N_defect_with_feedback`。为避免无关反馈导致的分母污染和覆盖，进一步规定：

**统计单位**：

- 以唯一 `defect_id` 为统计单位。

**分母范围（采纳相关反馈）**：

- 分母为统计范围内，**至少存在 1 条采纳相关反馈**的唯一缺陷数；
- **采纳相关反馈**类型限定为：`correct_defect`、`suggestion_update`、`false_positive`、`false_negative`；
- **无关反馈**（`knowledge_gap`、`new_rule_candidate` 等）：
  - **不进入分母**；
  - **不覆盖**先前的采纳决定；
  - 仅作为审计历史保留。

**分子判定**：

- 在分母范围内的缺陷，其**最新一条采纳相关反馈**明确表明：
  - `correct_defect`（明确确认缺陷成立），或
  - `suggestion_update` 且 `suggestion_adopted ∈ {yes, partial}`；
- 排序依据：以服务端 `created_at` 和稳定的反馈 ID 排序，**仅在同类型采纳相关反馈之间**比较；
- 草稿、失败请求和无效反馈不参与统计。

**无关反馈不覆盖原则（关键）**：

- 若某缺陷先收到 `suggestion_update`（`adopted=yes`），后收到 `knowledge_gap`：
  - 该缺陷**仍在分母**；
  - 分子判定**仍为"已采纳"**；
  - `knowledge_gap` **不改变**分子/分母。
- 若某缺陷**仅有 `knowledge_gap`**，无采纳相关反馈：
  - **不进入分母**；
  - 不参与 EAR 计算。

**撤回语义**：

- 若最新的**采纳相关反馈**明确撤回或否定此前采纳（如 `false_positive`），按最新**采纳相关反馈**重算；
- **`knowledge_gap` 等无关反馈不构成撤回**。

**反馈历史**：

- 反馈历史必须保留；EAR 按当前最新**采纳相关反馈**计算，但可从历史记录复算历史时间点的结果。

**样本量**：

- 有效采纳相关反馈样本少于 20 个唯一缺陷时，EAR 只作为观察指标，不判定目标通过或失败；
- 达到 20 个后才应用全局 EAR ≥ 60% 的目标；
- 无采纳相关反馈或无有效分母时输出 `N/A`。

**必测场景（对应 §23.8）**：

- 同一缺陷多次采纳相关反馈，取最后一次；
- 同一缺陷最新采纳相关反馈为 `false_positive`（撤回），重算后不计入分子；
- **先采纳建议、后提交 knowledge_gap**：分子/分母不变；
- **仅有 knowledge_gap 反馈**：不进入分母；
- **仅有 false_positive 等采纳相关反馈**：进入分母，分子=0；
- 无反馈样本 / 有效样本 19/20 条边界；
- CSV、API 与报告结果一致。

### 23.4 Evidence 引用完整性与结论正确性分开验收

每条 Evidence 至少包含 `source`、`section`、`reason`；按场景提供 `ir_ref` 和/或 `rag_ref`。引用校验分两层：

1. **引用完整性（机器可测）**：`ir_ref` 必须解析到当前 Case 的真实 IR 节点、元件、网络或属性；`rag_ref` 必须解析到检索库中真实存在的文档/知识块。不存在、跨 Case、格式错误或无法定位的引用，不能通过自动质量门槛。
2. **证据相关性与工程结论正确性（需评估）**：`reason` 解释该引用为何支持此结论，而不是只复述引用 ID。引用真实存在不代表推理正确；B 类 Case 应由专家金标和测试校验结论相关性。

对于无 RAG 依据的 IR 结构问题，不强制要求 `rag_ref`；对于声称基于 Datasheet/规范的结论，应提供可追溯知识来源，否则降级为 `NEED_EXPERT_REVIEW` 或 `LOW_CONFIDENCE`。禁止生成虚构的 IR/RAG 引用。

### 23.5 Golden Case 覆盖矩阵要求

在 `data/cases/` 中维护 10 个 Case 的覆盖矩阵，至少包括：`case_id`、A/B 类别、覆盖的规则 ID、预期规则命中、预期规则不命中、专家确认的缺陷金标、允许的 AI-only 候选类型、所需 Evidence 引用、专家标注状态。

- 6 个 A 类案例主要验证已有/新增规则的正反例与回归；4 个 B 类案例验证 AI 辅助发现、证据链、专家决定和指标计算。
- 每条规则至少关联一个可执行测试；若受 IR 能力限制无法构造 Case，必须记录原因、替代单测和风险，不得声称已被 Golden Case 覆盖。
- B 类 Case 的 AI-only 候选必须由专家确认预期范围；未完成标注的 Case 可用于功能演示，但不得进入 AI_NER/EAR 的正式金标基准。
- CI 校验覆盖矩阵中的引用和 Case ID，防止目录、测试、文档之间 ID 不一致。

### 23.6 Sprint 容量与优先级裁剪

初步估算约 16 人工作日，不能在未确认团队容量的情况下默认承诺两周全部交付。Sprint 启动时由负责人按实际可用人日确认容量，并按以下顺序裁剪；不得通过跳过 P0 验收来压缩周期。

| 优先级 | 交付内容 | 处理原则 |
|---|---|---|
| P0 必须交付 | Sprint 0 回归；可重建数据库与 Alembic 迁移；10 条规则及 `POWER_003` 关键测试；AI 输出解析/证据校验/状态分离；至少一个 A 类与一个 B 类 E2E；专家决定持久化；**AI_NER / EAR 公式（含采纳相关反馈口径）及单元测试**；无真实上传路径 | 不满足不得宣布 Sprint 1 核心验收通过 |
| P1 应交付 | 全部 10 个 Case 覆盖矩阵；完整 10 Case Benchmark/CSV；六类反馈闭环；API/OpenAPI 与 Web MVP 联调；Mock LLM/Fake Embedding CI | 容量不足时由负责人明确延后项和影响，不得将未完成项标记通过 |
| P2 可延后 | 非关键页面视觉优化、非关键展示增强、额外诊断报表与便利脚本 | 记录到 backlog，不阻塞核心链路 |

实际任务板必须为每项任务记录负责人、预计人日、依赖、优先级、状态和验收证据。若团队容量不足，应缩小 P1 交付而非扩大范围；真实文件上传、上传数量/次数/频率限制仍明确不在本 Sprint 范围。

### 23.7 数据库重置与迁移验收

开发数据库可以完全重置，但 Schema 仍必须由 Alembic 迁移唯一管理。验收至少包含：

1. 在确认是开发环境后，通过显式开发脚本重置开发数据库/数据卷；脚本必须检查环境标识并二次确认，禁止在生产配置下执行。
2. 空数据库执行 `alembic upgrade head` 成功；迁移后 ORM 模型与实际表、字段、索引、外键一致。
3. 应用启动初始化可重复执行，不应每次启动时隐式重建或破坏业务数据。
4. CI 至少验证空库迁移和关键表/约束；如迁移链支持 downgrade，则验证可支持的回退路径。
5. 不允许通过手工建表或 ORM `create_all` 形成第二套 Schema 来源。

> 注意：alembic downgrade仅用于开发调试，**绝对禁止在生产环境执行downgrade回滚迁移脚本**。

### 23.8 必须加入的验收测试

- **专家决策状态矩阵**：质量门槛通过但专家决定仍为 `pending`；专家分别 `confirmed`、`rejected`、`needs_verification`；重复提交反馈保留历史且最新决定可确定。
- **发现去重**：同一缺陷被规则和 AI 同时发现时只计规则；AI 同义重复、跨节点重复只计一次；不同 Case 的独立实例按 Case 分别计数。
- **AI_NER 重算**：含确认、驳回、待核验、重复和零分母数据集；CSV、API 与报告结果一致。
- **EAR 重算**：
  - 同一缺陷多次**采纳相关反馈**、最新采纳/撤回、无反馈、有效样本 19/20 条的边界；
  - **"先采纳建议、后提交 knowledge_gap"** → 分子/分母**不因无关反馈变化**；
  - **"仅有 knowledge_gap 反馈"** → **不进入分母**；
  - **"仅有 false_positive 等采纳相关反馈"** → 进入分母，分子=0；
  - CSV、API 与报告结果一致。
- **Evidence 引用**：有效 IR/RAG 引用通过；不存在、跨 Case、损坏的引用不能通过质量门槛；真实引用但结论错误的案例应由金标测试识别。
- **干净环境 E2E**：重置开发数据库 → Alembic 迁移 → Case 校验 → 发起评审 → 生成报告 → 专家决定 → 提交反馈 → 运行 Benchmark → 导出 CSV，全流程可重复执行。

### 23.9 统一接口与代码基线

- 反馈 API 只保留一个规范路由；最终路径以仓库现状为准，并同步 OpenAPI、后端测试、前端类型和 Demo。兼容旧路径时必须明确弃用策略，不能让两条路由产生不同语义。
- 本计划中的模块路径、Case 目录名、字段名和依赖版本须在 Sprint 启动时对照真实仓库确认；以代码仓库的冻结 Schema 和依赖锁文件为准，文档不能替代代码事实。
- P0 事项全部完成并有自动化测试证据后，负责人可将本文件标记为开发冻结基线；未完成的 P1/P2 项应在任务板中明确记录，不得仅在文档中宣称完成。

**P0 冻结基线触发条件**（对齐 §23.6）：

> 说明：满足以下全部P0条件，即可标记本文件为开发冻结基线；P1、P2允许存在未完成项，记录至backlog，不阻碍Sprint1核心验收。

以下条件**全部满足**后，负责人可将本文件标记为 **开发冻结基线**：

| # | 触发条件 | 验收证据 |
| :--- | :--- | :--- |
| 1 | Sprint 0 回归通过 | 回归测试报告 |
| 2 | Alembic 空库迁移 + Schema 一致 | `alembic upgrade head` + ORM 校验通过 |
| 3 | 10 条规则加载 + `POWER_003` 关键测试 | 规则单元测试报告 |
| 4 | AI 输出解析 / 证据校验 / 状态分离 | `test_output_parser.py` 通过 |
| 5 | 至少 1 个 A 类 + 1 个 B 类 E2E 通过 | E2E 测试报告 |
| 6 | 专家决定持久化 | `expert_decision` 状态矩阵测试通过 |
| 7 | AI_NER / EAR 公式 + 单元测试 | 公式单元测试报告 |
| 8 | 无真实上传路径 | 代码审查确认 |

**未满足任一条件** → **不得标记为冻结基线**，不得宣布 Sprint 1 核心验收通过。

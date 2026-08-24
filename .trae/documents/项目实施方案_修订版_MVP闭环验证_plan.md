# 项目实施方案（修订版：第一阶段 MVP 业务闭环验证 → 第二阶段 vLLM 本地化）

> 基于 [总体技术框架.txt](file:///e:/projects/Schematic_Design_Review/总体技术框架.txt) 与原方案 [项目实施方案_第一阶段OpenAI_第二阶段vLLM.md](file:///e:/projects/Schematic_Design_Review/.trae/documents/项目实施方案_第一阶段OpenAI_第二阶段vLLM.md) 进行收敛优化修订。

---

## 1. Summary（方案定位与总目标）

### 1.1 总体定位 【保留】
长期技术路线不变：
> **EDA 数据底座 + 规则引擎 + AI Agent** 三层协同。

### 1.2 第一阶段目标 【调整】
**原目标**：打通业务闭环，EDA 用模拟数据。
**新目标**：**实现 AI 原理图评审业务闭环验证（MVP）**。

验证对象（严格按顺序）：

```
Schematic IR 输入
    ↓
确定性规则检查（只由规则发现候选问题）
    ↓
RAG 知识增强（Datasheet / Reference Design / 历史案例）
    ↓
LLM 专家分析（只做：根因分析 / 风险解释 / 修改建议）
    ↓
结构化评审报告（必须含证据链+分析过程+修改建议）
    ↓
专家反馈沉淀（正确/误报/漏检/修改建议）
```

### 1.3 第一阶段明确「不做」的范围 【新增】
为保证 MVP 聚焦可落地，以下能力 **第一阶段暂不重点实现，保留扩展接口**：
- 多 EDA 真实解析（仅留 EDA Adapter 占位）
- MCP Server 完整实现（工具以「内置原子函数」形式存在，边界对齐未来 MCP）
- 多 Agent 自主规划（只使用 LangGraph Workflow，不做 Planner/ReAct 自主决策）
- 完整知识图谱（只做 pgvector 向量检索 + 文档来源标签）
- 企业级权限体系（可做简单 API Key / 登录占位，不做 RBAC + 审计）
- 大规模规则管理平台（规则用 YAML 文件管理 + Python 执行引擎，不做 Web 端规则配置后台）

### 1.4 第二阶段目标 【保留】
在不改业务代码（或极小改动）的情况下，将 LLM 推理从 OpenAI 切换到 vLLM 本地推理服务（OpenAI Compatible），实现本地化/私有化部署。

---

## 2. Current State Analysis（当前状态）【保留】

当前仓库仅含两份架构文档：
- [总体技术框架.txt](file:///e:/projects/Schematic_Design_Review/总体技术框架.txt)
- [总体架构图.txt](file:///e:/projects/Schematic_Design_Review/总体架构图.txt)

实施从 **工程化骨架搭建 + 五类核心资产建设** 并行开始。

---

## 3. Proposed Repository Layout（目录结构与文件清单）【调整 + 新增】

```
Schematic_Design_Review/
  frontend/                          # 【保留】Vue 专家界面（MVP最小集）
  backend/                           # 【保留】FastAPI + LangGraph 后端
  infra/                             # 【保留】Nginx / Docker Compose
  docs/                              # 【调整】IR / 规则 / API 等关键文档必须化
  data/                              # 【新增】第一阶段核心资产（黄金测试集 + 规则 + 报告样例）
    cases/                           #   黄金测试集：10~20 个典型评审案例
    rules/                           #   规则定义（规则与执行引擎分离）
    knowledge/                       #   RAG 知识原文（Datasheet/RefDesign/历史案例）
    assets/                          #   IR 示例、报告示例、Feedback 示例
  .env.example
  README.md
  总体技术框架.txt
  总体架构图.txt
```

### 3.1 frontend/（前端交互层）【保留 + 收敛】

```
frontend/
  package.json
  vite.config.ts
  tsconfig.json
  index.html
  src/
    main.ts                          # P0 入口
    App.vue                          # P0
    api/
      http.ts                        # P0 axios/fetch 封装
      review.ts                      # P0 任务 API
      feedback.ts                    # P0 反馈 API
    pages/
      UploadPage.vue                 # P1 上传/发起评审（MVP 可简化为传 case_id）
      TaskListPage.vue               # P1 任务列表
      TaskDetailPage.vue             # P0 展示报告：规则缺陷 + AI 研判 + 证据链
      FeedbackPanel.vue              # P0 内嵌于 TaskDetail：确认/误报/漏检/修改建议
    components/
      DefectCard.vue                 # P0 单条缺陷卡片（含 severity、rule_id、evidence、root_cause、suggestion）
      EvidenceChain.vue              # P0 证据链展示（RAG 引用片段 + 来源）
      RuleBasisTag.vue               # P1 规则依据展示
    types/
      review.ts                      # P0 前端 ReviewReport/Defect/Feedback 类型
```

**收敛说明**：移除第一阶段不做的「知识库管理页」，改为聚焦报告查看 + 反馈。

### 3.2 backend/（后端：AI 业务调度层）【保留 + 调整 + 新增】

```
backend/
  pyproject.toml / requirements.txt  # P0
  requirements-dev.txt               # P1
  Dockerfile                         # P0
  app/
    main.py                          # P0 FastAPI 入口
    __init__.py

    core/
      config.py                      # P0 环境变量 + LLM Provider 选择 + DB
      logging.py                     # P0 结构化日志（含 task_id/trace_id）
      security.py                    # P2 简单鉴权占位

    api/
      router.py                      # P0 v1 路由聚合
      deps.py                        # P0 依赖注入（DB Session）
      v1/
        health.py                    # P0 /healthz
        review.py                    # P0 任务：创建/状态/报告
        feedback.py                  # P0 专家反馈：确认/误报/漏检/建议
        cases.py                     # P0 黄金测试集：列出 case_id、加载 case
        knowledge.py                 # P2 知识导入（MVP 可脚本化导入，不放 Web）

    domain/
      schemas/                       # 【调整】从单文件拆分，业务模型齐全
        task.py                      # P0 TaskCreate/TaskStatus/TaskOut
        ir.py                        # P0 Schematic IR Schema（Pydantic）
        rule.py                      # P0 RuleDef / RuleResult
        report.py                    # P0 ReviewReport / Defect / Evidence
        feedback.py                  # P0 Feedback（correct/fp/fn/suggestion）
        knowledge.py                 # P1 DocChunk / RetrievalResult
      enums.py                       # P0 Severity/DocumentType/FeedbackType

    persistence/
      db.py                          # P0 SQLAlchemy engine/session
      models.py                      # P0 Task / ReportDefect / Feedback / DocChunk
      migrations/                    # P1 Alembic（MVP 第一阶段可 create_all 替代）

    ir/                              # 【增强】Schematic IR 资产化
      schema.py                      # P0 IR Pydantic 模型（Design/Component/Pin/Net/Attribute/Hierarchy）
      schema.json                    # P0 输出的 JSON Schema（供外部校验）
      serializer.py                  # P0 load_ir / dump_ir
      examples/
        sample_buck.json             # P0 Buck 样例 IR
        sample_ldo.json              # P0 LDO 样例 IR
        sample_xtal.json             # P0 晶振样例 IR
      validator.py                   # P1 IR 校验工具（字段合规/引用一致性）

    adapters/
      eda/
        __init__.py
        case_source.py               # P0 从 data/cases 加载 IR（取代原 stub_source，更贴近真实）
        stub_source.py               # P2 随机/固定 IR 生成
        cadence_placeholder.py       # P2 第二阶段 Cadence 接入口占位
        api_placeholder.py           # P2 第二阶段 EDA API 接入口占位

    rules/                           # 【新增 + 调整】规则定义与执行引擎分离
      engine.py                      # P0 规则执行引擎：加载 rule_defs → 匹配 applicable → 执行 check
      registry.py                    # P0 规则注册中心（按 rule_id 查找）
      builtin/                       # P0 内置 check 函数（Python）
        power_checks.py              # P0 电源（去耦电容 / 电压等级匹配 / 电源地完整性占位）
        clock_checks.py              # P0 时钟（晶振负载电容 / 端接占位）
        interface_checks.py          # P0 接口（UART TX/RX 交叉 / CAN 终端电阻占位）
        param_checks.py              # P1 参数/封装/安规（可逐步补）
      rule_defs/                     # P0 规则定义（YAML，规则不硬编码）
        POWER_001.yaml               #   rule_id / rule_name / applicable_condition / check_logic / severity / rule_basis / suggestion
        POWER_002.yaml
        CLOCK_001.yaml
        IFACE_001.yaml
        IFACE_002.yaml
        PARAM_001.yaml

    rag/                             # 【调整】收敛 RAG 范围 + 来源标签
      ingest.py                      # P1 文档入库（Markdown/PDF→chunk→embedding→pgvector）
      retriever.py                   # P0 检索：按 rule_id/component/net 召回，返回 source+引用
      chunking.py                    # P1 切分策略
      sources.py                     # P0 DocumentType 定义：datasheet/reference_design/review_case
      examples/
        seed_knowledge.md            # P0 种子知识（几条 Datasheet/RefDesign/案例摘要片段，用于跑通 RAG）

    llm/
      provider.py                    # P0 抽象：chat / embeddings
      openai_provider.py             # P0 OpenAI（可配 base_url）
      vllm_provider.py               # P0 兼容层（vLLM OpenAI Compatible，走 base_url）
      prompts/
        expert_analysis.md           # P0 Prompt：「规则候选 + RAG 引用 + IR 片段」→「根因+风险+建议+证据链」
        expert_analysis_schema.md    # P0 强制 LLM 输出 JSON Schema（ReviewReport/Defect）
      output_parser.py               # P0 LLM 输出校验：结构化 JSON 解析 + 字段补全/兜底

    skills/                          # 【保留 + 命名向 Workflow Node 对齐】
      load_ir.py                     # P0 Node1: load_ir
      deterministic_check.py         # P0 Node2: deterministic_check
      knowledge_retrieve.py          # P0 Node3: knowledge_retrieve
      expert_analysis.py             # P0 Node4: expert_analysis（根因/风险/建议）
      report_generate.py             # P0 Node5: report_generate
      feedback_save.py               # P0 Node6: feedback_save

    workflows/                       # 【调整】固定 Workflow，不做 Agent Planner
      state.py                       # P0 Graph State：task_id / ir / rule_results / rag_contexts / llm_analysis / report
      review_graph.py                # P0 固定 6 个 Node，顺序执行，保留未来扩展分支

    services/
      task_service.py                # P0 任务创建、状态流转、线程池执行
      report_service.py              # P0 报告融合与落库
      case_service.py                # P0 黄金测试集加载/查询
      benchmark_service.py           # P1 验收指标计算：准确率/漏检率/误报率/采纳率

    utils/
      ids.py                         # P0
      time.py                        # P0

  tests/                             # 【新增】与黄金测试集联动
    unit/
      test_rules_engine.py           # P0 规则引擎：对 10~20 case 跑，比对 expected_review
      test_ir_schema.py              # P0 IR JSON Schema 校验
    e2e/
      test_review_flow_cases.py      # P0 用 data/cases 端到端跑，出 Benchmark 指标
    conftest.py
```

### 3.3 infra/（网关 + 一键运行）【保留】

```
infra/
  nginx/
    nginx.conf                       # P0 / → 前端；/api → 后端
  docker/
    docker-compose.yml               # P0 nginx + backend + postgres(pgvector)
    docker-compose.vllm.yml          # P0 vLLM 覆盖文件（第一阶段先写好，第二阶段启用）
    initdb/
      001_pgvector.sql               # P0 CREATE EXTENSION vector
```

### 3.4 docs/（关键文档必须化）【调整】

```
docs/
  ir/
    schematic_ir_schema.md           # P0 IR Schema + 字段说明 + JSON 示例
    test_case_format.md              # P0 黄金测试案例格式规范（case_id/ir/expected/comment）
  rules/
    rule_format.md                   # P0 规则 YAML 格式 + 示例
    rule_list.md                     # P0 规则清单（rule_id/名称/依据/风险等级）
  reports/
    review_report_schema.md          # P0 ReviewReport JSON Schema + 每条 Defect 字段说明
  api/
    openapi.md / openapi.yaml        # P1 可由 FastAPI 自动生成
  prompts/
    prompt_governance.md             # P1 Prompt 变更记录与版本化策略
  benchmark/
    metrics_definition.md            # P0 缺陷识别准确率/漏检率/误报率/采纳率定义
```

### 3.5 data/（五类核心资产 + 黄金测试集）【新增 · 第一阶段必须同步建设】

```
data/
  cases/                                              # P0 资产 2：黄金测试集（10~20 案例）
    case001_buck_converter/
        info.yaml                                     #   原始设计信息：名称/版本/场景/关键器件
        schematic_ir.json                             #   IR 输入
        expected_review.json                          #   专家评审结果（ReviewReport 格式）
        expert_comments.md                            #   文字版评审思路、参考依据、注意事项
    case002_ldo/
        info.yaml
        schematic_ir.json
        expected_review.json
        expert_comments.md
    case003_crystal_osc/
        ...
    case004_uart_loopback/
        ...
    case005_can_bus/
        ...
    case006_ddr3_interface/
        ...
    case007_mcu_peripheral/
        ...
    case008_power_sequence/
        ...
    case009_esd_protection/
        ...
    case010_reset_circuit/
        ...
    # 建议：Buck/LDO/晶振/UART/CAN/DDR/MCU外围/电源时序/ESD/复位 各 ≥ 1 个

  rules/                                              # P0 资产 3：规则定义（与 backend/app/rules/rule_defs 同源，可做软链或 CI 同步）
    POWER_001.yaml
    POWER_002.yaml
    CLOCK_001.yaml
    IFACE_001.yaml
    IFACE_002.yaml
    PARAM_001.yaml

  knowledge/                                          # P1 资产：RAG 原始知识
    datasheets/
      xxx_dcdc_datasheet.md                           #   （可节选关键页 / 摘要片段）
    reference_designs/
      buck_ref_design.md
    review_cases/
      case_buck_fp1.md                                #   专家历史误判/正确案例文本

  assets/                                             # P0 资产 1、4、5 的 Schema + 示例
    ir_schema.json                                    #   Schematic IR JSON Schema
    review_report_schema.json                         #   ReviewReport JSON Schema
    feedback_schema.json                              #   Feedback JSON Schema
    sample_review_report.json                         #   一份完整报告示例
    sample_feedback.json                              #   一份反馈示例
```

---

## 4. 五类核心资产设计（第一阶段同步建设）【新增】

> 这是 MVP 成功的关键：**代码 + 数据资产必须并行交付**。

### 4.1 资产 1：Schematic IR 标准（P0）
- 文件：
  - `backend/app/ir/schema.py`（Pydantic）
  - `data/assets/ir_schema.json`（JSON Schema 导出）
  - `docs/ir/schematic_ir_schema.md`（字段说明 + JSON 示例）
  - `docs/ir/test_case_format.md`（黄金案例格式规范）
- 必含实体：`Design / Component / Pin / Net / Attribute / Hierarchy`
- 每个字段要求：类型 / 是否必填 / 说明 / 示例值

### 4.2 资产 2：黄金测试集（P0，10~20 案例）
路径：`data/cases/caseXXX_xxx/`
每个 case 必含：
  - `info.yaml`：case 元信息
  - `schematic_ir.json`：IR 输入（通过 IR Schema 校验）
  - `expected_review.json`：专家评审结果（必须满足 ReviewReport JSON Schema，含每条缺陷的 location/severity/rule_id/root_cause/suggestion）
  - `expert_comments.md`：评审思路、注意事项（用于 Prompt/RAG/训练数据）
必覆盖类型：Buck / LDO / 晶振 / UART / CAN / DDR / MCU 外围 / 电源时序 / ESD / 复位（≥ 10 条，P0）；建议扩充到 15~20 条（P1）。

### 4.3 资产 3：规则库（P0）
- 路径：`data/rules/*.yaml`（与 `backend/app/rules/rule_defs/` 保持一致）
- 每条规则 YAML 必含字段：
  - `rule_id`（如 POWER_001）
  - `rule_name`
  - `applicable_condition`（可基于 component_type/net_name/attribute 等匹配）
  - `check_logic`（`builtin.power_checks.check_decoupling_cap` 这样指向内置 check 函数 + 参数）
  - `severity`（low/medium/high/critical）
  - `rule_basis`（引用哪条规范/哪页 Datasheet）
  - `suggestion`（默认修改建议模板，可被 LLM 重写/优化）
- 规则定义与执行引擎完全分离：Python 只实现原子 check 函数，规则组合靠 YAML。

### 4.4 资产 4：评审报告模型（ReviewReport）（P0）
- JSON Schema：`data/assets/review_report_schema.json`
- Pydantic：`backend/app/domain/schemas/report.py`
- 文档：`docs/reports/review_report_schema.md`

每条 Defect 必须字段：
```jsonc
{
  "defect_id": "DEF-20260701-00001",
  "location": {
    "sheet": "PAGE1",
    "path": "U1.C8 / Net:VCC_3V3",
    "coords": {"x": 100, "y": 200}
  },
  "component": "C8",
  "net": "VCC_3V3",
  "severity": "high",
  "rule_id": "POWER_001",
  "evidence": [
    { "type": "rule_hit", "detail": "U1 下未检测到 100nF 去耦电容" },
    { "type": "rag_ref", "source_type": "datasheet", "source_title": "XXX_DC-DC_DS.pdf", "page": 12, "snippet": "建议在 VIN 引脚近处放置 100nF 陶瓷电容" },
    { "type": "ir_ref", "detail": "Component[U1].Pins[VIN] 连接到 Net[VIN_12V]" }
  ],
  "root_cause": "输入电源缺少高频去耦，可能引起纹波过大与开关噪声",
  "suggestion": "在 U1 VIN 引脚 1cm 以内补加 100nF X7R 0603 陶瓷去耦电容 Cx 到 GND。",
  "confidence": 0.92
}
```
> 强制 AI 输出必须具备：**证据链 evidence + 根因 root_cause + 修改建议 suggestion**。

### 4.5 资产 5：专家反馈数据（P0）
- JSON Schema：`data/assets/feedback_schema.json`
- Pydantic：`backend/app/domain/schemas/feedback.py`
- 必含字段：
```jsonc
{
  "feedback_id": "FB-xxx",
  "task_id": "...",
  "defect_id": "DEF-xxx",              // 对单条缺陷的反馈（或对整条报告 feedback_scope=report/defect）
  "feedback_scope": "defect",
  "feedback_type": "correct" | "false_positive" | "false_negative",  // 正确 / 误报 / 漏检
  "expert_suggestion": "建议改为...",
  "commented_by": "engineer_A",
  "commented_at": "2026-07-01T00:00:00Z",
  "attached_refs": [                    // 可附 Datasheet 页码 / 截图引用 / 历史案例编号
    {"type": "datasheet", "title": "...", "page": 15, "remark": "..."}
  ]
}
```

---

## 5. AI 评审流程（LLM 定位收敛）【调整】

### 5.1 原流程 【保留】
`规则 → RAG → LLM`

### 5.2 第一阶段禁止行为 【新增】
- ❌ 禁止直接把整份原理图/整份 IR 丢给 LLM 自由分析；
- ❌ 禁止 LLM 独立「发现电路问题」；所有候选问题必须来源于 **确定性规则检查输出**（`rule_results`）。

### 5.3 LLM 定位（设计专家解释助手）【调整】
对**每条规则命中的候选缺陷**，LLM 仅负责：
1. **根因分析**（为什么这是个问题？机理是什么？）
2. **风险解释**（不修改会怎样？影响哪些指标/场景？）
3. **修改建议**（给出可执行、可落到器件选型/连线的建议）
4. **证据链整合**（把 rule_hit + RAG 引用 + IR 片段整理成 `evidence[]`）

### 5.4 具体数据流 【新增】
```
Schematic IR
  ↓
[Node2] deterministic_check  →  rule_results[]（每条含 rule_id/location/severity）
  ↓
[Node3] knowledge_retrieve：
    按每条 rule_result 的 rule_id + component + net 检索 pgvector
    每个检索结果打标签 document_type ∈ {datasheet, reference_design, review_case}
    并返回 source_title/page/snippet（确保可追溯）
  ↓
[Node4] expert_analysis（LLM，每条 rule_result 独立调用或批量调用）
    Prompt 上下文由 4 部分组成：
      1) Schematic IR 相关子图片段（只给该缺陷涉及的 component/net/pin）
      2) rule_result（rule_id/applicable_condition/check_logic/severity/rule_basis）
      3) RAG 检索结果（带来源标签 + 页码 + snippet）
      4) ReviewReport JSON Schema 强制输出约束
    输出：每条 rule_result → 一条 Defect（含 evidence/root_cause/suggestion/confidence）
  ↓
[Node5] report_generate：
    把所有 Defect 组装成 ReviewReport，落库 reports/defects 表
```

---

## 6. RAG 设计（范围收敛 + 来源可追溯）【调整】

### 6.1 第一阶段 RAG 支持范围 【新增】
**优先支持（P0/P1）**：
- Datasheet（器件手册摘要/关键章节）
- Reference Design（官方参考设计说明）
- 历史评审案例（专家历史误判/正确判断 + 结论）

**暂不追求（P2）**：
- 大规模「设计规范库」全量入库（可先用 Markdown 摘要的形式提供关键规范条目）

### 6.2 知识来源标签 【新增】
每段 `DocChunk` / 每条检索结果必须带：
- `document_type ∈ { datasheet, reference_design, review_case, design_standard }`（第一阶段前三个用得最多）
- `source_title`（文件名/文档名）
- `source_page` / `source_section`（页码/章节，可追溯）
- `snippet`（原文片段，用于「证据链」展示）

### 6.3 检索策略（MVP 简化）【调整】
第一阶段不过度追求复杂检索：
- 对每条 rule_result：`rule_id 关键词 + component_type + part_number + net_name` 拼接 query
- 向量相似度 TopK（K=3~5） + 简单重排
- LLM Prompt 中把 `document_type / source_title / source_page / snippet` 都带进去，强制 LLM 在 `evidence[]` 中引用这些来源。

---

## 7. LangGraph 设计（固定 Workflow 6 节点）【调整】

### 7.1 不做的事 【新增】
- 不做 Planner / ReAct / 多 Agent 自主规划；
- 不做循环重试（除非明确需要 retry，MVP 先不做）；
- 不做人在回路（先做完再给专家在 Web 上反馈）。

### 7.2 固定 Node（严格 6 步）【新增】
Graph State（`workflows/state.py`）建议字段：
```python
{
  "task_id": str,
  "case_id": str | None,               # 若从黄金测试集跑
  "ir": SchematicIR | None,            # Node1 产出
  "rule_results": list[RuleResult],    # Node2 产出
  "rag_contexts": list[RetrievalResult],  # Node3 产出
  "llm_analysis": list[Defect],        # Node4 产出
  "report": ReviewReport | None,       # Node5 产出
  "error": str | None,
}
```

节点顺序（固定 DAG，无分支）：
1. **Node1 load_ir**：从 case_source（case_id）或上传（MVP 先 case_id）加载 IR；校验 IR Schema；写入 state.ir
2. **Node2 deterministic_check**：调用 rules/engine 跑所有匹配规则；写入 state.rule_results
3. **Node3 knowledge_retrieve**：对每条 rule_result 召回 RAG；写入 state.rag_contexts（带 document_type/source_*）
4. **Node4 expert_analysis**：按 Defect JSON Schema 调用 LLM 生成结构化根因/风险/建议；校验输出；写入 state.llm_analysis
5. **Node5 report_generate**：组装 ReviewReport；落库 tasks + report_defects；写入 state.report
6. **Node6 feedback_save**（专家反馈链路入口）：本 Node 在「用户提交反馈」时单独跑或在 Task 提交时把 feedback 条目落库（与报告异步）

> 【保留】未来扩展点：LangGraph 边界已留，后续可以插入 Planner / Agent 节点，不动现有 Node。

---

## 8. 第一阶段实施里程碑（重新规划）【新增】

### 阶段 0：数据准备（与代码并行启动，P0 必做）
**目标**：五类核心资产先有最小集，让代码能「有东西可跑 + 可验收」。
- 交付物：
  - Schematic IR JSON Schema + 字段说明文档 + 3~5 个样例 IR（Buck/LDO/晶振/UART/CAN 各一个）
  - 黄金测试集：先出 10 个 case（覆盖 P0 必测类型），每个 case 的 `schematic_ir.json + expected_review.json + expert_comments.md`
  - 规则库：先出 6 条 YAML 规则（POWER_001/002、CLOCK_001、IFACE_001/002、PARAM_001）
  - ReviewReport / Defect / Feedback JSON Schema + 示例文件各一份
  - RAG 种子知识：5~10 条 Markdown 摘要片段（datasheet / reference_design / review_case 都要有）

### 阶段 1：核心评审链路（P0 必做）
**目标**：在没有 Web 的情况下，用 `data/cases` 端到端跑通并产出 Benchmark。
- 交付物：
  - Backend 骨架 + main.py 启动
  - ir/ 加载/校验 + rules/ 执行引擎 + 6 条内置 check 函数
  - LLM Provider 抽象（OpenAI + vLLM 兼容层 base_url）+ Prompt + 强制 JSON 输出解析
  - LangGraph review_graph 6 Node 跑通
  - `tests/e2e/test_review_flow_cases.py`：10 个 case 端到端跑，对比 expected_review
  - `benchmark_service.py` 输出：准确率 / 漏检率 / 误报率（基于 expected_review 计算）
  - PostgreSQL 基础表：tasks / report_defects / feedback / doc_chunks

### 阶段 2：Web 界面（P1）
**目标**：工程师能点选 case 发起评审、看报告、做反馈。
- 交付物：
  - FastAPI 路由：`/review/tasks`（POST/GET）、`/feedback`（POST/GET）、`/cases`（GET 列表/单个 case）
  - Frontend：UploadPage（简化为 case 选择） + TaskDetail（报告 + 证据链 + 反馈面板）
  - Nginx + Docker Compose 一键启动

### 阶段 3：知识增强（P1）
**目标**：把 RAG 真正接进来，报告里出现「引用 Datasheet 页码/RefDesign/历史案例」。
- 交付物：
  - 文档 ingest 脚本/命令行（PDF→chunk→embedding→入库，MVP 支持 Markdown + 简单 PDF 文本提取）
  - retriever 与 expert_analysis Prompt 打通
  - 报告 evidence[] 中出现 RAG 引用且可追溯（document_type + source_title + source_page + snippet）

### 阶段 4：真实 EDA 接入（P2，第一阶段不强制）
**目标**：替换 case_source，从真实 Cadence 工程 / EDA API 产出 Schematic IR。
- 交付物：
  - `cadence_adapter.py` 或 `eda_api_adapter.py` 首个可工作版本
  - 与现有 load_ir Node 无缝切换（同一 EDA Adapter 接口）

---

## 9. 验收与业务指标（新增 AI 效果验收）【新增】

### 9.1 可运行性验收（基础，P0 必过）
- Docker Compose 一键启动：`nginx / backend / postgres` 全部 healthy
- `GET /api/v1/healthz` 返回 ok
- 通过 Web/API 跑 10 个黄金 case，都能得到 ReviewReport（HTTP 200 + 字段齐全）
- 提交专家反馈 → DB feedback 表有记录 → 任务详情页能看到反馈

### 9.2 业务验收（AI 效果，P0 必测、P1 指标达标）
- 指标定义文档：`docs/benchmark/metrics_definition.md`
- 指标（基于 `expected_review.json` 计算，每条缺陷匹配用 rule_id+location 近似对齐）：
  - **缺陷识别准确率（Precision）**：报告缺陷中被专家判定 correct 的占比
  - **漏检率（False Negative Rate）**：专家期望存在但系统未报告的缺陷占比
  - **误报率（False Positive Rate）**：系统报告但被专家判定 false_positive 的占比
  - **专家采纳率**：报告的 `suggestion` 被专家标记为采纳/部分采纳的比例
- 输出形式：`tests/e2e/test_review_flow_cases.py` 跑完成后输出 CSV / Markdown 表格，列出每个 case 的 4 项指标 + 总体均值。

> 第一阶段不设硬阈值，但要求**指标可被持续追踪**；后续版本再设定 ≥ 85% 采纳率等目标。

---

## 10. 运行环境与部署 【保留】

### 10.1 保留项
- FastAPI（P0）
- LangGraph（P0，仅 Workflow 模式）
- Schematic IR（P0）
- Rule Engine（P0，YAML + Python 执行引擎）
- RAG（P1，pgvector）
- LLM Provider 抽象（OpenAI + vLLM 切换，P0）
- Docker Compose 部署（P0）
- PostgreSQL 15/16 + pgvector（P0）

### 10.2 Python / Node 版本 【保留】
- Backend：Python 3.11
- Frontend：Node.js 20 LTS
- 数据库：PostgreSQL 15/16 + pgvector

### 10.3 环境变量（P0）【保留】
- `APP_ENV=dev|prod`
- `DATABASE_URL=postgresql+psycopg://...`
- `LLM_PROVIDER=openai|vllm`
- `LLM_BASE_URL=...`
- `LLM_MODEL_NAME=...`（如 `gpt-4o-mini` 或本地 `Qwen2.5-14B-Instruct`）
- `OPENAI_API_KEY=...`（第一阶段必须）
- `EMBEDDING_PROVIDER=openai|...`
- `MAX_WORKERS=...`
- `DATA_DIR=/app/data`（挂载黄金测试集/规则/知识目录）

### 10.4 第二阶段 vLLM 切换 【保留】
通过 `LLM_PROVIDER=vllm` + `LLM_BASE_URL=http://vllm:8000/v1` 切换；启用 `docker-compose.vllm.yml` overlay；业务代码零改动。

---

## 11. 第一阶段实际开发优先级（P0 / P1 / P2）【新增】

### P0（必须完成，否则 MVP 不成立）
- 阶段 0 数据资产：
  - IR Schema + 文档 + 5 个样例 IR
  - 10 个黄金测试 case（info + ir + expected_review + expert_comments）
  - 6 条规则 YAML + 对应的内置 check 函数
  - ReviewReport / Feedback JSON Schema + 示例文件
  - RAG 种子知识（5~10 条摘要片段，用于 Node3/4 联调占位）
- 阶段 1 代码：
  - Backend 工程骨架 + main.py + healthz
  - `ir/`：schema / serializer / validator / examples
  - `rules/`：engine + registry + 6 条内置 check
  - `llm/`：provider 抽象 + OpenAI 实现 + vLLM 兼容层 + 强制 JSON 输出 parser
  - `llm/prompts/expert_analysis.md` + `*_schema.md`（严格禁止 LLM 自由发挥）
  - `workflows/`：state + review_graph（6 Node 顺序）
  - `persistence/`：DB 模型（tasks/report_defects/feedback/doc_chunks）+ create_all
  - `services/benchmark_service.py`：四项指标计算
  - `tests/e2e/test_review_flow_cases.py`：跑 10 case，输出指标 CSV
- Docker：`infra/docker/docker-compose.yml` + `.env.example`

### P1（建议在第一阶段完成，便于真正被工程师使用）
- Web 界面：Frontend + FastAPI 路由（任务/反馈/cases）
- Docker Compose：Nginx + 前端构建集成
- RAG ingest + retriever 真实跑通：doc_chunks 表入库 + 检索 + 报告 evidence[] 含 RAG 引用
- 黄金测试集扩充到 15~20 个 case；加 DDR/高速接口、电源时序、ESD、复位等
- Alembic 迁移；日志 / 错误处理增强

### P2（后续扩展，第一阶段保留接口）
- 真实 EDA 接入（Cadence 解析 / EDA API）
- MCP Server 抽离（把内置工具改为独立 MCP 服务）
- 复杂 Agent / 多轮规划 / 人在回路
- 规则管理后台 + 知识图谱 + 权限体系
- 更高质量 PDF 解析 / OCR / Table 提取

---

## 12. Assumptions & Decisions 【调整】

- 已确认：
  - 第一阶段部署：Docker Compose【保留】
  - 存储：PostgreSQL + pgvector【保留】
  - 规则引擎：后端内嵌库【保留】
  - 第一阶段 EDA：模拟数据 / 占位接口【调整为「黄金测试集 case_source + 数据资产」，更贴近真实】
- 新增决定：
  - LLM 第一阶段**不直接发现缺陷**，只做「解释助手」；候选缺陷 100% 来自规则命中
  - 报告/缺陷/反馈强制 JSON Schema 输出 + 可追溯证据链
  - 五类核心资产（IR、黄金测试集、规则库、ReviewReport 模型、Feedback 模型）与代码同步 P0 交付
  - 验收除接口通、容器起，额外必须跑 10 case 输出四项业务指标（Precision / FN / FP / 采纳率）

---

## 13. Verification Steps（实施后验证步骤清单）【调整 + 新增】

### 13.1 资产验证（P0）
- IR Schema：用 `backend/app/ir/schema.py` 对 `data/cases/*/schematic_ir.json` 全量校验通过
- 规则库：`rules/engine.py` 对 10 case 执行后，输出 rule_results 条目数 ≥ 每 case 期望值（见 `expected_review.json`）
- ReviewReport JSON Schema：报告输出 100% 通过 `data/assets/review_report_schema.json` 校验
- Feedback Schema：`sample_feedback.json` 通过校验

### 13.2 端到端 + 业务指标（P0）
- `pytest tests/e2e/test_review_flow_cases.py`：
  - 10/10 case 成功产出 ReviewReport
  - 控制台输出 metrics.csv：每个 case 的 Precision / FN Rate / FP Rate / Adopt Rate + 整体均值
  - 任意抽查 2~3 条 Defect：evidence[] 至少含 1 条 rule_hit + 1 条 rag_ref/ir_ref
  - 任意抽查 2~3 条 Defect：root_cause / suggestion 非空且非模板套话

### 13.3 Web 验证（P1）
- Docker Compose up 后：
  - 前端打开首页，选择 case → 创建任务 → 轮询到 finished → 看到报告（每条缺陷卡片含证据链）
  - 对 1 条缺陷提交「误报 + 建议」→ 刷新后反馈面板可见
  - `/healthz` 正常，日志含 task_id 结构化字段

### 13.4 vLLM 切换（第二阶段，P0 先验证配置路径可用）
- `.env` 修改 `LLM_PROVIDER=vllm` + `LLM_BASE_URL=http://vllm:8000/v1`
- 启动 `docker-compose.vllm.yml` overlay（或本地 vLLM），跑 1~2 个 case，报告结构不变、字段齐全

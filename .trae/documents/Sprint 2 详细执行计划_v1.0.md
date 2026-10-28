# Sprint 2 融合版详细执行计划 V1.3

> **文档状态：正式执行基线（V1.3 冻结）。**
> **项目：AI 原理图评审 · 证据链 · 闭环演进系统（第一阶段 MVP）**
> **版本：V1.3**
> **日期：2026-10-27**
> **融合依据：** `Sprint2_详细执行计划_V1.0.md`（完整 MVP 范围）与 `Sprint 2 详细执行计划_v1.0.md`（工程执行细节）。
>
> 本文档是 Sprint 2 的统一执行基线。若旧版两份计划与本文冲突，以本文冻结结论为准。本文不代表功能已经实现；验收项须以代码、测试和实际演示结果为准。

---

## 0. 执行摘要

Sprint 2 的目标不是只做技术接口或演示页面，而是完成一个**可供硬件评审工程师实际操作的产品闭环**，在已有 Sprint 1 技术骨架上补齐业务工作流、资产管理、规则治理、知识缺口处理、指标验证、导出与交付能力。

闭环主路径：

1. 选择或管理 Golden Case；
2. 发起单 Case 或批量评审；
3. 查看 AI 评审报告及可追溯 Evidence；
4. 对缺陷进行专家最终判定，并提交过程反馈；
5. 将反馈转成规则候选或知识缺口；
6. 人工审核、编辑、Benchmark 规则候选；
7. 经人工明确批准后，将候选发布到正式规则库并升级版本；
8. 重新执行 Benchmark，比较发布前后结果；
9. 处理知识缺口、维护知识库；
10. 查看指标、导出报告与批量结果；
11. 通过 CI、回归测试和完整 Demo 验证交付质量。

### 0.1 本 Sprint 必须交付的产品功能

以下功能**全部纳入 Sprint 2**，不得以"留待 Sprint 3"作为默认裁剪结论：

- 任务状态、筛选、详情指标与专家判定交互；
- 规则候选完整审核与正式规则库发布；
- 知识缺口 Backlog 管理；
- Case 管理、资产预览与单 Case Benchmark；
- 评审报告 Markdown/PDF 导出；
- 批量评审与 Benchmark CSV 导出；
- RAG 知识库简易管理；
- 指标聚合看板；
- CI、资产校验、回归、Demo、用户/运维/开发文档；
- vLLM OpenAI 兼容接口验证与报告。

允许功能在首版存在体验、边界或性能不足，但必须有可运行的最小实现、明确验收标准和已知问题记录。不得把"页面存在"当作"功能完成"。

### 0.2 不纳入本 Sprint 的边界

- 真实 EDA 原理图解析与 Cadence 适配器；
- 外部用户上传原理图文件；
- PDF/OCR 自动解析；
- MCP Server、多 Agent 自主规划；
- 知识图谱；
- 企业级 RBAC 和完整用户身份体系；
- 拖拽式可视化规则编辑器（本 Sprint 采用受控 YAML 文本编辑）。

---

## 1. 冻结的业务决策与指标原则

### 1.1 规则批准必须进入正式规则库

规则候选审核是产品闭环的关键环节。Sprint 2 必须实现：

- 候选列表、详情、来源反馈与 Case 追溯；
- YAML 预览与编辑；
- 候选 Benchmark；
- 人工批准、驳回与理由记录；
- **批准后发布到正式规则库**：更新正式规则资产、数据库中的规则定义及版本信息；
- 发布后执行回归 Benchmark，验证规则变更效果。

安全约束：

1. **禁止系统自动批准或自动发布规则**；发布必须由用户明确执行批准动作。
2. 发布前必须校验 YAML 语法、必需字段、规则 ID 唯一性、规则结构与可执行性。
3. 发布操作须具备备份、原子写入/事务补偿、版本记录及失败恢复机制，避免文件与数据库状态不一致。
4. 发布后运行相关规则单测和 Golden Case 回归；若验证失败，应清楚显示失败原因，并保留恢复到上一正式版本的能力。
5. 候选状态与正式规则版本状态分离。`accepted` 不得被解释为"发布一定成功"；必须明确区分"已批准、发布中、已发布、发布失败"。
6. 正式规则目录以仓库既定 `data/rules/` 为准；实际路径、数据库同步策略和版本机制在编码前核查现有实现，不得另造第二套事实来源。

### 1.2 Seed 与真实指标严格分层

Seed 数据是**产品研发和调试工具**，目的是在没有专家参与时验证反馈写入、指标聚合、UI 空值/非空值展示和规则候选工作流。Seed 不是专家证据，不能冒充真实业务效果。

必须区分以下两类结果：

| 类型 | 用途 | 是否可作为真实业务指标 |
|---|---|---|
| Seed / simulated 数据 | 开发调试、接口和 UI 验证、流程演示 | 否 |
| 真实专家反馈与判定 | 评估实际专家采纳、AI 有效发现表现 | 是，且须满足指标定义中的样本门槛 |

具体要求：

- 每条模拟数据必须可识别（如 `simulated=true` 或等价标记），并能从真实指标聚合查询中排除。
- Seed 验收只要求证明：数据生成/清理可重复、聚合函数按定义工作、无样本/样本不足时返回正确空值、前端正确显示 `N/A`。
- Seed 结果不得用于宣称"真实 EAR 达标"或"真实 AI_NER 达标"。
- 当真实专家样本不足时，指标显示 `N/A · 样本不足` 或等价提示，并显示有效样本量；不得以 Seed 补足门槛。
- `metrics_definition.md` 是指标公式、样本定义、过滤条件和空值语义的唯一事实来源；Sprint 2 更新文档版本时必须同步测试与 UI 说明。
- 真实指标的业务验收和 Seed 技术验收分开记录。没有真实专家数据时，不阻塞产品研发闭环，但不得把"真实业务指标已达标"列为通过项。

### 1.3 任务状态与评审质量状态分离

- `task_status` 表示任务执行结果；
- `review_status` 表示 AI 对缺陷/结论的质量状态；
- `expert_decision` 表示专家对单条缺陷的最终判定；
- `feedback_item` 表示专家反馈过程记录。

四者不可互相替代。专家判定接口不得改写 AI 的 `review_status`，也不得隐式创建一条反馈记录。

**Sprint 2 单任务状态冻结为 `pending / done / failed`，不包含 `running`。** Sprint 1 的 `create_task` 是同步执行：调用返回 `task_id` 时评审通常已经结束，因此 `running` 对当前 API 不可观测。Sprint 2 不主动写入无法被用户观察的 `pending`，也不为显示中间态而重复生成 `review_result`。

`pending` 仅作为 ORM `server_default` 兜底值，用于覆盖极端场景（如迁移过程中断、Graph 中途崩溃导致未走到落库节点）；**Sprint 1 的 `review_result` 表没有 `task_status` 列，不存在历史 pending 数据**，Sprint 2 也不主动写入 pending。新建任务必须可靠记录 `done/failed`。若 Sprint 3 引入异步队列，再单独设计 `running`、轮询与重试语义。

### 1.4 批量评审执行模型

Sprint 2 不默认引入 Celery 或外部队列。首选受最大 Case 数限制的同步批量接口，并在前端明确提示同步执行与超时风险；不得在客户端超时后自动重跑整个批次。

**批量上限默认值冻结为 5 个 Case，但必须由 S2-A 基线压测确认**。压测方法：对 5 个 Case 的批量接口连续执行 3 次，记录 P95 耗时；若 P95 > 120 秒，降至 3 个 Case；若 3 个 Case 的 P95 仍 > 120 秒，**必须在进入 UI 开发前提交后台执行架构决策**（不进入 UI 开发，不得静默降级为"批量退化为单 Case"）。

**批量接口必须使用独立超时（建议 180 秒），不复用 Sprint 1 `http.ts` 的 20 秒单任务超时**。前端批量页面在超时后必须提示"任务可能仍在执行"并提供 `batch_id` 查询入口，不得提示"任务未创建"。

**批量中失败的 Case 不产生 `review_defect`，不可提交缺陷级反馈，但可提交 Case 级反馈**（如 `false_negative` 标记整体漏检）。`feedback_item.review_defect_id` 在 Sprint 1 中已为 `Optional`（允许 null），无需改表。

不得仅为了避免超时而随意启动无管理的内存线程。内存后台线程若被采用，必须记录其进程重启会丢失任务、无跨进程协调和不具备持久队列等限制，并将持久化队列作为后续改进项。

### 1.5 指标计算规则

- 指标公式、分母/分子、去重方式、样本门槛以 `docs/benchmark/metrics_definition.md` 为准。
- EVC、真实 AI_NER、真实 EAR 与 `review_status` 分布在详情接口聚合后返回。
- 真实反馈是异步产生的，指标采用实时聚合，避免静态列过期；如项目现有架构确有缓存，应明确失效策略。
- 无有效样本、样本不足或分母为 0 时按定义返回 `null`/`N/A`，不得伪造 0%。
- Seed/simulated 数据在真实指标查询中必须被排除。
- Benchmark 结果须记录数据来源/运行模式，避免模拟值和真实模型结果混淆。

**AI_NER 样本门槛冻结（V1.3 新增）**：Sprint 1 的 `metrics_definition.md` v3.1 只冻结了 EAR ≥ 20 条样本门槛（§9.6），**AI_NER 的样本门槛未冻结**。Sprint 2 必须在 `metrics_definition.md` v3.2 中补冻结 **AI_NER 样本门槛 ≥ 10 条 `ai_discovered` 缺陷**（与 EAR 的 20 条区分），并写入代码测试与 `summary_metrics.sample_info.ai_ner_sample_threshold` 的返回值。若 S2-A 基线核查发现 AI_NER 门槛确实无法冻结，必须在附录 C 第 9 项登记原因，不得以 `null` 含糊带过。

---

## 2. Sprint 1 基线与兼容要求

### 2.1 开发前基线核查

执行前记录实际 Git HEAD、数据库迁移版本、测试结果、Benchmark 结果。两份旧计划记录的 HEAD 和测试数字不同，不能直接把其中任一数字当成当前事实。

- [ ] 记录当前 Git commit 与工作区状态。
- [ ] 执行全量非 integration 测试并记录结果。
- [ ] 按项目既定方式执行 LLM integration Benchmark；记录数据来源、TP/FP/FN 和运行时间。
- [ ] 验证现有 Case、规则 YAML、知识库资产和历史评审数据。
- [ ] 核对 `models.py`、Alembic 迁移、API schema、服务层和前端 TypeScript 类型。
- [ ] 将基线结果写入 `sprint2_changelog.md`，不覆盖历史记录。

### 2.2 向后兼容

- 保留 Sprint 1 已有 API 的路径和原有字段语义；新增字段采用向后兼容方式。
- 不以 `errors` 字段推测任务状态；任务状态应以持久化的 `task_status` 为准。
- `GET /review/tasks` 原有排序和分页行为保持稳定；新增筛选应在总数计算和分页之前生效。
- 专家判定更新必须校验 defect 属于指定 task。
- 报告快照类型与实时 API 缺陷类型分开维护，避免把不同来源/用途的字段混成一个类型。
- 数据库迁移须可从 Sprint 1 基线升级；不得删除正式 Case、规则、知识库或历史业务数据。
- **bench 基线不可退化（V1.3 新增）**：Sprint 2 结束时，真实 LLM integration Benchmark 必须保持 Sprint 1 收口的 `tp=16/fp=0/fn=0`（或经基线核查后的等价基线），否则视为严重回归。此项写入 §8.2 准出清单。

---

## 3. 功能范围与验收矩阵

| ID | 功能 | Sprint 2 最小交付 | 主要验收证据 |
|---|---|---|---|
| F01 | 任务状态与筛选 | 状态持久化、列表按状态筛选、失败原因可查看 | 迁移/服务/API 测试 |
| F02 | 专家判定 | 确认/驳回/待核验；审计字段；错误回滚 | PATCH 测试、前端异常场景 |
| F03 | 真实指标与 Seed | 实时聚合；Seed 可重复生成/清理；模拟数据排除 | 指标单测、样本边界测试 |
| F04 | 规则候选管理 | 列表、详情、反馈来源、YAML 编辑、Benchmark、批准/驳回 | API/UI/服务测试 |
| F05 | 正式规则发布 | 人工批准后发布正式 YAML、同步规则定义与版本、失败恢复 | 发布测试、回归 Benchmark、发布记录 |
| F06 | 知识缺口 Backlog | 列表、关联任务/缺陷/反馈、状态处理、跳转知识库入口 | API/UI/服务测试 |
| F07 | Case 管理 | 10 个 Golden Case 列表、元数据/资产预览、单 Case Benchmark、历史结果 | UI/API/资产校验 |
| F08 | 报告导出 | Markdown/PDF，保留缺陷字段、Evidence、风险、根因、建议 | 导出文件与字段核对 |
| F09 | 批量评审 | 多选内部 Case、批量执行、状态/错误反馈 | E2E 与部分失败测试 |
| F10 | Benchmark CSV | 批量结果导出、指标与运行来源明确 | CSV 内容测试 |
| F11 | RAG 知识库管理 | Markdown 片段导入/查看/启停；从知识缺口跳转 | API/UI/服务测试 |
| F12 | 指标看板 | EVC、AI_NER、EAR、质量状态分布、候选/知识缺口统计 | API/UI 与空值测试 |
| F13 | CI 与资产治理 | 迁移检查、资产校验、规则测试、Mock 回归、非 integration 测试 | CI 运行记录 |
| F14 | vLLM 兼容验证 | OpenAI 兼容配置跑通可运行的 Golden Case 集，输出报告 | 配置、日志、适配报告 |
| F15 | 文档与 Demo | 一键演示、用户/部署运维/开发文档、变更记录 | 从干净环境按文档复现 |

> "完成"意味着最小工作流可以真实执行并有自动化或可复现的验收证据。若某项存在已知缺陷，必须登记影响、绕行方案和后续修复项，不得直接标记为无缺陷完成。

---

## 4. 后端、数据与 API 设计

### 4.1 数据库与迁移

基于现有 ORM 做增量变更，编码前先核查字段和约束，避免重复建字段/表。

重点任务：

1. `ReviewResult.task_status`：增加状态字段、索引和合法值约束；为历史记录制定安全回填策略。
2. `RuleCandidate`：补齐候选状态、拒绝理由、Benchmark 结果与时间戳；如已有字段则复用，不重复迁移。
3. `FeedbackItem.rule_candidate_ref`：删除候选时反向置空软引用，避免悬空引用。
4. `RuleDefinition`：正式规则发布时同步 YAML 内容、启用状态和版本信息；定义文件与数据库一致性校验。
5. `KnowledgeDoc/KnowledgeChunk`：复用现有结构支撑 Markdown 知识条目导入、查看、启停；确有缺项才新增字段。
6. `KnowledgeGap`：**Sprint 2 采用独立持久化表**，独立管理状态、来源追溯、解决时间、处理人和备注，并按实际关系关联来源 `FeedbackItem`、Case/task/defect。不得仅用 `FeedbackItem.feedback_type='knowledge_gap'` 代替 backlog 状态管理。
7. 补齐 `updated_at` 触发器、外键/索引和数据库约束；提供可重放迁移链。
8. 提供仅清理测试/Seed 数据的脚本，必须保留正式资产和真实业务记录。清理脚本必须默认 dry-run；执行真实删除必须显式传入 `--confirm-delete`，并要求明确指定目标数据集/环境。脚本执行前输出将删除的记录数与范围；禁止无参数全库清理，禁止根据模糊名称匹配删除。为删除行为增加测试，验证正式记录不受影响。

验收：

- [ ] 从 Sprint 1 数据库升级到最新 Schema 成功。
- [ ] 新库初始化与升级路径均通过测试。
- [ ] 历史数据回填后可正常查询。
- [ ] 删除候选不会留下无效反馈引用。
- [ ] 规则文件与数据库同步检查可发现不一致。

### 4.2 任务状态、列表和详情指标

建议接口契约：

- `GET /api/v1/review/tasks?status=done&limit=20&offset=0`
- `GET /api/v1/review/tasks/{task_id}` 新增 `summary_metrics`
- `PATCH /api/v1/review/tasks/{task_id}/defects/{defect_id}` 更新专家判定

`summary_metrics` 字段契约冻结如下，前后端共享同一 schema，并同步写入 `docs/benchmark/metrics_definition.md` v3.2：

```json
{
  "evidence_coverage": 0.5542,
  "ai_new_effective_rate": null,
  "expert_adoption_rate": null,
  "review_status_distribution": {},
  "sample_info": {
    "ear_sample_size": 12,
    "ear_sample_threshold": 20,
    "ai_ner_sample_size": 0,
    "ai_ner_sample_threshold": 10
  }
}
```

`sample_info` 中样本数仅统计符合指标定义的真实数据；Seed/simulated 不计入。**EAR 阈值（20）与 AI_NER 阈值（10）均由后端指标定义输出，前端不得硬编码**。指标为 `null` 时，UI 显示 `N/A` 和具体样本原因（`N/A · 样本不足` / `N/A · 待专家反馈`）。

`task_status` 与 `review_status` 使用不同字段名和不同枚举。非法筛选值返回 422；任务不存在返回 404；`summary_metrics` 不得改变旧字段行为。

### 4.3 专家判定 PATCH

请求体示例：

```json
{
  "expert_decision": "confirmed",
  "expert_decision_by": null
}
```

- `expert_decision` 仅允许 `pending / confirmed / rejected / needs_verification`。
- `expert_decision_by` 在没有用户系统时允许 `null`，不得伪造真实用户 ID。
- 服务层校验任务存在、缺陷存在且属于该任务。
- 更新 `expert_decision`、`expert_decision_updated_at`、`expert_decision_by`。
- 不改写 `review_status`；不隐式创建 `feedback_item`。
- API 错误语义：不存在/不属于该任务 404；枚举或请求体不合法 422。

### 4.4 规则候选 API

路由命名统一使用 kebab-case，并沿用项目既有 `/api/v1` 前缀：

- `GET /api/v1/rule-candidates?status=...`
- `GET /api/v1/rule-candidates/{candidate_id}`
- `PUT /api/v1/rule-candidates/{candidate_id}`（编辑 YAML/元数据）
- `POST /api/v1/rule-candidates/{candidate_id}/benchmark`
- `POST /api/v1/rule-candidates/{candidate_id}/approve`
- `POST /api/v1/rule-candidates/{candidate_id}/reject`

批准流程与发布流程必须在业务上清晰可见。若采用单个 approve 端点完成"批准并发布"，响应必须明确返回发布状态、版本号和回归结果；若采用分步操作，则需避免批准后无人知道是否已发布。前后端、API 文档、测试统一使用同一命名。

Benchmark 结果应持久化，至少记录：候选版本/规则哈希、运行时间、Case 范围、运行来源、指标、错误与状态。重复查看页面不得无提示地重复调用 LLM。

### 4.5 知识缺口、Case、导出与批量接口

#### KnowledgeGap 数据模型（冻结）

Sprint 2 新增独立 `KnowledgeGap` 表，建议字段至少包括：

- `id` / `gap_id`；
- `source_feedback_id`（指向来源 `FeedbackItem`，Sprint 2 冻结为一对一：一条 `feedback_type='knowledge_gap'` 反馈最多生成一个缺口，`source_feedback_id` 加唯一约束；一对多留 Sprint 3，若需要则改为中间表 `gap_feedback_link(gap_id, feedback_id)`）；
- 可选的 `case_id`、`task_id`、`defect_id` 关联；
- `title`、`description` / `note`；
- `status`：`open / in_progress / resolved / wontfix`；
- `created_at`、`updated_at`、`resolved_at`；
- `resolved_by`（无用户系统时允许 `null`）。

禁止把 backlog 状态仅保存在 JSONB/comment 中。

建议 API 形态（编码前与现有路由核对，避免重复）：

- `GET /api/v1/knowledge-gaps`
- `PATCH /api/v1/knowledge-gaps/{gap_id}`
- `GET /api/v1/cases`
- `GET /api/v1/cases/{case_id}`
- `POST /api/v1/cases/{case_id}/benchmark`
- `GET /api/v1/cases/{case_id}/benchmarks`
- `GET /api/v1/review/tasks/{task_id}/export?format=md|pdf`
- `POST /api/v1/cases/batch-review`
- `GET /api/v1/benchmark/batch` 或项目已有等价查询接口
- `GET /api/v1/benchmark/batch/export?format=csv`
- `GET /api/v1/dashboard/summary`
- 知识库 Markdown 条目的列表、导入、启用/禁用接口

接口名是统一设计目标，不代表当前仓库中这些路由已存在。实现前先盘点已有接口，能复用则复用。

### 4.6 正式规则发布事务与恢复

**V1.3 发布状态机与恢复策略（优先于本节其他概括性措辞）：**

- 状态为 `approved → publishing → published | publish_failed`；获批不代表已发布。
- 执行顺序：临时文件校验 → 备份正式文件 → 同目录原子替换 → DB 事务同步 → 规则引擎内存重载 → 状态落为 `published`。只有文件、DB、重载均成功才标记 `published`。
- **关键语义（V1.3 明确）**：文件原子替换后、DB 同步提交前，**规则引擎不得重载**（引擎不自动监听文件，内存仍是旧规则）。这样「文件成功、DB 失败」时，引擎状态与 DB 一致（都是旧规则），恢复备份文件即可，**不需要「引擎回滚」**。
- DB 同步失败时恢复备份文件；重载失败时恢复文件与 DB 到上一一致版本并记录补偿结果。
- 发布后 Benchmark 回归失败**不自动回滚**，保留变更和证据，要求人工选择接受当前版本或执行受控回滚。
- Sprint 2 不自动执行 `git commit`；由人工审查提交，CI 校验规则文件与 DB 定义一致。

若文件系统与数据库无法使用同一个原子事务，必须实现补偿/恢复步骤，并为"文件成功、数据库失败"和"数据库成功、文件失败"分别编写测试。

---

## 5. 前端交付范围

### 5.1 任务列表与详情

- 任务列表支持按 `task_status` 筛选，保留原排序和分页。
- 详情页显示 AI 评审结果、Evidence、`review_status`、专家判定与审计信息。
- 缺陷卡片提供"确认 / 驳回 / 待核验"操作。
- 操作可采用乐观更新，但失败时必须回滚并展示可理解错误。
- `expert_decision_by=null` 显示"操作者未记录"；无审计信息时不显示空白占位行。
- 摘要卡片显示 EVC / AI_NER / EAR / 状态分布及样本可用性。
- 指标为空时显示 `N/A` 和原因，不把 `null` 渲染成 0%。

### 5.2 反馈表单

- `suggestion_diff` 使用旧建议/新建议的结构化输入，不要求用户编辑裸 JSON。
- `rule_candidate` 使用标题、描述、严重程度提示等结构化表单。
- 根据反馈类型动态显示相关字段；其他反馈类型不渲染无关表单。
- 提交失败保留用户输入并展示错误；成功后明确提示结果。
- 反馈过程记录与缺陷最终专家判定保持分离。

### 5.3 规则候选管理页面

必须支持：

- 候选列表与状态筛选；
- 候选详情、来源反馈、关联 Case；
- YAML 预览/编辑和校验反馈；
- 运行候选 Benchmark、查看结果与正式基线对比；
- 人工批准/驳回、驳回理由；
- 发布状态、正式规则版本、发布失败信息；
- 发布后重新执行相关 Benchmark，并展示变化。

### 5.4 知识缺口、Case、知识库页面

- 知识缺口页：列表、来源追溯、状态更新、跳转知识条目处理。
- Case 页：展示 Golden Case 元数据和资产预览，包括 `evaluation.yaml`、`expert_reasoning.md`、Evidence 相关附件（按实际资产存在情况显示）。
- Case 详情支持单 Case Benchmark 与历史运行结果。
- 知识库页：Markdown 条目导入、查看、启用/禁用；不做 PDF/OCR。
- 批量评审：选择内部 Case、启动任务、查看每个 Case 的成功/失败状态、处理部分失败并导出 CSV。

### 5.5 UI 与 CLI 双通路

核心运维/业务操作应尽量同时提供 UI 与 CLI 等价入口，至少包括：Seed 生成/清理、Case 资产校验、Benchmark 执行、规则 YAML 校验、规则发布前检查、回归 Benchmark、知识库 Markdown 导入/启停和 Demo。CLI 必须复用同一 Service 层，不能另写一套业务逻辑。无法在 Sprint 2 提供等价 CLI 的操作须登记遗留项及原因；高风险操作（规则发布、真实数据清理）仍须有明确确认步骤。

### 5.6 资产变更审计（P1）

有容量时提供简易审计记录，至少记录规则/知识条目的对象 ID、变更前后哈希或 diff 快照、操作者（无身份系统时允许 null）、时间、动作、结果和关联版本。此项为 P1，不得挤占规则发布安全、知识缺口持久化或 Seed 清理防护的测试资源。

### 5.7 导出与指标看板

- 报告导出支持 Markdown 和 PDF，保留缺陷字段、`review_status`、`expert_decision`、Evidence 三要素、风险、根因、建议及任务元信息。
- PDF 不得通过截图代替结构化报告；导出失败需有错误提示。
- 看板展示 EVC、真实 AI_NER、真实 EAR、质量状态分布、专家判定分布、规则候选与知识缺口统计。
- 所有统计显示统计范围、运行来源和有效样本量；Seed 数据应显著标记且不能混入真实业务指标。

---

## 6. CI、测试与资产治理

### 6.1 CI 组成

建议至少拆为以下工作：

1. 后端单元/服务/API 测试；
2. 前端 type-check 与构建；
3. Alembic 迁移与 Schema 检查；
4. Case 资产校验；
5. 规则 YAML 语法/结构校验；
6. Mock LLM + Fake Embedding 的 Golden Case 回归；
7. Benchmark 指标比较与退化告警；
8. 发布流程、知识缺口、导出、批量任务的关键回归；
9. vLLM 适配验证作为独立、可配置工作，不因外部服务未配置而让普通 PR CI 不稳定。

真实 LLM integration 测试不必每次默认在 CI 中运行，但必须有明确的手动/定期运行命令、报告和失败处理流程。Mock 回归不能替代真实模型表现验证。

### 6.2 必须覆盖的测试

- [ ] 任务状态落库、筛选、历史数据兼容。
- [ ] 专家判定合法更新、非法枚举、任务/缺陷不存在、缺陷不属于该任务。
- [ ] 前端 404、422、超时、重复点击、刷新持久化、乐观更新回滚。
- [ ] 指标无样本、分母为 0、样本不足、Seed 排除、真实数据计算。
- [ ] AI_NER 样本门槛（≥ 10 条 `ai_discovered`）生效：门槛不足时返回 `null`，达到门槛时返回数值。
- [ ] 规则候选 YAML 校验、Benchmark 结果持久化、批准/驳回。
- [ ] 规则发布状态机、文件原子替换、DB 同步、引擎内存重载、发布失败补偿、回归失败人工接受/回滚。
- [ ] Seed/测试数据清理脚本默认 dry-run；无 `--confirm-delete` 不删除任何数据。
- [ ] 正式规则发布成功、规则 ID 冲突、写文件失败、DB 更新失败、回归失败、恢复上一版本。
- [ ] 知识缺口关联与状态更新；一对一约束生效。
- [ ] Case 资产缺失、无效资产、单 Case Benchmark。
- [ ] Markdown/PDF 导出字段完整；批量 CSV 列名、行数与来源正确。
- [ ] 批量评审全部成功、部分失败、全部失败。
- [ ] 批量评审同步超时（前端 20 秒 vs 后端 180 秒）→ 前端提示「任务可能仍在执行」+ 提供 `batch_id` 查询入口。
- [ ] RAG 条目导入、查看、启用/禁用。
- [ ] 全量非 integration 测试、前端 type-check/build、Mock Golden Case 回归。

### 6.3 Case 资产校验

CI 至少校验：

- IR/Case 资产结构；
- `expected_review.json` 必需字段；
- B 类 Case 所需资产组合与引用完整性；
- 规则 ID 与 Case 引用一致；
- 每条规则至少有可执行测试或明确的替代测试/风险说明；
- 资产、测试、文档之间的 Case ID 不漂移。

---

## 7. 分阶段执行顺序与依赖

以下是依赖顺序，不是固定工时承诺。原详细执行计划的 11.5 人日仅覆盖较窄的任务集合，不能直接套用到本融合版全范围。

| 阶段 | 工作包 | 主要交付 | 依赖 |
|---|---|---|---|
| S2-A | 基线回归、Schema/ORM、状态持久化、迁移与数据清理 | 基线记录、迁移、状态测试 | Sprint 1 当前基线 |
| S2-B | 专家判定 PATCH 与审计字段 | API/Service/Schema 测试 | S2-A |
| S2-C | 任务筛选、详情聚合、真实指标函数 | API、聚合测试、指标口径说明 | S2-A、S2-B 可并行部分 |
| S2-D | 前端专家判定交互 | 缺陷卡片、错误回滚、审计行 | S2-B |
| S2-E | 摘要卡片、反馈表单增强、指标空值语义 | Task Detail UI | S2-C |
| S2-F | 规则候选 UI、Benchmark 持久化、正式规则发布 | 候选审核到发布的完整流程 | 数据/服务层基线、规则引擎核查 |
| S2-G | 知识缺口 Backlog、Case 管理、单 Case Benchmark | 资产与知识闭环页面/API | 反馈/Case/知识模型盘点 |
| S2-H | 报告导出、批量评审、批量 CSV、RAG 简易管理、指标看板 | 业务补齐功能 | 相关 Service/API |
| S2-I | CI、资产校验、Mock 回归、vLLM 验证 | CI 与适配报告 | 各功能稳定 |
| S2-J | Demo、文档、全量验收与遗留项 | 可复现交付包 | 全部阶段 |

### 7.0 S2-A 首个交付物：基线核查报告

S2-A 的第一项任务必须产出 `docs/sprint2/baseline_audit.md`。在报告通过评审前，不进入后续依赖性较强的 Schema/发布流程开发。报告至少核查：

1. 当前 Git HEAD、工作区状态、全量测试与 Benchmark 基线；
2. `task_status` 当前 ORM/迁移/历史数据情况；
3. `RuleCandidate`、`FeedbackItem`、`KnowledgeDoc/KnowledgeChunk` 实际字段；
4. 正式规则加载路径、DB/YAML 同步与规则引擎重载机制；
5. 当前候选状态机、规则校验和 CLI 能力；
6. `metrics_definition.md` 当前版本、EAR/AI_NER 公式、阈值与 Seed 过滤字段；
7. 已存在的 Case、Benchmark、导出、批量、知识库和看板 API/服务；
8. 前后端路径、Pydantic Schema 与 TypeScript 类型的一致性；
9. `metrics_definition.md` 中 AI_NER 样本门槛是否已冻结；若未冻结，S2-A 必须补冻结（默认 ≥ 10 条 `ai_discovered`）。

若核查结果与计划假设不符，先修订本计划和验收契约，再编码；不得为了遵守计划而重复建模或覆盖已有实现。

### 7.0.1 功能优先级与容量红线

| 优先级 | 范围 | 交付规则 |
|---|---|---|
| P0 必须完整交付 | 基线核查、迁移/兼容、任务状态、专家判定、Seed/真实指标隔离、规则候选审核与正式发布、知识缺口 Backlog、Case 管理、Markdown 报告导出、批量评审与 CSV、RAG 简易管理、指标看板、CI/Mock 回归、Demo 与核心文档 | 必须满足该功能的 API/服务、前端/CLI 入口、测试、文档四件套；若某功能本质无 UI，则提供等价 CLI/API 操作说明 |
| P1 最小可用交付 | PDF 导出、资产变更审计日志、额外 UX、部分非关键自动化 | 必须给出最小可运行实现或经负责人显式批准延期；延期项登记到遗留清单，不得声称已完成 |
| P2 条件性增强 | 更复杂的批量异步队列、完整用户身份/RBAC、额外运营分析与自动化 | 本 Sprint 不作为承诺项；如有剩余容量再做 |

容量估算规则：

- 按工作包分别估算开发、测试、代码审查、文档、集成与返工时间，不能只估编码时间。
- `估算总工作量 > 团队可用容量 × 1.2` 时，必须在开工前由负责人显式裁剪或调整排期，并记录决策、影响、替代方案和验收变化。
- 不允许把所有功能同时列为"承诺完成"后再静默降级；P0 若被裁剪，必须重新确认 Sprint 目标。
- 任何功能要标记完成，必须满足其最小交付定义和可复现验收证据。

### 7.1 每阶段完成定义（DoD）

每个工作包必须同时满足：

- 代码与类型/Schema 更新完成；
- 服务/API/前端相关测试完成；
- 失败与边界场景有处理；
- 文档、示例或操作说明同步；
- 不破坏 Sprint 1 回归基线；
- 可从 UI 或约定 CLI/API 实际走通；
- 已知问题记录在 Sprint 2 changelog / 遗留项中。

### 7.2 容量与范围管理

本版明确要求所有核心产品功能进入 Sprint 2，但不虚构工时。启动开发前按团队实际容量对上述工作包逐项估算。若容量不足，优先减少视觉精细度、非关键交互和非必要自动化，不得静默删除：

- 正式规则发布；
- 知识缺口管理；
- Case 管理；
- Markdown/PDF 报告导出；
- 批量评审/CSV；
- RAG 简易管理；
- 指标看板；
- CI/验收与 Demo。

如需调整范围，必须由产品/项目负责人显式决策，记录被调整项、影响与替代方案；不得将未实现项标记为已完成。

---

## 8. Sprint 2 Demo 与最终验收

### 8.1 一键 Demo 主流程

`demo_sprint2.ps1`（或仓库既有等价脚本）必须支持 `--mode mock|real-llm`（PowerShell 参数风格可按仓库约定调整），共用流程骨架：

- `mock`：Mock Provider + Seed/模拟数据，不访问外部 LLM；CI 只运行该模式。
- `real-llm`：显式启用真实 LLM，手动执行并记录模型、配置、运行范围与费用/耗时；真实专家数据有则使用，没有则如实显示 N/A。

两个模式均须在输出和报告中标注数据来源（V1.3 冻结标注格式：Mock 模式产物文件名加 `_mock` 后缀，如 `report_mock.json`；真实模式加 `_real`；Seed 反馈在 `comment` 前缀 `[SEED]`，与 Sprint 1 已有约定一致）。Demo 主流程至少演示：

1. 列出并选择 Golden Case；
2. 执行评审并查看报告/Evidence；
3. 对缺陷进行专家判定；
4. 提交 false-negative 或其他适当反馈，生成规则候选；
5. 在 Web 页面查看候选、编辑 YAML、运行候选 Benchmark；
6. 人工批准候选，发布到正式规则库并展示新版本；
7. 重新执行相关 Benchmark，展示发布前后比较；
8. 处理一条知识缺口并新增/启用 Markdown 知识条目；
9. 导出 Markdown 报告（P0）；若 PDF P1 实现已完成，再导出 PDF；
10. 执行批量评审并导出 Benchmark CSV；
11. 展示看板的真实/模拟数据区分与样本量说明。

不得要求 CI 访问真实 LLM。真实轨是手动验收，不可用 Seed 结果证明真实业务指标达标。

如真实专家样本不存在，Demo 中使用 Seed 展示研发调试路径，并显著标注 `simulated`；不得演示成真实业务指标。

### 8.2 功能准出清单

- [ ] 当前 Sprint 1 基线、迁移版本与测试结果已记录。
- [ ] bench integration 保持 Sprint 1 收口基线（`tp=16/fp=0/fn=0` 或经基线核查确认的等价基线）。
- [ ] 任务状态持久化、筛选和错误显示可用。
- [ ] 专家判定 PATCH、审计字段与前端回滚可用。
- [ ] 指标实时聚合；Seed 被排除于真实指标；空值语义正确。
- [ ] 规则候选可编辑、Benchmark、批准/驳回。
- [ ] 人工批准后规则确实进入正式规则库，数据库/文件版本一致。
- [ ] 规则发布失败可检测、可恢复；发布后回归 Benchmark 可执行。
- [ ] 知识缺口 Backlog 可查看、追溯、更新状态。
- [ ] Case 管理、资产预览、单 Case Benchmark 和历史结果可用。
- [ ] Markdown/PDF 报告导出可用，字段与 Evidence 完整。
- [ ] 批量评审可用，部分失败可识别，CSV 可导出。
- [ ] RAG Markdown 知识条目可导入、查看、启停。
- [ ] 指标看板可用，Seed/真实指标清楚区分。
- [ ] CI 包含资产校验、迁移/测试、Mock Golden Case 回归与前端检查。
- [ ] vLLM 兼容验证报告完成，说明配置、运行范围、失败与限制。
- [ ] `demo_sprint2.ps1` 在约定环境可复现（mock + real-llm 双轨）。
- [ ] 用户手册、部署运维手册、开发手册、变更记录完成。
- [ ] 遗留缺陷有明确严重度、影响、绕行方式和后续归属。

### 8.3 真实业务指标准入

以下为真实业务指标验收，不得由 Seed 代替：

- [ ] 真实专家反馈样本来源可追溯；
- [ ] 样本量满足 `metrics_definition.md` 对该指标的门槛（EAR ≥ 20 条，AI_NER ≥ 10 条）；
- [ ] 真实 AI_NER/EAR 按冻结公式计算并可复核；
- [ ] 报告注明时间范围、有效样本量与数据来源。

若 Sprint 2 结束前没有足够真实专家数据：

- 产品功能与 Seed 技术验证仍可验收；
- 真实业务指标标记为"未具备验收数据 / N/A"，不能标记达标，也不以此掩盖数据缺失；
- 后续由真实使用过程积累数据后再做业务效果评估。

---

## 9. 交付物清单

### 9.1 代码与数据库

- Alembic 迁移、ORM 与 Schema 更新；
- 任务、专家判定、指标聚合、规则候选发布、知识缺口、Case、导出、批量评审、知识库和看板服务/API；
- 前端相应页面、组件与 API 类型；
- 正式规则发布的版本记录、备份和恢复机制；
- Seed 生成/清理脚本。

### 9.2 测试与报告

- 后端单元/服务/API 测试；
- 前端 type-check/build 与关键手动场景清单；
- Mock Golden Case 回归报告；
- 真实模型 Benchmark 报告（如运行环境可用）；
- vLLM 兼容验证报告；
- 规则发布前后 Benchmark 对比；
- 资产校验结果。

### 9.3 文档

- `sprint2_changelog.md`
- 更新后的 `docs/benchmark/metrics_definition.md`（v3.2，含 AI_NER 样本门槛）
- 用户使用手册（最小目录：任务/Case 评审、专家判定与反馈、规则候选审核/发布、知识缺口处理、报告导出、指标含义与 N/A）
- 部署/运维手册（最小目录：环境变量、数据库迁移、启动/停止、规则发布与恢复、Seed 清理安全、日志/故障排查、备份）
- 开发手册（最小目录：架构与 Service 边界、API/Schema、数据库迁移、指标定义、Case/规则资产结构、规则发布状态机、测试/CI/Demo）
- Demo 运行说明与已知问题清单

---

## 10. 风险与处理原则

| 风险 | 影响 | 处理原则 |
|---|---|---|
| 规则 YAML 与数据库更新不一致 | 规则库状态不可信 | 备份、原子写入、事务/补偿、恢复测试 |
| Seed 混入真实指标 | 业务结论失真 | 模拟标记、聚合排除、独立验收 |
| 同步任务状态设计与实际执行方式不匹配 | UI 状态不可信 | 先核查当前 `create_task` 与 Graph 持久化，再冻结可见语义 |
| 旧 API/历史数据被破坏 | Sprint 1 回归 | 增量迁移、兼容测试、基线对比 |
| 规则发布后指标退化 | 正式规则质量下降 | 发布后回归、版本记录、可恢复 |
| 批量评审同步超时 | 前端误报任务失败 | 独立 180 秒超时 + `batch_id` 查询入口 + 不自动重试 |
| 范围超出容量 | Sprint 延期或功能半成品 | 全部功能保留为目标，优先压缩非核心体验；范围变更须显式审批 |
| vLLM/LLM 外部环境不可用 | 验证无法复现 | 适配验证独立执行，记录配置和失败，不阻塞普通 CI |

---

## 14. 分阶段执行计划

> 下表为执行顺序和交付物。**具体工时见 Sprint 启动后的拆分文档**；本表格不承诺固定工时。已知 V1.3 全范围（F01~F15）初步估算约 30~35 人工作日，**远超原始 2 周 Sprint 容量**；Sprint 启动前必须按 §7.0.1 完成 P0/P1/P2 优先级裁剪，划分必须交付 / 可延后任务。本表阶段编号与 §7 的 S2-A ~ S2-J 一一对应。

| 阶段 | 重点工作 | 交付物 | 阶段验收 | 优先级(P0/P1/P2) |
| :--- | :--- | :--- | :--- | :--- |
| **S2-A. 基线回归与数据库** | 回归 Sprint 1；输出 `baseline_audit.md`（9 项核查）；统一 Alembic 为 Schema 唯一来源；`review_result` 加 `task_status`；`RuleCandidate` 补 benchmark 字段；`KnowledgeGap` 建表；迁移与数据清理脚本 | 基线核查报告、migration、模型、迁移测试、清理脚本（默认 dry-run） | 空库迁移成功；Sprint 1 库可升级；`alembic current` = head；bench integration 保持 `tp=16/fp=0/fn=0` | P0 |
| **S2-B. 专家判定 PATCH** | `PATCH /review/tasks/{id}/defects/{id}`；Service 层双校验（task + defect 归属）；审计字段落库 | Schema、Service、API 端点、Mock 测试（200/422/404） | 合法更新返回 200；非法枚举 422；defect 不属于 task 返回 404；不改 `review_status`、不隐式创建 feedback | P0 |
| **S2-C. 筛选与真实指标聚合** | `GET /review/tasks?status=`；`summary_metrics` 实时聚合；`compute_real_ear` / `compute_real_ai_ner`；`sample_info` 字段；AI_NER 样本门槛 ≥ 10 条冻结 | Service 聚合函数、API 响应字段、`metrics_definition.md` v3.2、指标单测 | 分母 0 返回 `null`；`knowledge_gap` 不进 EAR 分母；Seed 被排除；AI_NER 门槛不足返回 `null`；`sample_info` 字段完整 | P0 |
| **S2-D. 前端专家判定交互** | `ExpertDecisionPanel.vue`；`DefectCard` 展开区集成；乐观更新 + 失败回滚；审计行展示 | 组件、API 封装、TS 类型、手动验证 8 场景 | 点击确认/驳回/待核验可落库；404/422/超时回滚并提示；刷新后持久；当前态按钮 disabled | P0 |
| **S2-E. 摘要卡片与反馈表单增强** | `TaskSummaryCard.vue`（EVC/AI_NER/EAR/质量分布）；`SuggestionDiffEditor.vue`；`RuleCandidateForm.vue`；`FeedbackPanel` 按类型动态渲染 | 组件、类型定义、表单校验 | 四格指标显示；EAR 空显示 `N/A · 样本不足`；AI_NER 空显示 `N/A · 待专家反馈`；`suggestion_diff` 双列表单；`rule_candidate` 结构化表单 | P1 |
| **S2-F. 规则候选审核与正式发布** | 候选列表/详情/YAML 编辑/候选 Benchmark；`approve` → `publishing` → `published` 状态机；文件原子替换 + DB 同步 + 引擎重载 + 发布后回归；失败补偿 | API（6 端点）、Service、前端页面、发布记录、恢复测试 | 候选可编辑/批准/驳回；批准后进入正式 `data/rules/`；文件/DB 一致；发布失败可恢复；回归失败不自动回滚 | P0 |
| **S2-G. 知识缺口与 Case 管理** | `KnowledgeGap` 独立表 + 状态流转（open/in_progress/resolved/wontfix）；Case 列表/资产预览/单 Case Benchmark/历史结果 | API（knowledge-gaps + cases）、前端页面、资产校验 | 缺口可追溯来源反馈；一对一约束生效；Case 资产预览完整；单 Case Benchmark 可跑 | P0（知识缺口）/ P1（Case 管理） |
| **S2-H. 导出、批量、RAG、看板** | Markdown/PDF 报告导出；批量评审（同步上限 5 个，P95 > 120s 降 3 个，独立 180s 超时）；Benchmark CSV；RAG Markdown 导入/启停；指标看板 | 导出 Service、批量 API、前端页面、CSV 导出、看板 API/UI | 导出字段完整（Evidence 三要素）；批量部分失败可识别；CSV 每行一 Case；RAG 条目可导入/启停；看板区分 Seed/真实 | P0（Markdown 导出 + 批量 + CSV）/ P1（PDF + RAG + 看板） |
| **S2-I. CI 与资产治理** | GitHub Actions 双 job（backend + frontend）；迁移检查；Case 资产校验；规则 YAML 校验；Mock Golden Case 回归；bench 退化告警；vLLM 兼容验证（独立、可配置） | `.github/workflows/ci.yml`、资产校验脚本、vLLM 适配报告 | CI 绿（backend + frontend）；资产校验通过；Mock 回归通过；bench 不退化为 `tp=16/fp=0/fn=0`；vLLM 无环境时标记「未验证」 | P0（CI + 资产 + Mock）/ P1（vLLM） |
| **S2-J. Demo、文档与验收** | `demo_sprint2.ps1` 双轨（mock / real-llm）；用户/部署运维/开发三份手册；`sprint2_changelog.md`；准出清单逐项打勾 + 证据链接 | Demo 脚本、三份手册、changelog、验收记录 | Demo 双轨可复现；三份手册最小目录齐全；§8.2 准出清单全部有证据；遗留项明确严重度/影响/归属 | P0（Demo + changelog + 验收）/ P1（三份手册完整版） |

### 14.1 建议执行顺序

1. **S2-A 必须先完成**：`baseline_audit.md` 通过评审前，不进入 S2-B 及之后的 Schema/发布流程开发。核查发现与计划假设不符时，**先修订计划再编码**。
2. **S2-B → S2-D 串行**：S2-D 依赖 S2-B 的 PATCH 接口。
3. **S2-C 可与 S2-B 部分并行**：聚合函数独立于 PATCH 接口，但 `summary_metrics` 的 API 响应需等 S2-C 完成后才能接前端。
4. **S2-F 风险最高，单独排期**：发布状态机的四个失败点（文件校验/原子替换/DB 同步/引擎重载）必须逐步验证，不允许「最后一起测」。
5. **S2-H 的批量评审必须先压测再写 UI**：P95 > 120s 时停下，提交后台执行架构决策，不写前端。
6. **S2-I 的 CI 逐步加项**：先 `pytest -m "not integration"` + `npm run type-check` 绿，再加迁移检查，再加资产校验，每加一项跑一次 CI。

### 14.2 阻塞规则

- **P0 阻塞**：基线核查未通过、迁移失败、专家判定无法落库、规则发布文件/DB 不一致、批量评审压测超限未决策——**立即升级项目负责人，当天处理**。
- **P1 阻塞**：前端交互异常、导出字段缺失、看板空值显示错误——记录到 changelog，周会升级，不阻塞 P0。
- **P2 阻塞**：视觉优化、非关键自动化——记录到 backlog，不阻塞 Sprint 验收。
- **需求边界或指标口径歧义**：先更新决策记录（附录 B/D）和 `metrics_definition.md`，再继续实现；不允许「先写代码再补文档」。
- **计划与代码事实冲突**：以代码事实为准，修订计划并更新 `baseline_audit.md`；不允许「为了遵守计划而重复建模或覆盖已有实现」。

### 14.3 与 §7 阶段编号的映射

| 本表阶段 | §7 阶段 | 说明 |
| :--- | :--- | :--- |
| S2-A | S2-A | 完全对应 |
| S2-B | S2-B | 完全对应 |
| S2-C | S2-C | 完全对应 |
| S2-D | S2-D | 完全对应 |
| S2-E | S2-E | 完全对应 |
| S2-F | S2-F | 完全对应（含规则发布） |
| S2-G | S2-G | 完全对应（知识缺口 + Case 管理） |
| S2-H | S2-H | 完全对应（导出 + 批量 + RAG + 看板） |
| S2-I | S2-I | 完全对应（CI + vLLM） |
| S2-J | S2-J | 完全对应（Demo + 文档 + 验收） |

### 14.4 每阶段完成定义（DoD）

每个阶段必须同时满足以下 7 项，缺一不算完成：

1. 代码与类型/Schema 更新完成；
2. 服务/API/前端相关测试完成（新增测试 + 全量非 integration 全绿）；
3. 失败与边界场景有处理（不允许「只测 happy path」）；
4. 文档、示例或操作说明同步（changelog 不留空）；
5. 不破坏 Sprint 1 回归基线（bench `tp=16/fp=0/fn=0`）；
6. 可从 UI 或约定 CLI/API 实际走通（不接受「单元测试过但接口不通」）；
7. 已知问题记录在 `sprint2_changelog.md` / 遗留项中（不允许「口头说已知」）。

### 14.5 容量估算与裁剪提示

- 按工作包分别估算「开发 + 测试 + 代码审查 + 文档 + 集成 + 返工（按 20% 返工率）」六项，**不能只估编码时间**。
- `估算总工作量 > 团队可用容量 × 1.2` 时，必须在开工前由负责人显式裁剪或调整排期，并记录决策、影响、替代方案和验收变化。
- 裁剪顺序：**先砍 P2 → 再砍 P1 中「资产审计/额外 UX」→ P0 若仍超，砍「非核心体验」但保留「规则发布 / 知识缺口 / 批量评审 / Markdown 导出」**。
- 不允许把所有功能同时列为「承诺完成」后再静默降级；P0 若被裁剪，必须重新确认 Sprint 目标。

---

## 附录 A：关键接口命名约定

- API 前缀统一使用 `/api/v1`。
- 资源路径统一采用 kebab-case，例如 `/rule-candidates`、`/knowledge-gaps`。
- 同一资源不可在前后端混用 `/rule_candidates` 与 `/rule-candidates`。
- 新接口先检查现有路由；本文件列出的路径是目标契约草案，最终以代码审查冻结的 OpenAPI 为准。
- API 响应中的业务状态、任务执行状态和专家判定状态必须语义分离。

## 附录 B：冻结决策总表

| 决策点 | 冻结结论 |
|---|---|
| Sprint 2 目标 | 完成可操作的产品业务闭环，而非只完成技术接口 |
| 规则批准 | 人工批准后进入正式规则库；不得自动发布 |
| 规则发布安全 | YAML 校验、版本记录、文件/DB 一致性、失败恢复、发布后回归 |
| Seed 定位 | 仅用于研发调试与流程验证，不代表真实专家数据 |
| 真实指标 | 排除 Seed；样本不足显示 N/A；真实业务效果单独验收 |
| 指标来源 | `metrics_definition.md` 是唯一公式与样本口径来源 |
| 任务状态 | 与 `review_status` 分离；单任务仅 `pending / done / failed`，Sprint 2 不引入 `running` |
| 专家判定 | 与反馈历史分离；只更新专家判定及审计字段 |
| 正式功能范围 | 知识缺口、Case 管理、报告导出、批量评审、RAG 简易管理、指标看板均属于 Sprint 2 |
| 不做范围 | 真实 EDA 上传解析、PDF/OCR、MCP、多 Agent、知识图谱、企业级 RBAC、拖拽式规则编辑器 |
| 工时 | 不沿用较窄方案的 11.5 人日作为全量承诺；启动前重新估算 |
| 兼容性 | 保留 Sprint 1 API、历史数据与指标基线，所有新增能力需回归验证 |

## 附录 C：执行前必须核查的事实

以下项目须在编码前以当前仓库代码、数据库与上游冻结文档核查，不能仅依据计划文本推断已存在：

1. 当前 Git HEAD、测试基线和 Benchmark 来源；
2. `task_status` 当前模型/迁移是否已存在；
3. `RuleCandidate`、`FeedbackItem`、`KnowledgeDoc/KnowledgeChunk` 的实际字段；
4. 正式规则加载路径、`RuleDefinition` 与 YAML 的同步机制；
5. 规则候选现有状态机与 CLI 能力；
6. `metrics_definition.md` 当前版本、真实指标公式与 Seed 过滤字段；
7. 现有 Case、Benchmark、导出、批量评审和知识库 API 是否已有可复用实现；
8. 前后端 API 路径命名与 Pydantic/TypeScript 类型是否一致；
9. `metrics_definition.md` 中 AI_NER 样本门槛是否已冻结；若未冻结，S2-A 必须补冻结（默认 ≥ 10 条 `ai_discovered`）。

## 附录 D：V1.3 新增冻结决策与待核查边界

| 主题 | 冻结结论 |
|---|---|
| 单任务状态 | `pending / done / failed`；Sprint 2 不新增 `running`；同步 `create_task` 不承诺可见执行中状态 |
| `pending` 语义 | 仅 ORM `server_default` 兜底；Sprint 1 无 `task_status` 列，不存在历史 pending 数据；Sprint 2 不主动写入 |
| 批量状态 | 批量记录独立于单任务；`done / partial_failed / failed` 只用于批次聚合，单 Case 仍用单任务状态 |
| 批量执行 | 默认同步、默认最多 5 个 Case；S2-A 压测 P95 > 120 秒则降至 3 个，仍超则提交后台执行架构决策；独立 180 秒超时；客户端不自动重试；失败 Case 可提交 Case 级反馈（`review_defect_id=null`） |
| 规则发布状态 | `approved → publishing → published / publish_failed`；候选获批不等于已发布 |
| 规则发布顺序 | 临时文件校验 → 备份 → 原子替换 → DB 事务同步 → 规则引擎内存重载 → 状态落为 published；文件替换后、DB 提交前不重载引擎；中途失败执行补偿 |
| 发布后回归失败 | 不自动回滚；标记异常并保留证据，由人工决定接受或回滚 |
| Git 边界 | Sprint 2 不自动 git commit；人工审查/提交，CI 校验文件与 DB 一致性 |
| KnowledgeGap | 新建独立表，具备状态、来源追溯、解决时间/处理人/备注；Sprint 2 冻结一对一（`source_feedback_id` 唯一约束） |
| 指标样本字段 | `summary_metrics.sample_info`；EAR 阈值 20、AI_NER 阈值 10 由后端提供，前端不硬编码 |
| AI_NER 样本门槛 | Sprint 2 冻结 ≥ 10 条 `ai_discovered` 缺陷；`metrics_definition.md` v3.2 同步更新 |
| 数据清理 | 默认 dry-run；真实删除必须显式 `--confirm-delete` 并明确指定目标 |
| Demo | `mock` 与 `real-llm` 双轨；CI 只跑 mock；真实轨手动运行；产物文件名 `_mock` / `_real` 后缀；Seed 反馈 `[SEED]` 前缀 |
| 优先级 | P0 核心闭环完整交付；P1 PDF/简易审计等可最小实现或显式延期；P2 不作 Sprint 2 承诺 |
| 容量红线 | 估算超过团队可用容量 × 1.2 时，开工前显式裁剪/调整排期 |
| vLLM | 仅做兼容验证，不做生产部署；无环境则记录未验证 |
| CLI | 核心操作尽量提供与 UI 等价的 CLI，CLI 复用 Service 层 |
| 架构限制 | 若采用内存线程，记录进程重启/无持久队列限制；记录 `rule_candidate_ref` 软引用风险及后续改进方向 |
| bench 基线 | Sprint 2 结束时保持 Sprint 1 收口基线（`tp=16/fp=0/fn=0` 或等价），写入 §8.2 准出清单 |

## 附录 E：API/CLI 与状态语义补充

### 批量评审结果示例

```json
{
  "batch_id": "batch-...",
  "status": "partial_failed",
  "total_cases": 5,
  "succeeded": 4,
  "failed": 1,
  "items": [
    {
      "case_id": "case001",
      "task_id": "task-...",
      "task_status": "done",
      "error": null
    },
    {
      "case_id": "case002",
      "task_id": "task-...",
      "task_status": "failed",
      "error": "简短错误摘要"
    }
  ]
}
```

示例仅说明字段语义，不代表现有 API 已实现。正式字段须在 S2-A 基线审计后写入 OpenAPI 与前后端类型。

### 核心 CLI 操作建议

CLI 命令名以仓库现有工具风格为准，至少提供或登记替代方案：

- 资产校验；
- Mock Golden Case 回归；
- Benchmark 执行与报告导出；
- Seed 数据生成/清理；
- 规则 YAML 校验与发布前检查；
- 规则发布后回归；
- Markdown 知识条目导入/启停；
- 双轨 Demo。

所有 CLI 均应复用 Service 层。Seed 清理和规则发布必须具备显式确认/保护机制。

## 附录 F：其他冻结要求

- **批量评审：** 批次记录与单 Case 任务分离；单 Case 使用 `pending/done/failed`，批次使用 `done/partial_failed/failed`。默认同步最多 5 个 Case（S2-A 压测 P95 > 120 秒则降至 3 个，仍超则提交后台执行架构决策）；独立 180 秒超时；客户端不得自动重试整个批次。CSV 每行代表一个 Case 结果。失败 Case 可提交 Case 级反馈（`review_defect_id=null`）。
- **KnowledgeGap：** 独立表，至少有状态 `open/in_progress/resolved/wontfix`、来源反馈（一对一唯一约束）、时间戳、解决人（可空）和处理备注。
- **UI/CLI 双通路：** 资产校验、Mock Benchmark、Seed 清理、规则校验/发布前检查、发布后回归、知识条目导入、Demo 应尽量提供 CLI，复用同一 Service 层。
- **vLLM：** 只做 OpenAI 兼容验证（配置/认证、最小请求、一个 Golden Case、错误与超时记录），不做生产部署；没有环境时标记"未验证"。
- **资产审计（P1）：** 有容量时记录规则/知识条目变更前后哈希或 diff、操作者、时间、动作和版本。
- **架构局限：** 若采用内存线程，记录进程重启会丢任务、无持久队列/跨进程协调；记录 `rule_candidate_ref` 软引用风险及后续改进方向。

## 附录 G：版本记录

| 版本 | 日期 | 变更 |
|---|---|---|
| V1.1 | 2026-10-27 | 融合两份 Sprint 2 计划；统一产品范围与工程执行原则；冻结正式规则发布、Seed/真实指标分层；纳入知识缺口、Case、导出、批量评审、RAG 管理、看板、CI、vLLM 与交付文档；修正工时承诺与任务状态语义风险。 |
| V1.2 | 2026-10-27 | 修复 7 项评审问题：`running` 移除、规则发布状态机与恢复策略、P0/P1/P2 优先级矩阵与容量红线、`summary_metrics.sample_info` 冻结、KnowledgeGap 独立表、批量评审 `partial_failed` 语义、API 命名约定。新增附录 D/E/F，补 §7.0 基线核查报告、§7.0.1 优先级矩阵。 |
| V1.3 | 2026-10-27 | 修复 6 项 V1.2 遗留问题：版本号统一为 V1.3（标题/正文/附录 G 一致）、AI_NER 样本门槛冻结为 ≥ 10 条 `ai_discovered`（补 §1.5 / §4.2 / §7.0 / 附录 C 第 9 项）、批量评审独立 180 秒超时 + `batch_id` 查询入口 + 失败 Case 可提交 Case 级反馈（补 §1.4 / §6.2 / §8.2）、KnowledgeGap 一对一冻结（补 §4.5 / 附录 D / 附录 F）、规则发布「文件替换后 DB 提交前不重载引擎」明确（补 §4.6 / 附录 D）、`pending` 语义修正为「仅 server_default 兜底，S1 无历史 pending 数据」（补 §1.3 / 附录 D）。新增 §2.2 bench 基线不可退化 + §8.2 准出项，补 §6.2 批量超时测试、§8.1 Demo 数据来源标注格式。 |
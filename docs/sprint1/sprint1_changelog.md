# Sprint 1 变更与阻塞记录

> 起始：2026-10-19
> 最后更新：2026-10-21
> 状态：阶段 A ✅ / 阶段 B ✅（B 类 4 个 Case 待硬件专家校准）/ 阶段 C ✅
> 依据：`Sprint1_融合版开发落地规划_v1.0.md`、`Sprint1前置条件确认清单.md`

---

## 阶段 B · 规则设计与 Case 资产（2026-10-19）

> 状态：🟢 阶段 B 完成；B 类 4 个 Case 待硬件专家校准
> 责任：Tech Lead B → hardware-expert

### B.1 已解决的阻塞

| 原阻塞 | 解决方案 | 状态 |
|---|---|---|
| `CTRL_001/002` 的 `category: control` 与 `rule_format.md` 枚举冲突 | 改用 `interface` / `timing` | ✅ 已解决 |
| `POWER_003` 降额依据缺失 | 用通用降额准则（GJB/Z 35）占位；后续替换 | ✅ 已解决（占位） |
| `POWER_002` 8051 专向 | 加 `reset_polarity` 参数；用 `pin_type` 排除 MCU | ✅ 已解决 |
| 5 条新规则的设计依据缺失 | 按"IR 可判定拓扑类缺陷"原则设计 | ✅ 已解决 |
| B 类 4 个 Case 的构造 | Design Agent 设计 + Review Agent 审查 | ✅ 已解决（PROVISIONAL） |
| 复用 case001 的 evidence 带缺陷引用 | 用"去缺陷引用"版替换 | ✅ 已解决 |

### B.2 剩余待办（B 类专家校准）

**B 类 4 个 Case 目前是 PROVISIONAL**，需硬件专家审阅并签字：

| Case | 需要专家审阅 | 状态 |
|---|---|---|
| case_b01_buck_thermal_derating_B | Buck 热降额计算、GJB/Z 35 条款、证据链 | ⏳ PROVISIONAL |
| case_b02_reset_rc_B | RC 容差叠加分析、最坏工况推导、Intel 手册引用 | ⏳ PROVISIONAL |
| case_b03_io_direction_B | I2C 开漏特性、上拉电阻取值、规则边界 | ⏳ PROVISIONAL |
| case_b04_power_sequence_B | 多路时序推导、灌电流路径、规则边界 | ⏳ PROVISIONAL |

**专家校准流程**：

1. hardware-expert 审阅 `expert_reasoning.md` + `expected_review.json` + `evidence/`
2. 修正电路 / AI-only 候选 / 证据
3. 签字 `CALIBRATED`
4. 更新 `info.yaml::calibration_status` 从 `PROVISIONAL` 到 `CALIBRATED`
5. 更新 `expected_review.json::expert_sign_off::reviewer` 从 `PROVISIONAL_AI_ASSISTANT` 到 `hardware-expert`

### B.3 不受阻塞的任务

以下任务**不依赖专家校准**，已全部完成：

- ✅ 10 条规则实现 + 单测
- ✅ 10 个 Case 资产（6A + 4B）
- ✅ 覆盖矩阵
- ✅ 规则命中测试
- ✅ B 类资产完整性测试
- ✅ A/B 类命名结构测试

### B.4 阶段 B 的 P0 目标完成情况

| # | 交付内容 | 状态 |
|---|---|---|
| A | Sprint 0 回归 + Alembic 唯一 Schema | ✅ |
| B | 10 个 Case 资产 + 覆盖矩阵 + 校验脚本 | ✅ 10/10 |
| B' | 10 条规则（含 POWER_003）+ 单测 | ✅ 10/10 |
| expert_decision 持久化 | ✅ |
| AI_NER / EAR 公式 + 单测 | ⏳ 阶段 F |

**阶段 B 的 P0 目标全部完成**（除阶段 F 的 AI_NER / EAR 公式）。

---

## 阶段 C · LLM 层（2026-10-20）

> 状态：✅ 已完成
> 依据：§7 / §14 / §23
> 测试结果：`tests/test_llm_zhipu.py` 28 passed；底座回归 167 passed

### C.1 交付清单

| # | 文件 | 说明 |
|---|---|---|
| 1 | `app/schemas/report.py` | 新增 `Evidence` / `Defect` / `ReviewReport`；`ReviewStatus` 保留 |
| 2 | `app/llm/provider.py` | `LLMResponse` + `LLMProvider` 抽象 + Mock / Zhipu / OpenAI占位 / VLLM占位 + `get_provider()` |
| 3 | `app/llm/output_parser.py` | 三级校验（字段 → Evidence → 状态）+ §5.2 五条强规则 + `ai_discovered` 硬拦截 |
| 4 | `app/llm/prompts/expert_analysis.md` | 四段输入 + 输出强约束；渲染约定用 `str.replace` |
| 5 | `app/llm/prompt_renderer.py` | `render_prompt()`，避免 `str.format()` 吞 JSON 花括号 |
| 6 | `app/llm/embeddings/{__init__,base,fake,zhipu,openai}.py` | Embedding 四实现；`zhipu` 1024 维 |
| 7 | `app/rag/embedding_factory.py` | 工厂收敛 + `BACKEND_DIM` + `assert_backend_matches_db` |
| 8 | `app/rag/ingest.py` | 工厂改 import + 写入前维度守卫 + `meta_json` 从实例读 |
| 9 | `app/rag/retriever.py` | 工厂改 import + `Retriever.__init__` 维度守卫 |
| 10 | `alembic/versions/c1_embedding_dim_1024.py` | `knowledge_chunk.embedding` 1536 → 1024 |
| 11 | `scripts/switch_embedding_backend.py` | 切换脚本（迁移 + 清库 + 改 models.py + 重灌 + 冒烟） |
| 12 | `scripts/init_db.py` | docstring 补 Sprint 1 说明 + 校验环节加维度只读检查 |
| 13 | `tests/test_llm_zhipu.py` | 整合版 28 条测试（智谱连通性 + Mock + Parser + Schema + Embedding 分发） |

### C.2 关键技术决策

#### C.2.1 对话 LLM 主用智谱

- `LLM_PROVIDER=zhipu`，模型 `glm-4-flash`，Key 取 `zhipuai_api_key or llm_api_key`。
- `openai` / `vllm` 保留占位，调用 `complete()` 抛 `NotImplementedError`，基类捕获为 `LLMResponse(error=...)`。
- CI 默认 `mock`，四场景（成功/超时/非法 JSON/异常）由 `MockProvider(preset=...)` 控制。

#### C.2.2 `LLMResponse` 职责边界

- 只承载"原始返回 + 解析结果 + 元信息 + 错误"，**不承担业务质量判定**。
- 字段：`raw_text` / `parsed` / `model` / `provider` / `latency_ms` / `error` / `finish_reason` / `request_id`。
- 质量判定（三态、Evidence 校验、模糊词）全部在 `output_parser.py`。

#### C.2.3 重试策略

- 单次超时 30s；网络类错误重试 1 次；4xx 不重试。
- 由基类 `LLMProvider.complete()` 统一处理，四实现不各写一遍。

#### C.2.4 `origin` 取代 `discovery_source`

- 测试（`test_case_b_validation.py` / `test_rules.py`）全部断言 `origin`，不引入别名。
- `origin` 取值：`rule` / `ai_discovered` / `hybrid`。

#### C.2.5 Embedding 智谱化：1024 维（非 2048）

- 智谱 `embedding-3` 默认 2048 维，**超过 pgvector HNSW 对 `vector` 类型的 2000 维上限**。
- 降为 1024 维（Matryoshka 降维），HNSW 友好，检索质量损失可忽略。
- 配套：`ZhipuEmbedding.dimension=1024`、`BACKEND_DIM["zhipu"]=1024`、迁移 `c1_embedding_dim_1024`。

#### C.2.6 单列 + 迁移 + 参数切换

- `knowledge_chunk.embedding` 单列，不双列并存。
- 切后端：改 `RAG_EMBEDDING_BACKEND` + 跑切换脚本（一次性）。
- 日常启动不跑切换脚本。

### C.3 踩坑与修正

#### 坑 1：pgvector HNSW 对 `vector` 类型上限 2000 维

**现象**：迁移 `c1_embedding_dim_2048` 执行 `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` 时报：

```text
psycopg.errors.ProgramLimitExceeded: column cannot have more than 2000 dimensions for hnsw index
```

**根因**：pgvector 硬限制，不是代码 bug。`vector` 类型 HNSW 最大 2000 维；`halfvec` 可到 4000 维。

**修正**：选 1024 维（Matryoshka 降维），迁移改为 `c1_embedding_dim_1024`。

**备选**：若将来必须保留 2048 维精度，改用 `halfvec(2048)` + `halfvec_cosine_ops`，检索时 `embedding::halfvec(2048)` cast。

#### 坑 2：`atttypmod - 4` 在部分 pgvector 版本偏差 4

**现象**：DB 列实际 `vector(1024)`，但 `atttypmod - 4` 算出 1020；`vector(1536)` 算出 1532。

**根因**：部分 pgvector 版本 `atttypmod` 存储约定与 `atttypmod - 4` 的假设不一致。

**修正**：维度读取改用 `format_type(atttypid, atttypmod)` 解析 `"vector(1024)"` 字符串，不依赖 `atttypmod` 编码细节。改动落在 `embedding_factory.assert_backend_matches_db` 和 `switch_embedding_backend._current_db_dim`。

#### 坑 3：切换脚本本地硬编码 `BACKEND_DIM`

**现象**：`embedding_factory.py` 已改 `zhipu=1024`，但切换脚本本地 `BACKEND_DIM` 仍是 `zhipu=2048`，日志显示"目标：2048"，把 `models.py` 改成了 `Vector(2048)`。

**根因**：维度有两个事实来源。

**修正**：切换脚本删除本地 `BACKEND_DIM`，`main()` 内从 `embedding_factory` import，单一事实来源。同时 `_sync_models_py` 改为只替换 `KnowledgeChunk.embedding` 行的 `Vector(N)`，不误伤注释。

#### 坑 4：`ingest_single_file` 的 `embeddings` 契约变强

**现象**：`test_ingest_uses_embed_documents_not_embed_query` 的 `SpyEmbedding` 报 `AttributeError: 'SpyEmbedding' object has no attribute 'provider_name'`。

**根因**：S1-C 起 `ingest_single_file` 从 `embeddings` 实例读 `provider_name` / `model` 写 `meta_json`，原契约只要求 `embed_documents` / `embed_query`。

**修正**：`SpyEmbedding` 补 `provider_name` / `model` 属性。**契约变化应记入 `ingest_single_file` docstring**（已建议补充，可选）。

#### 坑 5：`get_embedding_client()` 无参调用依赖环境变量

**现象**：`test_get_embedding_client_fake_mode` 断言 `len(vec) == 1536` 失败，实际 1024。

**根因**：`.env` 切到 `zhipu` 后，无参调用返回 `ZhipuEmbedding(1024)`，不再是 fake。

**修正**：测试显式传 `"fake"`，与后端配置解耦；维度从 `BACKEND_DIM` 读，不写死。

### C.4 遗留项

| 项 | 说明 | 归属 |
|---|---|---|
| B 类 4 个 Case 专家校准 | 见 B.2，仍 PROVISIONAL | hardware-expert，可与阶段 D 并行 |
| `test_dimension_guard_mismatch_raises` 严格化 | 当前"不一致就 skip"，未断言"确实抛了预期错误"；需 mock `BACKEND_DIM` 制造不一致 | 阶段 D 测试增强时补 |
| `ingest_single_file` docstring 补契约 | 说明 `embeddings` 需提供 `provider_name` / `model` | 阶段 D 顺手补 |
| `test_embedding` 与项目维度解耦 | 该测试测智谱 API 原生 2048 维，与项目 1024 维不冲突，保持现状 | 不处理 |
| `halfvec` 2048 维支持 | 若将来需要保留 2048 维精度 | 后续 Sprint |
| 本地化 MiniLM（384 维）接入 | 骨架已建，未注册、未迁移 | 见 `docs/sprint1/onprem_minilm_embedding_integration.md` |

### C.5 验收证据

| 项 | 结果 |
|---|---|
| `tests/test_llm_zhipu.py` | 28 passed |
| 底座回归（rules + rag_seed + case_b_validation） | 167 passed |
| 智谱真实 chat 调用 | 通 |
| 智谱真实 embedding 调用（2048 维原生） | 通 |
| 项目 embedding（1024 维）重灌 + 检索 | 通 |
| DB `knowledge_chunk.embedding` | `vector(1024)` + HNSW 索引 |
| `alembic_version` | `c1_embedding_dim_1024` |
| 维度守卫 | `test_dimension_guard_mismatch_raises` PASSED |

---

## 阶段 C → 阶段 D 的准入条件

阶段 C 已闭环，进入阶段 D（LangGraph）前需确认：

| # | 条件 | 状态 |
|---|---|---|
| 1 | 阶段 C 的 13 个交付物全部落地 | ✅ |
| 2 | `tests/test_llm_zhipu.py` 28 passed | ✅ |
| 3 | 底座回归 167 passed | ✅ |
| 4 | DB `vector(1024)` + HNSW 索引 + 知识库已重灌 | ✅ |
| 5 | B 类专家校准（不阻塞阶段 D） | ⏳ 可与 D 并行 |

---

## 阶段 D · LangGraph 6 节点工作流（2026-10-22）

> 状态：✅ 已完成
> 依据：§5.3 / §14
> 测试结果：`tests/test_review_graph_e2e.py` 6 passed；`tests/test_llm_zhipu.py` 29 passed / 1 skipped；全量回归 510 passed / 1 skipped

### D.1 交付清单

| # | 文件 | 说明 |
|---|---|---|
| 1 | `app/workflows/__init__.py` | 工作流包入口 |
| 2 | `app/workflows/state.py` | `ReviewGraphState` + `make_initial_state()`；任务状态枚举 |
| 3 | `app/workflows/review_graph.py` | 6 节点 DAG + `build_review_graph()` + `run_review_graph()` |
| 4 | `app/services/task_service.py` | `create_task` / `get_task` / `list_tasks` / `get_task_defects` |
| 5 | `tests/test_review_graph_e2e.py` | 双轨 E2E（4 mock + 2 zhipu） |
| 6 | `tests/test_llm_zhipu.py`（改） | `test_dimension_guard_mismatch_raises` 严格化为 2 条 |

### D.2 6 节点 DAG

```text
load_ir → rule_check → rag_retrieve → expert_analysis → report_generate → feedback_save
```

| 节点 | 职责 | 失败处理 |
|---|---|---|
| `load_ir` | 从 `data/cases/<case_id>/schematic_ir.json` 加载 + 严格校验 | 写 `errors`，后续节点检测前置缺失时跳过 |
| `rule_check` | `execute_defect_rules` 执行 10 条规则 | 写 `errors` |
| `rag_retrieve` | `Retriever` 按 `rule_hits` + `rule_ids` 检索 | 写 `errors`，`rag_context=[]`，LLM 走"无 RAG 依据"降级 |
| `expert_analysis` | `render_prompt` + `get_provider().complete()` | 写 `errors`，`llm_raw_output` 缺失 |
| `report_generate` | `parse_report` 三级校验 + 三态判定 | 写 `errors` + `task_status=failed` |
| `feedback_save` | 落库 `review_result` + `review_defect` | 写 `errors` + `task_status=failed` |

### D.3 关键技术决策

#### D.3.1 状态分离：任务状态 vs 缺陷质量状态

- **任务状态**：`pending` / `running` / `done` / `failed`，表达"评审任务本身是否完成"
- **缺陷质量状态**：`AI_CONFIRMED` / `NEED_EXPERT_REVIEW` / `LOW_CONFIDENCE`，表达"单条缺陷的可信度"
- 两者严格分离（Sprint 1 §23.1），`task_status` 不因缺陷全被降级而变 `failed`

#### D.3.2 `review_result.task_id` 由 Graph 决定，不由 LLM 回显

- `review_result.task_id` 的语义是"任务批次标识"，应由 `task_service` 生成
- LLM 输出里的 `task_id` 只是 Prompt 渲染时注入的回显值，模型可能漏写/写错/写死占位符（Mock 场景即如此）
- `_persist_review_report` 显式传 `state["task_id"]`，不依赖 `report.task_id`

#### D.3.3 `feedback_save` 节点只落库报告，不实现反馈提交

- 命名保留 `feedback_save` 是为了与 Sprint 1 §5.3 的 6 节点清单对齐
- 实际职责是"报告落库"（`review_result` + `review_defect`）
- 反馈提交入口在阶段 E 由 API 层调用 `feedback_service`

#### D.3.4 异常不中断 Graph

- 6 个节点全部 `try/except` + `_append_error`
- `run_review_graph` 最外层再兜底一次
- 最终 `review_report` 带 `errors` 落库，API 可返回给前端

#### D.3.5 不建 `tasks` 表

- Sprint 0 已冻结用 `review_result` 承担"一次评审"
- `task_id` 对应 `review_result.task_id`
- `get_task` 按 `task_id` 查 `review_result`

#### D.3.6 双轨 E2E

- **Mock 轨**（4 条）：`monkeypatch` 替换 `review_graph.get_provider`，不触网，CI 稳定
- **真实轨**（2 条）：调用智谱 API，无 `ZHIPUAI_API_KEY` 时 skip

### D.4 踩坑与修正

#### 坑 1：LangGraph `StateGraph` 与 `TYPE_CHECKING` 不兼容

**现象**：`StateGraph(ReviewGraphState)` 构造时报：

```text
NameError: name 'SchematicIRDocument' is not defined
```

**根因**：`state.py` 用 `TYPE_CHECKING` 保护重依赖类型的 import，但 LangGraph 的 `_get_channels()` 会对 TypedDict 做 `get_type_hints()` 求值，`TYPE_CHECKING` 在运行时不生效。

**修正**：去掉 `TYPE_CHECKING`，改为运行时真实 import。阶段 C 的五个类型模块均无重依赖（`zhipuai` 等在 `provider.py` 内延迟 import），运行时 import 安全。

#### 坑 2：`review_output_json` 是 NOT NULL，落库漏传

**现象**：

```text
psycopg.errors.NotNullViolation: null value in column "review_output_json"
```

**根因**：`_persist_review_report` 构造 `ReviewResult` 时漏传 `review_output_json`。

**修正**：补 `review_output_json=report.model_dump(mode="json")`。

#### 坑 3：`review_result.task_id` 被 LLM 回显覆盖

**现象**：Mock 场景下 `get_task(task_id)` 查不到，因为落库的 `task_id` 是 Mock payload 的 `"PLACEHOLDER"`。

**根因**：`_persist_review_report` 里 `rr.task_id = report.task_id`，而 `report.task_id` 来自 LLM 输出。

**修正**：`_persist_review_report` 签名加 `task_id` 参数，`node_feedback_save` 传 `state["task_id"]`。

#### 坑 4：`task_status` 被误判为 `done`

**现象**：LLM 抛异常时，`node_report_generate` 构造空报告，`node_feedback_save` 落库成功，`task_status` 被置 `done`。

**根因**：`node_report_generate` 在 `llm_raw_output` 缺失时没置 `failed`；`node_feedback_save` 无条件置 `done`。

**修正**：
- `node_report_generate` 两个失败分支（`llm_raw_output` 缺失、`parse_report` 异常）均置 `task_status=failed`
- `node_feedback_save` 不覆盖上游的 `failed`

#### 坑 5：测试 `defect_id` 全局唯一导致幂等跳过

**现象**：`get_task_defects` 返回空列表，日志显示 `review_defect 已存在，跳过`。

**根因**：`defect_id` 在 DB 层 `unique=True`，Mock payload 写死 `DEF-MOCK-E2E-001`，第二次跑测试时幂等跳过。

**修正**：Mock payload 的 `defect_id` 改为动态生成 `DEF-MOCK-E2E-{uuid}`，测试断言从 `report` 取。

### D.5 遗留项

| 项 | 说明 | 归属 |
|---|---|---|
| `rules_passed` 精确化 | 当前 `_build_rules_passed_text` 用固定文案占位，未暴露"执行但未命中"的规则清单 | 后续 Sprint（需规则引擎扩展） |
| E2E 测试数据残留 | `test_e2e_case001_mock_with_defects` 会落库 `TEST-` 前缀数据，未自动清理 | 阶段 G 写 API 测试时统一加清理 fixture |
| 任务列表按质量筛选 | `list_tasks` 当前返回 `review_status`，不支持"只看全降级任务" | 阶段 G 的 API 层加聚合字段 |
| `test_demo.py` 兼容性 | Sprint 0 的 Mock E2E，依赖旧路径，阶段 D 起忽略 | 阶段 J 文档收尾时清理 |

### D.6 验收证据

| 项 | 结果 |
|---|---|
| `tests/test_review_graph_e2e.py` | 6 passed（4 mock + 2 zhipu） |
| `tests/test_llm_zhipu.py` | 29 passed / 1 skipped |
| 全量回归（`pytest tests/ --ignore=tests/test_demo.py`） | 510 passed / 1 skipped |
| case001 E2E（Mock + 真实） | 通 |
| case_b01 E2E（Mock + 真实） | 通 |
| 异常隔离（LLM 抛异常） | 通，`task_status=failed` |
| 报告落库 | 通，`review_result.review_output_json` 非空 |
| `review_defect` 落库 | 通，`defect_id` 唯一 |

---

## 变更记录

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-10-19 | 阶段 B 初版阻塞记录 | 原 `sprint1_blockers.md` |
| 2026-10-20 | 阶段 C 初版交付 | 13 个文件，28 条测试 |
| 2026-10-20 | Embedding 2048 → 1024 | pgvector HNSW 2000 维上限 |
| 2026-10-20 | 维度读取改 `format_type` | 规避 `atttypmod` 偏差 |
| 2026-10-20 | 切换脚本去本地 `BACKEND_DIM` | 单一事实来源 |
| 2026-10-20 | 测试 2 条修正 | `fake` 显式传参 + `SpyEmbedding` 补属性 |
| 2026-10-20 | 合并阶段 B / C 记录为单文件 | 原 `sprint1_blockers.md` + `stage_c_changelog.md` 合并 |
| 2026-10-21 | 新增本地化 MiniLM 接入指南 | 见 `docs/sprint1/onprem_minilm_embedding_integration.md` |
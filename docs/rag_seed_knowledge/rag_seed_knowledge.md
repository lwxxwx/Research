# Phase H · RAG Seed Knowledge 工作总结

> **文档定位**：Sprint0 Phase H 完整工作总结，记录目标、交付物、设计决策、验证证据与演进计划。
> **对齐规范**：`Sprint0_详细执行计划_V1.2_最终版` §13；`第一阶段工程实施方案_V1.2_增强版` §6
> **冻结版本**：v1.7
> **冻结日期**：2026-09-17
> **Sprint0 收官追加**：`test_rag_seed.py` v1.8 新增 3 个并入逻辑集成测试 + 1 个 ingest 回归测试
> **结论**：PASS · v1.7 基线 34 passed（29 单测 + 5 集成）；v1.8 追加后 RAG 相关测试 ≥ 38 passed

---

## 1. Phase H 目标与边界

### 1.1 目标（Sprint0 §13 原文对齐）

1. 在 `data/knowledge/` 下准备 **≥ 5 段 Markdown 格式的种子知识**；
2. **注入向量库**，验证检索准确度；
3. 对齐 V1.2 §6.2「知识来源标签」：每段 chunk 必须带
   `document_type / source_title / source_section / source_page / snippet`。

### 1.2 边界（Sprint0 明确不做）

| 不做项 | 理由 |
| :--- | :--- |
| 真实 PDF / OCR 解析 | Sprint1 P1，MVP 用 Markdown 摘要 |
| 复杂 rerank / 混合检索 | Sprint0 只做向量 TopK + 元数据过滤 |
| Web 端知识管理 UI | V1.2 §1.3 明确不做 |
| 接入 LangGraph Workflow | Sprint0 约束：不接入 LLM、不接入 Workflow |

### 1.3 验收标准

- [x] 种子知识 ≥ 5 段，覆盖 datasheet / application_note / reference_design
- [x] `knowledge_doc / knowledge_chunk` 表创建并成功注入
- [x] 至少 5 条检索准确度验证（query → 期望命中）
- [x] 与 case001（Phase G）联动：POWER_001/POWER_002 可检索到对应 evidence

---

## 2. 交付物清单

### 2.1 种子知识（`data/knowledge/`，5 份）

| # | 文件 | source_type | 覆盖规则 | chunk 数 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `datasheets/stc89c55rc_oscillator_crystal.md` | datasheet | `[]`（晶振，Sprint0 无对应规则） | 1 |
| 2 | `datasheets/stc89c55rc_power_supply.md` | datasheet | `["POWER_001"]` | 1 |
| 3 | `application_notes/an_mcu_reset_circuit.md` | application_note | `["POWER_002"]` | 1 |
| 4 | `reference_designs/ref_stc89c55_minimum_system.md` | reference_design | `["POWER_001", "POWER_002"]` | 1 |
| 5 | `reference_designs/an_8051_io_port_feature.md` | application_note | `["IO_001"]` | 1 |

**覆盖三类来源**：
- datasheet × 2（晶振、电源）
- application_note × 2（复位、IO 端口）
- reference_design × 1（最小系统）

**未覆盖但已预留**：
- `review_case`（历史评审案例，Sprint1 补充）
- `design_standard`（设计规范，Sprint1 补充）

### 2.2 RAG 模块（`backend/app/rag/`）

| 文件 | 职责 | 关键实现 |
| :--- | :--- | :--- |
| `sources.py` | DocumentType 枚举 | 5 类：datasheet / reference_design / application_note / review_case / design_standard |
| `chunking.py` | frontmatter 解析 + 切片 | v1.0（固定字符）+ v1.1（语义优先）双入口 |
| `ingest.py` | 种子知识入库 CLI | `--config seed_ingest_list.yaml` + `--dry-run` |
| `retriever.py` | TopK 检索 + 元数据过滤 | `part_numbers ∩ related_rule_ids` 双维度过滤 |
| `knowledge.py` | Pydantic 模型 | `RetrievalResult`（直接映射 evidence）+ `IngestConfig` |

**Phase H 依赖的非 RAG 模块**：

| 文件 | 职责 | 提供 |
| :--- | :--- | :--- |
| `backend/app/persistence/db.py` | 数据库会话工厂 | `get_db_session()` 上下文管理器（自动 commit/rollback/close） |
| `backend/app/persistence/models.py` | ORM 模型 | `KnowledgeDoc` / `KnowledgeChunk`（映射 `knowledge_doc` / `knowledge_chunk` 表） |
| `backend/app/core/config.py` | 配置加载 | `settings.rag_embedding_backend` / `settings.database_url` |

### 2.3 数据库表（`infra/docker/initdb/`）

**DDL 脚本族**：
- `001_pgvector.sql`：开启 `vector` 扩展
- `002_schema.sql`：创建全部 12 张业务表（**唯一事实来源**）

**Phase H 涉及的表（2 张）**：

| 表 | ORM 类 | 关键列 | 索引 |
| :--- | :--- | :--- | :--- |
| `knowledge_doc` | `KnowledgeDoc` | id / title / source_type / content_md / `metadata`(JSONB) | — |
| `knowledge_chunk` | `KnowledgeChunk` | id / knowledge_doc_id(FK) / chunk_text / embedding(vector(1536)) / `metadata`(JSONB) | HNSW(vector_cosine_ops) |

**其余 10 张表**（由 Phase C 冻结）：
users / project / schematic_case / ir_document / rule_definition /
rule_execution / review_result / review_defect / feedback_item / rule_candidates

**前置依赖**：`knowledge_chunk.embedding vector(1536)` 依赖 `pgvector` 扩展。
若单独执行 `002_schema.sql`，需确保扩展已存在（由 `001_pgvector.sql` 开启）。

**关键设计**：`metadata` 列名与 ORM 属性 `meta_json` 的差异
- **DB 真实列名**：`metadata`
- **ORM 属性名**：`meta_json`
- **原因**：规避 SQLAlchemy `Base.metadata` 保留字冲突
- **记忆口诀**：**psql 用 `metadata`，Python 用 `meta_json`**

### 2.4 配置文件（`data/knowledge/seed_ingest_list.yaml`）

    ingest_entries:
      - file: "./data/knowledge/datasheets/stc89c55rc_oscillator_crystal.md"
      - file: "./data/knowledge/datasheets/stc89c55rc_power_supply.md"
      - file: "./data/knowledge/application_notes/an_mcu_reset_circuit.md"
      - file: "./data/knowledge/reference_designs/ref_stc89c55_minimum_system.md"
      - file: "./data/knowledge/reference_designs/an_8051_io_port_feature.md"
    case_evidence_scan: null

### 2.5 测试文件（`backend/tests/test_rag_seed.py`）

- **单元测试**（`-m "not integration"`）：29 条
- **集成测试**（`-m "integration"`）：5 条
- **合计**：34 条用例

---

## 3. 关键设计决策

### 3.1 Embedding Provider 抽象

**Sprint0 选型**：`langchain_core.embeddings.fake.DeterministicFakeEmbedding(size=1536)`

**优势**：
- 接口标准，与 `OpenAIEmbeddings` 完全一致
- **1536 维**与 OpenAI `text-embedding-3-small` 对齐，**Sprint1 切换无需 ALTER 维度**
- **CI 无 token 消耗**：`RAG_EMBEDDING_BACKEND=fake` 为默认值
- 不引入额外依赖（无 numpy 强制）

**Sprint1 切换路径**：

    RAG_EMBEDDING_BACKEND=openai
    OPENAI_EMBEDDING_API_KEY=sk-xxx

仅改环境变量，业务代码零改动。

### 3.2 向后兼容的切片策略

**v1.0（`split_text_chunk` / `process_markdown_file`）**：
- 固定字符切片
- 标注 `DeprecationWarning`
- Sprint0 保留

**v1.1（`split_text_chunk_v2` / `process_markdown_file_v2`）**：
- `##` 二级标题优先切分
- 段落聚合到 `chunk_size`
- 代码块 / 表格保护（整段不切）
- **chunk 级 `source_section`**（M3）
- **无 `##` 时回落文档级 section**（M3.1）
- **单段超长时字符硬切兜底**（BUG2）
- **不支持 overlap**（BUG1）

### 3.3 Chunk 级元数据对齐 evidence 三要素

`RetrievalResult` 字段与 V1.2 §3.2 evidence 三要素显式对应：

| RetrievalResult | evidence 字段 |
| :--- | :--- |
| `source_title` | `source` |
| `source_section` | `section` |
| `snippet` | `reason` |

**辅助字段**（不映射到 evidence，供 Sprint1 使用）：

| RetrievalResult | 用途 |
| :--- | :--- |
| `is_continuation` | Sprint1 重排时降权续块 |
| `score` | 向量距离 |
| `part_numbers` / `related_rule_ids` | 检索元数据过滤 |
| `doc_id` / `chunk_id` | 溯源 |

这使 Sprint1 Node4 的 evidence 拼装变为 **1 行代码**：

    evidence = {
        "type": "rag_ref",
        "source": hit.source_title,
        "section": hit.source_section,
        "reason": hit.snippet,
    }

### 3.4 元数据过滤：part_numbers ∩ related_rule_ids

`retriever.retrieve()` 支持两个过滤维度：

- `part_numbers`：器件型号（如 `STC89C55RC`）
- `rule_ids`：规则 ID（如 `POWER_001` / `IO_001`）

**对齐 V1.2 §6.3 检索策略**：

> 对每条 rule_result：`rule_id 关键词 + component_type + part_number + net_name` 拼接 query

Sprint1 Node3 可直接调用：

    hits = retriever.retrieve(
        query=rule_result.rule_id + " " + component.part_number,
        part_numbers=[component.part_number],
        rule_ids=[rule_result.rule_id],
        top_k=5,
    )

### 3.5 表结构唯一来源 = DDL 脚本

**硬约束**：
- `infra/docker/initdb/001_pgvector.sql`：开启 pgvector 扩展
- `infra/docker/initdb/002_schema.sql`：创建全部业务表
- 两者组合 = Database Schema v1.0 的**唯一事实来源**

**禁止**：`Base.metadata.create_all()` 或任何自动建表。

`models.py` 头部注释已锁定：

    class Base(DeclarativeBase):
        """注意：Sprint0禁止 Base.metadata.create_all()，
        schema唯一来源：infra/docker/initdb/002_schema.sql"""
        pass

**注**：`models.py` 注释虽只提 `002_schema.sql`，但应理解为「DDL 脚本族」（含 `001`）；
核心约束是**禁止 ORM 自动建表**，而非字面意义的「只有一个文件」。

---

## 4. 优化与 BUG 修复记录

### 4.1 优化来源总览

| 批次 | 编号 | 项 | 类型 |
| :--- | :--- | :--- | :--- |
| v1.1 | M1 | 表结构走 DDL | 硬约束确认 |
| v1.1 | M2 | 切片段落优先 | 增强 |
| v1.1 | M3 | chunk 级 source_section | 增强 |
| v1.1 | M4 | `joinedload` 防 N+1 | 性能 |
| v1.1 | M5 | JSONB dict 兼容 | 健壮性 |
| v1.1 | M3.1 | 无 `##` 回落文档级 | 补丁 |
| v1.2 | 优化 1 | (intro) 引言块继承 doc_section | 体验 |
| v1.2 | 优化 2 | 日志打印 section_count dict | 可观测 |
| v1.2 | 优化 3 | FakeMapping 补 `__len__` | 测试健壮性 |
| v1.2 | 优化 4 | deprecation 注释精确化 | 文档 |
| v1.2 | 优化 5 | `RetrievalResult.is_continuation` | 契约透传 |
| v1.2 | 优化 6 | 异常信息附 chunk 长度列表 | 排障 |
| v1.3 | BUG1 | 移除 overlap 拼接（跨 section 污染） | 阻断 BUG |
| v1.3 | BUG2 | 单段超长字符硬切兜底 | 阻断 BUG |
| v1.3 | BUG3 | `_as_dict` 增强 + WARNING 限流 | 阻断 BUG |
| v1.4 | 优化 1 | `import yaml` 移到文件头 | 规范 |
| v1.4 | 优化 2 | 集成测试统一用 `_as_dict()` | 契约一致 |
| v1.4 | 优化 3 | `pytest.raises` 恢复精确匹配 | 契约锁定 |
| v1.4 | 优化 4 | `retriever` 加 `assert doc is not None` | 数据完整性 |
| v1.4 | 优化 5 | `_hard_split` 标点查找简化 | 可读性 |
| v1.4 | 优化 6 | 段落聚合断言加 `strip()` | 健壮性 |
| v1.4 | 优化 7 | `Chunk` 删除未使用 `meta` 字段 | 清理 |
| v1.5 | 优化 1 | 删未使用 `Union` / `Session` | PEP 8 / F401 |
| v1.5 | 优化 2 | `_hard_split` 注释与循环语义对齐 | 文档 |
| v1.5 | 优化 3 | 新增 3 条 `_hard_split` 边界单测 | 覆盖 |
| v1.5 | 优化 4 | 删未使用 `import json` | PEP 8 / F401 |
| v1.5 | 优化 5 | 新增超长完整代码块不切割单测 | 覆盖 |
| v1.6 | 缺陷 1 | `choose_closest_punct_with_multiple` 补 `len == 20` | 测试精确性 |
| v1.6 | 缺陷 2 | `single_long_paragraph` 的 `+50` 余量加注释 | 测试文档化 |
| v1.7 | A.1 | `choose_closest_punct_with_multiple` 改字符串全等断言 | 测试强化 |
| v1.7 | A.2 | 新增 `HARD_SPLIT_TOLERANCE = 50` 常量 | 消除魔法数字 |
| v1.7 | A.3 | 头部 v1.7 变更注释同步 | 文档 |

### 4.2 v1.1 优化（M1~M5 + M3.1）

**M2 · 切片段落优先**
- `chunking.py`：新增 `split_text_chunk_v2`
- 策略：`##` 切章节 → 段落聚合到 `chunk_size` → 代码块 / 表格保护

**M3 · chunk 级 source_section**
- `chunking.py`：`Chunk.section` 字段
- `ingest.py`：落库到 `metadata.source_section`；`doc_source_section` 另存
- 检索时优先用 chunk 级 section

**M3.1 · 无 `##` 回落文档级**
- `_split_by_h2(body, doc_section)` 增加参数

**M4 · joinedload 防 N+1**
- `retriever.py`：`.options(joinedload(KnowledgeChunk.doc))`
- 集成测试执行时间 1.23s（无 N+1 延迟）

**M5 · JSONB dict 兼容**
- 见 v1.3 BUG3 的 `_as_dict`（继承 M5 扩展）

### 4.3 v1.2 优化（优化 1 / 2 / 3 / 4 / 5 / 6）

**优化 1 · (intro) 引言块继承 doc_section**
- `_split_by_h2` 的 intro 分支改为 `parts.append((doc_section or "(intro)", intro))`
- 避免 evidence.section 显示字面 `(intro)`

**优化 2 · 日志打印 section_count dict**
- `ingest.py`：`logger.info(f"... sections={section_count}")`
- 原：`sections={list(section_count.keys())}` 看不出数量

**优化 3 · FakeMapping 补 `__len__`**
- 测试 `test_as_dict_mapping_like` 补齐魔术方法

**优化 4 · deprecation 注释精确化**
- `split_text_chunk` / `process_markdown_file` 的 docstring 改为
  「计划在 Sprint1 或后续大版本中删除；删除前将通过 changelog 提前公告」

**优化 5 · RetrievalResult.is_continuation**
- `knowledge.py`：新增字段
- `retriever.py`：透传 `meta.get("is_continuation", False)`
- 为 Sprint1 重排降权续块铺路

**优化 6 · 异常信息附 chunk 长度列表**
- `process_markdown_file_v2` 异常 message 携带 `[len(c.text) for c in chunks]`

### 4.4 v1.3 阻断 BUG 修复（BUG1 / 2 / 3）

**BUG1 · 移除 overlap 拼接**
- **问题**：跨 section 拼接导致 chunk 正文与 source_section 语义不一致
- **修复**：`split_text_chunk_v2` 移除 `chunk_overlap` 参数与拼接逻辑
- **兼容**：`process_markdown_file_v2` 保留 `chunk_overlap` 参数但忽略

**BUG2 · 单段超长字符硬切兜底**
- **问题**：单个段落 > limit 且无 `\n\n`，会产出 1 个严重超长 chunk
- **修复**：新增 `_hard_split()` 辅助函数，优先在句子边界切
- **精确标记**：`is_continuation` 用「段内序号 > 0」，避免误标独立段落的第 1 块

**BUG3 · `_as_dict` 增强 + WARNING 限流**
- **问题**：静默返回 `{}` 掩盖类型问题；日志可能爆炸
- **修复**：
  1. 优先 `.items()` 路径
  2. `isinstance(dict)` 兼容 MutableDict
  3. 转换失败输出 WARNING
  4. 每种类型只 warn 一次（`_warned_types` + 锁）
  5. 仍返回 `{}`，不阻塞检索

### 4.5 v1.4 优化（优化 1~7）

**优化 1 · `import yaml` 移到文件头**
- 符合 PEP 8；pytest 收集阶段更快

**优化 2 · 集成测试统一用 `_as_dict()`**
- 替代手写 `dict(obj.meta_json)`
- 补 `is_continuation` 布尔断言

**优化 3 · `pytest.raises` 恢复精确匹配**
- `match="文件缺失 YAML frontmatter"`（含空格）
- `match="非法 source_type"`（含空格）
- 与业务文案完全对齐

**优化 4 · `retriever` 加防御断言**
- `assert doc is not None, "...数据完整性被破坏"`
- 删除 `if doc else` 三目分支（DB 层 FK + NOT NULL 保证 doc 必存在）

**优化 5 · `_hard_split` 标点查找简化**
- 单层 offset 遍历替代双层
- 用 `puncts_set` 加速成员判断

**优化 6 · 段落聚合断言加 `strip()`**
- `assert c.text.strip().endswith("。")`，容忍首尾空白

**优化 7 · `Chunk` 删除未使用 `meta` 字段**
- 同时移除 `from dataclasses import field` 的 `field` import

### 4.6 v1.5 优化（优化 1~5）

**优化 1 · 删未使用 import（`Union` / `Session`）**
- `retriever.py`：`from typing import List, Optional`；`from sqlalchemy.orm import joinedload`
- 避免 ruff F401

**优化 2 · `_hard_split` 注释与循环语义对齐**
- 显式说明 `offset=0` 检查 `remaining[limit-1]`（不含 limit 位置自身）
- 代码逻辑不变（已通过测试）

**优化 3 · 新增 3 条 `_hard_split` 边界单测**
- `test_hard_split_choose_closest_punct_near_limit`：验证就近标点截断
- `test_hard_split_choose_closest_punct_with_multiple`：多标点优先最近
- `test_hard_split_falls_back_when_no_punct_within_window`：窗口内无标点硬切

**优化 4 · 删未使用 `import json`**
- `test_rag_seed.py`：避免 ruff F401

**优化 5 · 新增超长完整代码块不切割单测**
- `test_split_v2_long_code_block_not_cut`：代码块总长 > chunk_size 仍整段保留

### 4.7 v1.6 轻微缺陷修复（缺陷 1 / 2）

**缺陷 1 · `test_hard_split_choose_closest_punct_with_multiple` 补 `len == 20`**
- **问题**：原断言仅验证末尾标点类型，未锁定切点的**精确位置**
- **风险**：若 `_hard_split` 逻辑调整导致切点偏移，原断言可能仍通过
- **修复**：增加 `assert len(pieces[0]) == 20`，直接验证切在标点之后
- **收益**：锁定精确切点位置

**缺陷 2 · `test_split_v2_single_long_paragraph` 的 `+50` 加注释**
- **问题**：`assert len(c.text) <= 400 + 50` 中的 `+50` 来源不明
- **分析**：当前 `_hard_split` 保证 `cut_pos ∈ [limit-99, limit]`，每块长度 `<= limit`；`+50` 是防御性余量
- **修复**：加注释说明「当前实现理论无需 +50；保留以防未来回退窗口或边界逻辑调整」
- **收益**：读者理解 `+50` 的用意

### 4.8 v1.7 方案 A 应用（A.1 / A.2 / A.3）

**A.1 · 改用字符串全等断言**
- **位置**：`test_hard_split_choose_closest_punct_with_multiple`
- **问题**：v1.6 的断言是 `len == 20` + `endswith("。")` + `not endswith("！")` 三条
- **分析**：
  - 三条断言**部分重叠**（长度 + 末位内容）
  - 与同组的 `test_hard_split_choose_closest_punct_near_limit`（用字符串全等）**风格不一致**
- **修复**：改用 **`assert pieces[0] == "A" * 15 + "！" + "A" * 3 + "。"`**
  - 单条断言同时锁定**内容、长度、切点位置**
  - 失败信息更直观（直接显示差异字符串）
- **收益**：测试更强、风格统一、可读性更好

**A.2 · 新增 `HARD_SPLIT_TOLERANCE = 50` 常量**
- **位置**：`test_rag_seed.py` 文件顶层（import 下方）
- **问题**：`test_split_v2_single_long_paragraph` 中 `400 + 50` 的 `50` 是**魔法数字**
- **修复**：
  - 新增常量 `HARD_SPLIT_TOLERANCE = 50`
  - 断言改为 `assert len(c.text) <= 400 + HARD_SPLIT_TOLERANCE`
  - 加注释说明「描述 `_hard_split` 的防御性余量」
- **命名选择**：`HARD_SPLIT_TOLERANCE`（比 `CHUNK_SIZE_TOLERANCE` 更精确，因为它描述的是 `_hard_split` 的行为）
- **收益**：
  - 消除魔法数字
  - 未来调整容差只需改一处
  - 常量名自解释

**A.3 · 头部 v1.7 变更注释同步**
- **位置**：文件顶部 docstring
- **问题**：A.1/A.2 改变后，v1.6 的头部注释已不准确
- **修复**：新增 v1.7 变更段，说明方案 A.1/A.2/A.3 的具体内容
- **收益**：读者从文件头即可了解最新版本演进

---

## 5. 验证证据

### 5.1 单元测试（29 passed）

    docker compose `
      -f infra/docker/docker-compose.yml `
      -f infra/docker/docker-compose.dev.yml `
      exec backend uv run pytest tests/test_rag_seed.py -m "not integration" -v

**预期**：`29 passed, 5 deselected`

**分类**：
- v1.0 兼容：10 条
- v1.1 M2/M3/M5：8 条
- v1.2 优化 1/3/5：6 条
- v1.3 BUG1/2/3：9 条
- v1.5 优化 3/5：4 条（其中 2 条在 v1.6/v1.7 中**多次加强断言**）

### 5.2 集成测试（5 passed）

    docker compose `
      -f infra/docker/docker-compose.yml `
      -f infra/docker/docker-compose.dev.yml `
      exec backend uv run pytest tests/test_rag_seed.py -m "integration" -v

**预期**：`6 passed`（新增 `test_ingest_uses_embed_documents_not_embed_query`）

| 用例 | 验证点 |
| :--- | :--- |
| `test_retriever_basic_query` | Retriever 可用 + 元数据过滤 |
| `test_knowledge_table_has_seed_data` | `doc_count ≥ 5, chunk_count ≥ 5`；`_as_dict` 读取 |
| `test_chunk_meta_has_chunk_level_section` | M3 落库 + `is_continuation` bool |
| `test_retriever_returns_chunk_level_section` | M3 + M4 端到端 |
| `test_retriever_passes_is_continuation` | 优化 5 透传 |

### 5.3 DB 层验证

    SELECT id, knowledge_doc_id, metadata->>'source_section' AS section,
           metadata->>'is_continuation' AS is_cont,
           left(chunk_text, 40) AS preview
    FROM knowledge_chunk ORDER BY id;

**结果**（5 条 chunk）：

| id | section | is_cont |
| :--- | :--- | :--- |
| 1 | Oscillator Circuit 晶振电路设计 | false |
| 2 | Power Supply 电源供电 | false |
| 3 | 上电复位电路 | false |
| 4 | Minimum Working Circuit 最小工作条件 | false |
| 5 | P0/P1/P2/P3端口电气特性 | false |

**验证**：M3.1 在 DB 层生效；当前 5 份种子均为单 chunk，`is_continuation` 均为 false。

### 5.4 数据统计

| 指标 | 值 |
| :--- | :--- |
| `knowledge_doc` 行数 | 5 |
| `knowledge_chunk` 行数 | 5 |
| `embedding_provider` | `fake`（无外网） |
| 全测试套件 | **44 passed**（RAG 相关；截至 Sprint 0 收官，整体套件 263 passed） |
| 单元测试耗时 | ~1.2s |
| 集成测试耗时 | ~1.2s |

---

## 6. 目录结构（Phase H 相关）

    Schematic_Design_Review/
    ├── data/
    │   └── knowledge/
    │       ├── seed_ingest_list.yaml
    │       ├── datasheets/
    │       │   ├── stc89c55rc_oscillator_crystal.md
    │       │   └── stc89c55rc_power_supply.md
    │       ├── application_notes/
    │       │   └── an_mcu_reset_circuit.md
    │       └── reference_designs/
    │           ├── ref_stc89c55_minimum_system.md
    │           └── an_8051_io_port_feature.md
    ├── backend/
    │   ├── app/
    │   │   ├── rag/
    │   │   │   ├── sources.py
    │   │   │   ├── chunking.py
    │   │   │   ├── ingest.py
    │   │   │   ├── retriever.py
    │   │   │   └── knowledge.py
    │   │   └── persistence/
    │   │       ├── db.py
    │   │       └── models.py
    │   └── tests/
    │       └── test_rag_seed.py
    └── infra/docker/initdb/
        ├── 001_pgvector.sql
        └── 002_schema.sql

---

## 7. 遗留问题与 Sprint1 待办

### 7.1 遗留问题（不阻塞冻结）

| 编号 | 项 | 优先级 |
| :--- | :--- | :--- |
| S1 | 硬编码 1536 → `settings.embedding_dim` | P2 |
| S2 | `ingest.py` 重复执行会追加（非 upsert） | P1 |
| S3 | v1.0 废弃 API 未删除 | P1 |
| S4 | 种子 md 无 `##` 标题 | P1 |
| S5 | 2 条 `DeprecationWarning` 已用 `pytest.warns` 显式断言 | ✅ 已闭环 |
| S6 | 全局 warning 治理未启用 | P3 |
| S7 | `002_schema.sql` 缺 `CREATE EXTENSION vector`（依赖 001） | P1 |
| S8 | `feedback_item.check_feedback_item_type` 约束不幂等 | P1 |

### 7.2 Sprint1 待办清单

| # | 任务 |
| :--- | :--- |
| 1 | 删除 v1.0 废弃 API（`split_text_chunk` / `process_markdown_file`） |
| 2 | `ingest.py` 改为 upsert（按 `source` 路径唯一键） |
| 3 | Embedding 切换到 OpenAI `text-embedding-3-small` |
| 4 | PDF → Markdown 时产出 `##` 标题 |
| 5 | 补充 `review_case` / `design_standard` 类型种子 |
| 6 | `scan_case_evidence` 启用：扫描 B 类 case evidence |
| 7 | 加入重排（Rerank）；用 `is_continuation` 降权续块 |
| 8 | `002_schema.sql` 补 `CREATE EXTENSION IF NOT EXISTS vector;`（S7） |
| 9 | `check_feedback_item_type` 用 `DO $$` 块包裹保证幂等（S8） |
| 10 | 硬编码 1536 → `settings.embedding_dim`（S1） |
| 11 | 全局 warning 治理 `filterwarnings = ["error"]`（S6） |
| 12 | 加 `pg_session_rollback` fixture（事务回滚），隔离 `test_ingest_uses_embed_documents_not_embed_query` 的 DB 污染 |
| 13 | 可选：`sources.py` 支持 `MCU_xxx → POWER_xxx` 别名（若 Sprint 1 需要保留旧规则标识） |

---

## 8. Sprint0 整体进度（截至 Phase H 完成）

| Phase | 状态 | 关键证据 |
| :--- | :--- | :--- |
| A · Docker + uv | ✅ | 容器运行 |
| B · Backend Framework | ✅ | FastAPI 骨架 |
| C · Database | ✅ | 12 张表 |
| D · Schematic IR | ✅ | case001 校验通过 |
| E · Golden Case | ✅ | case001 校准 |
| F · Rule System | ✅ | POWER_001/002 命中 |
| G · Benchmark | ✅ | Precision=1.0 / Recall=1.0 / EVC=0.6667 |
| **H · RAG Seed Knowledge** | ✅ **v1.7 冻结** | **34 passed** |
| I · Feedback 6 类 | ⏳ | — |
| J · Demo 全链路 | ⏳ | 依赖 Phase I |
| K-L · 验收与风险管理 | ⏳ | 依赖 Phase J |

**Sprint0 完成度：8/12 Phase（A~H 全部完成）**

---

## 9. 与上下游 Phase 的衔接

### 9.1 上游依赖

| Phase | 依赖内容 |
| :--- | :--- |
| Phase C | `002_schema.sql` 提供 `knowledge_doc` / `knowledge_chunk` 表 |
| Phase E | case001 evidence 提供种子知识的内容来源 |
| Phase G | `_evidence_is_complete` 用于 evidence 三要素校验 |

### 9.2 下游衔接

| Phase | 衔接点 |
| :--- | :--- |
| **Phase I** | `RetrievalResult` 作为 feedback 的 evidence 引用 |
| **Phase J** | Demo 展示「规则命中 → RAG 引用 Datasheet」链路 |
| **Sprint1 Node3** | `Retriever.retrieve()` 直接调用 |
| **Sprint1 Node4** | `RetrievalResult` → evidence `rag_ref` 拼装 |
| **Sprint1 Rerank** | 用 `is_continuation` 降权续块 |

### 9.3 与 case001（Phase G）联动验证

| case001 evidence | RAG 种子知识 |
| :--- | :--- |
| `STC89C55RC Datasheet V2.1` §7.1 Power Supply Decoupling | `stc89c55rc_power_supply.md` |
| `AN-8051-001 8051 Minimum System Design Guide` §3 | `an_mcu_reset_circuit.md` |
| `8051 Minimum System Reference Design Guide` §2/§4 | `ref_stc89c55_minimum_system.md` |

---

## 10. 关键经验总结

### 10.1 工程经验

1. **数据资产优先于代码工程**：Phase H 的核心是「知识内容 + 元数据规范」，而非「代码优雅」
2. **生态对齐优于自研**：`DeterministicFakeEmbedding` 替代自研 hash
3. **显式契约优于隐式约定**：`RetrievalResult` 字段直接对应 evidence 三要素
4. **测试锁定契约**：用 `_evidence_is_complete` 在 Phase H 测试中锁定 Phase G 契约
5. **CI 无副作用**：`RAG_EMBEDDING_BACKEND=fake` 默认值保证无 token 消耗
6. **渐进式冻结**：v1.1 → v1.7 逐版叠加，每版都跑通再冻结
7. **测试断言的精度**：末端匹配（`endswith`）无法锁定切点位置，
   需配合长度断言或**字符串全等断言**双重锁定
8. **魔法数字抽常量**：即使只 1 处使用，抽为常量也有价值（自解释 + 单点调整）

### 10.2 踩坑教训

1. **函数签名漏改**：M3.1 补丁合并时漏改 `_split_by_h2` 签名，导致 `NameError`
   - 教训：用 IDE 重构功能或本地跑测试再提交
2. **测试断言过度精确**：`match="缺失YAML frontmatter"` 与代码文案的空格不一致
   - 教训：断言用关键词或**完整前缀（含空格）**
3. **测试逻辑反转**：`"。"[:-1]` 求值为 `""`，`endswith("")` 恒为 True
   - 教训：边界断言的正确性需独立验证
4. **ORM/DB 列名差异**：`meta_json` ↔ `metadata`（规避 SQLAlchemy 保留字）
   - 教训：psql 调试时用真实列名
5. **DDL 幂等性**：`ALTER TABLE ADD CONSTRAINT` 无 `IF NOT EXISTS`，重跑会失败
   - 教训：非幂等语句用 `DO $$ ... IF NOT EXISTS ... END $$;` 包裹
6. **overlap 跨 section 污染**：字符级 overlap 会混入相邻 section 内容
   - 教训：语义切片下不支持 overlap；如需 overlap 用 v1.0
7. **单段超长无兜底**：无 `\n\n` 的超长段落会产出超大 chunk
   - 教训：字符硬切兜底必须优先在句子边界切
8. **测试断言过于宽松**：仅 `endswith` 无法锁定切点位置
   - 教训：精确测试需「内容 + 长度」双重断言，或**字符串全等断言**
9. **魔法数字可读性差**：`400 + 50` 中的 `50` 来源不明
   - 教训：常量化 + 注释说明来源

### 10.3 命名规范建议

建议在 `CONTRIBUTING.md` 补充：

| 表 | DB 列名 | ORM 属性名 | 原因 |
| :--- | :--- | :--- | :--- |
| knowledge_doc | metadata | meta_json | 规避 Base.metadata 保留字 |
| knowledge_chunk | metadata | meta_json | 同上 |

规则：新增列优先使用无冲突的列名（`meta_json` / `extra_json` / `attributes`）；
只有历史列已冻结时才用 `mapped_column(..., name="...")` 映射。

---

## 11. 冻结签字

| 项 | 内容 |
| :--- | :--- |
| Phase H 版本 | v1.7 |
| 冻结日期 | 2026-09-17 |
| 冻结结论 | **PASS** |
| 测试证据 | Phase H 冻结时 **34 passed**（29 单测 + 5 集成）；Sprint 0 收官时 RAG 相关 **44 passed**（整体套件 263 passed） |
| DB 证据 | 5 doc + 5 chunk，chunk 级 section 各不相同 |
| 交付物完整性 | 5 种子 + 5 模块 + 2 表 + 1 配置 + 1 测试文件 |
| 优化批次 | v1.1（M1~M5+M3.1）+ v1.2（优化 1~6）+ v1.3（BUG1~3）+ v1.4（优化 1~7）+ v1.5（优化 1~5）+ v1.6（缺陷 1~2）+ v1.7（方案 A.1~A.3） |
| 遗留问题 | 8 项（Sprint1 处理，不阻塞冻结） |

**签字**：
- Phase H 负责人：______
- 架构组：______
- 日期：2026-09-17

---

> **文档版本**：v1.7
> **编写日期**：2026-09-17
> **上游文档**：
> - `Sprint0_详细执行计划_V1.2_最终版_plan.md` §13
> - `第一阶段工程实施方案_V1.2_增强版_plan.md` §6
> - `docs/freeze/phase_h_expert_calibration.md`（同期冻结纪要）
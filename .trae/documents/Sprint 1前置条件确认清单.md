# Sprint 1 前置条件确认清单（执行依据）

> **文档定位**：本文件是 Sprint 1 进入执行阶段的**正式前置条件冻结记录**，基于 Sprint 1 融合版 §21 + §23.6，以及 2026-10-12 的实际仓库核对结果。
> **使用方式**：本文件是阶段 A 启动的**唯一依据**。任何与本文不一致的文档表述，以本文为准。
> **生效范围**：Sprint 1 阶段 A 至阶段 J 全程有效。

---

## 0. 文档状态

| 项 | 值 |
|---|---|
| 前置条件总数 | 6 + 2（含今晚补充验证项） |
| 已关闭数 | **8 / 8** |
| 阻塞项 | 无 |
| 可进入阶段 A | **是** |
| 冻结日期 | 2026-10-12 |
| 依据文档 | `Sprint1_详细执行计划_V1.0_正式版_plan.md` §21 / §23.6 |

---

## 1. 前置条件总表

| # | 前置条件 | 责任人 | 状态 | 冻结结论 |
|---|---|---|---|---|
| 1 | Defect 10 字段清单确认 | 技术负责人 | ✅ 关闭 | 以 `models.py::ReviewDefect` 为唯一源 |
| 2 | `expert_decision` 存储位置确认 | 技术负责人 | ✅ 关闭 | `review_defect` 加 3 字段 + CHECK + 索引 |
| 3 | 10 个 Case 真实目录名确认 | 数据负责人 | ✅ 关闭 | `case001` 保留 + 新增 9 个 |
| 4 | Alembic 初始迁移方案确认 | 技术负责人 | ✅ 关闭 | 新建完整初始迁移，`002_schema.sql` 退居历史基线 |
| 5 | 依赖版本确认 | 技术负责人 | ✅ 关闭 | `langgraph==0.2.35` / `alembic==1.20.0` / `openai==1.109.1` |
| 6 | P0/P1/P2 容量裁剪确认 | 项目负责人 | ✅ 关闭 | 档位 B：2 人 × 8 工作日 = 16 人日 |
| 7 | `DATABASE_URL` 环境变量可用 | 技术负责人 | ✅ 关闭 | 容器内可见，格式符合 Sprint 0 §17.1 |
| 8 | `models.py` 新增字段撞名检查 | 技术负责人 | ✅ 关闭 | 无撞名，可安全新增 |

---

## 2. 前置条件 1：Defect 10 字段清单

### 2.1 冻结结论

Defect 10 字段以 `backend/app/persistence/models.py::ReviewDefect` 为**唯一事实来源**。

| # | 字段 | ORM 类型 | DB 类型 | 必填 | 说明 |
|---|---|---|---|---|---|
| 1 | `defect_id` | `String(64)` | `VARCHAR(64) NOT NULL UNIQUE` | 是 | 唯一 ID |
| 2 | `category` | `String(64)` | `VARCHAR(64) DEFAULT 'other'` | 是 | 分类枚举 |
| 3 | `location` | `JSONB` | `JSONB NOT NULL` | 是 | `{sheet, path, coords}` |
| 4 | `component` | `String(100)` | `VARCHAR(100)` | 否 | 元件 refdes |
| 5 | `net` | `String(100)` | `VARCHAR(100)` | 否 | 网络名 |
| 6 | `risk` | `String(16)` | `VARCHAR(16) DEFAULT 'medium'` | 是 | 风险等级（与 `severity` 同义映射） |
| 7 | `evidence` | `JSONB` | `JSONB NOT NULL DEFAULT '[]'::jsonb` | 是 | Evidence 三要素数组 |
| 8 | `root_cause` | `Text` | `TEXT NOT NULL` | 是 | 根因 |
| 9 | `suggestion` | `Text` | `TEXT NOT NULL` | 是 | 建议 |
| 10 | `confidence` | `Float` | `FLOAT DEFAULT 0.5` | 是 | 置信度 |

### 2.2 固有追踪字段（不计入 10 字段，但必须存在）

| 字段 | 类型 | 说明 |
|---|---|---|
| `review_status` | `String(32) DEFAULT 'AI_CONFIRMED'` | 三态质量状态 |
| `origin` | `String(32) DEFAULT 'rule'` | 来源追踪：`rule` / `ai_discovered` / `hybrid` |
| `rule_id` | `String(64)` | 关联规则 ID |

### 2.3 落地位置

- **代码**：`app/persistence/models.py::ReviewDefect`（唯一源）
- **文档**：`docs/ai_validation/ai_quality_control.md` 引用上述源
- **迁移**：Alembic 初始迁移

---

## 3. 前置条件 2：`expert_decision` 存储位置

### 3.1 冻结结论

在 `review_defect` 表新增 3 个字段，**不改任何已有字段名**。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `expert_decision` | `VARCHAR(32) NOT NULL` | `'pending'` | 专家最终判定 |
| `expert_decision_updated_at` | `TIMESTAMPTZ` | NULL | 判定时间（审计） |
| `expert_decision_by` | `BIGINT FK → users.id` | NULL | 判定人（审计） |

### 3.2 枚举值

| 值 | 含义 |
|---|---|
| `pending` | 尚无专家最终判定 |
| `confirmed` | 专家确认该候选缺陷成立 |
| `rejected` | 专家判定该候选缺陷不成立 |
| `needs_verification` | 需要补充资料或进一步核验 |

### 3.3 约束

- `CHECK (expert_decision IN ('pending','confirmed','rejected','needs_verification'))`
- `INDEX idx_review_defect_expert_decision (expert_decision)`

### 3.4 与 `feedback_item` 的边界

- **`review_defect.expert_decision`**：当前专家决定（快照），便于查询。
- **`feedback_item`**：每次反馈的历史记录，**不可覆盖**。
- 两者关系：`expert_decision` 是"最终态"，`feedback_item` 是"过程态"。
- `AI_CONFIRMED` **绝不**自动把 `expert_decision` 设为 `confirmed`。

---

## 4. 前置条件 3：10 个 Case 目录名

### 4.1 冻结清单

| # | 目录名 | 类别 | 用途 | 优先级 |
|---|---|---|---|---|
| 1 | `case001` | A | Buck 转换器（**已存在，不改名**） | P0 |
| 2 | `case002` | A | LDO 线性稳压 | P0 |
| 3 | `case003` | A | 晶振 / 时钟电路 | P0 |
| 4 | `case004` | A | UART / 串口接口 | P0 |
| 5 | `case005` | A | CAN 总线 | P0 |
| 6 | `case006` | A | MCU 外围 / 复位电路 | P0 |
| 7 | `case_b01_buck_thermal_derating_B` | B | Buck 热降额（AI 辅助发现） | P0 |
| 8 | `case_b02_reset_rc_B` | B | Reset RC 拓扑 | P1 |
| 9 | `case_b03_io_direction_B` | B | IO 方向冲突 | P1 |
| 10 | `case_b04_power_sequence_B` | B | 电源时序 | P1 |

### 4.2 命名规则

- **A 类**：`case<3位数字>`，纯数字无后缀。
- **B 类**：`case_b<2位数字>_<短描述>_B`，描述用下划线分词，结尾 `_B` 标明 B 类。
- `info.yaml` 里 `case_id` 必须等于目录名。
- `case_class` ∈ `{A, B}`。

### 4.3 每个 Case 的资产清单

**A 类最少**：
- `info.yaml`
- `schematic_ir.json`
- `expected_review.json`
- `evaluation.yaml`

**B 类必须**：
- `info.yaml`
- `schematic_ir.json`
- `expected_review.json`
- `evaluation.yaml`
- `expert_reasoning.md`
- `evidence/{datasheet,reference_design,application_note}/`（至少一类非空）

### 4.4 当前状态

- `data/cases/case001/` 已存在，结构完整（含 `evidence/` / `expert_reasoning.md` / `evaluation.yaml` / `expected_review.json` / `info.yaml` / `schematic_ir.json` / `circuit_notes.md`）。
- `case002`~`case006`、`case_b01`~`case_b04` **尚未建立**，阶段 B 补齐。

### 4.5 CI 校验

阶段 B 加 `tests/unit/test_case_naming.py`，校验：
- 目录名与 `info.yaml::case_id` 一致。
- A/B 类命名规则符合上述模式。
- B 类三件套齐全。

---

## 5. 前置条件 4：Alembic 初始迁移方案

### 5.1 冻结结论

**Sprint 1 的完整数据库 Schema 以 Alembic migration 为唯一事实来源。**

### 5.2 执行动作

1. `backend/alembic/` 与 `backend/alembic.ini` **已由 `alembic init` 生成**（容器内 `/app/`，已同步宿主机）。
2. **改 `alembic.ini`**：注释掉 `sqlalchemy.url`，连接串由 `env.py` 从环境变量读。
3. **改 `env.py`**：`target_metadata = Base.metadata`；从 `DATABASE_URL` 环境变量读连接。
4. **新建完整 Sprint 1 初始迁移**：含 12 张业务表 + Sprint 1 新增字段。
5. **`002_schema.sql` 退居历史基线**：移入 `infra/docker/initdb/_deprecated/`，不再自动执行。
6. **`initdb/` 只保留 `001_pgvector.sql`**。

### 5.3 验收标准

- 空库 `alembic upgrade head` 成功。
- `alembic current` = head。
- `pg_tables` 中业务表数 = 12，加 `alembic_version` 共 13。
- 二次 `autogenerate` 产出空迁移（ORM 与 DB 零漂移）。

### 5.4 迁移文件必含（autogenerate 不会自动生成，需手补）

- `CREATE EXTENSION IF NOT EXISTS vector`
- `update_updated_at_column()` 触发器函数
- `check_feedback_item_type` CHECK 约束（6 类反馈）
- `check_review_defect_expert_decision` CHECK 约束
- 所有含 `updated_at` 表的触发器（`update_<table>_updated_at`）
- `knowledge_chunk.embedding` 为 `Vector(1536)`（需 import `pgvector.sqlalchemy.Vector`）

### 5.5 边界约束

- **禁止**用 `Base.metadata.create_all` 建表。
- **禁止**手工建表绕过 Alembic。
- **禁止**在生产环境执行 `downgrade`。

---

## 6. 前置条件 5：依赖版本

### 6.1 冻结版本

| 依赖 | 锁定版本 | 说明 |
|---|---|---|
| Python | `>=3.11,<3.12` | 基础镜像 `python:3.11-slim` |
| uv | 0.12.6 | 官方镜像 `ghcr.io/astral-sh/uv:0.12.6` |
| `fastapi` | 0.104.1 | 已有 |
| `uvicorn[standard]` | 0.24.0 | 已有 |
| `pydantic` | 2.5.2 | 已有 |
| `pydantic-settings` | 2.1.0 | 已有 |
| `sqlalchemy` | 2.0.23 | 已有 |
| `psycopg[binary]` | 3.1.18 | 已有，协议 `postgresql+psycopg://` |
| `pgvector` | 0.5.0 | 已有 |
| `langchain-openai` | 0.2.8 | 已有 |
| `langchain-core` | 0.3.59 | 锁文件解析 |
| **`langgraph`** | **0.2.35** | **Sprint 1 新增** |
| **`alembic`** | **1.20.0** | **Sprint 1 新增** |
| **`openai`** | **1.109.1** | **Sprint 1 显式固定** |

### 6.2 兼容性验证

- `langgraph==0.2.35` 与 `langchain-core==0.3.59` **兼容**，无冲突。
- `alembic==1.20.0` 与 `sqlalchemy==2.0.23` **兼容**。
- `openai==1.109.1` 与 `langchain-openai==0.2.8` 间接引入版本**一致**。
- `uv sync --frozen --extra dev` 验证通过。
- dev 工具链（`pytest 7.4.3` / `ruff 0.1.6`）已恢复。

### 6.3 待办（不阻塞阶段 A）

- 将 `pyproject.toml` 中 `alembic>=1.20.0` / `langgraph>=0.2.35` 改为 `==`，与 Sprint 0 §17.3 锁定风格一致。
- 回写 `Sprint1_详细执行计划...` §12.2 / §17 依赖表为实际锁定版本。

---

## 7. 前置条件 6：P0/P1/P2 容量裁剪

### 7.1 容量基线

- **团队配置**：2 人 × 8 工作日 = **16 人日**
- **原估算**：约 16 人日（Sprint 1 §14）
- **缓冲**：1.5 人日

### 7.2 角色分配

| 角色 | 承担范围 | 人日 |
|---|---|---|
| 技术负责人 A | DB/Alembic、LLM Provider、LangGraph、OutputParser、Benchmark、API | 8 |
| 技术负责人 B | Case 资产、10 条规则、RAG、反馈闭环、Web MVP、Demo | 8 |

### 7.3 裁剪表

| 优先级 | 交付内容 | 负责 | 人日 |
|---|---|---|---|
| **P0** | A. Sprint 0 回归 + Alembic 唯一 Schema + ORM/迁移一致 | A | 1.0 |
| **P0** | B. 10 个 Case 资产（6A+4B）+ 覆盖矩阵 + 校验脚本 | B | 2.5 |
| **P0** | B'. 10 条规则（含 POWER_003 降额算法）+ 单测 | B | 1.5 |
| **P0** | C. LLM Provider（mock/openai/vllm）+ Prompt + OutputParser | A | 1.5 |
| **P0** | D. LangGraph 六节点 + State + ≥1A+1B E2E | A | 1.5 |
| **P0** | E. 6 类反馈 + 规则候选草稿 + 知识缺口导出 | B | 1.0 |
| **P0** | expert_decision 持久化 + 状态矩阵测试 | A | 0.5 |
| **P0** | AI_NER / EAR 公式 + 单元测试 | A | 0.5 |
| **P0 小计** | | | **10.0** |
| **P1** | F. Benchmark V1.2（Rule Only + Full Review）+ CSV | A | 1.0 |
| **P1** | G. API 路由 + OpenAPI + API 测试 | A | 1.0 |
| **P1** | H. Web MVP | B | 1.5 |
| **P1** | I. `demo_sprint1.ps1` 端到端 Demo | B | 0.5 |
| **P1 小计** | | | **4.0** |
| **P2** | J. 文档收尾、风险清单、变更记录 | B | 0.5 |
| **P2 小计** | | | **0.5** |
| **合计** | | | **14.5 + 1.5 缓冲 = 16.0** |

### 7.4 裁剪原则

- **P0 全部保留**：不满足则不得宣布 Sprint 1 核心验收通过。
- **P1 全部保留**，但 H（Web MVP）和 I（Demo）排最后，阻塞时优先砍。
- **P2 只留 0.5 人日文档收尾**，其余进 backlog。
- **缓冲 1.5 人日**：吸收 Case 资产返工、`langgraph` 适配、PS 5.1 转义踩坑。

### 7.5 明确不在 Sprint 1 范围（拒绝扩容）

- 真实 EDA 解析 / Cadence 适配器
- vLLM 实际部署（只做兼容层 + 配置路径）
- 真实 PDF 解析 / OCR
- MCP Server 抽离 / Planner / ReAct / 多 Agent
- 企业级 RBAC / 审计 / 规则管理后台
- 文件上传 UI / 上传数量 / 频率限制 / 配额

### 7.6 阻塞规则

- P0 阻塞超过 1 人日 → 立即升级项目负责人。
- P1 阻塞 → 记录并延后，不影响 P0 验收。
- 需求边界或指标口径歧义 → 先更新决策记录，再继续实现。

---

## 8. 附加前置条件 7：`DATABASE_URL` 环境变量

### 8.1 验证结果

容器内 `DATABASE_URL` 可见，格式：
 `postgresql+psycopg://postgres:dev_secure_pwd_CHANGE_ME@postgres:5432/schematic_review` 

### 8.2 对齐检查

| Sprint 0 §17.1 约定 | 实际值 | 结果 |
|---|---|---|
| 协议 `postgresql+psycopg://` | ✅ | 对齐 |
| 主机用服务名（非 localhost） | `postgres` | 对齐 |
| 配置来源 compose `environment` | ✅ | 对齐 |

### 8.3 落地位置

- `infra/docker/docker-compose.yml` 第 120 行：`DATABASE_URL: ${DATABASE_URL:-postgresql+psycopg://...}`
- 优先级：`.env` 文件中 `DATABASE_URL` > 自动拼接默认值。

---

## 9. 附加前置条件 8：`models.py` 撞名检查

### 9.1 检查结果

| 新增字段 | 现有同名？ | 结论 |
|---|---|---|
| `ReviewDefect.expert_decision` | 无 | ✅ 安全新增 |
| `ReviewDefect.expert_decision_updated_at` | 无 | ✅ 安全新增 |
| `ReviewDefect.expert_decision_by` | 无 | ✅ 安全新增 |
| `ReviewResult.schematic_case_id` | 无（`IRDocument` 有，不同类） | ✅ 安全新增 |
| `RuleCandidate.schematic_case_id` | 无 | ✅ 安全新增 |

### 9.2 需删除字段

| 类 | 字段 | 原因 |
|---|---|---|
| `RuleCandidate` | `task_id` | Sprint 1 §8 明确删除（BigInteger 无 FK，语义重复） |

### 9.3 不改名原则

- **不修改任何已有字段名。**
- `ReviewResult.task_id`（String 64）**保留**为批次标识。
- `RuleCandidate.task_id`（BigInteger）**删除**。
- 两者同名不同表，删除后者不影响前者。

### 9.4 FK 语义差异说明

| 类 | 字段 | ondelete | 语义 |
|---|---|---|---|
| `IRDocument` | `schematic_case_id` | `CASCADE` | IR 属于 case，case 删则 IR 删 |
| `ReviewResult` | `schematic_case_id` | `SET NULL` | 评审独立，case 删则清空关联 |
| `RuleCandidate` | `schematic_case_id` | `SET NULL` | 候选独立，case 删则清空关联 |

**这是 Sprint 1 §8 冻结决策，非笔误。**

---

## 10. 阶段 A 执行清单（依据本文件）

| 任务 | 内容 | 验收 |
|---|---|---|
| 任务 1 | 改 `alembic.ini`：注释 `sqlalchemy.url` | `sqlalchemy.url` 无匹配 |
| 任务 2 | 改 `env.py`：`target_metadata = Base.metadata` + 从 `DATABASE_URL` 读 | `len(Base.metadata.tables) == 12` |
| 任务 3 | 改 `models.py`：最小改动版（5 新增 + 1 删除） | 无撞名、`task_id` 已删 |
| 任务 4 | `002_schema.sql` 移入 `_deprecated/` | `initdb/` 只剩 `001_pgvector.sql` |
| 任务 5 | 清库 + autogenerate + 手补约束 + `upgrade head` | 12 张表 + `alembic_version` |
| 任务 6 | 二次 autogenerate 确认空迁移 | ORM 与 DB 零漂移 |

**任务 1~6 全部通过 → 阶段 A 验收通过 → 进入阶段 B。**

---

## 11. 变更记录

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-10-12 | 初版 | 6 项前置条件 + 2 项补充验证全部关闭 |
| 2026-10-12 | `DATABASE_URL` 验证 | 容器内可见，格式对齐 Sprint 0 §17.1 |
| 2026-10-12 | `models.py` 撞名检查 | 无撞名，可安全新增 |
| 2026-10-12 | 依赖版本冻结 | `langgraph==0.2.35` / `alembic==1.20.0` / `openai==1.109.1` |

---

## 12. 引用文档

| 文档 | 作用 |
|---|---|
| `Sprint1_详细执行计划_V1.0_正式版_plan.md` | Sprint 1 执行基线 |
| `Sprint0_详细执行计划_V1.2_最终版_plan.md` | Sprint 0 工程基线（内容版本 V1.3） |
| `第一阶段工程实施方案_V1.2_增强版_plan.md` | 上游业务基线 |
| `项目实施方案_修订版_MVP闭环验证_plan.md` | 顶层 MVP 目标 |
| `backend/app/persistence/models.py` | Defect 10 字段唯一源 |
| `backend/pyproject.toml` / `backend/uv.lock` | 依赖版本唯一源 |
| `infra/docker/docker-compose.yml` | `DATABASE_URL` 定义 |
| `infra/docker/initdb/002_schema.sql` | 历史基线（Sprint 1 起废弃） |
| `backend/alembic/versions/` | Schema 唯一事实来源（Sprint 1 起） |

---

**文档版本**：V1.0
**冻结日期**：2026-10-12
**状态**：生效中，作为 Sprint 1 阶段 A 启动依据
**下一步**：按第 10 章阶段 A 执行清单启动任务 1



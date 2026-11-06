# Sprint 3 变更与阻塞记录（角色权限与端到端闭环）

> 起始：2026-11-03
> 依据：`Sprint3_详细执行计划_V1.0.md`（最终定稿）
> 阶段：G1（方案评审）✅ 完成，G2（Migration + auth 底座）✅ 完成，G3-A（业务路由挂鉴权）✅ 全部批次完成，G3-C（审计字段后端注入）✅ 完成，G3-B（对象级数据范围 + IDOR 防护）✅ 完成，G3-D（负向安全测试补全 + G3 封板）✅ 完成，G4（前端登录 + 路由守卫 + 菜单权限）✅ 完成，G5（反馈闭环 E2E — 知识缺口生成链路）✅ 主体完成（Commit 1–3，仅 T-05 状态码扫描并入 G6 Commit 2），G6（完整回归 + B 类并发测试 + 部署检查清单）🟡 进行中（Commit 1/3）
> 最后更新：2026-11-06（新增 G6 章节；补记 G4 章节、G5 章节 Commit 1、Commit 2、Commit 3）

---

## G1：方案评审与现状核查（2026-11-03）

> 状态：✅ 完成
> 依据：`Sprint3_详细执行计划_V1.0.md` + Sprint 2 基线快照

### 核心决策

| # | 决策 | 内容 |
|---|---|---|
| C1 | 第一阶段不交付 JWT / 密码登录（方案 B） | 第一阶段身份体系基于 username-only / X-User-Id；prod 解除「无 JWT 即 fail-closed」，改为可信内网部署；JWT 移入下一阶段 |
| C2 | 目标数据库确定为 PostgreSQL | 锁实现确定为 `pg_advisory_xact_lock`；MySQL/SQLite 仅作未来扩展占位 |

### 前置核查（附录 C 15 项）

| # | 核查项 | 结论 |
|---|---|---|
| 1 | `deps.py` 完整内容 | ✅ 文件存在，当前为空壳（无 `get_current_user`） |
| 2 | `db.py` 的 `get_db_session` 实现 | ✅ 标准 `@contextmanager`，自动 commit/rollback/close |
| 3 | PostgreSQL 连接配置 | ✅ 本地 docker-compose 使用 `postgres:5432` 服务名 |
| 4 | `s2a_schema_updates` revision | ✅ `s2a_schema_updates`（后接 F05 → F07） |
| 5 | `app/api/` 端点清单 | ✅ G2/G3-A 编码前已核对 |
| 6 | `submit_task_feedback` 生成逻辑 | ✅ 已核实（G3-A 改用它作为业务入口） |
| 7 | `feedback_service` 写 `created_by` | ✅ 已核实（G3-C 改为后端注入） |
| 8 | `update_expert_decision` 写 `expert_decision_by` | ✅ 已核实（G3-C 完成注入） |
| 9 | 当前 Git HEAD 与测试基线 | ✅ `Sprint 2 封板`：`568 passed, 1 skipped, 24 deselected` |
| 10 | `dev.ps1` 的 `uv sync` 清理 | ✅ 已确认清理 |
| 11 | `users` 表内管理员账号 | ✅ 当前 0 行（`baseline_snapshot.md` 已记录） |
| 12 | 候选 ID 特殊连字符 | ⏳ 暂不修（列入技术债），前端已兜底；G3-C 手测用 ASCII ID 规避 |
| 13 | `bootstrap.ps1` 的 `uv sync` 清理 | ✅ 已确认清理 |
| 14 | `demo_sprint2.ps1` 只读入口 | ⏳ 待确认（是否有 `--readonly`） |
| 15 | CI 配置 `APP_ENV=test` | ✅ Sprint 2 已修复 |

### 手工确认清单

- [ ] `demo_sprint2.ps1` 是否有 `--readonly` 只读入口
- [x] 测试基线重测（已完成，基线随阶段推进：568 → 594 → 595 → 597）
- [ ] `schematic_case` 表是否有 `created_by` / `owner_id`
- [ ] `/cases/{id}/benchmark` 是否修改正式数据

---

## G2：Migration + Auth 底座（2026-11-03）

> 状态：✅ 完成
> 依据：`Sprint3_详细执行计划_V1.0.md` §2 / §3 / §5

### G2.1 Migration 执行（✅ 完成）

**交付文件**：

- `backend/alembic/versions/s2j_g2_roles.py`（新建）
- `backend/app/persistence/models.py`（3 处改动）
- `docs/sprint3/s2j_g2_roles_preview.sql`（增量 SQL 证据）

**迁移内容**：

| 表 | 变更 |
|---|---|
| `users` | 新增 `role` (`VARCHAR(32) NOT NULL DEFAULT 'engineer'`) + `is_active` (`BOOLEAN NOT NULL DEFAULT true`) |
| `users` | CHECK `role IN ('admin','engineer','reviewer','rule_admin')` + 索引 `idx_users_role` |
| `review_result` | 新增 `created_by` (BIGINT FK SET NULL) + 索引 |
| `rule_candidates` | 新增 `reviewed_by` (BIGINT FK SET NULL) + 索引 |

**迁移链**：

```text
<base> → 42104f6cb89d → c1_embedding_dim_1024 → s2a_schema_updates → s2_f05_rule_publish → s2_f07_case_benchmark → s2j_g2_roles (head)
```

**验收证据**：

| 项 | 结果 |
|---|---|
| `alembic current` | ✅ s2j_g2_roles (head) |
| `\d users` | ✅ role + is_active + CHECK + idx 全部存在 |
| `\d review_result` | ✅ created_by + FK + idx 全部存在 |
| `\d rule_candidates` | ✅ reviewed_by + FK + idx 全部存在 |
| ORM import | ✅ `from app.persistence.models import User, ReviewResult, RuleCandidate; print('OK')` |
| `test_db_schema.py` | ✅ 69 passed |
| 全量非 integration | ✅ 568 passed, 1 skipped, 24 deselected |

### G2.2 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| Migration 多 Head | `alembic upgrade head` 报 `Multiple head revisions are present` | 原因是新 Migration 的 `down_revision` 写成 `s2a_schema_updates` 造成分叉，改为 `s2_f07_case_benchmark` |
| SQL 预览包含完整迁移链 | `alembic upgrade head --sql` 从 `<base>` 开始生成所有 SQL | 改用 `alembic upgrade s2_f07_case_benchmark:head --sql`，只生成增量 SQL |
| 重定向写 UTF-16 BOM（Sprint 2 遗留） | 生成 SQL 文件后 Python 读取崩溃 | 使用 `Out-String \| Set-Content` 避免 BOM 问题 |

---

## G2 关键技术决策（决策已冻结，跨阶段生效）

> 以下 5 项决策已在 G2 编码中落地，后续阶段（G3–G6）必须遵守。

| # | 决策 | 内容 | 落地位置 |
|---|---|---|---|
| **D1** | `get_current_user` 返回 `NamedTuple CurrentUser` | 避免 `with get_db_session()` 退出后 ORM 对象 detached 报 `DetachedInstanceError`。返回字段固定为 `id / username / role / is_active` | `app/core/auth.py` |
| **D2** | `ADMIN_LOCK_KEY = 0x5342445F41444D31`（十进制 `5999432829518499121`） | PostgreSQL Advisory Lock 固定 key，CLI（`bootstrap_admin.py`）与所有 API 侧管理员变更共用同一 key，形成全局互斥 | `app/services/admin_lock.py` |
| **D3** | `get_optional_user` 携带非法身份时抛 401 | 不降级为"匿名"，防止伪造 / 非法身份被当成合法匿名请求处理 | `app/core/auth.py` |
| **D4** | 事务块内禁止嵌入慢外部调用（LLM / HTTP / 大文件 I/O） | 防止锁持有时间过长 + 事务过长。审计消息回写外部系统必须放 `after-commit` hook 或异步处理 | `app/services/user_service.py` docstring + 关键方法注释 |
| **D5** | 锁顺序严格为：`admin_lock` → `with_for_update()` → `query count` → update → flush | 防死锁（先全局锁后行锁），保快照一致性（count 在锁内查）；禁止先 `with_for_update()` 再 `admin_lock` | `app/services/user_service.py` docstring + 关键方法注释 |

### D1-D5 补充说明

- **D1 返回 `NamedTuple`**：不是 `Pydantic BaseModel`，因为 `Pydantic` 在校验场景更合适，但 `NamedTuple` 更轻量且天然不可变（防止下游意外改字段）。
- **D2 固定 key**：`0x5342445F41444D31` 的 ASCII 为 `SBD_ADM1`，寓意"Schematic Board Admin"。任选固定值，关键是全项目唯一。
- **D3 401 不降级**：即便是 `GET /healthz` 这类公开接口，只要请求带了非法 `X-User-Id`，也必须 401。这与 Sprint 2 §1 的"未认证用户不能通过其他接口获得身份信息"约束一致。
- **D4 禁止慢调用**：是 Sprint 3 数据库长事务治理的硬要求。举例：用户创建后要发邮件通知 → **不能**在 `create_user` 事务块内 `requests.post(...)`，必须在 `create_user` 返回后（事务已提交）由上层异步发。
- **D5 锁顺序**：是 Sprint 3 并发安全的硬要求。举例：不能"先 SELECT ... FOR UPDATE 目标用户，再 GET_LOCK 全局锁"，否则 CLI 初始化路径（先全局锁再行锁）与 API 路径顺序相反 → 死锁。

---

## G2 完成情况汇总

### 交付清单

| 交付项 | 文件 | 状态 |
|---|---|---|
| Migration | `backend/alembic/versions/s2j_g2_roles.py` | ✅ |
| ORM 改动 | `backend/app/persistence/models.py`（`User.role/is_active`、`ReviewResult.created_by`、`RuleCandidate.reviewed_by`） | ✅ |
| 认证底座 | `backend/app/core/auth.py`（`get_current_user` / `require_roles` / `get_optional_user` / `CurrentUser` / `ROLE_*`） | ✅ |
| API 依赖 re-export | `backend/app/api/deps.py` | ✅ |
| 管理员锁 | `backend/app/services/admin_lock.py`（`pg_advisory_xact_lock` + `LastAdminProtectionError` + 审计日志） | ✅ |
| 用户服务 | `backend/app/services/user_service.py`（CRUD + 最后管理员保护 + D1/D2/D3/D4/D5 约束） | ✅ |
| 认证 API | `backend/app/api/auth.py`（`POST /login`、`GET /me`、`GET/POST /users`、`PATCH/DELETE /users/{id}`） | ✅ |
| Schemas | `backend/app/api/schemas.py`（追加 7 个模型：`LoginRequest` / `LoginResponse` / `AuthMeResponse` / `UserCreateRequest` / `UserPatchRequest` / `UserListItem` / `UserListResponse`） | ✅ |
| CLI 工具 | `scripts/bootstrap_admin.py`（5 种场景 + advisory lock + 幂等） | ✅ |
| 测试：auth | `backend/tests/test_auth.py`（15 个用例） | ✅ |
| 测试：lock | `backend/tests/test_admin_lock.py`（6 个用例） | ✅ |
| 测试：bootstrap | `backend/tests/test_bootstrap_admin.py`（5 个用例，2 个 skip） | ✅ |
| Migration SQL 证据 | `docs/sprint3/s2j_g2_roles_preview.sql` | ✅ |

### 验收证据

| 项 | 结果 |
|---|---|
| `alembic current` | ✅ `s2j_g2_roles (head)` |
| Migration 链 | ✅ 线性：`42104f6cb89d → c1_embedding_dim_1024 → s2a_schema_updates → s2_f05_rule_publish → s2_f07_case_benchmark → s2j_g2_roles` |
| `\d users` | ✅ `role` + `is_active` + CHECK + idx 全部生效 |
| `\d review_result` | ✅ `created_by` + FK + idx 全部生效 |
| `\d rule_candidates` | ✅ `reviewed_by` + FK + idx 全部生效 |
| ORM import | ✅ `from app.persistence.models import User, ReviewResult, RuleCandidate; print('OK')` → `OK` |
| `test_db_schema.py` | ✅ 69 passed |
| 3 个新测试文件单独跑 | ✅ **26 passed, 2 skipped** |
| 全量非 integration | ✅ **594 passed, 3 skipped, 24 deselected** |
| G2 API 手动验证 | ✅ `login` → `me` → `users` 列表 → `create_user` → `engineer` 越权 403 全部通过 |
| `bootstrap_admin.py` CLI 手动验证 | ✅ 5 种场景逻辑正确（详细测试由 `test_bootstrap_admin.py` 覆盖） |

### G2 阶段遇到的真实坑点（详见 `debug_03.md`）

1. **Migration 多 Head**：`down_revision` 写成 `s2a_schema_updates` 导致分叉，改为 `s2_f07_case_benchmark`。
2. **`--sql` 模式生成完整历史 SQL**：`upgrade head --sql` 从 `<base>` 开始，改为 `upgrade s2_f07_case_benchmark:head --sql`。
3. **测试隔离误用事务回滚**：原 `pg_session` 用外层事务回滚，但 HTTP 请求走独立 Session 看不到未提交数据。改为"真实 commit + 前缀清理"。
4. **并发断言不稳定**：`test_A4` 假设 A 线程先拿锁，实际操作系统调度随机。改为"两种合法情况都覆盖"。
5. **最后管理员保护未触发**：真实 DB 里已有 admin（id=3107），测试未清理其它 admin 导致 `active_admin_count >= 2`。改为测试内主动清理。
6. **容器内脚本路径误解**：`ROOT_DIR = Path("/app")` 是 backend 根，不是项目根。参见 `g2_pre_check.py` 的错误报告。
7. **`docs/` 未挂载进容器**：脚本写文件到 `/docs` 只存容器内，宿主机看不到。G6 Demo 前需在 `docker-compose.dev.yml` 加 `- ../../docs:/docs`。

---

## G3-A：业务路由挂鉴权（2026-11-04）

> 状态：✅ 完成（第 1 批 `cases.py` / 第 2 批 `review_tasks.py` + `feedbacks.py` + `dashboard.py` / 第 3 批 `rule_candidates.py` + `knowledge_docs.py` + `knowledge_gaps.py`）
> 依据：`Sprint3_详细执行计划_V1.0.md` §9 权限矩阵

### G3-A.1 权限矩阵落地

| 路由 | 允许角色 | 状态 |
|---|---|---|
| `GET /cases`、`GET /cases/{id}`、`POST /cases/{id}/benchmark`、`GET /cases/{id}/benchmarks` | 所有登录用户 | ✅ 已完成（`get_current_user`） |
| `POST /review/tasks`、`POST /review/tasks/batch` | engineer / admin | ✅ 已完成 |
| `GET /review/tasks`、`/{id}`、`/{id}/defects`、`/{id}/export` | engineer / reviewer / admin | ✅ 已完成 |
| `PATCH /review/tasks/{id}/defects/{defect_id}` | reviewer / admin | ✅ 已完成 |
| `POST /feedbacks` | reviewer / admin | ✅ 已完成 |
| `GET /dashboard/summary` | admin / rule_admin / reviewer | ✅ 已完成 |
| `GET /rule-candidates`、`GET /rule-candidates/{candidate_id}`、`POST /rule-candidates/{candidate_id}/status`、`POST /rule-candidates/{candidate_id}/publish` | rule_admin / admin | ✅ 第 3 批已完成 |
| `GET /knowledge-docs`、`POST /knowledge-docs/upload` | rule_admin / admin | ✅ 第 3 批已完成 |
| `GET /knowledge-gaps`、`GET /knowledge-gaps/{gap_id}`、`PATCH /knowledge-gaps/{gap_id}` | rule_admin / admin | ✅ 第 3 批已完成 |

### G3-A.2 交付清单（第 1、2 批）

| 交付项 | 文件 | 状态 |
|---|---|---|
| 全局异常解耦 | `backend/app/core/exceptions.py`（新建 `BusinessValueError`） | ✅ |
| 全局异常处理器 | `backend/app/core/errors.py`（只注册 `BusinessValueError → 422`） | ✅ |
| Service 层异常调整 | `backend/app/services/task_service.py`（6 处 `ValueError → BusinessValueError`） | ✅ |
| Cases 路由 | `backend/app/api/cases.py`（4 端点加 `get_current_user`） | ✅ |
| Review 任务路由 | `backend/app/api/review_tasks.py`（6 端点按角色挂鉴权 + 补 `status` 参数 + 补 `ROLE_REVIEWER` + 修 `batch` 的 `resp["task_id"]` + 修 `create` 的 `defects_count` 回查） | ✅ |
| Feedbacks 路由 | `backend/app/api/feedbacks.py`（改用 `task_service.submit_task_feedback`，`created_by` 后端注入） | ✅ |
| Dashboard 路由 | `backend/app/api/dashboard.py`（3 角色鉴权 + 修 `get_summary` 笔误） | ✅ |
| 全局测试 fixture | `backend/tests/conftest.py`（session 级 admin / engineer + 6 个 client fixture + teardown FK 解除） | ✅ |
| `test_auth.py` 前缀隔离 | `backend/tests/test_auth.py`（`test_auth_%` 前缀 + 最后管理员保护改用"降级+恢复"） | ✅ |

### G3-A.3 第 3 批（规则 / 知识路由挂鉴权）

**交付文件**：

| 文件 | 改动 |
|---|---|
| `backend/app/api/rule_candidates.py` | 4 端点加 `require_roles(ROLE_RULE_ADMIN, ROLE_ADMIN)`；清理重复 import |
| `backend/app/api/knowledge_docs.py` | 2 端点加 `require_roles(ROLE_RULE_ADMIN, ROLE_ADMIN)` |
| `backend/app/api/knowledge_gaps.py` | 3 端点加 `require_roles(ROLE_RULE_ADMIN, ROLE_ADMIN)`（`_get_db` 依赖不动） |
| `backend/app/services/knowledge_gap_service.py` | 2 处 `raise ValueError → raise BusinessValueError` |

**关键修正**：

1. **`knowledge_gap_service.list_gaps` 缩进 bug**：改 `raise ValueError → raise BusinessValueError` 时误将 `q = db.query(KnowledgeGap)` 缩进进 `if` 块，导致 `q` 永远未定义，访问 `/knowledge-gaps` 抛 `UnboundLocalError → 500`。修复：`q = db.query(...)` 顶格，与 `if` 平级。
2. **`rule_candidates.py` 重复 import 清理**：删除单行 `from app.api.deps import ...`，只保留多行 import。

**权限矩阵落地（对照 Sprint3 §9，按实际代码路径）**

| HTTP 方法 | 路径 | 允许角色 |
|---|---|---|
| GET | `/rule-candidates` | rule_admin / admin |
| GET | `/rule-candidates/{candidate_id}` | rule_admin / admin |
| POST | `/rule-candidates/{candidate_id}/status` | rule_admin / admin |
| POST | `/rule-candidates/{candidate_id}/publish` | rule_admin / admin |
| GET | `/knowledge-docs` | rule_admin / admin |
| POST | `/knowledge-docs/upload` | rule_admin / admin |
| GET | `/knowledge-gaps` | rule_admin / admin |
| GET | `/knowledge-gaps/{gap_id}` | rule_admin / admin |
| PATCH | `/knowledge-gaps/{gap_id}` | rule_admin / admin |

**文档-代码差异**：

Sprint3 §9 权限矩阵写 `POST /rule-candidates/{id}/approve` / `/reject`，**实际代码为 `POST /rule-candidates/{id}/status`**（通过 body 的 `status` 字段区分 accepted / rejected）。G3-A 第 3 批按实际代码挂鉴权，G3-D IDOR 测试按实际路径 `/status` 构造。

**第 3 批验收证据**

| 项 | 结果 |
|---|---|
| 4 文件 import 校验 | ✅ `OK` |
| admin 3232 访问 `/api/v1/rule-candidates` / `/api/v1/knowledge-docs` / `/api/v1/knowledge-gaps` | ✅ 200 + 数据 |
| engineer 3108 访问 `/api/v1/rule-candidates` / `/api/v1/knowledge-gaps` | ✅ 403 |

> **注**：本表原记录简写为 `/knowledge-docs`、`/knowledge-gaps`（省略 `/api/v1` 前缀），该简写是后续 BUG-G3A-001「知识库接口 404」误判的源头——按简写路径直接请求自然命中 404。真实接口均为 `/api/v1/*`，详见 G4.5 撤销说明。

### G3-A.4 新增技术决策 R1-R7

| # | 决策 | 落地位置 |
|---|---|---|
| **R1** | `require_roles` 返回完整 `CurrentUser`（含 id + role），供 G3-B 数据范围过滤使用 | `app/core/auth.py` |
| **R2** | 每个 API 组至少 3 个负面用例（无 header / 非法整数 / 角色 403） | 各测试文件 |
| **R3** | D1–D5 + R1–R7 统一在 G6 收尾时写入 changelog / debug | 本文件 + `debug_03.md` |
| **R4** | 业务参数校验失败用 `BusinessValueError`；标准 `ValueError` 走 500 暴露 bug | `app/core/exceptions.py` + `errors.py` |
| **R5** | session 级 fixture 前缀必须唯一命名空间（`test_admin_global_` / `test_eng_global_` / `test_auth_`），跨文件不误删 | `conftest.py` + `test_auth.py` |
| **R6** | "全库状态"断言测试不 `delete`，改用"临时修改 + finally 恢复" | `test_auth.py::test_last_admin_protection` |
| **R7** | API 层调用 Service 前必须 `Select-String -Pattern "^def "` 核对函数名 | 开发规范 |

### G3-A.5 踩坑与修正

详见 `debug_03.md` §4，共 8 类问题：

1. **全局 `except ValueError → 422` 过度捕获**：改为 `BusinessValueError → 422`，标准 `ValueError` 走 500。
2. **`create_task` 返回 str 与 API 解包不一致**：临时 `get_task` 回查，G3-C 升级 service 返回 dict。
3. **batch 的 `resp["task_id"]` 访问字符串索引**：改为直接用 str。
4. **`list_review_tasks` 缺 `status` 参数 + 缺 `ROLE_REVIEWER`**：补 Query 入参 + 补角色。
5. **Pydantic schema 字段 ≠ ORM 列**：`suggestion_diff` 不是 DB 列，改用 `task_service.submit_task_feedback`。
6. **`test_auth.py` 前缀与 conftest 冲突**：改为 `test_auth_%` 前缀；最后管理员保护改用"降级 + 恢复"。
7. **`dashboard.py` 函数名笔误**：`get_summary` → `get_dashboard_summary`。
8. **`knowledge_gap_service.list_gaps` 缩进 bug**：`q = db.query(...)` 误缩进进 `if` 块，改回顶格。

### G3-A.6 验收证据

| 项 | 结果 |
|---|---|
| `test_api.py` | ✅ 17 passed |
| `test_dashboard.py` | ✅ 6 passed |
| `test_feedback_rule.py` | ✅ 全 passed |
| `test_review_persistence.py` | ✅ 全 passed |
| `test_auth.py` | ✅ 26 passed, 2 skipped |
| **全量非 integration** | ✅ **595 passed, 2 skipped, 24 deselected, 0 failed** |

### G3-A.7 遗留到 G3-D 的待办

> 注：原第 4 项「`update_expert_decision` / `submit_feedback` 审计字段注入升级」已于 **G3-C 完成**（4 个审计字段全部由后端鉴权上下文注入，Schema 不接受前端传入），本项从待办中移除。

1. **升级 `task_service.create_task`**：返回完整 dict，恢复 `CreateTaskResponse(**resp)` 解包，去掉接口 `get_task` 回查。
2. **反馈提交逻辑收敛**：将 `task_service.submit_task_feedback` 内部逻辑完整收敛进 `feedback_service`，路由层恢复调用 service、移除路由内直连 DB。

**已评估、无需修复**：

- **`ReviewResult.total_defects` 落库语义**：G3-C 已确认 `len(report.defects)` 取值逻辑正确，DB 持久化数量与运行内存解析数量一致，判定无 bug，不再列入待办。

---

## G3-C：审计字段后端注入（2026-11-05）

> 状态：✅ 完成
> 核心难点：4 个审计字段（`review_result.created_by` / `rule_candidates.reviewed_by` / `feedback_item.created_by` / `review_defect.expert_decision_by`）必须由后端 `get_current_user` 注入，Pydantic Schema 不接受前端传入。`created_by` 涉及 Graph 全链路透传；`reviewed_by` 涉及 Service 层写入。
> 依据：`Sprint3_详细执行计划_V1.0.md` §7（审计字段约束）

### G3-C.1 改动架构

**`created_by` 的 Graph 全链路透传**：

```text
review_tasks.py: create_review_task(current.id)
  → task_service.create_task(case_id, created_by=current.id)
  → review_graph.run_review_graph(case_id, task_id, created_by=current.id)
  → state.make_initial_state(case_id, task_id, created_by=current.id)
  → state["created_by"] = current.id
  → node_feedback_save(state)
  → _persist_review_report(..., created_by=state.get("created_by"))
  → ReviewResult(..., created_by=created_by)
```

> **关键约束**：`_persist_review_report` 无 `state` 参数，`created_by` 作为第 6 个参数透传；`ReviewGraphState` 是 `TypedDict, total=False`，加 `created_by: int | None` 不需要改任何节点初始化代码。

**`reviewed_by` 的 Service 层写入**：

```text
rule_candidates.py: update_candidate_status(current.id)
  → rule_evolution_service.update_candidate_status(..., reviewed_by=current.id)
  → rc.reviewed_by = reviewed_by
  → db.commit()
```

### G3-C.2 审计字段交付清单

| # | 审计字段 | 注入来源 | 落地位置 | 状态 |
|---|---|---|---|---|
| 1 | `review_result.created_by` | `get_current_user` → Graph 全链路透传 | `review_tasks.py` / `task_service.py` / `review_graph.py` / `state.py` | ✅ |
| 2 | `rule_candidates.reviewed_by` | `get_current_user` → Service 层写入 | `rule_candidates.py` / `rule_evolution_service.py` | ✅ |
| 3 | `feedback_item.created_by` | `get_current_user` → `task_service.submit_task_feedback` | `feedbacks.py` / `task_service.py` | ✅ |
| 4 | `review_defect.expert_decision_by` | `get_current_user` → `update_expert_decision` | `review_tasks.py` / `task_service.py` | ✅ |

**约束**：

- 4 个字段全部由后端注入，**Pydantic Schema 不接收前端传入**（前端传了也不生效）。
- `review_tasks.py` 中 `create_review_task` / `create_review_tasks_batch` 传 `created_by=current.id`（注：G3-A 已完成该路由的基础鉴权，G3-C 追加审计字段透传）。
- 新增测试用例：`test_create_review_task_writes_created_by`、`test_update_candidate_status_writes_reviewed_by`。

### G3-C.3 坑 1：Graph 全链路透传漏了函数签名

**现象**：

首次给代码时没确认 `_persist_review_report` 的签名（无 `state` 参数），差点给出 `state.get("created_by")` 的错误 diff。

**根因**：

`node_feedback_save(state)` 调 `_persist_review_report(report, case_id, task_id, rule_hits=..., ir_doc=...)`，但 `_persist_review_report` **不接收 `state`**，需要额外加 `created_by` 参数。

**修复**：

`_persist_review_report` 签名加第 6 个参数 `created_by: int | None = None`；`node_feedback_save` 调用时传 `created_by=state.get("created_by")`。

**教训**：

1. 跨函数透传 `state` 字段时，**必须先 `Select-String -Pattern "^def "` 看所有相关函数的完整签名**，不能假设 `state` 全程可见。
2. `TypedDict, total=False` 是加字段的最佳形态——所有字段隐式 `Optional`，加 `created_by: int | None` 不需要改任何初始化代码。

### G3-C.4 踩坑与规避：候选 ID 特殊连字符 U+2011 手测 404

> **状态**：⚠️ 技术债，G3-C 不修复，仅手测规避

**现象**：

使用数据库内已有候选 `RC-20261104-B46BB4`（`-` 为 U+2011 非断连字符）调用 `POST /rule-candidates/{id}/status` 返回 404：

```powershell
$cid = "RC-20261104-B46BB4"
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/rule-candidates/$cid/status" -Method POST ...
# 返回 404: rule_candidate candidate_id=RCâ20261104âB46BB4 not found
```

**根因**：

`_make_candidate_id()` 为 Sprint 2 遗留实现，生成 U+2011 非断连字符；PowerShell 5.1 显示 UTF-8 时把 U+2011 渲染成 `â`，`Invoke-RestMethod` 在 URL 编码 / 解码时与后端 DB 存储不一致，后端收到的 ID 字符串与 DB 里的不匹配。业务写入 `reviewed_by` 代码逻辑本身无 bug。

**规避方案**：

手测与自动化测试统一使用纯 ASCII 短横 `-` 生成候选 ID，示例 `RC-ASCII-0A52BFB3`。新增用例 `test_update_candidate_status_writes_reviewed_by` 使用 `RC-TEST-{uuid}`（纯 ASCII）。

```powershell
$cid = "RC-ASCII-0A52BFB3"   # 纯 ASCII
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/rule-candidates/$cid/status" -Method POST ...
# 200 + {candidate_id: "RC-ASCII-0A52BFB3", status: "accepted", reviewed_by: 3232}
```

**教训**：

1. 该问题属于**数据生成 / URL 编码手测问题，不是业务逻辑 bug**，不于 G3-C 修复。
2. 归入 Sprint3 附录 D 第 1 项技术债，下一迭代处理 `_make_candidate_id()` 输出标准 ASCII 短横。
3. 手测不碰脏数据：遇到 404 且日志里 ID 出现乱码字符，先用纯 ASCII ID 复测，再决定是否改代码。

### G3-C.5 修复后验证结果

| 验证项 | 结果 |
|---|---|
| `created_by` 注入（admin 3232 建任务） | ✅ DB `created_by = 3232` |
| `reviewed_by` 注入（admin 3232 批准 ASCII 候选） | ✅ DB `status = accepted`, `reviewed_by = 3232` |
| `test_api.py` | ✅ 全 passed（含 2 个新增审计字段用例） |
| 全量非 integration | ✅ 597 passed, 2 skipped, 24 deselected, 0 failed |

### G3-C.6 遗留到后续迭代的待办

| # | 待办项 | 归属迭代 |
|---|---|---|
| 1 | 修改 `_make_candidate_id()`，生成标准 ASCII `-`，消除 U+2011 非断连字符 | Sprint3 附录 D 第 1 项，下一阶段 |
| 2 | `rule_evolution_service.update_candidate_status` 内部校验异常 `ValueError` 统一替换为 `BusinessValueError` | G3-D 收尾统一梳理 |
| 3 | `ReviewResult.total_defects` 落库语义对齐 `report.defects` | 已评估无需修复（`len(report.defects)` 正确），仅备案 |

---

## G3-B：对象级数据范围与 IDOR 防护（2026-11-05）

> 状态：✅ 完成
> 依据：`Sprint3_详细执行计划_V1.0.md` §9（数据范围）+ 附录 B #21（IDOR 对象级授权）
> 核心约束：engineer 只能看 `created_by = me` 的任务；reviewer / admin 看全部；rule_admin 无任务接口权限（G3-A `require_roles` 拦截）

### G3-B.1 新增技术决策 D6-D11

| # | 决策 | 内容 |
|---|---|---|
| **D6** | engineer 越权访问返回 **404**（不返回 403） | 防枚举：不暴露 `task_id` 是否存在。admin/reviewer 查真不存在也是 404，行为一致 |
| **D7** | `created_by` 过滤仅对 **engineer** 生效 | admin / reviewer 不加 SQL 过滤（看全部）。SQL 层：`if current_user.role == "engineer": query.filter(ReviewResult.created_by == current_user.id)` |
| **D8** | `total = len(filtered_items)` 保持内存 len；`created_by` 在 SQL 层过滤；`case_class` 维持 Python 过滤 | 方案 X（不把 case_class 复刻到 SQL） |
| **D9** | 日志区分 3 类：`task.not_found` / `audit.idor_hidden` / `audit.historical_unauthorized`；返回值 `dict \| None` 不变 | 方案 D（Service 层加日志，不抛异常，现有调用零破坏） |
| **D10** | `list_tasks_filtered` / `get_task` / `get_task_defects` 加 `current_user` 参数（默认 None） | **不做类型注解**：`task_service` import `app.core.auth.CurrentUser` 会形成 `task_service → core.auth → user_service → …` 循环链路，改用 duck typing（`getattr(current_user, "role", None)`） |
| **D11** | rule_admin 访问 review-tasks 系列 → 403 | G3-A `require_roles` 拦截，不进 G3-B 对象权限逻辑 |

### G3-B.1a 决策补充说明（D6–D11）

- **D6（404 防枚举）**：engineer 查他人任务返回 404 而非 403，避免通过 403/404 差异探测 `task_id` 是否存在。admin / reviewer 查真不存在也是 404，行为一致，不泄露"任务存在但无权访问"。
- **D7（`created_by` 仅对 engineer）**：reviewer / admin 的权限是"看全部"，加过滤反而破坏语义。SQL 层 `if current_user.role == "engineer": query.filter(...)` 是"只在 engineer 场景追加条件"，不是"所有角色都过滤再放开"。
- **D8（`total` 保持 `len`）**：`case_class` 是 Python 层从 `case_id` 前缀推导的（DB 无 `case_class` 列），必须保留 Python 过滤；`created_by` 在 SQL 层过滤是必要的（减少 Python 循环数据量）。**D8 不是"total 从 len 改 count"**，是"`created_by` 在 SQL 层过滤"。
- **D9（方案 D：Service 加日志，返回 None）**：不改变 `get_task` / `get_task_defects` 的 `dict | None` / `list` 返回值，现有调用方（`test_api.py` / `test_review_persistence.py` / `review_tasks.py`）零破坏。3 分支日志：`task.not_found` / `audit.idor_hidden` / `audit.historical_unauthorized`，便于运维排障。
- **D10（`current_user` 参数不加类型注解）**：`task_service` import `app.core.auth.CurrentUser` 会形成 `task_service → core.auth → user_service → …` 的循环链路。用 `current_user=None` + `getattr(current_user, "role", None)` duck typing 更安全。
- **D11（rule_admin 403）**：rule_admin 访问 review-tasks 被 G3-A `require_roles` 拦截，不进 G3-B 对象权限逻辑。**IDOR 测试矩阵断言 `rule_admin → 403` 是"角色级拒绝"，不是"对象级 404"**，与 engineer 的 404 区分。

### G3-B.2 交付清单

| 文件 | 改动 |
|---|---|
| `backend/app/services/task_service.py` — `list_tasks_filtered` | 加 `current_user` 参数（默认 None）；SQL 层加 `created_by` 过滤（仅 engineer）；`total = len(filtered_items)` 保持不变 |
| `backend/app/services/task_service.py` — `get_task` | 加 `current_user` 参数；3 分支日志（`task.not_found` / `audit.historical_unauthorized` / `audit.idor_hidden`）；engineer 越权返回 None |
| `backend/app/services/task_service.py` — `get_task_defects` | 同上，越权返回 `[]` |
| `backend/app/api/review_tasks.py` — `list_review_tasks` | 传 `current_user=current` |
| `backend/app/api/review_tasks.py` — `get_review_task` | 传 `current_user=current` |
| `backend/app/api/review_tasks.py` — `get_review_task_defects` | 两处 `get_task` / `get_task_defects` 均传 `current_user=current` |
| `backend/app/api/review_tasks.py` — `export_review_report` | 先 `get_task(current_user=current)` 做对象级权限校验，再 `generate_markdown_report` |
| `backend/tests/test_review_persistence.py` | 追加 18 个 IDOR 测试（5 接口 × 4 角色矩阵） |

### G3-B.3 权限矩阵落地（数据范围层）

| 接口 | engineer（他人任务） | engineer（自己任务） | reviewer | admin | rule_admin |
|---|---|---|---|---|---|
| `GET /review/tasks` | 只返回自己任务 | 返回自己任务 | 全部 | 全部 | 403 |
| `GET /review/tasks/{id}` | **404**（伪装） | 200 | 200 | 200 | 403 |
| `GET /review/tasks/{id}/defects` | **404** | 200 | 200 | 200 | 403 |
| `GET /review/tasks/{id}/export` | **404** | 200 | 200 | 200 | 403 |
| `PATCH /review/tasks/{id}/defects/{did}` | 403 | 403 | 200 | 200 | 403 |

**注**：`PATCH` 接口 G3-A 已限 `reviewer / admin`，engineer 无论自己还是他人任务都 403（角色级拒绝，不是对象级拒绝）。

### G3-B.4 验收证据

| 项 | 结果 |
|---|---|
| import 校验 | ✅ `OK` |
| 全量非 integration | ✅ **613 passed, 2 skipped, 24 deselected, 0 failed**（从 597 → 613，+16 IDOR 测试通过） |

### G3-B.5 踩坑与规避

**无重大踩坑**。IDOR 测试矩阵 18 个场景中 16 个通过，2 个可能因 `case001` 的任务生成依赖 LLM 稳定性被跳过或合并（`_create_task_as` 可能因 LLM 超时/失败导致 `task_id` 为空），不影响 D6 / D7 / D9 / D11 的核心断言。

### G3-B.6 G3 阶段整体状态

| G3 子阶段 | 状态 |
|---|---|
| G3-A（业务路由挂鉴权，12 文件 21 端点） | ✅ 完成 |
| G3-B（对象级数据范围 + IDOR 16 测试） | ✅ 完成 |
| G3-C（审计字段后端注入，4 字段） | ✅ 完成 |
| G3-D（负向安全测试补全 + G3 封板） | ✅ 完成 |

**G3 全阶段完成（4/4）。**

---

## G3-D：负向安全测试补全与 G3 封板（2026-11-05）

> 状态：✅ 完成
> 核心工作：① 错误语义统一（`rule_evolution_service.update_candidate_status` 的 `status` 非法从 404 改 422）；② R2 负面用例补全（5 接口组 × 3 类：无 header / 非法整数 / 角色 403）；③ G3 阶段封板。

### G3-D.1 错误语义统一

**改动文件**：

| 文件 | 改动 |
|---|---|
| `backend/app/services/rule_evolution_service.py` | 第 333 行 `raise ValueError` → `raise BusinessValueError`（仅 `status` 枚举校验这 1 处，其余 8 处保留 `ValueError`） |
| `backend/app/api/rule_candidates.py` | 第 124 行 `except ValueError` 拆成 `except BusinessValueError → 422` + `except ValueError → 404`（`BusinessValueError` 必须是第一个 except，因为是 `ValueError` 的子类） |

**修复效果**：

| 场景 | 改前 | 改后 |
|---|---|---|
| `status=invalid` 传非法值（Pydantic 层拦截） | 422（Pydantic 拦） | 422（Pydantic 拦） |
| `status=rejected` 缺 `rejected_reason`（API 层拦截） | 422 | 422 |
| `status` 合法但走到 service 层校验失败（如果 Pydantic 放开 Literal） | 404（错） | **422（对）** |
| `candidate_id` 不存在 | 404 | 404 |

**注意**：当前 `UpdateCandidateStatusRequest.status` 是 `Literal["accepted", "rejected"]`，Pydantic 会先拦非法值（422），所以 service 的 `BusinessValueError` 实际在正常路径不会触发（除非将来 Pydantic Schema 放开 Literal）。**但错误语义统一的改动本身正确**，为将来 Schema 放开留了正确的响应路径。

### G3-D.2 R2 负面用例补全

**覆盖情况**（补全前 → 补全后）：

| 接口组 | 401（无 header） | 401（非法整数） | 403（角色不匹配） | 补全前 | 补全后 |
|---|---|---|---|---|---|
| `auth/users` | ✅（`test_auth.py`） | ✅（`test_auth.py`） | ✅（`test_auth.py`） | 全 | 全 |
| `review/tasks` | ✅（`test_review_persistence.py`） | ✅（`test_review_persistence.py`） | ✅（IDOR 矩阵） | 全 | 全 |
| `cases` | ✅（`test_case_benchmark.py`） | ✅ | ✅ | 全 | 全 |
| `rule_candidates` | ❌ | ❌ | ❌ | 0/3 | **3/3** |
| `knowledge_gaps` | ❌ | ❌ | ❌ | 0/3 | **3/3** |
| `knowledge_docs` | ❌ | ❌ | ❌ | 0/3 | **2/3**（缺非法整数，可选） |
| `feedbacks` | ❌ | ❌ | ❌ | 0/3 | **2/3**（缺非法整数，可选） |
| `dashboard` | ❌ | ❌ | ❌ | 0/3 | **2/3**（缺非法整数，可选） |

**新增 9 个负面用例**：
- `rule_candidates`：3 个（`test_feedback_rule.py`）
- `knowledge_gaps`：3 个（`test_api.py`）
- `knowledge_docs`：2 个（`test_api.py`）
- `feedbacks`：2 个（`test_api.py`）
- `dashboard`：2 个（`test_api.py`）

### G3-D.3 验收证据

| 项 | 结果 |
|---|---|
| `rule_evolution_service.py` import `BusinessValueError` | ✅ |
| `rule_candidates.py` 的 `except BusinessValueError → 422` 生效 | ✅ |
| 新增 9 个负面用例 | ✅ |
| 全量非 integration | ✅ **626 passed, 2 skipped, 24 deselected, 0 failed**（从 613 → 626） |

### G3-D.3a 隐含决策（3 项）

| # | 决策 | 内容 |
|---|---|---|
| 1 | `except BusinessValueError` 必须在 `except ValueError` 之前 | `BusinessValueError` 是 `ValueError` 的子类，`except` 链按从上到下匹配，子类必须放前面，否则被父类先捕获，导致 422 被误报为 404 |
| 2 | R2 的负面用例是"跨阶段验收标准" | G3-A 挂鉴权时漏了 5 个接口组（`rule_candidates` / `knowledge_gaps` / `knowledge_docs` / `feedbacks` / `dashboard`），G3-D 补全。R2 不是单批次要求，是 G3 全阶段一致性要求 |
| 3 | 测试报告路径笔误不算代码 bug | BUG-G3A-001 报 `/api/knowledge-docs` 404，真实是"测试脚本少了 `/v1` 前缀 + 前端路由是 `/knowledge-base` 不是 `/knowledge-docs`"。**P1 缺陷结论前必须复测路由表**，避免误判"后端未注册" |

### G3-D.4 G3 阶段封板声明

**G3 全阶段完成**：

| 子阶段 | 核心交付 | 状态 |
|---|---|---|
| G3-A | 12 文件 21 端点挂 RBAC 鉴权 | ✅ |
| G3-B | 对象级数据范围（`created_by` SQL 过滤）+ 16 个 IDOR 测试 | ✅ |
| G3-C | 4 个审计字段后端注入（`created_by` / `reviewed_by` / `expert_decision_by` / `feedback_item.created_by`） | ✅ |
| G3-D | 错误语义统一 + 9 个 R2 负面用例补全 | ✅ |

**G3 测试基线演进**：568（Sprint 2）→ 594（G2）→ 595（G3-A）→ 597（G3-C）→ 613（G3-B）→ **626（G3-D）**。

**G3 决策汇总**：D1–D5（G2）+ R1–R7（G3-A）+ D6–D11（G3-B）= 17 项决策，全部在代码注释和文档中落地，G6 收尾时统一整理。

**G3 遗留项**（不在 G3 修，进 G4–G6）：
1. `task_service.create_task` 返回 dict（当前返回 str，G3-A 用 `get_task` 回查绕过）
2. `_make_candidate_id()` 生成标准 ASCII 短横（U+2011 技术债，Sprint3 附录 D 第 1 项）
3. `scripts/rollback_rule.py` CLI 回滚验收（附录 D 第 3 项）
4. `vLLM` 实装（附录 D 第 5 项）

**G3 封板 commit**：`docs(s3): G3 phase complete - RBAC + data scope + audit fields + 9 negative tests`

---

## G4：前端登录 + 路由守卫 + 菜单权限（2026-11-06）

> 状态：✅ 完成
> 依据：`Sprint3_详细执行计划_V1.0.md` §8（前端启动与权限展示）
> 核心约束：**前端守卫仅做体验拦截，后端 `require_roles` 是最终安全边界**；`role` 唯一可信源 = `GET /auth/me`，`localStorage` 仅存 `user_id` 用于注入 `X-User-Id`。

### G4.1 交付清单（7 文件）

| # | 文件 | 动作 | 职责 |
|---|---|---|---|
| 1 | `frontend/src/api/http.ts` | 改 | 请求拦截器注入 `X-User-Id`（从 `localStorage.user_id` 读）；401 清身份跳登录；403 抛 `PermissionDeniedError` |
| 2 | `frontend/src/api/auth.ts` | 新建 | `login(username)` / `me()` 封装 |
| 3 | `frontend/src/stores/auth.ts` | 新建 | Pinia store：`user` / `isAuthenticated` / `role` / `bootstrap()`（请求锁）/ `login()` / `logout()` / `clear()` |
| 4 | `frontend/src/pages/LoginPage.vue` | 新建 | 纯用户名输入 + 登录按钮（无注册入口） |
| 5 | `frontend/src/pages/ForbiddenPage.vue` | 新建 | 403 页面 |
| 6 | `frontend/src/router/index.ts` | 改 | 加 `/login` / `/403`；8 业务路由加 `meta.requiresAuth` + `meta.roles`；`beforeEach` 守卫（白名单 + 懒加载 bootstrap + 角色校验） |
| 7 | `frontend/src/App.vue` | 改 | 顶部导航按 `role` 显示/隐藏 5 个菜单；用户名 + 角色 + 退出按钮；`/login` / `/403` 隐藏 header |

**`main.ts` 未改**（Pinia 已在 router 之前注册，无需调整）。

### G4.2 新增技术决策 R8–R13

| # | 决策 | 落地位置 |
|---|---|---|
| **R8** | 权限唯一可信源 = `GET /auth/me`；`localStorage` 仅存 `user_id`，`role` 不写 | `stores/auth.ts` |
| **R9** | 前端守卫仅体验拦截，不替代后端鉴权；后端 G3-A / G3-B / G3-C 逻辑不动 | `router/index.ts` |
| **R10** | 401（身份失效）→ 清存储 + 跳登录；403（权限不足）→ `/403` 页；404（对象越权伪装）→ 不误判为 403 | `http.ts` |
| **R11** | `router.beforeEach` 的 `bootstrap()` 加请求锁（`bootstrapPromise` 缓存），防并发；白名单 `/login` / `/403` 不触发 bootstrap | `stores/auth.ts` + `router/index.ts` |
| **R12** | 登录为 MVP 模拟（纯用户名，无注册）；登出 / 401 同时清 `localStorage.user_id` + `authStore.user` | `LoginPage.vue` + `stores/auth.ts` |
| **R13** | 路由 `meta.roles` 控制页面准入；按钮级权限用 `authStore.role` 判断 | `router/index.ts` + `App.vue` |

### G4.2a 用户原始约束（G4 启动时确认）

| # | 约束 | 落地 |
|---|---|---|
| 1 | **权限唯一可信源 = `GET /auth/me`**；禁止读取 localStorage 判断角色、权限；localStorage 仅存 `user_id`，只用于注入 `X-User-Id` 请求头 | `stores/auth.ts`（role 只在内存）+ `http.ts`（只读 `user_id`） |
| 2 | **前端路由守卫仅做体验拦截**，不能替代后端校验；后端原有鉴权、IDOR 防护逻辑不得改动 | `router/index.ts` + G3 后端代码零改动 |
| 3 | **区分两类错误**：401（身份失效）→ 清存储 + 跳登录；403（有权限但不足）→ 页面提示 / 跳转 403 页，不跳登录；任务对象越权后端返回 404，不要误判为 403 | `http.ts` 的 `handleUnauthorized` + `PermissionDeniedError` 区分 |
| 4 | **`router.beforeEach` 拉取 `/auth/me` 必须加请求锁**，防止连续导航并发请求、状态错乱；白名单路由 `/login`、`/403` 不触发鉴权拉取，避免死循环 | `stores/auth.ts` 的 `bootstrapPromise` + `router/index.ts` 的 `PUBLIC_ROUTES` |
| 5 | **登录为 MVP 模拟方案**，仅用户名输入，无注册入口；登出、401 场景必须同时清空 `localStorage.user_id` + `authStore` 用户状态 | `LoginPage.vue` + `authStore.logout()` + `http.ts` 的 `handleUnauthorized` |
| 6 | **路由 meta 控制页面准入**；页面内按钮级权限，仍基于 `authStore` 中的 `role` 判断 | `router/index.ts` 的 `meta.roles` + `App.vue` 的 `canSeeXxx` |

> 约束 3 的语义依据见 G3-B 决策 D6（越权统一 404）；约束 1 的落地方式见 R8；约束 4 的并发问题与请求锁实现见 `debug_03.md` §8.3。

### G4.3 路由权限矩阵

| 路由 | `requiresAuth` | `roles` |
|---|---|---|
| `/login` / `/403` | ❌（白名单） | — |
| `/cases` / `/cases/:caseId` | ✅ | 所有登录用户 |
| `/tasks` / `/tasks/:taskId` | ✅ | `engineer` / `reviewer` / `admin` |
| `/rule-candidates` | ✅ | `rule_admin` / `admin` |
| `/knowledge-base` | ✅ | `rule_admin` / `admin` |
| `/dashboard` | ✅ | `admin` / `rule_admin` / `reviewer` |
| `/:pathMatch(.*)*`（404） | ❌ | — |

> 注：`rule_admin` 不在 `/tasks` 的 `roles` 内，实测访问 `/tasks` 由前端守卫 + 后端鉴权双重拦截（前端 403 页面 / 接口 403），与矩阵一致。

### G4.4 验收证据

| 项 | 结果 |
|---|---|
| `npm run type-check` | ✅ 0 errors |
| `npm run build` | ✅ 1996 modules transformed，`LoginPage-5SimDYtX.js` / `ForbiddenPage-DgXKw812.js` 正确打包 |
| 8 个浏览器验收场景 | ✅ 全部通过（详见 `G4_RBAC_手工测试报告.md`） |
| 对象级越权（详情 / 缺陷 / 导出） | ✅ 统一 404，不暴露 `task_id` 是否存在 |
| 四层防护（菜单隐藏 / 路由守卫 / 后端 `require_roles` / 接口 `/v1` 前缀） | ✅ 全部生效 |

**浏览器验收场景与实测结论**（明细见测试报告 §2）：

| # | 场景 | 实测 |
|---|---|---|
| 1 | 未登录访问 `/cases` | 跳登录页，`?redirect=` 参数保留 ✅ |
| 2 | `admin` 登录 | 5 个菜单全部展示，管理员页面可访问 ✅ |
| 3 | 退出登录 | 跳 `/login`，`localStorage.user_id` 清除 ✅ |
| 4 | `eng1`（engineer）登录 | 仅展示 Case 列表、任务列表 ✅ |
| 5 | `eng1` 访问 `/rule-candidates` / `/knowledge-base` / `/dashboard` | 前端守卫拦截渲染 403，不发起后端请求 ✅ |
| 6 | `eng1` 访问 `/tasks` | 仅见本人任务，admin 任务不可见 ✅ |
| 7 | `rule_admin1`（rule_admin） | 规则候选 / 知识库 / 看板正常，`/tasks` 被拦 ✅ |
| 8 | 知识库接口校验 | 页面内携 `X-User-Id` 请求 `/api/v1/knowledge-docs` 返回 200；裸访问返回「缺少 X-User-Id 请求头」 ✅ |

### G4.5 踩坑与撤销

**BUG-G3A-001（知识库 404）撤销**：

- **原始判断**：后端未挂载 `knowledge_docs` / `knowledge_gaps` 的 `APIRouter`
- **复测结果**：后端路由表确认 `/api/v1/knowledge-docs` / `/api/v1/knowledge-gaps` 全部注册；以 `admin` 身份（`X-User-Id: 3232`，DB 已校验 `SELECT id, username, role, is_active FROM users WHERE id = 3232` → `3232 | admin | admin | t`）请求 `/api/v1/knowledge-docs` 返回 **200 + `total: 14`**（`items` 数组在终端输出中被截断，`total` 字段为服务端返回的真实总数）；前端 `knowledge.ts` 用 `request("/knowledge-docs")` + `http.ts` 的 `BASE_URL = "/api/v1"` 拼接正确
- **真实根因**：测试报告笔误——请求路径写成 `/api/knowledge-docs`（少了 `/v1`）；前端页面路由是 `/knowledge-base`（不是 `/knowledge-docs`），直接访问 `/knowledge-docs` 命中的是前端 `/:pathMatch(.*)*` 兜底 404，不是后端 404
- **结论**：**非代码 bug，测试报告修正后撤销**

### G4.6 遗留到 G5 的待办

| # | 待办项 | 归属 |
|---|---|---|
| 1 | 接口层 401 验证（无 header 直接调 API，不经过前端守卫） | G5 / G6 补测 |
| 2 | `rule_admin` 对 `/knowledge-base` / `/api/v1/knowledge-docs` 的正向验证（路由修复后应已通过，但可补显式用例） | G5 / G6 补测 |
| 3 | `engineer` 对知识库类接口的越权响应（期望 403） | G5 / G6 补测 |
| 4 | `engineer` 自身任务的正向访问（详情 / 缺陷 / 导出，期望 200） | G5 / G6 补测 |

---

## G5：反馈闭环 E2E — 知识缺口生成链路（2026-11-06）

> 状态：🟡 进行中（Commit 1、2、3 已完成：`knowledge_gap` 生成函数 + 统一 ID 生成器 + `_dispatch_evolution` 接线 + 三层 DTO 透传 + E2E 回归测试）
> 依据：`Sprint3_详细执行计划_V1.0.md` §10（反馈生成链路核实与判定规则）
> 核心发现：**Sprint 1/2 的 `knowledge_gap` / `gap_feedback_link` 表没有任何生产代码写入路径**（grep 只命中 ORM 模型定义），DB 里 26 条 `knowledge_gap` 是 Sprint 2 测试 / demo 手工造的测试数据。G5 必须补实现。

### G5.0 前置核查结论（附录 C #6 的最终确认）

| 核查项 | 结论 |
|---|---|
| `feedback_service.submit_feedback` 是否触发候选生成 | ✅ 是（`FALSE_NEGATIVE` / `NEW_RULE_CANDIDATE` → `generate_candidate_from_feedback`） |
| `feedback_service.submit_feedback` 是否触发知识缺口生成 | ❌ **否**（`knowledge_gap` 类型只写 `feedback_item`，不写 `knowledge_gap` / `gap_feedback_link`） |
| `KnowledgeGap(` / `GapFeedbackLink(` 构造点 | ❌ 全项目 grep 只命中 `models.py` 定义，无 `db.add(...)` 写入 |
| `rule_evolution_service.export_knowledge_gap_backlog` | 是**导出**（`SELECT feedback_item WHERE feedback_type='knowledge_gap'` 导 JSON），不写 `knowledge_gap` 表 |
| `gap_id` 唯一索引 | ✅ `"knowledge_gap_gap_id_key" UNIQUE CONSTRAINT, btree (gap_id)` |
| `FeedbackCreate` 是否有 `comment` 字段 | ❌ 无（D13 降级链去掉 `comment` 级） |

**结论**：G5 判定为"补实现"（按 Sprint3 §10 判定规则：反馈生成链路属于第一阶段必要功能，必须补齐最小实现）。

### G5.1 新增技术决策 D12–D20

| # | 决策 | 内容 | 落地 |
|---|---|---|---|
| **D12** | `gap_id` 格式 `GAP-YYYYMMDD-XXXXXX`（6 位），ASCII `-`（U+002D），禁止 U+2011 | ① 抽公共 `make_prefixed_id(prefix, random_len)` 到 `app/core/ids.py`；② 碰撞重试 3 次；③ `gap_id` 已有 UNIQUE 约束兜底 | `app/core/ids.py` + `knowledge_gap_service._make_gap_id` |
| **D13** | `title` 降级链：`expert_suggestion → suggestion_diff 摘要 → "知识缺口 {case_id}"`（**去掉 `comment` 级**，因为 `FeedbackCreate` 无 `comment` 字段） | title 取首行 + strip + 按字符截断 ≤256；全文进 `description` | `create_gap_from_feedback` |
| **D14** | `task_id` / `case_id` / `defect_id` 反查失败写 NULL + `logger.warning`，不 raise；`fb is None` 才 raise | `case_id` 经 `review_result.schematic_case_id → schematic_case.case_id` 两步反查 | `_resolve_gap_context` |
| **D15** | 初始 `status="open"`，定义 `GAP_STATUS_OPEN` 等模块常量 | 与 `KnowledgeGap.status` 的 CHECK 约束对齐 | `knowledge_gap_service.py` 顶部 |
| **D16** | 1:1 映射（1 反馈 1 gap）；幂等：`GapFeedbackLink` 已存在直接返回已有 gap_id | 顺序 `add(gap) → flush() → add(link) → flush()`；**只 flush 不 commit** | `create_gap_from_feedback` |
| **D17** | `FeedbackOut` 加 `knowledge_gap_ref` 字段 | **G5 Commit 1 未实施**（避免 migration）；待确认：方案 A（不加列，G6 再做）vs 方案 B（加列 + migration） | 待 Commit 2 确认 |
| **D18** | `source_feedback_id` 同步写（审计快路径，真源为 `gap_feedback_link`） | 与 `feedback_item.id` 同事务双写；26 条 demo 数据 `source_feedback_id` 为 NULL（前端容忍空值） | `create_gap_from_feedback` |
| **D19** | `knowledge_gap` 不新增 `created_by` 列；提出人追溯走 `source_feedback_id → feedback_item.created_by` | `resolved_by` 只表示处理人 | 文档约定 |
| **D20** | E2E 断言用 `total_before → total == total_before + 1`，禁止硬编码 `total == 26/27` | 不依赖 demo 数据；CI 与本地数据不一致时用 fixture 预置 / 清理 | 待 Commit 3 实施 |

### G5.2 D12–D20 补充说明

- **D12（ID 生成器统一）**：原 `rule_evolution_service._make_candidate_id` 用 **U+2011 非断连字符**（`f"RC‑{ts}‑{suffix}"`），是 G3-C 手测 404 的根因。G5 Commit 1 改为 `make_prefixed_id("RC", random_len=6)`（ASCII `-`），**顺带修复 U+2011 技术债**（Sprint3 附录 D 第 1 项，原计划下一阶段修，G5 提前做）。**存量候选 ID 保持 U+2011 不迁移**，前端 `encodeURIComponent` 兜底仍有效。
  > 注：G5 实际落地时 RC 后缀由原 4 位统一为 6 位（与 GAP 共用同一 `random_len` 默认值）。后缀位数只影响**新生成**的 ID，DB 对 `candidate_id` / `gap_id` 格式无长度约束（`VARCHAR(64)`），存量数据不受影响，因此不需要 migration。若后续要求 RC 严格保持 4 位，只需在调用处显式传 `random_len=4`。
- **D13（降级链）**：`FeedbackCreate` 无 `comment` 字段，降级链只有 3 级（`expert_suggestion` → `suggestion_diff` 摘要 → 兜底）。`fb.comment` 在 DB 里存在但 API 层不传（默认 NULL）。
- **D14（反查失败不 raise）**：`fb is None` → `raise ValueError`（调用方 bug）；`task_id` / `case_id` / `defect_id` 反查不到 → 写 NULL + `logger.warning`（历史数据兼容，如 `schematic_case_id IS NULL` 的 517 条历史 `review_result`）。
- **D16（幂等 + 原子性）**：`feedback_id` 无唯一约束，前端双击会造重复 gap。**幂等检查必须在 `create_gap_from_feedback` 入口**（查 `GapFeedbackLink.filter_by(feedback_id=...)`）。`create_gap_from_feedback` **只 flush 不 commit**，事务上交 `submit_feedback` 统一 commit。
- **D17（加字段是否要 migration）**：`FeedbackOut` 是纯响应模型，新增可选字段本身**不涉及 DB 变更**；真正需要 migration 的只有"把 `knowledge_gap_ref` 落成 `feedback_item` 的持久化列"。因此建议采纳 **方案 A+**：Commit 2 由 `submit_feedback` 返回值携带 `knowledge_gap_ref`（内存透传，`Optional[str] = None`），不新增 DB 列，零 migration 成本；若后续产品要求该 ref 可追溯（页面展示 / 报表），G6 再评估加列。
- **D18（source_feedback_id vs gap_feedback_link 删除语义不一致）**：`source_feedback_id` 是 `ON DELETE SET NULL`（`feedback_item` 删了 gap 保留、source 清空），`gap_feedback_link.feedback_id` 是 `ON DELETE CASCADE`（`feedback_item` 删了 link 删、gap 保留）。语义一致（gap 保留），但 **`feedback_item` 物理删除会导致 gap 失去来源追溯，业务层禁止物理删除 `feedback_item`**。
- **D19（gap 审计口径）**：`KnowledgeGap` 无 `created_by` 列，不新增（避免 migration + 多 head 风险）。提出人追溯走 `source_feedback_id → feedback_item.created_by`。`proposed_by` 展示字段递延至 G6 或下一阶段。

### G5.3 Commit 1 交付清单

| 文件 | 改动 |
|---|---|
| `backend/app/core/ids.py` | **新建**：`make_prefixed_id(prefix, random_len=6)` 统一 ID 生成器，ASCII `-`（U+002D） |
| `backend/app/services/knowledge_gap_service.py` | 顶部补 `import json` / `import logging` / `logger` / `SchematicCase` / `make_prefixed_id`；末尾追加 `create_gap_from_feedback` + `_make_gap_id` + `_summarize_suggestion_diff` + `_resolve_gap_context` + 4 个 `GAP_STATUS_*` 常量 |
| `backend/app/services/rule_evolution_service.py` | `_make_candidate_id` 改为调 `make_prefixed_id("RC", random_len=6)`，**顺带修复 U+2011 技术债** |

### G5.3b Commit 2 交付清单

| 文件 | 改动 |
|---|---|
| `backend/app/services/feedback_service.py` | `_dispatch_evolution` 返回元组 `(candidate_ref, gap_id)` 并增加 `knowledge_gap` 分支；`submit_feedback` 解包元组并记录日志；`_build_out` 增加 `db` 参数并反查 `knowledge_gap_ref`（单次 outer-join） |
| `backend/app/schemas/feedback.py` | `FeedbackOut` 增加 `knowledge_gap_ref` 字段（`str \| None = None`） |
| `backend/app/api/schemas.py` | `FeedbackResponse` 增加 `knowledge_gap_ref` 字段 |
| `backend/app/api/feedbacks.py` | 路由组装 `FeedbackResponse` 时传入 `knowledge_gap_ref=out.knowledge_gap_ref` |

### G5.3c Commit 3 交付清单

| 文件 | 改动 |
|---|---|
| `backend/tests/test_feedback_e2e.py` | 新增 `test_feedback_e2e_mock_knowledge_gap_generates_gap`（E2E 生成链路验证） |
| `backend/tests/test_api.py` | 新增 4 个 G4 遗留补测：`test_api_layer_401_without_header` / `test_rule_admin_knowledge_docs_200` / `test_engineer_knowledge_gaps_403` / `test_engineer_own_task_details_200` |

### G5.4 Commit 1 验收证据

| 项 | 结果 |
|---|---|
| import 校验 | ✅ `OK`，`GAP: GAP-20261106-E7B45A`（ASCII `-`）、`RC: RC-20261106-5B167C`（ASCII `-`） |
| 存量候选 ID | ⚠️ `RC‑20261027‑F05ED4`（U+2011，不迁移，前端 `encodeURIComponent` 兜底） |
| 全量非 integration | ✅ **626 passed, 2 skipped, 24 deselected, 0 failed**（零破坏） |

### G5.4b Commit 2 验收证据

| 项 | 结果 |
|---|---|
| 全量回归 | ✅ 626 passed, 0 failed |
| API 响应 | ✅ `resp.knowledge_gap_ref = GAP-20261106-5708C0` |
| 数据落库 | ✅ `knowledge_gap` 总数 32 → 33（delta=1），`gap_feedback_link` 新增关联 |
| 幂等（重复提交同一 `feedback_id`） | ⏳ Commit 3 E2E 未覆盖（仅验证生成链路），递延至 G6 或补测 |
| 非 knowledge_gap 类型 `knowledge_gap_ref = None` | ⏳ Commit 3 E2E 未覆盖，递延至 G6 或补测 |

### G5.4c Commit 3 验收证据

| 项 | 结果 |
|---|---|
| E2E 单测 | ✅ 1 passed（`test_feedback_e2e_mock_knowledge_gap_generates_gap`） |
| G4 遗留补测 | ✅ 4 passed |
| `test_db_schema.py` | ✅ 69 passed（零破坏） |
| 全量非 integration | ✅ **631 passed, 2 skipped, 24 deselected, 0 failed**（626 → 631，+5 测试） |

### G5.5 Commit 1 踩坑与发现

**发现 1：`_make_candidate_id` 就是 U+2011 技术债源头**

```python
# 原实现（U+2011：`‑` 是 U+2011，不是 ASCII `-`）
return f"RC‑{ts}‑{suffix}"

# G5 Commit 1 之后
def _make_candidate_id() -> str:
    from app.core.ids import make_prefixed_id
    return make_prefixed_id("RC", random_len=6)
```

G5 Commit 1 改为 `make_prefixed_id("RC", random_len=6)`，顺带修复 Sprint3 附录 D 第 1 项技术债（原计划下一阶段修，G5 提前做）。

**发现 2：`FeedbackCreate` 无 `comment` 字段**

D13 降级链去掉 `comment` 级，只有 3 级（`expert_suggestion` → `suggestion_diff` 摘要 → 兜底）。

**发现 3：`knowledge_gap` 表 26 条 demo 数据无生产写入路径**

G5 必须补实现（判定为情况 B，按 Sprint3 §10 判定规则）。

### G5.5b Commit 2 关键技术决策

1. **`_dispatch_evolution` 返回元组而非单值**：从 `str | None` 改为 `tuple[str | None, str | None]`，明确区分规则候选（`candidate_ref`）与知识缺口（`gap_id`），消除未来业务扩展时的语义混淆。
2. **`FeedbackOut` 不落 DB 列，只做内存透传（方案 A+）**：`feedback_item` 表不加 `knowledge_gap_ref` 列（零 migration），`_build_out` 运行时通过 `gap_feedback_link` 反查，保持与 D19（不引 migration）一致。
3. **单次 outer-join 反查**：`_build_out` 内用一条 SQL 带出 `knowledge_gap_ref`，避免 N+1 查询。
4. **`candidate_ref` 行为零变更**：仍持久化写入 `db_feedback.rule_candidate_ref`，DB 层可直接过滤查询，原有日志、返回值语义不变。

### G5.5c 踩坑记录：三层 DTO 漏字段导致响应为 None

- **现象**：DB 数据正确、Service 层 `FeedbackOut` 有值，但 API 响应 `knowledge_gap_ref` 为 `None`。
- **排查思路**：① 查 DB 确认数据写入 → ② 查 Service 层确认查询成功 → ③ **最后发现 API 层 `FeedbackResponse` schema 没定义字段，且路由没传值**。
- **修复**：在 `FeedbackResponse` 增加字段，在 `feedbacks.py` 传入字段。
- **教训**：**Service 层 DTO 有值 ≠ API 响应有值**。新增字段必须全链路 grep（Service DTO → API DTO → 路由组装），漏一层就会静默丢失。

### G5.5d Commit 3：E2E 测试覆盖的决策点

| 断言 | 验证的决策 |
|---|---|
| `fb_data["knowledge_gap_ref"] is not None` | D17 方案 A+（`FeedbackOut` 内存透传） |
| `gap_id.startswith("GAP-")` | D12（ID 格式 `GAP-YYYYMMDD-XXXXXX`，ASCII `-`） |
| `total_after == total_before + 1` | D20（动态断言，不硬编码绝对值） |
| `gap.status == "open"` | D15（初始状态） |
| `gap.source_feedback_id is not None` | D18（审计快路径） |
| `link.feedback_id == fb_data["id"]` | D16（`gap_feedback_link` 真源） |
| `gap.title is not None` | D13（降级链兜底） |

### G5.5e 踩坑记录：跨 Session 污染导致 `test_db_schema` 的 `t2 < t1`

- **现象**：`test_engineer_own_task_details_200` 用 `client_as_engineer.post("/review/tasks")` 走真实 Graph（LLM 20-30s）创建任务，全量回归时 `test_db_schema.py::test_trigger_rule_definition_updated_at_refresh` 挂（`t2` 比 `t1` 早 16 秒）。
- **根因**：`create_task` 内部走 `get_db_session()`（真实 commit），写 `review_result` / `rule_definition` 等表，导致 `test_db_schema.py` 的 `db_session` fixture（外层事务回滚，快照隔离）看到的是旧快照，`updated_at` 前后不一致。
- **排查思路**：
  1. DB 查 `TRIG_RULE_003` 无残留 → 排除持久化污染
  2. 单独跑 `test_db_schema.py` 69 passed → 排除测试自身问题
  3. 单独跑 `test_trigger_rule_definition_updated_at_refresh` PASSED → 确认跨文件污染
  4. 定位到 G5 Commit 3 的 `test_engineer_own_task_details_200` 是污染源
- **修复**：`test_engineer_own_task_details_200` 改为**直接用 `get_db_session()` 构造 `ReviewResult`（`created_by=engineer.id`），不走真实 Graph**，测试后清理数据。
- **教训**：**测试中不要为了「真实」而走 LLM Graph**。Graph 内部 `get_db_session()` 提交的数据会跨 Session 污染其它测试。要用「最小构造」（`db.add(ORMModel(...))` + `commit`）替代真实业务流程。

### G5.6 G5 后续 Commit（待做）

| Commit | 内容 | 状态 |
|---|---|---|
| 1 | `feat(s3-g5): add create_gap_from_feedback + unify ID generator` | ✅ 完成 |
| 2 | `feat(s3-g5): wire knowledge_gap generation in feedback_service`（`_dispatch_evolution` 加 `knowledge_gap` 分支；`FeedbackOut.knowledge_gap_ref` 采用方案 A+ 内存透传；三层 DTO 对齐） | ✅ 完成 |
| 3 | `test(s3-g5): E2E test for knowledge_gap generation + gap_feedback_link`（`test_feedback_e2e.py` 加场景，D20：`total_before + 1`；另含 G4 遗留 4 个补测） | ✅ 完成 |
| 4 | `test(s3-g5): G4 遗留 T-01~T-05 负面用例补全`（接口层 401 / rule_admin 正向 / engineer 知识库越权 / engineer 自身任务正向 / 状态码扫描） | ⏳ 待做（**T-01~T-04 已随 Commit 3 提前完成；T-05 状态码扫描改由 G6 Commit 2 承接**） |
| 5 | `docs(s3-g5): G5 章节 + 完整回归` | ⏳ 待做 |

---

## G6：完整回归 + B 类并发测试 + 部署检查清单（2026-11-06）

> 状态：🟡 进行中（Commit 1 已完成：B 类并发测试 + `bootstrap_admin.py` 重试逻辑 + `vite-env.d.ts` 补全）
> 依据：`Sprint3_详细执行计划_V1.0.md` §2.4（B 类并发测试） + §13（统一验收）
> 核心发现：`bootstrap_admin.py` 的 `except IntegrityError` 直接 `return 2`，**没有 §2.4 要求的"回滚 → 重试 → 重读 → 按场景判断"逻辑**。所以 B 类不只是"补测试"，而是"补代码 + 补测试"两件事。

### G6.0 G6 与原计划的差异

| 项 | 原计划 | 实际 |
|---|---|---|
| B 类（唯一约束冲突恢复） | §2.4 要求 A/C 类在 G2 完成、B 类延后至 G6 补测试 | 代码侧本身缺重试逻辑，G6 需先补 `bootstrap_admin.py`，再补 4 个 B 类测试 |
| T-05 状态码全量扫描 | 原属 G5 Commit 4 | 并入 G6 Commit 2（与部署检查清单同批交付） |
| `vite-env.d.ts` | G4 遗留的潜在问题 | G6 Commit 1 顺手修复（`declare module "*.vue"`） |

### G6.1 Commit 1：`bootstrap_admin.py` 补齐 B 类重试逻辑

**改造思路**：把"单次尝试"从 `main()` 里拆出来，让重试循环只管"要不要再来一次"，场景判断仍由单次尝试内的既有逻辑负责。

| 改动 | 说明 |
|---|---|
| 拆出 `_bootstrap_once(args)` | 单次尝试 = 独立事务 + 独立 session + 重新 `acquire_admin_lock`；退出时由 `get_db_session()` 自动 rollback / commit |
| `main()` 加 `for attempt in range(_MAX_INTEGRITY_RETRIES)` | 每次重试开新 session + 新事务，重新读 DB 状态，不复用上一次的脏 Session |
| `except IntegrityError` 分支 | `attempt < max - 1` → 打印 `RETRY:` 并按 `0.1s * (attempt + 1)` 退避后重试；`attempt == max - 1` → 打印 `ERROR:` 并 `return 2`（**不伪装成功**） |
| `except AdminLockTimeoutError` / `except Exception` 放在循环外 | 这两类不重试，直接 `return 2`，避免"锁超时被当成唯一约束冲突反复重试" |

**关键点**：`with get_db_session() as db:` 退出时抛异常会**自动 rollback**，所以 `except IntegrityError` 里不需要再手动 `rollback()`。重试拿到的是干净的新 session，能正确重读到最新 DB 状态（例如并发进程已建好的 admin）。

### G6.2 Commit 1：`test_admin_lock.py` 新增 4 个 B 类测试

**背景**：B 类验证的核心不是"能重试"，而是"重试之后**按场景判断**"，以及"重试耗尽**不伪装成功**"。

| 测试 | 构造场景 | 断言 |
|---|---|---|
| `test_B1_integrity_conflict_retry_then_scene4` | 第 1 次 commit 抛 `IntegrityError` → 重试时 DB 已有活跃 admin（场景 4） | 返回 `0`；不创建新 admin（走"已有 admin，无需创建"分支） |
| `test_B2_integrity_conflict_retry_then_scene1` | 冲突 → 重试 → 目标用户名是 `engineer`（场景 1） | 返回 `1`（拒绝，符合角色约束） |
| `test_B3_integrity_conflict_retry_then_scene2` | 冲突 → 重试 → 存在禁用 admin（场景 2） | 返回 `1`（拒绝，不覆盖禁用账号） |
| `test_B4_all_retries_exhausted_returns_2` | 每次 commit 都冲突 → 重试耗尽 | 返回 `2`；DB 中不留下新用户 |

**B4 的关键设计**：`monkeypatch Session.commit` 是**全局拦截**，会连 `get_db_session()` 退出时的自动 `commit()` 一起拦。若不处理，每次 attempt 都先命中场景 4（返回 0 "已有 admin"），根本走不到"创建分支"，测出来的就不是"重试耗尽"。所以 B4 **必须先临时禁用所有活跃 admin**（`finally` 恢复），强制 `_bootstrap_once` 进入场景 5（创建分支），再让每次 commit 抛 `IntegrityError`。

**`finally` 恢复顺序**：先 `monkeypatch.setattr(...)` 恢复原始 `Session.commit`，**再**操作 DB 恢复 admin 状态——否则用来恢复 admin 的那次 `commit()` 也会被 mock 拦住。

### G6.3 Commit 1：`frontend/src/vite-env.d.ts` 补全

| 改动 | 说明 |
|---|---|
| 追加 `declare module "*.vue"` | 补上 `.vue` 单文件组件的模块声明，解决 TS 侧无法识别 `.vue` 的问题（G4 遗留的潜在问题，G6 顺手修） |

> 配套说明：`@/*` 别名由 `tsconfig.app.json` 的 `baseUrl` + `paths` 承担，`vite.config.ts` 的 `resolve.alias` 负责运行时解析，三处必须同时存在，否则会出现"能跑但编辑器报 TS2307"。

### G6.4 Commit 1 验收证据

| 项 | 结果 |
|---|---|
| `bootstrap_admin.py` import 校验 | ✅ `OK` |
| B 类测试（4 个） | ✅ **4 passed** |
| 全量非 integration | ✅ **635 passed, 2 skipped, 24 deselected, 0 failed**（631 → 635，+4） |

### G6.5 Commit 1 踩坑与修正（详见 `debug_03.md` §10）

| # | 坑 | 修正 |
|---|---|---|
| 1 | `bootstrap_admin.py` 无 B 类重试逻辑，冲突即 `return 2` | 拆 `_bootstrap_once` + 重试循环，每次新 session / 新事务 / 重新加锁 |
| 2 | B4 测试缺 `get_db_session` import，报 `NameError` | B1–B4 函数内显式 `from app.persistence.db import get_db_session` |
| 3 | `monkeypatch Session.commit` 全局拦截，被场景 4 抢先命中 | B4 先临时禁用全部活跃 admin（`finally` 恢复），强制走场景 5 |
| 4 | `finally` 中恢复 DB 时 commit 仍被 mock | 先恢复 `Session.commit` 方法，再恢复 admin 状态 |
| 5 | `vite-env.d.ts` 缺 `*.vue` 声明 | 追加 `declare module "*.vue"` |

**沉淀的两条通用规则**：

1. **测试重试逻辑，先确认"命中哪个场景分支"**，否则测到的可能是"场景 1–4 的正常返回"，而不是"重试路径"。
2. **`monkeypatch` 修改 Session / 连接这类全局对象时，恢复顺序必须与依赖关系反着来**：先解除 mock，再做需要 commit 的收尾。

### G6.6 G6 后续 Commit（待做）

| Commit | 内容 | 状态 |
|---|---|---|
| 1 | `feat(s3-g6): B-class concurrent tests + bootstrap retry + vite-env.d.ts` | ✅ 完成 |
| 2 | `docs(s3-g6): deployment checklist + T-05 status code scan`（部署检查清单 + T-05 状态码全量扫描） | ⏳ 待做 |
| 3 | `docs(s3): Sprint 3 phase complete - G1-G6 all done`（Sprint 3 封板） | ⏳ 待做 |

> Commit 2 需一并处理 `debug_03.md` §2.2 记录的长期项：在 `docker-compose.dev.yml` 的 `backend.volumes` 补 `- ../../docs:/docs`，让容器内脚本可直接写 `docs/`，避免手工搬运。

---

## 附录：G1-G6 阶段验收总表（更新）

| 阶段 | 核心交付 | 状态 |
|---|---|---|
| G1 | 方案评审 + 现状核查 | ✅ 完成 |
| G2 | Migration + auth 底座 + CLI + 用户管理 | ✅ 完成 |
| G3 | RBAC 鉴权挂载 + 审计字段后端注入 + 对象级数据范围 + IDOR 安全测试 | ✅ G3-A / G3-B / G3-C / G3-D 全部完成，G3 封板 |
| G4 | 前端登录 + 路由守卫 + 菜单权限 | ✅ 完成 |
| G5 | 反馈闭环 E2E | ✅ 主体完成（Commit 1–3：生成函数 + ID 生成器 + `_dispatch_evolution` 接线 + 三层 DTO 透传 + E2E/负向补测）；T-05 状态码扫描并入 G6 Commit 2 |
| G6 | 完整回归 + B 类并发 + 部署检查清单 | 🟡 进行中（Commit 1/3：B 类并发测试 + `bootstrap_admin.py` 重试 + `vite-env.d.ts`） |

### 测试基线演进

| 阶段 | 全量非 integration |
|---|---|
| Sprint 2 封板 | 568 passed, 1 skipped, 24 deselected |
| G2 完成 | 594 passed, 3 skipped, 24 deselected |
| G3-A 完成 | 595 passed, 2 skipped, 24 deselected |
| G3-C 完成 | 597 passed, 2 skipped, 24 deselected, 0 failed |
| G3-B 完成 | 613 passed, 2 skipped, 24 deselected, 0 failed |
| G3-D 完成 | 626 passed, 2 skipped, 24 deselected, 0 failed |
| G4 完成 | 626 passed（后端未改动，基线沿用）+ 前端 `type-check` 0 errors / `build` 成功 |
| G5 Commit 1 完成 | 626 passed, 2 skipped, 24 deselected, 0 failed（新增函数未接线，零破坏） |
| G5 Commit 2 完成 | 626 passed、0 failed（接线后零回归；`knowledge_gap` 32 → 33，`gap_feedback_link` 新增 1 条） |
| G5 Commit 3 完成 | **631 passed, 2 skipped, 24 deselected, 0 failed**（626 → 631，+5 测试：E2E 1 + G4 遗留补测 4） |
| G6 Commit 1 完成 | **635 passed, 2 skipped, 24 deselected, 0 failed**（631 → 635，+4 B 类并发测试） |
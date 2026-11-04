# Sprint 3 变更与阻塞记录（角色权限与端到端闭环）

> 起始：2026-11-03
> 依据：`Sprint3_详细执行计划_V1.0.md`（最终定稿）
> 阶段：G1（方案评审）✅ 完成，G2（Migration + auth 底座）✅ 完成，G3-A（业务路由挂鉴权）✅ 第 1、2 批完成（第 3 批待做）

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
| 5 | `app/api/` 端点清单 | ⏳ 待 G2 编码前最终核对 |
| 6 | `submit_task_feedback` 生成逻辑 | ⏳ 待核实 |
| 7 | `feedback_service` 写 `created_by` | ⏳ 待核实（当前不含 `created_by` 参数） |
| 8 | `update_expert_decision` 写 `expert_decision_by` | ⏳ 待核实 |
| 9 | 当前 Git HEAD 与测试基线 | ✅ `Sprint 2 封板`：`568 passed, 1 skipped, 24 deselected` |
| 10 | `dev.ps1` 的 `uv sync` 清理 | ✅ 已确认清理 |
| 11 | `users` 表内管理员账号 | ✅ 当前 0 行（`baseline_snapshot.md` 已记录） |
| 12 | 候选 ID 特殊连字符 | ⏳ 暂不修（列入技术债），前端已兜底 |
| 13 | `bootstrap.ps1` 的 `uv sync` 清理 | ✅ 已确认清理 |
| 14 | `demo_sprint2.ps1` 只读入口 | ⏳ 待确认（是否有 `--readonly`） |
| 15 | CI 配置 `APP_ENV=test` | ✅ Sprint 2 已修复 |

### 手工确认清单

- [ ] `demo_sprint2.ps1` 是否有 `--readonly` 只读入口
- [ ] 测试基线重测（预期 `568 passed`）
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

```
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

> 状态：✅ 完成（第 1、2 批：`cases.py` / `review_tasks.py` / `feedbacks.py` / `dashboard.py`；第 3 批 `rule_candidates.py` / `knowledge_docs.py` / `knowledge_gaps.py` 待做）
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
| `GET /rule-candidates`、`PATCH /rule-candidates/{id}`、`POST /rule-candidates/{id}/approve`、`POST /rule-candidates/{id}/reject` | rule_admin / admin | ⏳ 第 3 批 |
| `GET /knowledge-docs`、`POST /knowledge-docs/upload` | rule_admin / admin | ⏳ 第 3 批 |
| `GET /knowledge-gaps`、`PATCH /knowledge-gaps/{id}` | rule_admin / admin | ⏳ 第 3 批 |

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

### G3-A.3 新增技术决策

| # | 决策 | 落地位置 |
|---|---|---|
| **R1** | `require_roles` 返回完整 `CurrentUser`（含 id + role），供 G3-B 数据范围过滤使用 | `app/core/auth.py` |
| **R2** | 每个 API 组至少 3 个负面用例（无 header / 非法整数 / 角色 403） | 各测试文件 |
| **R3** | D1–D5 + R1–R3 统一在 G6 收尾时写入 changelog / debug | 本文件 + `debug_03.md` |
| **R4** | 业务参数校验失败用 `BusinessValueError`；标准 `ValueError` 走 500 暴露 bug | `app/core/exceptions.py` + `errors.py` |
| **R5** | session 级 fixture 前缀必须唯一命名空间（`test_admin_global_` / `test_eng_global_` / `test_auth_`），跨文件不误删 | `conftest.py` + `test_auth.py` |
| **R6** | "全库状态"断言测试不 `delete`，改用"临时修改 + finally 恢复" | `test_auth.py::test_last_admin_protection` |
| **R7** | API 层调用 Service 前必须 `Select-String -Pattern "^def "` 核对函数名 | 开发规范 |

### G3-A.4 踩坑与修正

详见 `debug_03.md` §4，共 7 类问题：

1. **全局 `except ValueError → 422` 过度捕获**：改为 `BusinessValueError → 422`，标准 `ValueError` 走 500。
2. **`create_task` 返回 str 与 API 解包不一致**：临时 `get_task` 回查，G3-C 升级 service 返回 dict。
3. **batch 的 `resp["task_id"]` 访问字符串索引**：改为直接用 str。
4. **`list_review_tasks` 缺 `status` 参数 + 缺 `ROLE_REVIEWER`**：补 Query 入参 + 补角色。
5. **Pydantic schema 字段 ≠ ORM 列**：`suggestion_diff` 不是 DB 列，改用 `task_service.submit_task_feedback`。
6. **`test_auth.py` 前缀与 conftest 冲突**：改为 `test_auth_%` 前缀；最后管理员保护改用"降级+恢复"。
7. **`dashboard.py` 函数名笔误**：`get_summary` → `get_dashboard_summary`。

### G3-A.5 验收证据

| 项 | 结果 |
|---|---|
| `test_api.py` | ✅ 17 passed |
| `test_dashboard.py` | ✅ 6 passed |
| `test_feedback_rule.py` | ✅ 全 passed |
| `test_review_persistence.py` | ✅ 全 passed |
| `test_auth.py` | ✅ 26 passed, 2 skipped |
| **全量非 integration** | ✅ **595 passed, 2 skipped, 24 deselected, 0 failed** |

### G3-A.6 遗留到 G3-C 的待办

1. 升级 `task_service.create_task` 返回完整 dict，恢复 `CreateTaskResponse(**resp)`。
2. 修复 `ReviewResult.total_defects` 落库语义，保证与运行内存解析数量一致。
3. 反馈提交逻辑从 `task_service.submit_task_feedback` 收敛进 `feedback_service`。
4. `update_expert_decision` / `submit_feedback` 的审计字段注入升级（`updated_by` / `created_by`）。

---

## 附录：G1-G6 阶段验收总表（更新）

| 阶段 | 核心交付 | 状态 |
|---|---|---|
| G1 | 方案评审 + 现状核查 | ✅ 完成 |
| G2 | Migration + auth 底座 + CLI + 用户管理 | ✅ 完成 |
| G3 | 业务路由挂鉴权 + IDOR（G3-A 第 1、2 批完成） | ⏳ 进行中（第 3 批 + G3-B 待做） |
| G4 | 前端登录 + 路由守卫 + 菜单权限 | ⏳ 待开始 |
| G5 | 审计字段 + 反馈闭环 E2E | ⏳ 待开始 |
| G6 | 完整回归 + 负向安全测试 + B 类并发 | ⏳ 待开始 |
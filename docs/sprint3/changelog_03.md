# Sprint 3 变更与阻塞记录（角色权限与端到端闭环）

> 起始：2026-11-03
> 依据：`Sprint3_详细执行计划_V1.0.md`（最终定稿）
> 阶段：G1（方案评审）✅ 完成，G2（Migration + auth 底座）✅ 完成

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

## 附录：G1-G6 阶段验收总表（更新）

| 阶段 | 核心交付 | 状态 |
|---|---|---|
| G1 | 方案评审 + 现状核查 | ✅ 完成 |
| G2 | Migration + auth 底座 + CLI + 用户管理 | ✅ 完成 |
| G3 | 业务路由挂鉴权 + IDOR | ⏳ 待开始 |
| G4 | 前端登录 + 路由守卫 + 菜单权限 | ⏳ 待开始 |
| G5 | 审计字段 + 反馈闭环 E2E | ⏳ 待开始 |
| G6 | 完整回归 + 负向安全测试 + B 类并发 | ⏳ 待开始 |
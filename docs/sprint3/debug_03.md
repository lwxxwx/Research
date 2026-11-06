# Sprint 3 开发与调试问题记录（Debugging Notes）

> **用途**：沉淀 Sprint 3 开发过程中遇到的典型 Bug、环境问题及排查思路，供后续开发、联调及运维部署参考。
> **归档日期**：2026-11-03
> **最后更新**：2026-11-06（新增 S3-G4-FrontendAuth、S3-G5-KnowledgeGap〈Commit 1、2、3〉）

---

## 目录

1. [S3-G2-Migration：多 Head 问题与增量 SQL 生成](#1-s3-g2-migration多-head-问题与增量-sql-生成2026-11-03)
2. [S3-G2-PreCheck：脚本路径与容器挂载陷阱](#2-s3-g2-precheck脚本路径与容器挂载陷阱2026-11-03)
3. [S3-G2-TestIsolation：测试隔离与并发断言修正](#3-s3-g2-testisolation测试隔离与并发断言修正2026-11-03)
4. [S3-G3A-API：G3-A 鉴权改造后的 API 层排障](#4-s3-g3a-apig3-a-鉴权改造后的-api-层排障2026-11-04)
5. [S3-G3C-AuditFields：审计字段注入与手测踩坑](#5-s3-g3c-auditfields审计字段注入与手测踩坑2026-11-05)
6. [S3-G3B-DataScope：对象级数据范围与 IDOR 防护](#6-s3-g3b-datascope对象级数据范围与-idor-防护2026-11-05)
7. [S3-G3D-ErrorSemantics：错误语义统一与负面测试补全](#7-s3-g3d-errorsemantics错误语义统一与负面测试补全2026-11-05)
8. [S3-G4-FrontendAuth：前端登录 + 路由守卫 + 菜单权限](#8-s3-g4-frontendauth前端登录--路由守卫--菜单权限2026-11-06)
9. [S3-G5-KnowledgeGap：知识缺口生成链路补实现](#9-s3-g5-knowledgegap知识缺口生成链路补实现2026-11-06)
10. [S3-G2-AuthBase：SQLAlchemy Session Autobegin 陷阱](#10-s3-g2-authbasesqlalchemy-session-autobegin-陷阱g3-阶段验证)
11. [附：版本与适用范围](#11-附版本与适用范围)

---

## 1. S3-G2-Migration：多 Head 问题与增量 SQL 生成（2026-11-03）

> **状态**：✅ 已修复
> **核心难点**：新增 Migration 后 Alembic 出现双 Head（`s2_f07_case_benchmark` 和 `s2j_g2_roles` 并行），导致 `alembic upgrade head` 直接报错。

### 1.1 坑点与修复全过程

#### 坑 1：Migration 分叉（Multiple head revisions）

**现象**：

```text
$ alembic upgrade head
ERROR [alembic.util.messaging] Multiple head revisions are present for given argument 'head';
please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads
```

**根因**：

新建的 `s2j_g2_roles.py` 里 `down_revision = "s2a_schema_updates"`，但 Sprint 2 已经有 `s2a_schema_updates → s2_f05_rule_publish → s2_f07_case_benchmark` 的链，导致两条时间线并行：

```text
s2a_schema_updates
├─→ s2_f05_rule_publish → s2_f07_case_benchmark (head 1)
└─→ s2j_g2_roles (head 2) ← 这里分叉了
```

**修复**：

把 `s2j_g2_roles.py` 的 `down_revision` 改为 `"s2_f07_case_benchmark"`（接在 Sprint 2 最新 Migration 之后）。

```python
# 改前
down_revision = "s2a_schema_updates"

# 改后
down_revision = "s2_f07_case_benchmark"
```

修复后的链：

```text
s2a_schema_updates → s2_f05_rule_publish → s2_f07_case_benchmark → s2j_g2_roles (head)
```

#### 坑 2：`alembic upgrade head --sql` 生成的 SQL 包含完整历史

**现象**：

执行 `alembic upgrade head --sql > preview.sql` 后，生成的 SQL 文件里包含了：

- `CREATE TABLE alembic_version`
- `CREATE TABLE users`
- `CREATE TABLE knowledge_doc`
- ……

全部从 `<base>` 开始的完整建表语句，不是仅包含 G2 增量 SQL。

**根因**：

`alembic upgrade head --sql` 模式下，Alembic 不知道当前库的真实版本（因为 `--sql` 是纯离线模式，不连接数据库读 `alembic_version` 表），所以假设从空库开始，打印完整迁移链的 SQL。

**修复**：

指定起点 revision，用 `alembic upgrade s2_f07_case_benchmark:head --sql`，只打印从 `s2_f07_case_benchmark` 到 head 的增量 SQL。

```powershell
docker compose ... exec backend alembic upgrade s2_f07_case_benchmark:head --sql | Out-String | Set-Content docs\sprint3\s2j_g2_roles_preview.sql
```

**生成结果（已验证正确）**：

```sql
BEGIN;
-- Running upgrade s2_f07_case_benchmark -> s2j_g2_roles
ALTER TABLE users ADD COLUMN role VARCHAR(32) DEFAULT 'engineer' NOT NULL;
ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT true NOT NULL;
ALTER TABLE users ADD CONSTRAINT check_users_role CHECK (role IN ('admin','engineer','reviewer','rule_admin'));
CREATE INDEX idx_users_role ON users (role);
ALTER TABLE review_result ADD COLUMN created_by BIGINT;
ALTER TABLE review_result ADD CONSTRAINT fk_review_result_created_by_users FOREIGN KEY(created_by) REFERENCES users (id) ON DELETE SET NULL;
CREATE INDEX idx_review_result_created_by ON review_result (created_by);
ALTER TABLE rule_candidates ADD COLUMN reviewed_by BIGINT;
ALTER TABLE rule_candidates ADD CONSTRAINT fk_rule_candidates_reviewed_by_users FOREIGN KEY(reviewed_by) REFERENCES users (id) ON DELETE SET NULL;
CREATE INDEX idx_rule_candidates_reviewed_by ON rule_candidates (reviewed_by);
UPDATE alembic_version SET version_num='s2j_g2_roles' WHERE alembic_version.version_num = 's2_f07_case_benchmark';
COMMIT;
```

这个文件现在可以作为 G2 Migration 的验收证据 commit 进 git。

### 1.2 核心认知与总结

1. **新增 Migration 必须显式确认 `down_revision`**：每次新增 Migration 前，先跑 `alembic heads` 看当前 head。把 `down_revision` 设成当前 head，形成线性链。忘了改会导致分叉，之后 `upgrade head` 直接报错。
2. **`--sql` 模式必须指定起点 revision**：`alembic upgrade head --sql` 从 `<base>` 开始，不是从当前库版本开始。要生成增量 SQL，用 `alembic upgrade <current>:head --sql`。

---

## 2. S3-G2-PreCheck：脚本路径与容器挂载陷阱（2026-11-03）

> **状态**：⚠️ 临时规避（存在长期优化点）
> **核心难点**：为 G2 前期核查写的 `g2_pre_check.py` 脚本，因容器内路径与宿主机路径不一致，导致核查结果全部误判。

### 2.1 坑点与修复全过程

#### 坑 1：容器内 ROOT_DIR 误解

**现象**：

`g2_pre_check.py` 报告 `deps.py` 缺失、`db.py` 缺失、CI 配置缺失，几乎全部 ❌。但实际这些文件在宿主机上都存在。

**根因**：

脚本的 `ROOT_DIR = Path("/app")` 在容器内是 backend 的根（因为 `docker-compose.dev.yml` 里 `../../backend:/app`），但脚本里检查 `backend/app/api/deps.py` 变成了 `/app/backend/app/api/deps.py`（不存在）。实际路径是 `/app/app/api/deps.py`。

**修复**：

不改脚本逻辑，直接用 PowerShell 手工跑 4 条命令：

```powershell
Get-ChildItem backend\app\api\deps.py, backend\app\persistence\db.py
Get-Content backend\app\api\deps.py
Select-String -Path "backend\app\persistence\db.py" -Pattern "def get_db_session" -Context 0,15
Select-String -Path "backend\alembic\versions\s2a_schema_updates.py" -Pattern "^revision\s*="
```

**确认结果**：

- `deps.py` 存在，为空壳（预留位置：认证 / 限流 / trace）
- `db.py` 的 `get_db_session` 是标准 `@contextmanager`，auto commit/rollback/close
- `s2a_schema_updates` revision = `"s2a_schema_updates"`

#### 坑 2：`docs/` 目录未挂载进容器

**现象**：

脚本 `REPORT_PATH = Path("/app/../docs/sprint3/g2_pre_check_audit.md")` 想写进 `docs/sprint3/`，但执行后宿主机 `docs/sprint3/` 里看不到文件。

**根因**：

`docker-compose.dev.yml` 的 backend 服务 volumes 只有：

```yaml
volumes:
  - ../../backend:/app
  - ../../data:/data
  - ../../scripts:/scripts
  - ../../infra:/infra
  - ../../out:/out
```

`docs/` 未挂载，所以容器内的 `/docs` 是容器内部目录，宿主机看不到。而 `/app/../docs` 被解析为 `/docs`（根目录），也是容器内部。

**修复**：

改用 `/scripts/g2_pre_check_audit.md`（`scripts/` 已挂载），跑完后手动 `Move-Item` 到 `docs/sprint3/`。

**长期建议**：

在 `docker-compose.dev.yml` 的 `backend.volumes` 里加一行 `- ../../docs:/docs`，让容器内的 `/docs` 直接等于宿主机 `docs/`，所有脚本都能直接写。这个改动建议在 G6 Demo 脚本升级前完成。

### 2.2 核心认知与总结

1. **容器内视角 vs 宿主机视角**：写脚本时，路径要按容器内挂载点来算（`/app/app/api/deps.py`），而不是宿主机路径（`backend/app/api/deps.py`）。
2. **写文件前先确认挂载点**：不要在未挂载的目录（如 `/docs`、`/tmp` 之类）写文件，否则文件只存在于容器内，`docker compose restart` 就丢。
3. **优先用已挂载目录做输出**：`/scripts/`、`/data/`、`/out/` 都已挂载，直接写这些目录就不会丢文件。

---

## 3. S3-G2-TestIsolation：测试隔离与并发断言修正（2026-11-03）

> **状态**：✅ 已修复
> **核心难点**：3 个测试文件首次编写时全部失败，暴露出 3 个工程化问题。

### 3.1 坑 1：pg_session fixture 用「事务回滚」导致 HTTP 请求看不到数据

**现象**：

`test_auth.py` / `test_bootstrap_admin.py` 首次跑时大量 FAILED，报 `user not found`、`admin 不存在` 等。

**根因**：

原 `pg_session` fixture 用：

```python
conn = engine.connect()
trans = conn.begin()
sess = Session(bind=conn)
yield sess
trans.rollback()  # ← 试图用外层事务隔离
```

但 `create_user` / `login` 等 HTTP 请求走 `get_db_session()`，开的是完全独立的新连接。外层事务里 `db.add(user)` + `db.flush()` 的数据，HTTP 请求的 Session 根本看不到（PostgreSQL 默认 READ COMMITTED 也不会跨越未提交事务）。

**修复**：改为「真实 commit + 事后清理」：

```python
@pytest.fixture()
def pg_session():
    engine = create_engine(settings.database_url)
    sess = Session(bind=engine)
    yield sess
    # 测试结束按前缀清理
    sess.query(User).filter(
        User.username.like("test_admin_%") |
        User.username.like("test_eng_%") |
        User.username.like("test_dis_%")
    ).delete(synchronize_session=False)
    sess.commit()
    sess.close()
```

`admin_user` / `engineer_user` / `disabled_user` fixture 也都改为 `pg_session.commit()` 真正落库。

**教训**：

- 测试隔离不能靠事务回滚（跨 Session 场景下根本隔离不了）。
- 只能靠「唯一前缀 + 事后清理」（`test_admin_xxx` 前缀确保不撞生产数据）。
- 这是 Sprint 2 `test_review_persistence.py` 已经踩过的坑，但 G2 第一次写测试时没复用这个模式。

### 3.2 坑 2：并发测试假设线程调度确定，断言不稳定

**现象**：

`test_A4_concurrent_acquire_serialized` 时而 PASS 时而 FAIL，特别是 `results.index("A:released") < results.index("B:acquired")` 这条断言。

**根因**：

原代码假设「A 线程先拿到锁，B 线程后拿」。但 Python GIL + 操作系统调度是随机的——B 线程完全可能先跑到 DB 拿到锁。断言写死「A 先拿」会随机 FAIL。

**修复**：改为「两种合法情况都覆盖」：

```python
if idx_a_acq < idx_b_acq:
    # A 先拿锁，则必须 A 释放后 B 才能 acquire
    assert idx_a_rel < idx_b_acq
else:
    # B 先拿锁，则必须 B 释放后 A 才能 acquire
    idx_b_rel = results.index("B:released")
    assert idx_b_rel < idx_a_acq
```

**教训**：

并发测试不能假设线程调度顺序，只能验证「互斥语义」本身（谁先拿都行，但拿到锁时对方必须已释放）。

### 3.3 坑 3：最后管理员保护未触发（真实 DB 有历史 admin）

**现象**：

`test_last_admin_protection` 首次 FAILED，`active_admin_count` 实际是 2，不是 1，所以保护逻辑没触发。

**根因**：

`admin_user` fixture 创建的 admin（`test_admin_xxx`）和真实 DB 里已有的 admin（id=3107）同时是启用状态。`count(*) WHERE role='admin' AND is_active=true` 返回 2，> 1 所以允许降级。

**修复**：测试内主动清理其它所有 admin：

```python
def test_last_admin_protection(client, admin_user, pg_session):
    admin_id, _ = admin_user
    # 关键：删除其它所有活跃 admin，保证当前是系统仅剩的活跃管理员
    pg_session.query(User).filter(
        User.id != admin_id,
        User.role == "admin",
        User.is_active.is_(True)
    ).delete(synchronize_session=False)
    pg_session.commit()
    # 然后才 PATCH
```

**教训**：

- 测试的前置条件必须由测试自己保证，不能靠「碰运气不撞上历史数据」。
- 涉及「全库状态」的断言（如 `count`），必须先收窄到「本测试构造的场景」，避免被生产数据污染。

### 3.4 修复后验证结果

| 验证项 | 执行结果 |
|---|---|
| 3 个测试文件单独跑 | ✅ 26 passed, 2 skipped（2 skip 是场景 4/5，需隔离 DB 环境） |
| 全量非 integration | ✅ 594 passed, 3 skipped, 24 deselected（从 568 → 594，新增 26 个测试） |

---

## 4. S3-G3A-API：G3-A 鉴权改造后的 API 层排障（2026-11-04）

> **状态**：✅ 已修复
> **核心难点**：G3-A 给业务路由挂 RBAC 鉴权后，`test_api.py` / `test_dashboard.py` / `test_feedback_rule.py` / `test_review_persistence.py` 合并跑时出现 8 类问题，横跨 API 层、Service 层、全局异常处理器、ORM 模型、测试脚手架。
> **最终结果**：全量非 integration `595 passed, 2 skipped, 0 failed`。

### 4.1 坑 1：全局 ValueError handler 过度捕获（掩盖代码 bug）

**现象**：

`task_service.list_tasks_filtered` 对非法 `case_class` 抛 `ValueError`，路由层没做 `try/except`，最初注册了全局 `@app.exception_handler(ValueError) → 422`。测试 `test_list_tasks_invalid_params_422` 通过。但同类 `review_status` 非法值却返回 200（因为路由未透传 Query 参数）。

**根因**：

1. **全局 `except ValueError → 422` 过度捕获**：`json.loads` 解析 DB 脏数据失败、`int()` 类型转换失败等代码 bug 也会被伪装成 422，掩盖真实故障。
2. **路由 Query 参数漏声明**：`review_status` / `status` 参数未在 `@router.get` 签名里声明，FastAPI 静默丢弃，service 收到 `None`，校验分支不触发。

**修复**：

1. **新建 `app/core/exceptions.py` 定义 `BusinessValueError(ValueError)`**（不 import 任何 service，避免循环依赖）。
2. **`errors.py` 只注册 `BusinessValueError → 422`**，删除全局 `ValueError` handler，标准 `ValueError` 走兜底 500 暴露代码 bug。
3. **只把"用户输入参数校验失败"的 6 处改为 `raise BusinessValueError`**：
   - `task_service.list_tasks_filtered` 5 处（`case_class` / `review_status` / `status` / `limit` / `offset`）
   - `task_service.update_expert_decision` 1 处（`expert_decision` 枚举）
4. 其余 `raise ValueError` **一律保留**：
   - 路由层已有 `try/except` 转 400 的（`case_service.py` 2 处、`report_service.py` 2 处）
   - 服务内部数据一致性校验（`rule_evolution_service.py` 7 处、`benchmark_service.py` 1 处）
   - 被注释掉的旧代码（`feedback_service.py` 4 处）
   - G3-A 第 3 批：`knowledge_gap_service.py` 2 处（挂鉴权时同步改）
5. **`task_service.py` import 从 `app.core.exceptions` 而非 `app.core.errors`**，避免 `task_service → errors → feedback_service` 的耦合链。

**教训**：

业务异常必须用**自定义类型**与标准异常区分，不能图省事用宽泛的 `except ValueError` / `@app.exception_handler(ValueError)` 一刀切。否则"系统内部坏了"被伪装成"用户输入错了"，掩盖真实故障。

### 4.2 坑 2：`task_service.create_task` 返回类型与 API 解包不一致

**现象**：

```text
TypeError: CreateTaskResponse() argument after ** must be a mapping, not str
```

**根因**：

`task_service.create_task()` 返回**字符串 task_id**，不是 dict。`CreateTaskResponse(**resp)` 只能解包 mapping。

**修复（临时，G3-A）**：

```python
task_id = task_service.create_task(body.case_id)
detail = task_service.get_task(task_id)
if detail is None:
    return CreateTaskResponse(task_id=task_id, status="failed", defects_count=0)
return CreateTaskResponse(
    task_id=detail["task_id"],
    status=detail["status"],
    defects_count=detail["defects_count"],
)
```

> 遗留（G3-C）：升级 `task_service.create_task` 返回完整 dict 后恢复 `CreateTaskResponse(**resp)`。

### 4.3 坑 3：批量创建 `resp["task_id"]` 访问字符串索引

**现象**：

`POST /review/tasks/batch` 每次调用抛 `TypeError: string indices must be integers`，被 `except Exception` 吞掉，所有子任务记 `status="failed"`。

**根因**：

`task_service.create_task` 返回 str，但 batch 里 `resp["task_id"]` 当 dict 访问。

**修复**：

```python
resp = task_service.create_task(cid)   # 返回 str
items.append({
    "case_id": cid,
    "task_id": resp,                   # ← 直接用 str
    "status": "success",
    "error": None,
})
```

### 4.4 坑 4：`list_review_tasks` 缺 status 参数 + 缺 `ROLE_REVIEWER` 角色

**现象**：

1. 前端传 `?status=pending` 被静默丢弃，service 收到 None，无过滤。
2. reviewer 角色访问 `GET /review/tasks` 返回 403（权限矩阵要求 engineer / reviewer / admin 三角色）。

**根因**：

路由签名漏 `status: Optional[str] = Query(None)`；`require_roles` 漏 `ROLE_REVIEWER`。

**修复**：

```python
@router.get("", response_model=TaskListResponse)
def list_review_tasks(
    current: CurrentUser = Depends(
        require_roles(ROLE_ENGINEER, ROLE_REVIEWER, ROLE_ADMIN)   # ← 补 REVIEWER
    ),
    status: Optional[str] = Query(None, description="pending / done / failed"),   # ← 补
    case_class: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    limit: int = Query(20, ge=0),
    offset: int = Query(0, ge=0),
):
    result = task_service.list_tasks_filtered(
        status=status,             # ← 补透传
        case_class=case_class,
        review_status=review_status,
        limit=limit,
        offset=offset,
    )
    return result
```

### 4.5 坑 5：Pydantic schema 字段 ≠ SQLAlchemy ORM 列

**现象**：

```text
TypeError: 'suggestion_diff' is an invalid keyword argument for FeedbackItem
```

**根因**：

`feedbacks.py` 手动构造 `FeedbackItem(suggestion_diff=...)`，但 `suggestion_diff` 是 Pydantic 结构字段，DB 列名是 `suggestion_diff_json`；`rule_candidate` 同样不是 DB 列（列名是 `rule_candidate_ref`）。

**修复**：

不在 API 路由层手动 `db.add(ORMModel(...))`，改用项目已封装的业务入口 `task_service.submit_task_feedback(...)`，内部完成 task/defect 翻译、ORM 映射与 FeedbackService 调用。

**教训**：

Pydantic schema 字段 ≠ DB ORM 列。写 DB 前先确认目标 ORM 模型的真实列名，优先复用 service 已有业务入口，不自行拼 ORM。

### 4.6 坑 6：`test_auth.py` 的 pg_session 前缀与 conftest 冲突

**现象**：

单跑 `test_auth.py` 全过，全量跑时 `test_api.py` / `test_feedback_rule.py` 出现大量 `ForeignKeyViolation` 或 401 ERROR。

**根因**：

1. `test_auth.py` 的 `pg_session` teardown 用 `like("test_admin_%")` 清理，误删了 `conftest.py` 的 session 级 `test_admin_global_xxx`，而后者被其它测试的 `feedback_item.created_by` / `review_result.created_by` 引用。
2. `test_last_admin_protection` 里 delete 其它所有 admin（包括 conftest 的 session 级 admin），触发 FK 违反，且后续 fixture 找不到 admin，一连串 ERROR。

**修复**：

**1. `pg_session` 前缀命名空间隔离：**

- `test_auth.py` 的 `pg_session` teardown 只用 `like("test_auth_%")`；
- 3 个 user fixture 的前缀改为 `test_auth_admin_` / `test_auth_eng_` / `test_auth_dis_`。

**2. `test_last_admin_protection` 不删其它 admin，改用"临时降级 + finally 恢复"：**

```python
def test_last_admin_protection(client, admin_user, pg_session):
    admin_id, _ = admin_user
    # 1. 其它 active admin 临时降级为 engineer（不删，避免 FK 违反）
    other_admins = pg_session.query(User).filter(
        User.id != admin_id,
        User.role == "admin",
        User.is_active.is_(True),
    ).all()
    original_roles = {u.id: u.role for u in other_admins}
    for u in other_admins:
        u.role = "engineer"
    pg_session.commit()

    try:
        resp = client.patch(
            f"/api/v1/auth/users/{admin_id}",
            headers={"X-User-Id": str(admin_id)},
            json={"role": "engineer"},
        )
        assert resp.status_code == 422, resp.text
        assert "最后一个启用管理员" in resp.json()["detail"]
    finally:
        # 3. 恢复其它 admin
        for uid, original_role in original_roles.items():
            pg_session.query(User).filter(User.id == uid).update(
                {User.role: original_role}, synchronize_session=False
            )
        pg_session.commit()
```

**教训**：

1. session 级 fixture 的前缀必须唯一命名空间，避免跨文件误删。
2. 涉及"全库状态"的测试，不能直接 delete，用"临时修改 + finally 恢复"更安全。
3. 删除被业务表引用的 user 前，先想清楚 FK 链；即使 FK 是 ON DELETE SET NULL，并发事务下也可能报错。

### 4.7 坑 7：`dashboard.py` 跨层调用函数名笔误

**现象**：

6 个 dashboard 测试全挂：

```text
AttributeError: module 'app.services.dashboard_service' has no attribute 'get_summary'
```

**根因**：

G3-A 给 dashboard.py 挂鉴权时，凭记忆写了 `dashboard_service.get_summary()`，但实际函数名是 `get_dashboard_summary()`。

**修复**：

```python
# 改前
return DashboardSummaryResponse(**dashboard_service.get_summary())

# 改后
return DashboardSummaryResponse(**dashboard_service.get_dashboard_summary())
```

**教训**：

API 层调用 Service 时，函数名必须先用 `Select-String -Path "backend\app\services\<file>.py" -Pattern "^def "` 核对，不能凭记忆写。这类笔误如果测试覆盖了会立刻暴露；如果没测试覆盖，会静默 404/500，最坑。

### 4.8 坑 8：`knowledge_gap_service.list_gaps` 缩进 bug（改 `raise` 类型时误伤后续语句）

**现象**：

admin 访问 `GET /knowledge-gaps` 返回 500，日志：

```text
File "/app/app/services/knowledge_gap_service.py", line 60, in list_gaps
total = q.count()
UnboundLocalError: cannot access local variable 'q' where it is not associated with a value
```

**根因**：

把 `raise ValueError` 改成 `raise BusinessValueError` 时，误把紧随其后的 `q = db.query(KnowledgeGap)` 也缩进到了 `if` 块内部。

```python
# 改后（错误版本）
if status is not None and status not in _ALLOWED_STATUS:
    raise BusinessValueError(f"status 必须为 {_ALLOWED_STATUS}，实际: {status!r}")
    q = db.query(KnowledgeGap)          # ← 错误缩进，进入 if 块
if status is not None:
    q = q.filter(...)
```

> `raise` 执行会直接中断控制流，`q = db.query(...)` 永远不会执行；后续访问 `q.count()` 触发 `UnboundLocalError`。

**修复**：

`q = db.query(KnowledgeGap)` 与 `if` 语句平级，顶格书写，raise 后空一行隔开：

```python
# 改后（正确）
if status is not None and status not in _ALLOWED_STATUS:
    raise BusinessValueError(f"status 必须为 {_ALLOWED_STATUS}，实际: {status!r}")

q = db.query(KnowledgeGap)
if status is not None:
    q = q.filter(...)
```

**教训**：

1. 修改 `raise` 所在 if 块代码，务必完整检查 if 的缩进边界，不要仅修改 raise 一行。
2. raise 之后缩进在同一个 if 块内的代码永远不会被执行。
3. 该类 `UnboundLocalError` 静态检查工具（ruff / pylint F821/F841）可以提前捕获，G6 CI 建议开启该规则。

**验证**：修复后 admin 访问 `/knowledge-gaps` 返回 200，全量回归 `595 passed, 0 failed`。

### 4.9 修复后验证结果

| 验证项 | 执行结果 |
|---|---|
| test_api.py 单文件 | ✅ 17 passed（含 2 个 422 关键用例） |
| test_dashboard.py 单文件 | ✅ 6 passed |
| 全量非 integration | ✅ 595 passed, 2 skipped, 24 deselected, 0 failed |

### 4.10 遗留到 G3-D 的待办

> 注：原第 4 项「审计字段注入升级」已于 **G3-C 完成**（4 个审计字段全部由后端 `get_current_user` 注入，Schema 不接受前端传入，详见 §5），本项从待办中移除。

1. **升级 `task_service.create_task`**：返回完整 dict，恢复 `CreateTaskResponse(**resp)` 解包，去掉接口 `get_task` 回查。
2. **反馈提交逻辑收敛**：将 `task_service.submit_task_feedback` 内部逻辑完整收敛进 `feedback_service`，路由层恢复调用 service、移除路由内直连 DB。

**已评估、无需修复**：

- **`ReviewResult.total_defects` 落库语义**：G3-C 已确认 `len(report.defects)` 取值逻辑正确，DB 持久化数量与运行内存解析数量一致，判定无 bug，不再列入待办。

### 4.11 经验沉淀速查

| # | 坑 | 一句话结论 |
|---|---|---|
| 1 | 全局 ValueError handler 过度捕获 | 业务参数校验异常要用自定义类型（`BusinessValueError`），标准 `ValueError` 留给兜底 500 暴露 bug |
| 2 | service 返回 str，API `**resp` 解包 | `Model(**x)` 只能解包 dict，返回 str 会 TypeError |
| 3 | batch 的 `resp["task_id"]` 访问字符串索引 | 确认 service 返回类型后再取字段 |
| 4 | 路由漏声明 Query 参数 + 缺角色 | 路由签名每个 Query 都要声明并透传；角色矩阵逐一核对 |
| 5 | Pydantic 字段 ≠ ORM 列 | 不手动 `db.add(ORMModel(pydantic字段))`，优先复用 service 业务入口 |
| 6 | fixture 前缀冲突 + 全库状态测试 | session fixture 用唯一前缀命名空间；全库状态测试用"临时修改 + finally 恢复"而非 delete |
| 7 | 跨层函数名笔误 | 调用前用 `Select-String "^def "` 核对，不凭记忆 |
| 8 | if-raise 块误伤后续变量定义 | raise 会中断流程，检查 if 块全部缩进边界；开启静态检查 F821/F841 |

---

## 5. S3-G3C-AuditFields：审计字段注入与手测踩坑（2026-11-05）

> **状态**：✅ 已修复
> **核心难点**：4 个审计字段（`review_result.created_by` / `rule_candidates.reviewed_by` / `feedback_item.created_by` / `review_defect.expert_decision_by`）必须由后端 `get_current_user` 注入，Pydantic Schema 不接受前端传入。`created_by` 涉及 Graph 全链路透传；`reviewed_by` 涉及 Service 层写入。
> **最终结果**：全量非 integration `597 passed, 2 skipped, 0 failed`。

### 5.1 改动架构

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

### 5.2 坑 1：Graph 全链路透传的 5 处改动

**现象**：

首次给代码时没确认 `_persist_review_report` 的签名（无 `state` 参数），差点给出 `state.get("created_by")` 的错误 diff。

**根因**：

`node_feedback_save(state)` 调 `_persist_review_report(report, case_id, task_id, rule_hits=..., ir_doc=...)`，但 `_persist_review_report` **不接收 `state`**，需要额外加 `created_by` 参数。

**修复**：

`_persist_review_report` 签名加第 6 个参数 `created_by: int | None = None`；`node_feedback_save` 调用时传 `created_by=state.get("created_by")`。

**教训**：

1. 跨函数透传 `state` 字段时，**必须先 `Select-String -Pattern "^def "` 看所有相关函数的完整签名**，不能假设 `state` 全程可见。
2. `TypedDict, total=False` 是加字段的最佳形态——所有字段隐式 `Optional`，加 `created_by: int | None` 不需要改任何初始化代码。

### 5.3 坑 2：候选 ID 特殊连字符导致手测路径卡住

**现象**：

用 DB 里已有候选 `RC-20261104-B46BB4` 调 `POST /rule-candidates/{id}/status`：

```powershell
$cid = "RC-20261104-B46BB4"
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/rule-candidates/$cid/status" -Method POST ...
# 返回 404: rule_candidate candidate_id=RCâ20261104âB46BB4 not found
```

**根因**：

`RC-20261104-B46BB4` 里的 `-` 是 `U+2011` 非断连字符（Sprint 2 遗留的 `_make_candidate_id()` 生成），不是 ASCII `-`。
PowerShell 5.1 显示 UTF-8 时把 `U+2011` 渲染成 `â`，`Invoke-RestMethod` 在 URL 编码/解码时与后端 DB 存储不一致，后端收到的 ID 与 DB 里的不匹配。

**规避**：

手测时用纯 ASCII 候选 ID（如 `RC-ASCII-xxx` / `RC-TEST-xxx`）：

```powershell
$cid = "RC-ASCII-0A52BFB3"   # 纯 ASCII
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/rule-candidates/$cid/status" -Method POST ...
# 200 + {candidate_id: "RC-ASCII-0A52BFB3", status: "accepted", reviewed_by: 3232}
```

**教训**：

1. **手测路径不碰脏数据**：Sprint 2 遗留的 `U+2011` 候选 ID 是已知技术债（Sprint3 附录 D 第 1 项），手测时用 ASCII ID 绕开，不在 G3-C 里修。
2. **测试文件里也用 ASCII ID**：新增的 `test_update_candidate_status_writes_reviewed_by` 用 `RC-TEST-{uuid}`（纯 ASCII），避免测试环境被 `U+2011` 污染。
3. **不要把这当 bug 修**：这是测试路径问题，不是代码问题。代码 `update_candidate_status` 的 `reviewed_by` 写入逻辑正确，只是手测时被候选 ID 卡住。

### 5.4 修复后验证结果

| 验证项 | 结果 |
|---|---|
| `created_by` 注入（admin 3232 建任务） | ✅ DB `created_by = 3232` |
| `reviewed_by` 注入（admin 3232 批准 ASCII 候选） | ✅ DB `status = accepted`, `reviewed_by = 3232` |
| 全量非 integration | ✅ 597 passed, 2 skipped, 24 deselected, 0 failed |

### 5.5 未纳入 G3-C 的遗留项

| 项 | 归属 |
|---|---|
| `_make_candidate_id()` 生成标准 ASCII `-`，解决 `U+2011` 问题 | Sprint3 附录 D 第 1 项，下一阶段 |
| `update_candidate_status` 的 `raise ValueError` 改 `BusinessValueError`（错误语义统一） | G3-D 收尾时统一梳理 |
| `ReviewResult.total_defects` 落库语义（与 `report.defects` 一致） | G3-A 已确认无 bug（`len(report.defects)` 正确），不需修 |

---

## 6. S3-G3B-DataScope：对象级数据范围与 IDOR 防护（2026-11-05）

> **状态**：✅ 已完成
> **核心难点**：engineer 只能看 `created_by = me` 的任务，reviewer / admin 看全部；engineer 越权访问他人任务必须返回 404（防枚举），但日志要区分"真 404"、"越权 404"、"历史数据无权"。`total` 的改造不能把 `case_class` 的 Python 过滤语义复刻到 SQL。
> **最终结果**：全量非 integration `613 passed, 2 skipped, 0 failed`（+16 IDOR 测试）。

### 6.1 6 项设计决策（D6-D11）

| # | 决策 | 落地 |
|---|---|---|
| **D6** | engineer 越权访问返回 404（不返回 403），防枚举 | `get_task` / `get_task_defects` / `export` 越权时返回 None / []，API 层统一 404 |
| **D7** | `created_by` 过滤仅对 **engineer** 生效 | SQL 层 `if current_user.role == "engineer": query.filter(...)` |
| **D8** | `total = len(filtered_items)` 保持内存 len；`created_by` 在 SQL 层过滤 | 方案 X（不把 case_class 复刻到 SQL） |
| **D9** | 日志区分 3 类：`task.not_found` / `audit.idor_hidden` / `audit.historical_unauthorized` | 方案 D（Service 层加日志，返回 None 不变） |
| **D10** | Service 方法加 `current_user` 参数 | `CurrentUser \| None` 类型（含 id + role） |
| **D11** | rule_admin 访问 review-tasks → 403 | G3-A `require_roles` 拦截，不进 G3-B 对象权限逻辑 |

### 6.2 坑 1：`current_user` 参数类型避免循环 import

**现象**：

最初想给 `list_tasks_filtered` 加 `current_user: CurrentUser | None` 类型注解，但 `app.core.auth` 已经 import `app.services.user_service`，`task_service` import `CurrentUser` 会形成 `task_service → core.auth → user_service → ...` 的循环链路。

**根因**：

`CurrentUser` 是 `NamedTuple`，但 `task_service` 不应该依赖 `app.core.auth`（auth 是上层，task_service 是业务层）。

**修复**：

`current_user` 参数不加类型注解（`current_user=None`），运行时 duck typing：

```python
def list_tasks_filtered(
    *,
    current_user=None,   # 不加类型注解，避免循环 import
    ...
) -> dict[str, Any]:
    if current_user is not None and getattr(current_user, "role", None) == "engineer":
        query = query.filter(ReviewResult.created_by == current_user.id)
```

用 `getattr(current_user, "role", None)` 而非 `current_user.role`，防止 `None` 或非预期类型报错。

**教训**：

1. 业务 Service 不要 import 上层 `app.core.auth`，用 duck typing + `getattr` 更安全。
2. `current_user=None` 是合法状态（脚本 / CLI 调用不带用户上下文），必须防御性处理。

### 6.3 坑 2：越权返回 None 但日志要区分 3 种情况

**现象**：

`get_task` 内部如果直接 `if rr.created_by != current_user.id: return None`，日志只能打"任务不存在"，无法区分：

- 真不存在（DB 里没这条）
- 越权（engineer 访问他人任务）
- 历史数据（`created_by IS NULL`，Sprint 1 遗留）

**根因**：

`return None` 是统一出口，但 3 种情况的审计语义完全不同——越权要告警，历史数据要提醒迁移，真不存在是正常 404。

**修复**：

3 分支日志（方案 D）：

```python
if rr is None:
    # 分支 1：真不存在
    logger.info("task.not_found: task_id=%s ...", task_id, ...)
    return None

if current_user and current_user.role == "engineer":
    if rr.created_by is None:
        # 分支 2：历史数据无权
        logger.warning("audit.historical_unauthorized: task_id=%s ...", ...)
        return None
    if rr.created_by != current_user.id:
        # 分支 3：真实越权
        logger.warning("audit.idor_hidden: task_id=%s current_user_id=%s owner_id=%s",
                       task_id, current_user.id, rr.created_by)
        return None
```

**教训**：

1. "统一 404" 和 "区分日志" 不矛盾：对外响应一致，对内日志分层，运维排障能看到是哪类 404。
2. 日志前缀用 `task.not_found` / `audit.idor_hidden` / `audit.historical_unauthorized` 便于 grep。
3. 越权日志必须打 `current_user_id` + `owner_id`，方便回溯。

### 6.4 坑 3：`case_class` 的 Python 过滤不能复刻到 SQL

**现象**：

G3-B 改造 total 时，想给 `list_tasks_filtered` 改 SQL 层 count（`query.count()`），但 `case_class` 是 Python 层从 `case_id` 前缀推导的（`_infer_case_class(case_id)`），DB 无 `case_class` 列。

**根因**：

`case_class` 的推导规则（A 类 `^case\d{3}$`、B 类 `^case_b\d{2}_[a-z0-9_]+_B$`）是 Sprint 2 冻结的语义，DB 里没有对应列，只能在 Python 层过滤。

**修复**：

方案 X（保留 `total = len(filtered_items)`）：

```python
# SQL 层：created_by 过滤（仅 engineer）
if current_user and current_user.role == "engineer":
    query = query.filter(ReviewResult.created_by == current_user.id)

rows = query.all()   # SQL 层过滤后拿数据

# Python 层：case_class 过滤（不变）
filtered_items = []
for rr, case_id in rows:
    item_case_class = _infer_case_class(case_id)
    if case_class is not None and item_case_class != case_class:
        continue
    filtered_items.append({...})

total = len(filtered_items)   # 保持内存 len
paginated = filtered_items[offset:offset+limit]
```

**教训**：

1. D8 的真正含义是"`created_by` 在 SQL 层过滤"，不是"total 从 len 改 count"。`case_class` 的 Python 过滤存在时，total 必须保持 `len(filtered_items)`。
2. 不要为了"性能优化"把已冻结的 Python 过滤语义复刻到 SQL——`case_class` 的 A/B 命名规则如果将来变化，SQL 层要同步改，违反"冻结语义不动"。
3. `created_by` 在 SQL 层过滤是必要的（不是性能优化，是语义）：engineer 的 total 必须是"自己任务的数"，不能先拉全部再 Python 过滤。

### 6.5 坑 4：`export_review_report` 的对象级权限校验

**现象**：

`export_review_report` 当前直接 `report_service.generate_markdown_report(task_id)`，没有对象级权限校验——engineer 可以导出他人任务报告。

**根因**：

`generate_markdown_report` 内部不区分调用者身份（Service 层只负责生成报告）。

**修复**：

在 `try` 之前先 `get_task(current_user=current)` 做权限校验：

```python
task_info = task_service.get_task(task_id, current_user=current)
if task_info is None:
    raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")

try:
    md = report_service.generate_markdown_report(task_id)
except Exception as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
```

**教训**：

1. 对象级权限校验优先在调用业务 Service 之前（用 `get_task` 这种"权限感知"的查询），而不是在 Service 内部。
2. `generate_markdown_report` 无需接收 `current_user`——因为已经 `get_task` 校验过权限了。

### 6.6 修复后验证结果

| 验证项 | 结果 |
|---|---|
| import 校验（3 个函数） | ✅ OK |
| 全量非 integration | ✅ 613 passed, 2 skipped, 24 deselected, 0 failed |

> +16 IDOR 测试（从 597 → 613），覆盖 5 接口 × 4 角色矩阵。

### 6.7 经验沉淀速查

| # | 坑 | 一句话结论 |
|---|---|---|
| 1 | `current_user` 参数类型 | 业务 Service 不 import `app.core.auth`，用 `current_user=None` + `getattr` 防循环 import |
| 2 | 越权返回 None 但日志区分 | "统一 404" + "分层日志" 不矛盾；3 分支：`task.not_found` / `audit.idor_hidden` / `audit.historical_unauthorized` |
| 3 | `case_class` Python 过滤 | D8 不是"total 改 count"，是"`created_by` 在 SQL 层"；`case_class` 的 Python 过滤保持不动 |
| 4 | export 对象级权限 | 在 `generate_markdown_report` 之前先 `get_task(current_user=current)` 校验 |

---

## 7. S3-G3D-ErrorSemantics：错误语义统一与负面测试补全（2026-11-05）

> **状态**：✅ 已完成
> **核心难点**：`rule_evolution_service.update_candidate_status` 的 `status` 非法从 404 改 422，需要拆 `except ValueError` 为两层（`BusinessValueError` 优先）；R2 负面用例补全（5 接口组 × 3 类）。
> **最终结果**：全量非 integration `626 passed, 2 skipped, 0 failed`。

### 7.1 坑 1：`except ValueError` 同时捕获"资源不存在"和"参数非法"

**现象**：

`rule_candidates.py` 的 `update_candidate_status` 端点：

```python
try:
    result = _svc.update_candidate_status(...)
except ValueError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
```

这个 `except ValueError` 同时捕获：

- `status` 非法（应为 422）
- `candidate_id` 不存在（应为 404）

两个语义完全不同的错误映射到同一个 404，用户传非法 `status` 会看到"candidate not found"的 404，误导。

**根因**：

`rule_evolution_service.update_candidate_status` 内部 2 个 `raise ValueError`：

1. `status` 枚举校验（应为 422）
2. `candidate_id` 不存在（应为 404）

API 层 `except ValueError → 404` 一刀切。

**修复**：

1. `rule_evolution_service.update_candidate_status` 的 `status` 校验 `raise ValueError` → `raise BusinessValueError`（保留 `candidate_id` 不存在的 `raise ValueError`）
2. `rule_candidates.py` 拆 2 层：

```python
except BusinessValueError as exc:   # 必须放前面（子类优先）
    raise HTTPException(status_code=422, detail=str(exc)) from exc
except ValueError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
```

**教训**：

1. `except Exception` 的宽泛捕获是错误语义混乱的根源。一个 `except ValueError` 背后往往有多种语义（参数非法 / 资源不存在 / 数据格式错误），需要用自定义异常类型区分。
2. `BusinessValueError` 必须是 except 链的第一个，否则被父类 `except ValueError` 先捕获。
3. 这一规则要和 G3-A 的 R4（业务异常用 `BusinessValueError`）一致：参数校验失败 = `BusinessValueError` = 422；资源不存在 = 保留 `ValueError` 但由 API 层针对性转 404。

### 7.2 坑 2：R2 负面用例漏了 5 个接口组

**现象**：

G3-D 开始前，R2 的"每个 API 组 3 个负面用例（无 header / 非法整数 / 角色 403）"覆盖率：

| 接口组 | 覆盖率 |
|---|---|
| `auth/users` / `review/tasks` / `cases` | 3/3 ✅ |
| `rule_candidates` / `knowledge_gaps` / `knowledge_docs` / `feedbacks` / `dashboard` | 0/3 ❌ |

**根因**：

G3-A 挂鉴权时，只在 `test_auth.py` / `test_review_persistence.py` / `test_case_benchmark.py` 里补了负面用例，G3-A 第 3 批（`rule_candidates` / `knowledge_docs` / `knowledge_gaps`）的负面用例没补，G3-B / G3-C 也没补。

**修复**：

在 `test_feedback_rule.py` + `test_api.py` 末尾追加 9 个负面用例，覆盖 5 个接口组 × 2–3 类。

**教训**：

1. R2 是跨阶段的验收标准，不是单个批次的局部要求。G3-A 挂鉴权时漏了，G3-D 必须补全。
2. 负面用例要按"接口组"组织，同一接口组的 3 个负面用例放一起（如 `test_rule_candidates_no_auth_401` / `test_rule_candidates_invalid_uid_401` / `test_rule_candidates_engineer_403`），便于维护。
3. `feedbacks` / `dashboard` 的负面用例在 G3-A 已经"逻辑上有"但"测试上没写"，这类"实现了但没测"的漏洞最隐蔽。

### 7.3 修复后验证结果

| 验证项 | 结果 |
|---|---|
| `rule_evolution_service.py` import `BusinessValueError` | ✅ |
| `rule_candidates.py` 的 `except BusinessValueError → 422` 生效 | ✅ |
| 新增 9 个负面用例 | ✅ |
| 全量非 integration | ✅ 626 passed, 2 skipped, 24 deselected, 0 failed（从 613 → 626） |

### 7.4 经验沉淀速查

| # | 坑 | 一句话结论 |
|---|---|---|
| 1 | `except ValueError` 混合语义 | 拆成 `except BusinessValueError → 422` + `except ValueError → 404`，子类优先 |
| 2 | R2 负面用例漏了 5 个接口组 | 跨阶段验收标准，G3-D 补全；"实现了但没测"最隐蔽 |

---

## 8. S3-G4-FrontendAuth：前端登录 + 路由守卫 + 菜单权限（2026-11-06）

> **状态**：✅ 已完成
> **核心难点**：① 权限唯一可信源 = `GET /auth/me`，`localStorage` 仅存 `user_id`（`role` 不写）；② 401 / 403 / 404 三类错误区分；③ `router.beforeEach` 的 `bootstrap()` 请求锁；④ 菜单按 `role` 动态显示。
> **最终结果**：`npm run type-check` 0 errors + `npm run build` 成功 + 8 个浏览器验收场景全部通过。

### 8.1 坑 1：`localStorage` 存 `role` 的陷阱

**现象**：

最初想在 `localStorage` 存 `{ user_id, role }` 方便前端判断菜单权限。但这违反 Sprint3 §8 约束"localStorage 不可信，role 以后端为准"——用户手动改 `localStorage.role = "admin"` 就能绕过前端菜单限制（虽然后端 `require_roles` 会拦，但前端体验会错乱：菜单显示了但接口 403）。

**根因**：

`localStorage` 是纯客户端存储，用户可任意修改。**`role` 必须从 `GET /auth/me` 实时拉取**，存内存（Pinia store），不能持久化到 `localStorage`。

**修复**：

- `localStorage` **只存 `user_id`**（用于 `X-User-Id` 请求头注入）
- `role` / `username` / `is_active` **只存 Pinia store**（内存），每次 `bootstrap()` 从 `GET /auth/me` 刷新
- 刷新页面后 store 丢失，`router.beforeEach` 触发 `bootstrap()` 重新拉取

**教训**：

1. **"权限唯一可信源 = 后端"是铁律**，前端只能缓存"用户 id"这种非敏感标识，不能缓存"角色"这种权限依据。
2. `localStorage` 只能当"会话锚点"（记住"我是谁"），不能当"权限凭证"（"我能干什么"）。

### 8.2 坑 2：401 / 403 / 404 混淆导致体验错乱

**现象**：

最初 `http.ts` 对所有 `!response.ok` 统一 `throw new Error(detail)`，导致：

- 401（身份失效）：页面只显示"HTTP 401"错误，不跳登录，用户困惑
- 403（权限不足）：用户体验和 404 一样，不知道是"没权限"还是"资源不存在"
- 404（对象越权伪装，G3-B D6）：engineer 越权访问他人任务返回 404，如果前端误判为 403，会显示"权限不足"——但实际后端是"任务不存在"（防枚举）

**根因**：

`http.ts` 未区分 HTTP 状态码，所有错误走同一条 `throw new Error` 路径。

**修复**：

`http.ts` 的 `!response.ok` 分支区分 3 类：

```typescript
if (response.status === 401) {
  handleUnauthorized();                     // 清 localStorage.user_id + 跳 /login
  throw new UnauthorizedError(detail);
}
if (response.status === 403) {
  throw new PermissionDeniedError(detail);  // 页面级 catch，显示"权限不足"
}

throw new Error(detail);                    // 404 / 422 / 500 保持原样
```

**教训**：

1. 401 / 403 / 404 语义完全不同：401 = 身份失效（清身份跳登录）；403 = 权限不足（页面提示）；404 = 资源不存在（业务处理，不特殊跳转）。
2. G3-B D6 决策"engineer 越权返回 404"要求前端不能把 404 误判为 403——否则会把"防枚举"变成"明示权限不足"，反而暴露资源存在性。
3. `handleUnauthorized` 用 `window.location.href = "/login"` 而非 Vue Router——因为 `http.ts` 在 router 之前加载，避免循环依赖。

### 8.3 坑 3：`router.beforeEach` 并发 bootstrap 导致状态错乱

**现象**：

用户快速点击多个路由链接时，`router.beforeEach` 会并发触发多次 `bootstrap()`（每次都 `GET /auth/me`），导致：

- 多次 API 请求浪费
- 后续响应覆盖前面响应，状态可能错乱
- 401 场景下 `handleUnauthorized` 可能被多次触发

**根因**：

`bootstrap()` 没有请求锁，每次调用都发新请求。

**修复**：

`stores/auth.ts` 的 `bootstrap()` 用 `bootstrapPromise` 缓存：

```typescript
let bootstrapPromise: Promise<AuthUser | null> | null = null;

async function bootstrap(): Promise<AuthUser | null> {
  if (bootstrapped.value) return user.value;
  if (bootstrapPromise) return bootstrapPromise;   // ← 请求锁
  bootstrapPromise = (async () => { ... })();
  return bootstrapPromise;
}
```

**教训**：

1. 任何"启动时拉取"的异步操作都要加请求锁，避免并发导航触发多次请求。
2. `bootstrapPromise` 必须是模块级变量（不是 store 的 ref），因为它不需要响应式。

### 8.4 坑 4：`http.ts` 的 401 handler 用 `window.location.href` 而非 `router.push`

**现象**：

最初想用 `router.push("/login")` 跳登录，但 `http.ts` 是纯 API 封装，不 import Vue Router（否则循环依赖）。而且如果在 `router.beforeEach` 里触发 401（`bootstrap()` 调 `me()` 返回 401），`router.push` 会和当前导航冲突。

**修复**：

`http.ts` 的 `handleUnauthorized` 用 `window.location.href = "/login"`（整页刷新，不走 SPA 路由），并加 `window.location.pathname !== "/login"` 判断防重复跳转。

**教训**：

1. 底层工具（`http.ts`）不依赖上层（Vue Router），用 `window.location` 是合理的。
2. 401 是"身份失效"的强信号，整页刷新更干净（清空所有内存状态）。

### 8.5 坑 5：BUG-G3A-001 误判（后端未注册 vs 测试路径笔误）

**现象**：

G3-A 手工测试报告报 `GET /api/knowledge-docs` → 404，初步判断为"后端未挂载 `APIRouter`"，定为 P1 阻塞。

**复测**：

```text
# 后端路由表（应用内打印 app.routes）
/app/api/v1/knowledge-docs
/app/api/v1/knowledge-docs/upload
/app/api/v1/knowledge-gaps
/app/api/v1/knowledge-gaps/{gap_id}
```

后端路由完整注册。以 `admin` 身份（`X-User-Id: 3232`，DB 校验：`3232 | admin | admin | t`）直接请求 `GET /api/v1/knowledge-docs` → **200 + `total: 14`**（`items` 数组在终端输出被截断，`total` 为服务端真实总数）；前端页面内同样请求亦为 200。

**真实根因**：

测试报告请求路径写成 `/api/knowledge-docs`（少了 `/v1` 前缀）。前端页面路由是 `/knowledge-base`（不是 `/knowledge-docs`），直接访问 `/knowledge-docs` 会走前端 `/:pathMatch(.*)*` 的 404 页面（不是后端 404）。

**教训**：

1. 前后端路径要分清：前端路由 `/knowledge-base`，后端接口 `/api/v1/knowledge-docs`，两者不同。
2. 测试报告 404 时要先看请求路径：是 `/api/v1/xxx` 还是 `/api/xxx`？是前端路由还是后端接口？
3. P1 缺陷结论前必须复测：本次 404 报告直接判"后端未注册"是过度推断，实际是测试脚本笔误。
4. 分层验证顺序：前端页面路由 → 后端接口路由 → 鉴权拦截，三层分开确认，再登记缺陷类型。

### 8.6 修复后验证结果

| 验证项 | 结果 |
|---|---|
| `npm run type-check` | ✅ 0 errors |
| `npm run build` | ✅ 1996 modules transformed，`LoginPage-5SimDYtX.js` / `ForbiddenPage-DgXKw812.js` 正确打包 |
| 8 个浏览器验收场景 | ✅ 全部通过（详见 `G4_RBAC_手工测试报告.md`） |
| 对象级越权（详情 / 缺陷 / 导出） | ✅ 统一 404 |
| BUG-G3A-001 | ✅ 撤销（测试报告笔误），后端路由与鉴权逻辑均正常 |

### 8.7 经验沉淀速查

| # | 坑 | 一句话结论 |
|---|---|---|
| 1 | `localStorage` 存 `role` | `role` 不写 `localStorage`，只存 Pinia 内存，每次从 `GET /auth/me` 刷新 |
| 2 | 401 / 403 / 404 混淆 | 401 清身份跳登录；403 页面提示；404 保持原样（不误判为 403） |
| 3 | `bootstrap()` 并发请求 | 用 `bootstrapPromise` 模块级缓存（请求锁） |
| 4 | `http.ts` 用 `window.location.href` | 底层工具不依赖 Vue Router，用 `window.location` 整页刷新 |
| 5 | BUG-G3A-001 误判 | 测试报告路径笔误（少 `/v1`）；前后端路径要分清；P1 结论前必须复测 |

---

## 9. S3-G5-KnowledgeGap：知识缺口生成链路补实现（2026-11-06）

> **状态**：🟢 基本完成（Commit 1、2、3 已完成，E2E 链路打通，631 passed；仅剩 T-05 状态码全量扫描与 G5 最终文档收尾）
> **核心发现**：Sprint 1/2 的 `knowledge_gap` / `gap_feedback_link` 表**没有任何生产代码写入路径**，DB 里 26 条 `knowledge_gap` 是 Sprint 2 测试 / demo 手工造的测试数据。G5 必须补实现。
> **Commit 1 结果**：`create_gap_from_feedback` 函数 + 统一 ID 生成器 `make_prefixed_id` 落地，全量 626 passed 零破坏。
> **Commit 2 结果**：`_dispatch_evolution` 接线 + `FeedbackOut`/`FeedbackResponse`/路由三层 DTO 透传，API 返回 `GAP-20261106-5708C0`，全量 626 passed。
> **Commit 3 结果**：E2E 生成链路用例 + G4 遗留 4 个补测落地，全量 **631 passed、0 failed**（跨 Session 污染已在 9.5c 解决）。

### 9.1 坑 1：`_make_candidate_id` 的 U+2011 技术债源头（G5 顺带修复）

**现象**：

G3-C 手测 `POST /rule-candidates/RC‑20261104‑B46BB4/status` 返回 404，PowerShell 显示 `RCâ20261104âB46BB4`（U+2011 渲染成 `â`）；后端 DB 存储的 `candidate_id` 是 `RC‑20261104‑B46BB4`（U+2011），前端用 `encodeURIComponent` 兜底。

**根因**：

`rule_evolution_service._make_candidate_id` 的实现：

```python
def _make_candidate_id() -> str:
    ts = datetime.now().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"RC‑{ts}‑{suffix}"   # ← `‑` 是 U+2011，不是 ASCII `-` (U+002D)
```

`f"RC‑{ts}‑{suffix}"` 里的 `‑` 是 U+2011 非断连字符，不是 ASCII `-`。这是 Sprint 2 遗留的 P1 技术债（Sprint3 附录 D 第 1 项），原计划下一阶段修，G5 提前修复。

**修复（G5 Commit 1）**：

改为调 `make_prefixed_id("RC", random_len=6)`（ASCII `-`）：

```python
def _make_candidate_id() -> str:
    from app.core.ids import make_prefixed_id
    return make_prefixed_id("RC", random_len=6)
```

**修复效果**：

```text
新生成：RC-20261106-5B167C    （ASCII `-`，URL 安全）
存量：  RC‑20261027‑F05ED4    （U+2011，不迁移，前端 encodeURIComponent 兜底）
```

**教训**：

1. **"看起来像 `-`"的字符不一定是 ASCII `-`**：U+2011（非断连字符）、U+2010（连字符）、U+2013（短破折号）、U+2014（长破折号）在视觉上都近似 `-`，但 URL 编码 / 数据库存储完全不同。写 ID 生成器必须显式用 `"-"`（ASCII U+002D）。
2. **技术债要在"顺带"时修**：G5 本来只加 `create_gap_from_feedback`，但抽公共 `make_prefixed_id` 时顺带把 `_make_candidate_id` 一起改了，成本极低（3 行改动），避免 Sprint 4 专门开一个 commit 修。
3. **存量数据不迁移**：存量候选 ID（U+2011）不迁移，前端 `encodeURIComponent` 兜底 + 后端 `get_candidate` 的 URL 解码能处理。迁移成本 > 收益。

### 9.2 坑 2：`knowledge_gap` 表 26 条 demo 数据无生产写入路径

**现象**：

G3-A 手测 `GET /api/v1/knowledge-gaps` 返回 `total: 26`，但全项目 grep `KnowledgeGap(` / `GapFeedbackLink(` 只命中 `models.py` 的 ORM 定义，没有 `db.add(...)` 写入。

**根因**：

Sprint 1/2 的 `knowledge_gap` 表只是模型定义 + 读取接口（`list_gaps` / `get_gap` / `update_gap_status`），没有从 `feedback_item` 生成 `knowledge_gap` 的写入路径。`rule_evolution_service.export_knowledge_gap_backlog` 是**导出**（`SELECT feedback_item WHERE feedback_type='knowledge_gap'` 导 JSON），不写 `knowledge_gap` 表。

**修复（G5 Commit 1 + 后续 Commit 2）**：

- Commit 1：加 `knowledge_gap_service.create_gap_from_feedback` 函数
- Commit 2：`feedback_service._dispatch_evolution` 加 `knowledge_gap` 分支，调 `create_gap_from_feedback`

**教训**：

1. **"表存在 + 有数据"不代表"有写入路径"**：26 条 demo 数据是 Sprint 2 测试 / demo 手工 `db.add(KnowledgeGap(...))` 造的，不是真实业务链路产生。必须 grep 写入点（`db.add(KnowledgeGap(`）才能确认。
2. **导出 ≠ 生成**：`export_knowledge_gap_backlog` 从 `feedback_item` 表 SELECT 出 `feedback_type='knowledge_gap'` 的记录导 JSON，不写 `knowledge_gap` 表。导出和生成是两条独立链路。
3. Sprint3 §10 的判定规则："反馈生成链路是知识反馈闭环的关键环节，若判定为第一阶段必要功能则必须补齐最小实现"——G5 判定为"补实现"，不是"补测试"。

### 9.3 坑 3：`FeedbackCreate` 无 `comment` 字段（D13 降级链修正）

**现象**：

D13 原设计降级链 `expert_suggestion → comment → suggestion_diff 摘要 → 兜底`，但 `FeedbackCreate` Pydantic 模型没有 `comment` 字段：

```python
class FeedbackCreate(BaseModel):
    review_result_id: int
    review_defect_id: int
    feedback_type: FeedbackType
    expert_suggestion: str | None = None
    suggestion_adopted: str | None = None
    suggestion_diff: list[SuggestionDiff] | None = None
    # ← 没有 comment 字段
```

**根因**：

`FeedbackItem.comment` 在 DB 里存在（`Mapped[Optional[str]]`），但 `submit_feedback` 的 Pydantic 入参不传 `comment`，所以通过 API 提交的 feedback 的 `comment` 是 NULL。`comment` 只有测试 / demo 里手动 `db.add(FeedbackItem(comment="[SEED] 测试"))` 才有值。

**修复**：

D13 降级链去掉 `comment` 级，只剩 3 级：

```python
title_raw = (
    (fb.expert_suggestion or "").strip()
    or _summarize_suggestion_diff(fb.suggestion_diff_json)
    or f"知识缺口{(' ' + final_case_id) if final_case_id else ''}"
)
```

**教训**：

1. **Pydantic 模型字段 ≠ ORM 模型字段**：`FeedbackItem.comment`（ORM）存在，但 `FeedbackCreate.comment`（Pydantic）不存在。写入路径的字段来源只能是 Pydantic 入参，不能假设"DB 里有就意味着 API 能传"。
2. **降级链的每一级都要能实际到达**：`comment` 级在 API 路径下永远不可达（`comment` 永远是 NULL），保留它只会误导读者。

### 9.4 坑 4：`feedback_item` 物理删除会导致 gap 失去来源追溯

**现象（D18 补充说明）**：

- `KnowledgeGap.source_feedback_id` 是 `ON DELETE SET NULL`（`feedback_item` 删了 gap 保留、source 清空）
- `GapFeedbackLink.feedback_id` 是 `ON DELETE CASCADE`（`feedback_item` 删了 link 删、gap 保留）

语义一致（gap 保留），但 `feedback_item` 物理删除会导致 gap 失去来源追溯：

- `source_feedback_id` → NULL
- `gap_feedback_link` 记录 → 被 CASCADE 删除
- 结果：gap 成孤儿，不知道从哪条 feedback 生成

**修复（文档约束，非代码改动）**：

业务层禁止物理删除 `feedback_item`，其"删除"语义应为软删除（但目前 `FeedbackItem` 没有 `is_deleted` 列，也没有软删除接口）。G5 不碰这个，D18 文档写明约束，G6 或下一阶段考虑加软删除。

**教训**：

1. **同一个删除事件的不同外键策略要对齐**：`source_feedback_id SET NULL` + `gap_feedback_link CASCADE` 在"gap 是否保留"上语义一致，但**追溯能力不一致**（link 没了）。如果追溯重要，`gap_feedback_link` 也应 `SET NULL`（但主键是 `(gap_id, feedback_id)`，`feedback_id` 不能 NULL，所以只能 CASCADE）。
2. **审计类表禁止物理删除**：`feedback_item` 是审计 / 追溯表，业务上不应物理删除。G5 记录约束，G6 或下一阶段加软删除。

### 9.5 Commit 1 验证结果

| 验证项 | 结果 |
|---|---|
| `make_prefixed_id("GAP", 6)` | ✅ `GAP-20261106-E7B45A`（ASCII `-`） |
| `make_prefixed_id("RC", 6)` | ✅ `RC-20261106-5B167C`（ASCII `-`） |
| 全量非 integration | ✅ 626 passed, 2 skipped, 24 deselected, 0 failed |
| Commit 2 后回归 | ✅ 626 passed、0 failed（见 9.5b 结果） |
| Commit 3 后全量 | ✅ 631 passed, 2 skipped, 24 deselected, 0 failed（见 9.5c 结果） |

### 9.5b 坑 5：API 层 DTO 漏传字段导致 `knowledge_gap_ref` 为空

**现象**：

`POST /api/v1/feedbacks` 提交 `knowledge_gap` 类型反馈后，DB 中 `knowledge_gap` 和 `gap_feedback_link` 均成功写入，但 API 响应中的 `knowledge_gap_ref` 字段始终为空。

**排查思路（逐步缩小范围）**：

1. **查 DB**：确认 `gap_feedback_link` 表有记录（`feedback_id=2528` 对应 `gap_id=188`），数据层无问题。
2. **查 Service 层**：在 `_build_out` 增加调试日志，确认 `db_feedback.id` 是整数（2528/2529），且独立 Session 手工 SQL 能查出 `GAP-...`，说明查询逻辑正确。
3. **最终定位**：检查 API 路由 `feedbacks.py` 发现，`FeedbackResponse` 组装时**根本没有传** `knowledge_gap_ref`，且 `FeedbackResponse` schema **也没有定义该字段**。

**根因**：

`FeedbackOut`（Service DTO）有 `knowledge_gap_ref`，但 `FeedbackResponse`（API DTO）未同步新增字段，路由组装时也未传值。三层链路（Service → API Schema → 路由）缺失两层。

**修复**：

1. `schemas.py` 的 `FeedbackResponse` 增加 `knowledge_gap_ref: str | None = None`。
2. `feedbacks.py` 的 `return FeedbackResponse(...)` 增加 `knowledge_gap_ref=out.knowledge_gap_ref`。

**结果**：

修复后 API 响应正常返回 `GAP-20261106-5708C0`，`knowledge_gap` 总数正确递增（32 → 33）。

**教训**：

1. **"Service 层有值"不等于"API 响应有值"**：手动验证必须看 `Invoke-RestMethod` 的最终响应，而不是 Service 函数的返回对象。
2. **DTO 字段必须逐层对齐**：新增字段必须同时检查 `Service DTO` → `API Schema` → `路由组装` 三处。
3. **全链路 grep 是终极大法**：遇到字段静默丢失，直接 `grep` 字段名，缺哪补哪。

### 9.5c 坑 6：E2E 测试中的真实 Graph 触发跨 Session 污染

**现象**：

G5 Commit 3 新增的 `test_engineer_own_task_details_200` 用 `client_as_engineer.post("/review/tasks")` 创建任务，走真实 Graph（Mock LLM 也要走 `task_service.create_task` → `run_review_graph` → `_persist_review_report`）。

全量回归时，`test_db_schema.py::test_trigger_rule_definition_updated_at_refresh` 挂：

```text
t1 = 2026-11-06 10:52:33.339680+00:00
t2 = 2026-11-06 10:52:17.510604+00:00
assert t2 > t1        # ← t2 比 t1 早 16 秒
```

**排查思路（逐步缩小范围）**：

1. **查 DB**：`SELECT * FROM rule_definition WHERE rule_id LIKE 'TRIG_RULE_%'` → **0 行**（测试数据已回滚）。
2. **单独跑用例**：`pytest tests/test_db_schema.py::test_trigger_rule_definition_updated_at_refresh -v` → **PASSED**。
3. **单独跑全文件**：`pytest tests/test_db_schema.py -v` → **69 passed**。
4. **全量跑**：`pytest -m "not integration"` → **1 failed**（`t2 < t1`）。
5. **定位污染源**：G5 Commit 3 的 `test_engineer_own_task_details_200` 走 `create_task`（真实 Graph），内部 `get_db_session()` 真实 commit，污染 `rule_definition` 的 `updated_at`。

**根因**：

- `test_db_schema.py` 的 `db_session` fixture 用 `@pytest.fixture()`（function 级），**外层事务回滚隔离**。
- 但 PostgreSQL 的事务快照隔离会导致：`db_session` 开始时的快照是旧的，而 `test_engineer_own_task_details_200` 在之前的测试里通过 `get_db_session()` **真实提交**了 `rule_definition` 的 `updated_at`，两相对比就出现 `t2 < t1`。
- **根本问题**：`create_task` 走真实 Graph 会写 DB（`review_result` / `review_defect` / 规则命中相关表），跨 Session 不可见，污染后续测试。

**修复**：

`test_engineer_own_task_details_200` 改为**直接构造 `ReviewResult`（`created_by=engineer.id`）+ `commit`**，不走 Graph：

```python
with get_db_session() as db:
    db.add(ReviewResult(
        task_id=fake_task_id,
        task_status="done",
        created_by=eng_id,
        review_output_json={"defects": [{"defect_id": "D1", ...}], "errors": []},
        evidence_coverage=0.8,
    ))
    db.commit()
```

测试后 delete 清理。

**结果**：

- G4 遗留补测：4 passed
- `test_db_schema.py`：69 passed
- 全量非 integration：**631 passed, 2 skipped, 24 deselected, 0 failed**

**教训**：

1. **测试中的 `create_task` 不要走真实 Graph**：Graph 内部 `get_db_session()` 真实 commit，会跨 Session 污染其它用事务回滚隔离的测试。
2. **测试数据「最小构造」**：直接 `db.add(ORMModel(...)) + commit`，比真实业务流程更可控。
3. **跨文件测试污染排查法**：① 查 DB 确认无残留 → ② 单独跑 PASS → ③ 全文件跑 PASS → ④ 全量跑 FAIL → 定位到「哪个测试真实提交了数据」。
4. **`db_session` 用事务回滚的局限**：PostgreSQL 事务快照隔离下，跨 Session 的数据变更可能导致同事务内的 `updated_at` 对比失败。

**同类问题预警**：

`test_engineer_own_task_details_200` 的修复模式（直接构造 DB 数据 + 清理）应推广到所有「需要真实任务 / 真实缺陷」的测试。`test_feedback_e2e.py` 的 `_run_task_and_get_defect_id` 走真实 Graph 是已知污染源，但它的测试在 `test_feedback_e2e.py` 内部有 `prepare_out_dir` 清理，暂未引发问题。

### 9.6 G5 后续 Commit（待做）

| # | Commit | 内容 | 状态 |
|---|---|---|---|
| 1 | `feat(s3-g5): add create_gap_from_feedback + unify ID generator` | 生成函数 + 统一 ID 生成器 | ✅ |
| 2 | `feat(s3-g5): wire knowledge_gap generation in feedback_service` | `_dispatch_evolution` 加 `knowledge_gap` 分支 + 三层 DTO 对齐（见 9.5b） | ✅ |
| 3 | `test(s3-g5): E2E test` | `test_feedback_e2e.py` 加场景（D20：`total_before + 1`）+ G4 遗留 4 个补测（见 9.5c） | ✅ |
| 4 | `test(s3-g5): G4 遗留 T-01~T-05` | 5 个负面用例 | ⏳ |
| 5 | `docs(s3-g5): G5 章节 + 完整回归` | 本文件补全 | ⏳ |

---

## 10. S3-G2-AuthBase：SQLAlchemy Session Autobegin 陷阱（G3 阶段验证）

> **状态**：⏳ 待验证（G3 阶段执行，G2 不触发）
> **规划依据**：[Sprint3_详细执行计划_V1.0.md §3.3](./Sprint3_详细执行计划_V1.0.md)

### 10.1 为什么 G2 未触发

G2 的 `get_current_user` 和 `update_user` / `soft_delete_user` 各自使用独立的 `with get_db_session() as db:`，不共享 Session。因此 admin_lock 和业务 Service 不冲突。

### 10.2 G3 阶段可能触发

G3 给业务路由挂鉴权时，同一请求内会同时触发 `get_current_user` 和业务 Service。此时：

- `get_current_user` 用 `with get_db_session() as db:` 查用户（autobegin 触发）
- 业务 Service 也 `with get_db_session() as db:` 查数据（另一个 Session）

如果 G3 把两者合并为同一个 Session，就会触发 autobegin 冲突。G3 编码时必须按 Sprint3 §3.3 的约束：

- **方案 A（推荐）**：Service 层不 `db.begin()`，直接复用 `get_db_session` 的事务。
- **方案 B**：最后管理员保护需要 READ COMMITTED，用独立 Session + `SET TRANSACTION ISOLATION LEVEL READ COMMITTED`。

### 10.3 验收测试用例（G3 阶段）

1. **正常路径**：GET /auth/me 鉴权 → PATCH /auth/users/{id} 降级非最后 admin，不抛 `InvalidRequestError`。
2. **受保护路径**：唯一 admin 降级 → 抛 `LastAdminProtectionError`，不抛 `InvalidRequestError`。
3. **session 复用**：先 `get_current_user`，再进业务 Service，断言 Service 能正确判断 Session 事务状态。

---

## 11. 附：版本与适用范围

- 前端：Vue 3 + Vite 5 + TypeScript
- 后端：FastAPI + Pydantic + uv
- 基础设施：Docker Compose + PostgreSQL 15
- 适用范围：Sprint 3 G2–G6 开发联调阶段
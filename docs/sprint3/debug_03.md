# Sprint 3 开发与调试问题记录（Debugging Notes）

> **用途**：沉淀 Sprint 3 开发过程中遇到的典型 Bug、环境问题及排查思路，供后续开发、联调及运维部署参考。
> **归档日期**：2026-11-03
> **最后更新**：2026-11-03（G2 Migration 与 Auth 底座开发）

---

## 目录

1. [S3-G2-Migration：多 Head 问题与增量 SQL 生成](#1-s3-g2-migration多-head-问题与增量-sql-生成2026-11-03)
2. [S3-G2-PreCheck：脚本路径与容器挂载陷阱](#2-s3-g2-precheck脚本路径与容器挂载陷阱2026-11-03)
3. [S3-G2-TestIsolation：测试隔离与并发断言修正](#3-s3-g2-testisolation测试隔离与并发断言修正2026-11-03)
4. [S3-G2-AuthBase：SQLAlchemy Session Autobegin 陷阱](#4-s3-g2-authbasesqlalchemy-session-autobegin-陷阱g3-阶段验证)
5. [附：版本与适用范围](#5-附版本与适用范围)

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

## 4. S3-G2-AuthBase：SQLAlchemy Session Autobegin 陷阱（G3 阶段验证）

> **状态**：⏳ 待验证（G3 阶段执行，G2 不触发）
> **规划依据**：[Sprint3_详细执行计划_V1.0.md §3.3](./Sprint3_详细执行计划_V1.0.md)

### 4.1 为什么 G2 未触发

G2 的 `get_current_user` 和 `update_user` / `soft_delete_user` 各自使用独立的 `with get_db_session() as db:`，不共享 Session。因此 admin_lock 和业务 Service 不冲突。

### 4.2 G3 阶段可能触发

G3 给业务路由挂鉴权时，同一请求内会同时触发 `get_current_user` 和业务 Service。此时：

- `get_current_user` 用 `with get_db_session() as db:` 查用户（autobegin 触发）
- 业务 Service 也 `with get_db_session() as db:` 查数据（另一个 Session）

如果 G3 把两者合并为同一个 Session，就会触发 autobegin 冲突。G3 编码时必须按 Sprint3 §3.3 的约束：

- **方案 A（推荐）**：Service 层不 `db.begin()`，直接复用 `get_db_session` 的事务。
- **方案 B**：最后管理员保护需要 READ COMMITTED，用独立 Session + `SET TRANSACTION ISOLATION LEVEL READ COMMITTED`。

### 4.3 验收测试用例（G3 阶段）

1. **正常路径**：GET /auth/me 鉴权 → PATCH /auth/users/{id} 降级非最后 admin，不抛 `InvalidRequestError`。
2. **受保护路径**：唯一 admin 降级 → 抛 `LastAdminProtectionError`，不抛 `InvalidRequestError`。
3. **session 复用**：先 `get_current_user`，再进业务 Service，断言 Service 能正确判断 Session 事务状态。

---

## 5. 附：版本与适用范围

- 前端：Vue 3 + Vite 5 + TypeScript
- 后端：FastAPI + Pydantic + uv
- 基础设施：Docker Compose + PostgreSQL 15
- 适用范围：Sprint 3 G2–G6 开发联调阶段
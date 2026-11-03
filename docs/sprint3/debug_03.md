# Sprint 3 开发与调试问题记录（Debugging Notes）

> **用途**：沉淀 Sprint 3 开发过程中遇到的典型 Bug、环境问题及排查思路，供后续开发、联调及运维部署参考。
> **归档日期**：2026-11-03
> **最后更新**：2026-11-03（G2 Migration 与 Auth 底座开发）

---

## 目录

1. [S3-G2-Migration：多 Head 问题与增量 SQL 生成](#1-s3-g2-migration多-head-问题与增量-sql-生成)
2. [S3-G2-PreCheck：脚本路径与容器挂载陷阱](#2-s3-g2-precheck脚本路径与容器挂载陷阱)
3. [S3-G2-AuthBase：SQLAlchemy Session Autobegin 陷阱（规划中）](#3-s3-g2-authbasesqlalchemy-session-autobegin-陷阱规划中)
4. [附：版本与适用范围](#4-附版本与适用范围)

---

## 1. S3-G2-Migration：多 Head 问题与增量 SQL 生成（2026-11-03）

> 状态：✅ 已修复
> 核心难点：新增 Migration 后 Alembic 出现双 Head（`s2_f07_case_benchmark` 和 `s2j_g2_roles` 并行），导致 `alembic upgrade head` 直接报错。

### 一、坑点与修复全过程

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

### 二、核心认知与总结

1. **新增 Migration 必须显式确认 `down_revision`**：每次新增 Migration 前，先跑 `alembic heads` 看当前 head。把 `down_revision` 设成当前 head，形成线性链。忘了改会导致分叉，之后 `upgrade head` 直接报错。
2. **`--sql` 模式必须指定起点 revision**：`alembic upgrade head --sql` 从 `<base>` 开始，不是从当前库版本开始。要生成增量 SQL，用 `alembic upgrade <current>:head --sql`。

---

## 2. S3-G2-PreCheck：脚本路径与容器挂载陷阱（2026-11-03）

> 状态：✅ 已绕过
> 核心难点：为 G2 前期核查写的 `g2_pre_check.py` 脚本，因容器内路径与宿主机路径不一致，导致核查结果全部误判。

### 一、坑点与修复全过程

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

### 二、核心认知与总结

1. **容器内视角 vs 宿主机视角**：写脚本时，路径要按容器内挂载点来算（`/app/app/api/deps.py`），而不是宿主机路径（`backend/app/api/deps.py`）。
2. **写文件前先确认挂载点**：不要在未挂载的目录（如 `/docs`、`/tmp` 之类）写文件，否则文件只存在于容器内，`docker compose restart` 就丢。
3. **优先用已挂载目录做输出**：`/scripts/`、`/data/`、`/out/` 都已挂载，直接写这些目录就不会丢文件。

---

## 3. S3-G2-AuthBase：SQLAlchemy Session Autobegin 陷阱（规划中）

> 状态：⏳ 待开发时验证
> 规划依据：`Sprint3_详细执行计划_V1.0.md` §3.3

### 一、预期坑点

`Depends(get_current_user)` 和业务 Service 都用 `with get_db_session() as db:`。

`get_current_user` 执行时会触发 autobegin（SQLAlchemy 2.x 的默认行为），导致 Session 内部已经隐式开启了事务。

之后 Service 层若调用 `with db.begin():`，会抛 `InvalidRequestError: A transaction is already begun`。

### 二、预期修复方向

**方案 A（推荐）**：Service 层不再 `db.begin()`，直接复用 `get_db_session` 提供的事务：

```python
# ✅ 正确：Service 直接用 db
def update_user_role(db: Session, ...):
    # db 已经 autobegin，直接用
    target = db.query(User).filter(...).first()
    ...
    db.flush()  # 不用 commit，get_db_session 统一 commit
```

**方案 B**：独立 Session + 显式隔离级别（用于最后管理员保护需要 READ COMMITTED 的场景）：

```python
# 独立 Session，不依赖请求 Session
from app.persistence.db import SessionLocal
with SessionLocal() as db:
    db.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": LOCK_KEY})
    ...
    db.commit()
```

**注意**：方案 B 的独立 Session 必须与 `get_current_user` 的请求 Session 完全隔离，且在同一个独立 Session 内保持「锁 → 查询 → 更新」顺序，不允许拆事务。

### 三、验收测试用例

开发完成后必须跑通 3 个用例（真实 PostgreSQL）：

1. **正常路径**：`GET /auth/me` 鉴权通过后，`PATCH /auth/users/{id}` 降级非最后 admin，断言返回 200、DB role 已更新，不抛 `InvalidRequestError`。
2. **受保护路径**：`GET /auth/me` 鉴权通过后，PATCH 试图降级唯一启用 admin，断言返回 `LastAdminProtectionError`，不抛 `InvalidRequestError`，DB 状态未变更。
3. **session 复用**：同一请求内先触发 `get_current_user`，再进入管理员变更 Service，断言 Service 能正确判断 Session 事务状态、不复用旧快照、不重复 `db.begin()`。

---

## 4. 附：版本与适用范围

- 前端：Vue 3 + Vite 5 + TypeScript
- 后端：FastAPI + Pydantic + uv
- 基础设施：Docker Compose + PostgreSQL 15
- 适用范围：Sprint 3 G2–G6 开发联调阶段
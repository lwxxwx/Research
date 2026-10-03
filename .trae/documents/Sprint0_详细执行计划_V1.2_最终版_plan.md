# Sprint 0 开发落地规划
## Docker + uv 可复现开发环境 V1.3（方案A 并列挂载）
---
## 0. 文档定位
本文件是基于「AI 原理图评审 MVP」项目架构设计的 **Sprint 0 顺序落地指南**。
**核心变更**：从原来的"宿主机 Python venv + pip"迁移为 **"Docker 全运行环境 + uv Python 依赖管理"**。

**V1.2 修订说明**：
1. 修复 P0 级 Docker Healthcheck（改用 Python urllib，移除 curl 依赖）。
2. 完整补齐 Phase B-L 业务阶段详情，确保"执行型文档"属性。
3. 补齐 Phase L 风险管理矩阵（Risk -> Impact -> Mitigation -> Acceptance）。
4. 明确 5 项 Business Freeze 具体内容与 Environment Freeze 边界。
5. 统一 Docker / Compose 执行指令规范，明确 Frontend Placeholder 验收边界。
6. 工程规范调整：全部 Python 测试脚本统一放置于 `backend/tests`，scripts 目录仅保留运维脚本。

**V1.3 修订说明（挂载策略定稿）**：
1. 挂载策略从「5 条嵌套挂载（`/app/xxx`）」迁移为「5 条并列挂载（`/xxx`）」——**方案A**。
2. 根因归档：Docker Desktop for Windows 在「父 target `/app` + 子 target `/app/xxx`」嵌套 bind mount 场景下，会按「父 source + target 相对路径」在宿主机 `backend/` 下预创建 `data/scripts/infra/out` 空目录，污染源码目录。
3. 容器内路径映射更新：
   - `/app`     <- `backend/`
   - `/data`    <- `data/`
   - `/scripts` <- `scripts/`
   - `/infra`   <- `infra/`
   - `/out`     <- `out/`
4. `PYTHONPATH` 更新为 `/app:/`（追加根目录 `/`，使 `import scripts.xxx` 能命中挂在 `/scripts` 的脚本目录）。
5. `DATA_DIR` 更新为 `/data`。
6. 脚本调用约定：注入 `$env:PROJECT_ROOT`、`$f1`/`$f2` 使用绝对路径、所有 `docker compose` 调用用 `cmd /c` 包装、启动前预创建 bind mount 源目录。
7. 测试文件中的 U+2011 非断行连字符统一替换为标准 ASCII 连字符（U+002D）。
---
## 1. 上游基准与本次架构调整
### 1.1 上游基准
- [第一阶段工程实施方案 V1.2 增强版](file:///e:/projects/Schematic_Design_Review/.trae/documents/%E7%AC%AC%E4%B8%80%E9%98%B6%E6%AE%B5%E5%B7%A5%E7%A8%8B%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88_V1.2_%E5%A2%9E%E5%BC%BA%E7%89%88_plan.md)
- [项目实施方案_修订版_MVP闭环验证](file:///e:/projects/Schematic_Design_Review/.trae/documents/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88_%E4%BF%AE%E8%AE%A2%E7%89%88_MVP%E9%97%AD%E7%8E%AF%E9%AA%8C%E8%AF%81_plan.md)

### 1.2 原规划 → 新规划迁移说明
| 原规划特性 | 新规划 (Docker+uv) | 处理方式 |
| :--- | :--- | :--- |
| **宿主机 Python** | **Backend Container (Python 3.11.x)** | 删除宿主机依赖 |
| **venv / pip** | **uv (pyproject.toml + uv.lock)** | 替换为现代依赖管理 |
| **Venv 位置** | **容器内 /opt/venv** | 隔离源码挂载冲突 |
| **Docker Compose** | **Dev/Prod 分离** | 明确职责边界 |
| **Healthcheck** | **Python urllib 实现** | 移除 curl，提高镜像纯净度 |
| **Bootstrap** | **全自动化 Healthz 验证** | 强化环境就绪检查 |
| **`data/scripts/infra/out` 挂 `/app/xxx`（V1.3 前）** | **挂 `/data`、`/scripts`、`/infra`、`/out`（与 `/app` 并列）** | **消除嵌套 bind mount，避免 Docker Desktop 在 `backend/` 下误建空目录** |
---
## 2. Sprint 0 最终目标
1. **环境闭环**：开发者只需 Git + Docker Desktop，即可一键启动后端、数据库及前端占位。
2. **数据闭环**：完成 Buck IR 验证，Case001 专家校准，3 条 Power 规则在容器内运行通过。
3. **验收闭环**：在容器内执行 Rule Only Benchmark，建立规则引擎基线，评估指标，完成 Sprint 0 Demo。
4. **架构冻结**：完成业务 Schema 与开发环境双重冻结。
5. **挂载策略定稿**：完成方案A（并列挂载）落地，宿主机 `backend/` 下不再被 Docker Desktop 预创建 `data/scripts/infra/out` 空目录。
---
## 3. 开发环境架构
```text
Developer Host (Windows/WSL2)
├── Git & Trae/VS Code
└── Docker Desktop
         │
         ▼
    Docker Compose
         │
         ├── backend (Service: backend)
         │     ├── OS: Debian Slim (Linux)
         │     ├── Runtime: Python 3.11.x (Debian Slim) + uv (v0.12.6)
         │     │ └── 当前验证版本:Python 3.11.16 (python:3.11-slim 镜像获取)
         │     ├── Venv Path: /opt/venv (隔离挂载)
         │     └── Mount (Dev 模式，方案A 并列挂载):
         │           ./backend  -> /app        （源码）
         │           ./data     -> /data       （数据资产）
         │           ./scripts  -> /scripts    （脚本）
         │           ./infra    -> /infra      （基础设施）
         │           ./out      -> /out        （产物）
         │
         ├── postgres (Service: postgres)
         │     ├── Image: pgvector/pgvector:pg15
         │     └── Volume: pgdata (Named Volume)
         │
         └── frontend (Service: frontend - Placeholder)
               └── 边界说明：Frontend Placeholder 仅作为占位，不属于 Sprint 0 核心验收阻塞项。
```
---
## 4. 环境职责边界
- **宿主机**：Git、IDE、浏览器、Docker Desktop。**禁止在宿主机安装 Python/venv/pip/Postgres**。
- **Backend 容器**：所有 Python 逻辑（FastAPI、uv、pytest、ruff、benchmark）。
- **Postgres 容器**：持久化数据与向量扩展。
---
## 5. Repository Structure
```text
Schematic_Design_Review/
├── backend/                # 后端源码
│   ├── app/                # 业务逻辑
│   ├── tests/              # Python测试、测试CLI入口脚本，需存在__init__.py
│   ├── pyproject.toml      # uv 依赖定义
│   ├── uv.lock             # 锁定文件
│   └── Dockerfile          # 多阶段构建
├── infra/
│   └── docker/             # Docker 配置
│       ├── docker-compose.yml       # 基础配置
│       ├── docker-compose.dev.yml   # 开发挂载 (Dev Only)
│       ├── docker-compose.prod.yml  # 生产优化 (Prod Only)
│       └── initdb/                  # 数据库初始化脚本
├── scripts/                # 自动化脚本（运维 + CLI 入口）
│   ├── bootstrap.ps1       # 环境初始化（首次使用）
│   ├── dev.ps1             # 日常 start/stop/restart/logs/status
│   ├── demo_sprint0.ps1    # Sprint 0 演示
│   ├── demo_feedback_payload.py   # Demo 业务载荷脚本
│   └── init_db.py          # DB 初始化脚本
├── data/                   # 数据资产（挂载到 /data）
├── docs/                   # 项目文档
└── .env                    # 环境变量 (宿主机生成)
```
---
## 6. Phase A · Docker + uv 环境落地
### A1. Git / Branch workflow & 根文件
- **目标**：确立开发规范，初始化根目录文件。
- **步骤**：
    1. 确立分支：`main` (发布), `develop` (集成), `sprint0/base` (基准)。
    2. 创建 `.gitignore`, `.dockerignore`, `.env.example`, `README.md`, `CONTRIBUTING.md`。
- **验证**：`git branch` 确认分支结构；文件内容符合规划。

**A1.x 实际分支布局（V1.3 修订）**：
```text
1b4b1f5 (sprint0/base)               ← 停在 Phase I-J 基线
│
├── sprint0/base                  ← 主基线
│
├── feature/schema-b-symlink      ← B 方案（软链），冻结备选
│
└── feature/schema-a-flat-mounts  ← A 方案（并列挂载），当前开发主线
```

### A2. 迁移依赖到 pyproject.toml
- **目标**：使用 uv 管理依赖，锁定版本。
- **步骤**：将原有 `requirements.txt` 迁移至 `backend/pyproject.toml`，使用 `uv lock` 生成 `uv.lock`。

### A3. 编写工程级 Backend Dockerfile
- **目标**：固定环境版本，实现轻量化 Healthcheck，确保依赖在构建阶段固化。
- **修改文件**：`backend/Dockerfile`
- **核心实现**：
    1. 使用固定版本：`FROM ghcr.io/astral-sh/uv:0.12.6 AS uv_bin` 和 `FROM python:3.11-slim` (当前验证版本:Python 3.11.16)。
        - 使用 `python:3.11-slim` 可自动获取最新的 3.11.x 安全补丁版本。
        - 如需锁定具体版本，可使用 `python:3.11.16-slim`（已验证可用）。
    2. 构建阶段利用 `pyproject.toml` + `uv.lock` 执行 `uv sync --frozen`。
    3. 将 Python 虚拟环境安装在 `/opt/venv`，确保依赖在镜像构建阶段完成，而非启动后安装。
    4. 环境变量配置：`ENV UV_PROJECT_ENVIRONMENT=/opt/venv`。
    5. Healthcheck 实现：
       ```dockerfile
       HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
         CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/healthz')" || exit 1
      ```
### A4. 重新设计 Docker Compose (Dev/Prod 分离)

- **目标**：明确开发与生产边界，统一使用 named volume 持久化数据。
- **配置要点**：
  - **postgres**: 使用 named volume `pgdata` 进行持久化（`volumes: - pgdata:/var/lib/postgresql/data`）。
  - **docker-compose.dev.yml：方案 A 并列挂载（V1.3 定稿）**：
```docker-compose
volumes:
  - ${PROJECT_ROOT}/backend:/app        # 源码
  - ${PROJECT_ROOT}/data:/data          # 数据资产
  - ${PROJECT_ROOT}/scripts:/scripts    # 脚本
  - ${PROJECT_ROOT}/infra:/infra        # 基础设施
  - ${PROJECT_ROOT}/out:/out            # 产物
environment:
  - PYTHONPATH=/app:/                   # 追加根目录 /，使 import scripts.xxx 能命中 /scripts
  - DATA_DIR=/data
```
  - **根因说明**：原 5 条嵌套挂载（`data:/app/data` 等）在 Docker Desktop for Windows 上会触发「父 target `/app` + 子 target `/app/xxx`」的推导，按「父 source + target 相对路径」在宿主机 `backend/` 下预创建 `data/scripts/infra/out` 空目录。方案 A 把子挂载提升为并列挂载，消除父子 target 关系。
  - `docker-compose.prod.yml`: 不挂载源码，使用 Immutable Image，禁用 `--reload`。
  - `docker-compose.tools.yml`: **必须与基座 `docker-compose.yml` 叠加使用**，不可单独启动。
    - 原因：`pgadmin` 引用逻辑网络名 `backend_net`，其实际名称由基座 `${NETWORK_SUFFIX}` 决定；单独运行 tools.yml 会因网络未定义而报错。
    - 正确启动示例：
      ```powershell
      docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.tools.yml" --profile dev-tools up -d
      ```
### A5. 强化一键式 Bootstrap

- **目标**：环境就绪自动化验证。
- **脚本逻辑**：
  1. 预检 Docker Engine。
  2. 复制 `.env.example` -> `.env`。
  3. **脚本调用约定（V1.3 修订）**：
     - 注入 `$env:PROJECT_ROOT = $projectRoot`，供 `docker-compose.dev.yml` 的 `${PROJECT_ROOT}` 使用；
     - `$f1` / `$f2` 使用**基于项目根的绝对路径**（`Join-Path $projectRoot "infra/docker/..."`）；
     - **不使用 `--project-directory`**（PowerShell 5.1 下与 `-f` 组合会丢 `-f`）；
     - 所有 `docker compose` 调用使用 **`cmd /c` 包装**（规避 PowerShell 5.1 stderr 污染）；
     - 启动前**预创建 bind mount 源目录**（`backend/data/scripts/infra/out` + `out/.gitkeep`）。
  4. 执行启动命令（由 `dev.ps1` / `bootstrap.ps1` 内部封装）：
```powershell
# 等价于（在 dev.ps1 内部）：
cmd /c "docker compose -f `"$f1`" -f `"$f2`" up -d --build"
```
  5. 轮询检查 `postgres` 与 `backend` 的 Health 状态。
  6. 调用 `healthz` 接口，输出 `PASS` 及访问地址。
---
## 7. Phase B · Backend Framework
- **目标**：建立可扩展的 FastAPI 基础骨架。
- **实现步骤**：
    1. `app/core/config.py`: 实现 Pydantic Settings。
       - **配置来源优先级（V1.3 澄清）**：容器内**以 compose `environment` 注入为准**；`.env` 仅供 compose 在宿主机读取，**不挂载进容器**。
       - `SettingsConfigDict.env_file="/app/.env"` 保留作为 fallback（容器内该文件默认不存在，不报错）。
       - 下一阶段接入 LLM / RAG 密钥时，统一走 compose `environment` 注入（生产可外接密钥管理系统），**禁止依赖容器内 `.env`**。
    2. `app/core/logging.py`: 配置 JSON 结构化日志。
    3. `app/core/errors.py`: 统一全局异常处理器。
    4. `app/main.py`: 初始化 FastAPI App，挂载健康检查路由 `/api/v1/healthz`。
- **验证**：
    ```powershell
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
      exec backend uv run pytest tests/test_core.py
    ```
---
## 8. Phase C · Database
- **目标**：实现 10 张核心业务表与向量检索支持。
- **Source of Truth**: `infra/docker/initdb/002_schema.sql` 是 Sprint 0 Database Schema v1.0 的**唯一事实来源**。
- **职责边界**：
    - **models.py**: 仅负责运行时 ORM 映射。**禁止**使用 `Base.metadata.create_all` 等方式自动创建/修改 Schema。
    - **init_db.py**: 仅负责数据库连接管理、初始化执行与验证，不得重复维护另一套 Schema 定义。
- **P1 表校验策略（V1.3 补充）**：
    - `review_defect`：Sprint 0 建表 + 校验关键字段（`defect_id / category / risk / review_status / origin`），缺失则 `init_db.py` **raise**。
    - `rule_candidates`：Sprint 0 建表 + 校验表存在，缺失则 `init_db.py` **raise**（Sprint 0 Phase‑I 反馈闭环依赖该表）。
    - `tasks` / `benchmark_results`：Sprint 1 新增，**不在 Sprint 0 schema**。
    - **P0/P1 表总览**：P0 = 10 张核心业务表；P1 = `review_defect` + `rule_candidates`（Sprint 0 建表，P1 业务深度使用留 Sprint 1）。
- **IR 与 case 关联策略（V1.3 补充）**：
    - `ir_document.schematic_case_id` 为**可空外键**（→ `schematic_case.id`）。
    - 写入路径：`store_schematic_ir(session, case_id, ir_doc, schematic_case_id=None)`
        1. 若调用方显式传入 `schematic_case_id`，直接使用；
        2. 否则，用 `case_id`（业务字符串）查 `schematic_case` 表取**最新一条**补全；
        3. 查不到时：`logger.warning` 并**保持 NULL**（不阻断写入）。
    - 读取路径：`load_schematic_ir_by_case` 按 `ir_json->>'case_id'` 过滤，按 `id desc` 取最新。
    - **过渡实现（Sprint 0）**：以上为 Sprint 0 的**过渡实现**，作为单 Project 单 case 场景的临时方案；`ir_document` 表是**过渡表**。
    - **Sprint 0 的 IR 是样例文件**（`data/cases/case001/schematic_ir.json`），仅为跑通 Demo 用；`ir_document` 表是**过渡表**。
- **Sprint 1 目标（回归上游架构，V1.3 补充）**：
    - **IR 加载回归文件**：`data/cases/<case_id>/schematic_ir.json`，**不落 DB**（对齐上游方案 §5.4 / §7.2）；
    - **`ir_document` 表废弃**（或保留为"IR 快照缓存"）；
    - **不新建 `tasks` 表**：用 `schematic_case` 承担"case 主数据" + `review_result` 承担"一次评审"；
    - **`review_result` 加 `schematic_case_id BIGINT FK → schematic_case.id`**（ON DELETE SET NULL）；
    - **`rule_candidates` 加 `schematic_case_id` FK；删除 `task_id`**（BigInteger 无 FK，语义重复）；
    - **`review_result.task_id` 保留为"批次标识"**（String，如 `"RUN-20261004-001"`）；
    - **反馈按 case 聚合链路**：`feedback_item → review_result → schematic_case`（通过 `review_result.schematic_case_id`）；
    - **`case_id` 一致性校验**：`ir_doc.case_id` 与目录名不一致时 raise；
    - **`store_schematic_ir` 加 `project_id`**：按 Project 隔离查询；查不到 case → **raise**（不再 warning）。
    - **Sprint 1 实现文件清单**（供 Sprint 1 执行时对照）：
        1. `infra/docker/initdb/002_schema.sql`：`review_result` 加 `schematic_case_id` FK；`rule_candidates` 加 `schematic_case_id` + 删 `task_id`；
        2. `app/persistence/models.py`：`ReviewResult` 加 `schematic_case_id` + relationship；`RuleCandidate` 加 `schematic_case_id` + 删 `task_id`；
        3. `scripts/demo_feedback_payload.py`：`ReviewResult` 构造时设 `schematic_case_id`（从 `schematic_case` 查）；
        4. `app/ir/service.py`：`store_schematic_ir` 加 `project_id`；`load_schematic_ir_by_case` 改走 `schematic_case.case_path` 文件加载；
        5. `app/ir/service.py`：`store_schematic_ir` 查不到 case → **raise**；`ir_doc.case_id` 一致性校验；
        6. `ir_document` 表：**废弃**（或保留为"IR 快照缓存"，视需要）；
        7. **Alembic 迁移脚本**（Sprint 1 引入）：
            - `review_result.schematic_case_id` 加列 + FK；
            - `rule_candidates` 加 `schematic_case_id` + 删 `task_id`；
        8. **测试补充**：
            - `review_result.schematic_case_id` FK 测试；
            - `load_schematic_ir_by_case` 从 `case_path` 加载测试；
            - `store_schematic_ir` raise 测试（case 不存在）；
            - `ir_doc.case_id` 一致性校验测试。
- **`tasks` 表的平替方案（V1.3 补充）**：
    - **Sprint 0 未建 `tasks` 表**；上游方案的 `tasks` 职责由**两张已有表**承担：
        - **`schematic_case`**：承担"case 主数据"（`case_id` / `case_path` / `expected_review_json` / `evaluation_yaml`）；
        - **`review_result`**：承担"一次评审"（`id` = 评审任务 ID；`review_output_json` = Graph State 落库）。
    - **平替对照表**：
        | 上游 `tasks` 表职责 | Sprint 1 平替 |
        | :--- | :--- |
        | `tasks.id` | `review_result.id`（自增主键） |
        | `tasks.case_id` | `review_result.schematic_case_id`（FK） |
        | `tasks.status` | `review_result.review_status` |
        | `tasks.state_json` | `review_result.review_output_json` |
        | `tasks.created_at` | `review_result.created_at` |
    - **优点**：
        - **不新建表**（低成本）；
        - **语义清晰**（`schematic_case` = case，`review_result` = 评审）；
        - **与"IR 回归文件加载"兼容**（`schematic_case.case_path` 指向文件目录）；
        - **反馈链路短**（`feedback → review_result → schematic_case`）；
        - **可对齐上游"Graph State 落库"**（`review_result.review_output_json` 就是 Report 的 JSONB）。
- **Sprint 0 保持现状**：不改代码，不迁移 DB；`schematic_case` / `ir_document` / `review_result` 表保留为过渡。

- **实现步骤**：
    1. `app/persistence/models.py`: 建立与 `002_schema.sql` 一致的 SQLAlchemy 模型。
    2. `infra/docker/initdb/`: 编写 `001_pgvector.sql` (开启扩展) 与 `002_schema.sql` (创建表)。
    3. `scripts/init_db.py`: 编写执行与连接验证脚本。
- **执行指令**：
    ```powershell
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
      exec backend uv run python -m scripts.init_db
    ```
- **验证**：进入容器 `psql` 检查表结构及 `pgvector` 扩展是否生效。
---
## 9. Phase D · Schematic IR
- **目标**：定义并验证原理图中间表示标准。
- **实现步骤**：
    1. `app/ir/schema.py`: 定义 Component, Pin, Net, Attribute 的 Pydantic 模型。
    2. `data/cases/case001/schematic_ir.json`: 编写 Buck 转换器基准数据。
- **验证**：
    ```powershell
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
      exec backend uv run python -m app.ir.serializer --validate /data/cases/case001
    ```
- **V1.3 补充（双模式）**：`serializer.py` 支持 `--strict` 模式。
    - **普通模式**（默认）：`intent` / `context` 缺失仅 Warning，不阻断。
    - **严格模式**（`--strict`）：`intent` / `context` 缺失报 **Error**（Golden Case 验收专用）。
    - **目录/文件均可传参**：`--validate` 支持传目录（自动查找 `schematic_ir.json`）或直接传文件。
    - **Golden Case 严格校验命令**：
      ```powershell
      docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
        exec backend uv run python -m app.ir.serializer --validate /data/cases/case001 --strict
      ```
 **V1.3 修订**：路径从 `./data/cases/case001`（会解析为 `/app/data/...`）改为 `/data/cases/case001`。
 ---
## 10. Phase E · Golden Case
- **目标**：建立 10 个标准测试案例（6A+4B）。
- **实现步骤**：
    1. 补齐各案例目录：`expert_reasoning.md`, `evidence/`, `evaluation.yaml`。
    2. **Case001 专家校准**：完成 `docs/freeze/case001_expert_calibration.md`。
- **验收**：案例目录结构完整，`expected_review.json` 满足 10 字段规范。
---
## 11. Phase F · Rule System
- **目标**：实现确定性规则引擎与 **5 条内置规则**（3 条电源规则 + 2 条 IO 规则）。
- **实现步骤**：
    1. `app/rules/engine.py`: 实现基于 YAML 的规则解析与匹配逻辑。
    2. `data/rules/`: 编写 5 条规则 YAML：
      - `POWER_001`（去耦电容，severity=medium）
      - `POWER_002`（**复位 RC 拓扑**，severity=medium）—— 按代码口径修正（原方案写"反馈分压"为笔误）
      - `POWER_003`（降额，severity=medium）—— **Sprint0 为预留接口，check 函数返回空列表**；实际降额算法留 Sprint 1
      - `IO_001`（IO 引脚悬空，severity=low，汇总 per-component）
      - `IO_002`（IO 方向冲突，severity=critical，逐条 per-net，≥2 个 OUTPUT 引脚）
- **验证**：
    ```powershell
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
      exec backend uv run python -m tests.test_rules --case case001
    ```
**V1.3 修订**：规则引擎内部引用规则目录的路径已改为 `/data/rules`（详见 `app/rules/engine.py` 的 `RULES_DIR` 常量）。
- **标准**：Case001 至少命中 2 条规则。
- **V1.3 补充（双入口）**：
    - `execute_all_rules()`：返回**所有**命中（含 `severity=low` 的提示，如 `IO_001`）。
    - `execute_defect_rules()`：只返回 `severity ∈ {medium, high, critical}` 的命中，用于 **K7 断言 / Benchmark / 报告落库**。
    - **Benchmark / Report 必须用 `execute_defect_rules()`**，避免 `low` 级提示污染缺陷指标。
- **V1.3 补充（安全设计）**：
    - `app/rules/registry.py` 实现**显式白名单**注册表，防止规则 YAML 注入任意代码。
    - 规则 YAML 中 `check_logic.function` 只能引用注册表中的键（如 `"circuit_checks.check_decoupling"`），**禁止任意函数调用**。
    - 未注册的 function 引用 → `get_builtin_check_func()` 抛 `KeyError`，规则引擎跳过该规则并打印 warning。
- **V1.3 补充（适用条件匹配策略）**：
    - `applicable_condition.any_of` → 任一条件块匹配即适用。
    - `applicable_condition.all_of` → 全部条件块匹配才适用。
    - 未声明 `any_of` / `all_of` → 视为 **scope 级规则**，只要 `enabled=True` 即适用（如 `POWER_001` / `POWER_002` / `POWER_003`）。
    - 未知键 → 打印 warning，**不阻断**（Sprint0 MVP 宽松策略）。
    - 支持的条件块：`component_type_in` / `component_ref_in` / `net_name_in` / `net_is_power` / `attribute_exists` / `always`。
---
## 12. Phase G · Benchmark (Rule Only Baseline)
- **目标**：以 Case001 为主建立规则引擎基线，评估 Precision, Recall, EAR, EVC 四项指标。
- **评测定义**:
    - **Ground Truth**: 使用 Golden Case 的 `expected_review.json`。
    - **Prediction**: 使用 Rule Engine 的执行输出。
    - **计算依据**: 匹配规则与指标公式以 `docs/benchmark/metrics_definition.md` 为准。
    - **V1.3 补充（严格 IR 校验）**: Benchmark 加载 IR 时**强制 `strict_mode=True`**；IR 校验失败（如 Golden Case 三语义缺失）时 `run_single_case` 直接 `raise`，不静默降级。
- **说明**: 本阶段 Benchmark 仅针对规则引擎（Rule Only），不包含 AI Review 效能评估。
- **实现步骤**：
    1. `app/services/benchmark_service.py`: 实现四项指标的计算逻辑。
       - **V1.3 修订**：`BENCH_OUT_DIR = "/out"`、`RULES_DIR = Path("/data/rules")`。
    2. 运行脚本：
        ```powershell
        docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
          exec backend uv run python -m tests.test_benchmark
        ```
- **V1.3 修订**：`test_benchmark.py` 的 `case_list = ["/data/cases/case001"]`。
- **输出**：
  - **容器内**：`/out/bench_ruleonly_sprint0.csv`
  - **宿主机**：`out/bench_ruleonly_sprint0.csv`（与容器内 `/out` 同源）
- **验收标准**：以 Case001 为基准输出四项指标；10 Case 框架已预留用于后续扩展。
- **V1.3 补充（`case_class` 字段）**:
    - `CaseBenchResult.case_class` 当前**硬编码为 "A"**（Sprint 0 只有 case001）。
    - **Sprint 1 待办**：跑 6A+4B 时需从 `expected_review.json` 的 `case_class` 字段读取（或按目录约定），区分 A / B 类。
---
## 13. Phase H · RAG Seed Knowledge
- **目标**：注入评审所需的 Datasheet 与应用笔记知识。
- **实现步骤**：
    1. `data/knowledge/`: 准备 5 段 Markdown 格式的种子知识（**已就绪**）：
        - `datasheets/stc89c55rc_oscillator_crystal.md`
        - `datasheets/stc89c55rc_power_supply.md`
        - `application_notes/an_mcu_reset_circuit.md`
        - `reference_designs/ref_stc89c55_minimum_system.md`
        - `reference_designs/an_8051_io_port_feature.md`
    2. **配置清单**：`data/knowledge/seed_ingest_list.yaml`（**已就绪**）
        - `ingest_entries`: 5 个种子文件路径
        - `case_evidence_scan`: `null`（Sprint0 不启用 B-Case evidence 扫描）
    3. **注入向量库**（入口 `python -m app.rag.ingest`）：
        ```powershell
        # dry-run：只解析切片，不入库
        docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
          exec backend uv run python -m app.rag.ingest --config /data/knowledge/seed_ingest_list.yaml --dry-run

        # 正式导入
        docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
          exec backend uv run python -m app.rag.ingest --config /data/knowledge/seed_ingest_list.yaml
        ```
- **种子文件格式要求（V1.3 补充）**：
    - 必须为 Markdown，**首部含 YAML frontmatter**（`---` 开头结尾）。
    - 必填字段 5 个：
      - `source_type` ∈ `{datasheet, reference_design, application_note, review_case, design_standard}`（见 `app/rag/sources.py`）
      - `source_title`：文档标题
      - `source_section`：文档级章节
      - `part_numbers`：关联器件型号列表（数组，空写 `[]`）
      - `related_rule_ids`：关联规则 ID 列表（数组，空写 `[]`）
    - 正文至少含 1 个 `##` 二级标题；无 `##` 时整篇作为文档级 section 兜底。
    - **现有 5 文件覆盖 3 种 `source_type`**：`datasheet` × 2 / `application_note` × 1 / `reference_design` × 2。
- **切片策略（V1.3 补充，chunking.py v1.5）**：
    - **chunk_size = 800**；**不支持 overlap**（v1.3 BUG1 修复：语义切片不需要 overlap）。
    - **语义优先切片**：
        1. 按 `##` 二级标题切分为 section；
        2. 每 section 内按段落（`\n\n`）聚合至 ≤ `chunk_size`；
        3. 单段超长时按标点（`。！？；\n`）硬切（`_hard_split`）；
        4. 代码块 / 表格整体保护不切（宁可超长）；
        5. 末尾短尾块（< `EVIDENCE_MIN_LEN`）并入前一块，避免丢信息。
    - **chunk 级 `source_section`**：每 chunk 携带所在章节标题（v1.1 M3）。
    - **`is_continuation`**：超长章节拆分的续块标记 `True`；Sprint1 重排时用于降权。
    - **`EVIDENCE_MIN_LEN = 10`**（复用 `benchmark_service` 常量）。
- **Embedding 后端双模式（V1.3 补充）**：
    - `RAG_EMBEDDING_BACKEND=fake`（**默认**）：`DeterministicFakeEmbedding(size=1536)`，**不触网**，用于开发/CI。
    - `RAG_EMBEDDING_BACKEND=openai`：`OpenAIEmbeddings`，需 `OPENAI_EMBEDDING_API_KEY` 或 `LLM_API_KEY`。
    - `EMBEDDING_MODEL_NAME` 默认 `text-embedding-3-small`（`OPENAI_EMBEDDING_MODEL` 覆盖）。
    - **⚠️ CI 流水线必须用 `fake`**，禁止真实外网调用（对齐 §15.L 风险矩阵）。
- **检索器契约（V1.3 补充，retriever.py v1.5）**：
    - 入口：`Retriever().retrieve(query, part_numbers=None, rule_ids=None, top_k=3)`
    - 流程：`embed_query` → pgvector 余弦距离排序 → `joinedload` 消除 N+1 → `part_numbers` / `rule_ids` 集合交集过滤 → 过采样 `top_k * 4` 后精选 `top_k`。
    - 输出：`List[RetrievalResult]`（`app/rag/knowledge.py`），字段：
        - 映射为 `evidence` 字典：`source_type` / `source_title`→`source` / `source_section`→`section` / `snippet`→`reason`
        - 附加：`score` / `part_numbers` / `related_rule_ids` / `doc_id` / `chunk_id`
        - **`is_continuation`**：Sprint1 重排时用于降权续块
    - **Sprint 0 限制**：仅内部 Python API，**无重排、无 HTTP 接口**（Sprint 1 扩展）。
- **`_as_dict` JSONB 防御（V1.3 补充，retriever.py v1.3 BUG3）**：
    - JSONB 列可能返回 `dict` / `None` / psycopg `Json` 包装类型。
    - 四层 fallback：`isinstance(dict)` → `.items()` → `dict()` → 返回 `{}` + **限流 WARNING**（每种类型只 warn 一次）。
    - 目的：**单个 chunk 的 JSONB 解析失败不拖垮整个检索**。
- **数据完整性哨兵（V1.3 补充，retriever.py v1.4）**：
    - `doc = chunk.doc` 后 `assert doc is not None`。
    - 理由：DB 层 FK + NOT NULL 保证 `doc` 必存在；`assert` 用于捕获"FK 被绕过"（如手工删 doc 未级联）的数据完整性异常。
    - 若触发，**检索整体抛 `AssertionError`**，不静默降级（数据完整性不可妥协）。
---
## 14. Phase I-J · Feedback & Demo
- **Phase I**: 验证 6 类反馈（正确/误报/漏检等）的入库与闭环。
- **6 类反馈触发规则（V1.3 补充）**：
    | 反馈类型 | 触发规则候选生成？ | 说明 |
    | :--- | :--- | :--- |
    | `correct_defect` | ❌ | 仅落库 |
    | `false_positive` | ❌ | 仅落库 |
    | `false_negative` | ✅ | **生成 `rule_candidate` 草稿** |
    | `new_rule_candidate` | ✅ | **生成 `rule_candidate` 草稿** |
    | `suggestion_update` | ❌ | 仅落库 |
    | `knowledge_gap` | ❌ | 仅落库；由 `export_knowledge_gap_backlog` 导出 backlog |
- **外键约束（V1.3 补充）**：
    - `FeedbackCreate` 强制传 `review_result_id` / `review_defect_id`（**DB 外键**），**禁止直接传 `task_id` / `defect_id` 字符串**。
    - `feedback_item` 表**无** `task_id` / `defect_id` 列；这两个是**业务视图字段**，通过 JOIN 回填：
        - `task_id ← review_result.task_id`
        - `defect_id ← review_defect.defect_id`
    - 外键不存在时：`FeedbackNotFoundError`（继承 `ValueError`）。
    - `created_by` 存 `users.id`（**int 主键**），非用户名字符串。
- **`RuleCandidate` 11 字段契约（V1.3 补充，`app/schemas/feedback.py`）**：
    - 字段来源：`rule_format.md`（Rule YAML Format v1.0） + `rule_candidates` ORM 冗余字段。
    - **规则 YAML 必填字段（5）**：`rule_id` / `rule_name` / `category` / `severity` / `rule_basis`
    - **规则 YAML 必填、草稿阶段可空（3）**：`applicable_condition` / `check_logic` / `suggestion`
    - **`rule_candidates` ORM 冗余字段（3）**：`title` / `description` / `evidence_refs`
    - **排除字段（系统生成）**：`version` / `enabled` / `candidate_id` / `from_feedback_id` / `case_id` / `task_id` / `proposed_yaml` / `status`
    - **字段改名（v1.2）**：`rule_text → rule_name`；`rationale → rule_basis`
    - **必填性变更（v1.2）**：`category` 从可选改**必填**
- **`RuleEvolutionService` Sprint 0 边界（V1.3 补充）**：
    - 仅生成 `proposed` 状态 `rule_candidate` YAML 草稿。
    - **草稿禁止直接合并正式 `rules` 库**；必须人工编辑后 Sprint 1 跑 benchmark 校验。
    - **YAML 草稿头部带 `# WARNING: AUTO-GENERATED DRAFT RULE CANDIDATE`**。
    - **`rule_candidates` 表无 `task_id` 列**；task 信息通过 `from_feedback_id → feedback_item → review_result` JOIN 获取。
    - **`evidence_refs` 三要素校验**：每条必须含 `source` / `section` / `reason`，缺失 `raise ValueError`（不用 `assert`，避免 `python -O` 跳过）。
    - **CLI 三入口**：`--list` / `--export-yaml` / `--export-knowledge-gap`（互斥）。
- **`FeedbackService` v1.1 关键改进（V1.3 补充）**：
    - **异常类**：`FeedbackServiceError` / `FeedbackNotFoundError`（继承 `ValueError`，兼容旧 `except ValueError`）。
    - **`scalar_one_or_none`**：用 `db.execute(select(...)).scalar_one_or_none()` 替代 `db.scalar()`（多行时抛错，更严格）。
    - **依赖注入**：`__init__(rule_evolution=None)`，便于 mock / 复用。
    - **`commit` 开关**：`submit_feedback(commit=True)` 默认内部 commit；`commit=False` 由调用方控制事务。
    - **辅助方法**：`_validate_refs` / `_build_orm` / `_dispatch_evolution` / `_build_out` 拆分主流程。
    - **logger**：入口 / 外键失败 / 规则候选生成 / 完成节点打印日志。
    - **`comment` / `expert_suggestion`**：当前使用同一值（保留原逻辑），语义分离留后续。
- **`report.py` Sprint 0 边界（V1.3 补充）**：
    - Sprint 0 **不迁移存量 Defect 10 字段 Pydantic**，仍保留在原有模块。
    - `report.py` **仅存放 `ReviewStatus` 枚举**（`AI_CONFIRMED` / `NEED_EXPERT_REVIEW` / `LOW_CONFIDENCE`）。
    - **TD-007（Sprint 1）**：把 `ir/rag/rules/report` 全部 Pydantic 收拢到 `app/schemas/`。
- **待解决耦合（V1.3 补充）**：
    - **`FeedbackItem.rule_candidate_ref` 无 DB 外键约束**（String 字符串）：
        - 若 `RuleCandidate` 记录被删除，`FeedbackItem.rule_candidate_ref` 会残留**悬空字符串**，DB 不校验。
        - **Sprint 1 待办**：`RuleEvolutionService.delete_candidate()` 中先反向置空 `rule_candidate_ref`，再 `db.delete(rc)`。
    - **`FeedbackItem.review_defect_id` 使用 `ondelete="CASCADE"`**：
        - 删除 `ReviewDefect` 会**连带删除** `FeedbackItem`。
        - **Sprint 1 待办**：明确业务语义后决定是否改为 `SET NULL`。
    - **`ReviewResult` / `ReviewDefect` 的 `task_id` / `defect_id` 语义**：见 §8 "IR 与 case 关联策略"（同属跨模块待定项）。
- **Feedback 依赖（Phase-I 新增）**：无额外依赖（复用 SQLAlchemy + Pydantic 现有版本）。
- **Phase J**: 执行 `.\scripts\demo_sprint0.ps1`，展示"IR -> 规则 -> 评审 -> 反馈"全链路。
  - **`demo_feedback_payload.py` 双模式（V1.3 补充）**：
    - **普通模式**（默认）：执行完整业务 payload 链路（IR → 规则 → RAG → Mock Report → 幂等写库 → 输出 feedback json）。
    - **诊断模式**（`--diagnose`）：仅打印 Python 版本 / `sys.path` / 核心包导入状态，**不执行业务逻辑，无 DB / IR / RAG 副作用**。
    - **`demo_sprint0.ps1` Step4 调用 `--diagnose`**（不再用 ps1 内嵌 heredoc）。
  - **幂等写入策略（V1.3 补充）**：
    - `ReviewResult`：按 `task_id="DEMO-TASK-001"` 查询，存在则**复用主键**，不存在则新建。
    - `ReviewDefect`：按 `defect_id="<RULE_ID>-DEMO-01"` 查询，存在则**复用主键**，不存在则新建。
    - 重复运行 `demo_sprint0.ps1` **不会触发 `review_defect_defect_id_key` 唯一约束冲突**。
    - **`DetachedInstanceError` 修复**：session 关闭前把 `rr.id` / `rd.id` 拷贝到普通 `int` 变量；**with 退出后禁止访问 ORM 对象**。
  - **Mock ReviewReport 边界（V1.3 补充）**：
    - **Sprint 0 不接入 LLM**；`ReviewReport` 由 `demo_feedback_payload.py` 内存 Mock 组装。
    - **`RuleResult` 无 `defect_id` / `confidence` / `location` 字段**：
        - `defect_id`：demo 本地生成 `<RULE_ID>-DEMO-01`；
        - `location`：使用 V1.2 §3.1 结构（`sheet` / `path` / `coords` / `ir_refs`），`ir_refs` 保存原始 `evidence_ir_refs`；
        - `confidence`：固定 `0.92`；
        - `review_status`：`AI_CONFIRMED`。
    - **Sprint 1 TODO**：接入 LLM 后替换整块 Mock 代码，`defect_id` / `location` / `confidence` 全部由 LLM 生成。
  - **`demo_sprint0.ps1` 6 步流程（V1.3 补充）**：
    1. 预检 compose 配置文件存在 + Docker Desktop 运行。
    2. 检查 `postgres` / `backend` 容器状态。
    3. 检查 backend HTTP `/api/v1/healthz`（业务可用判定 = Docker healthy **OR** HTTP 200）。
    4. 容器 Python 诊断（`demo_feedback_payload.py --diagnose`）。
    5. **Phase I-J 业务 Demo**：
        - 5-1：容器内跑 `demo_feedback_payload.py`，生成 `ReviewResult` / `ReviewDefect` + `fb_false_neg.json` / `fb_knowledge_gap.json`；
        - 5-2：提交 `false_negative` 反馈 → 生成 `rule_candidate` 草稿；
        - 5-3：提交 `knowledge_gap` 反馈；
        - 5-4：导出 `rule_candidate` 列表 / YAML 草稿 / `knowledge_gap` backlog。
    6. 手动 Demo 提示（打印容器内手工调试命令）。
  - **业务日志三段式（V1.3 补充）**：
    - **落盘**：`demo_feedback_payload.py` 所有业务日志写入容器 `/out/demo_sprint0/payload_run.log`；**stdout 仅输出单行 JSON**（`{"review_result_id": int, "review_defect_id": int}`）。
    - **读取**：ps1 通过 `docker compose exec -T backend cat /out/demo_sprint0/payload_run.log` 打印。
    - **同步**：ps1 通过 `docker compose cp backend:/out/demo_sprint0/payload_run.log "$outDir/payload_run.log"` 拷回宿主机。
    - **理由**：规避 Windows PowerShell 5.1 GBK 转码破坏 `docker exec` stdout 中文文本。
  - **PS5.1 异常处理修复（V1.3 补充）**：
    - **问题**：`docker compose exec` 失败时抛 `RemoteException`，被 `$ErrorActionPreference=Stop` 直接中断。
    - **修复**：Step 5 执行 `docker` 前**临时** `$ErrorActionPreference="Continue"`，用 `$LASTEXITCODE` 判断；执行完恢复。
    - **理由**：`docker` 是外部命令，靠退出码判定，不靠 PS 异常。
  - **V1.3 修订**：`demo_feedback_payload.py` 中路径已对齐：
    - 输出：`/out/demo_sprint0`
    - IR：`/data/cases/case001/schematic_ir.json`
    - 规则目录：`/data/rules`
  - `demo_sprint0.ps1` 中所有 `/app/out/demo_sprint0` 已改为 `/out/demo_sprint0`。
  - **补充说明**（源自 `docs/feedback_demo.md` V1.17）：
    - 结构化产物（YAML/JSON）**容器内 python 落盘 + `docker compose cp` 二进制拷贝**回宿主机，**禁止走 `docker-exec stdout` 管道**（规避 Windows PowerShell 5.1 GBK 转码）；
    - **业务日志**统一写入容器 `/out/demo_sprint0/payload_run.log`，ps1 通过 `cat` 读取并 cp 回宿主机；**stdout 仅输出单行 JSON**；
    - **`demo_sprint0.ps1` Step4 诊断**调用 `demo_feedback_payload.py --diagnose`（不再用 ps1 内嵌 bash heredoc）；
    - **`tests/test_demo.py` 必须完整运行整个模块**（`pytest tests/test_demo.py`），**禁止挑选单条用例**；细粒度隔离测试全部下沉到 `test_feedback_rule.py`。
---
## 15. Phase K-L · 验收与风险管理
### K. Sprint 0 关键验收项 (12 项)
1. [ ] **Docker Healthy**: `backend` & `postgres` 均为健康状态。
2. [ ] **Healthz 200**: `/api/v1/healthz` 响应正常。
3. [ ] **DB Schema v1.0**: 10 张 P0 表 + 2 张 P1 表（`review_defect` / `rule_candidates`）创建完成。
4. [ ] **IR Schema v1.0**: 冻结 Component/Pin/Net 规范；Golden Case 三语义字段（intent/context）覆盖率 100%，且 `--strict` 模式校验通过。
5. [ ] **Case001 Calibration**: 专家校准纪要签字确认。
6. [ ] **5 Rules (3 Power + 2 IO)**: 规则 YAML 编写完成且在容器内可运行；`POWER_003` 为预留空实现。
7. [ ] **Case001 Rule Hit**: 命中条数 ≥ 2。
8. [ ] **Benchmark Metrics**: 包含 Precision, Recall, EAR, EVC 四项核心指标。
9. [ ] **Feedback 6 Categories**: 支持全分类反馈录入。
10. [ ] **5 Business Freezes**: 见第 18 章节详细清单。
11. [ ] **Environment Freeze**: 见第 17 章节详细清单。
12. [ ] **Sprint 0 Demo PASS**: 全链路演示通过。

> **V1.3 补充说明（前端边界）**：
> - `docker-compose.prod.yml` 的 `frontend` 服务当前**沿用基座 placeholder 挂载**（`./frontend-placeholder`），生产真实 Vue dist 挂载**留至 Sprint 1 处理**。
> - Sprint 0 不阻塞于前端生产化。

### L. Phase L · 风险管理矩阵
| 风险项 | 影响 | 缓解措施 | 验收方式 |
| :--- | :--- | :--- | :--- |
| **Golden Case 质量** | 基准失效，导致评测误导 | 引入专家二次审核机制 | 专家校准纪要签字 |
| **IR Schema 稳定性** | 频繁变更导致解析器重写 | Sprint 0 强制冻结 V1.0 | 变更需经过架构评审 |
| **DB Schema 变更** | 数据迁移成本高 | 使用 Alembic 管理迁移 (Sprint 1) | init_db 脚本原子化 |
| **Rule 误报/漏报** | 降低工具可信度 | 建立误报/漏报反馈闭环 | EAR 指标 ≥ 60% |
| **Benchmark 指标稳定性** | 指标波动大，无法客观评估进度 | 固化评测脚本与指标计算公式 | Benchmark Format 冻结文档 |
| **环境一致性风险** | Windows/Linux 表现不一 | 强制容器化开发，固定版本 | bootstrap 脚本统一验证 |
| **RAG种子知识库依赖OpenAI Embedding外网服务** | ingest导入失败；CI流水线必须mock OpenAIEmbeddings，禁止真实外网调用；密钥可独立配置`OPENAI_EMBEDDING_API_KEY` | 1.开发环境配置`.env`密钥；2.CI流水线必须mock OpenAIEmbeddings，禁止真实外网调用；3.密钥可独立配置`OPENAI_EMBEDDING_API_KEY` | 开发容器ingest可正常完成知识库导入；CI单元测试无OpenAI网络请求 |
| **Docker Desktop 嵌套挂载误建目录（V1.3 新增）** | 宿主机 `backend/` 下被创建 `data/scripts/infra/out` 空目录，污染源码目录 | 采用方案A 并列挂载（`/data`、`/scripts`、`/infra`、`/out` 与 `/app` 并列） | `docker inspect` 显示 5 条并列 bind；宿主机 `backend/` 下无这 4 个目录 |
---
## 16. Development Command Reference
所有指令推荐使用 `scripts/` 下的脚本；**手工执行时，`-f` 参数统一使用 `$PWD` 绝对路径**：

| 任务 | 统一指令 (Host 执行) |
| :--- | :--- |
| **首次启动环境** | `.\scripts\bootstrap.ps1` |
| **启动开发环境** | `.\scripts\dev.ps1 start -Build` |
| **停止环境** | `.\scripts\dev.ps1 stop` |
| **重启环境** | `.\scripts\dev.ps1 restart -Build` |
| **查看后端日志** | `.\scripts\dev.ps1 logs` |
| **查看容器状态** | `.\scripts\dev.ps1 status` |
| **初始化数据库** | `docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend uv run python -m scripts.init_db` |
| **运行单元测试** | `docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend uv run pytest` |
| **运行代码检查** | `docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend uv run ruff check .` |
| **运行 Benchmark** | `docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend uv run python -m tests.test_benchmark` |
| **运行 Sprint 0 Demo** | `.\scripts\demo_sprint0.ps1` |

**说明**：
- **首次建立环境使用 `bootstrap.ps1`；日常使用 `dev.ps1`**。
- 手工执行 `docker compose` 时，`-f` 参数统一使用 `"$PWD/infra/docker/..."` 绝对路径。
---
## 17. 环境架构冻结 (Environment Freeze)
### 17.1 运行时与依赖
- **Python Runtime**: 3.11.x (Debian Slim)
    - 基础镜像：`python:3.11-slim`
    - 当前验证版本:Python 3.11.16 (通过python:3.11-slim 镜像获取)
    - 备注：使用 `python:3.11-slim` 标签可自动获取最新补丁版本，确保安全性。
- **Dependency Manager**: uv 0.12.6 (ghcr.io/astral-sh/uv:0.12.6)
- **RAG 依赖（Phase‑H 新增）**：`langchain-openai==0.2.8` + `langchain-core`（`DeterministicFakeEmbedding` 来源；仅用于 Embedding 调用，Sprint0 不使用 LangGraph）
- **DB Driver（V1.3 补充）**: `psycopg` (v3.x)，SQLAlchemy 连接串协议 `postgresql+psycopg://`
- **Database**: PostgreSQL 15
- **Vector Extension**: pgvector (以 Sprint 0 实际构建验证结果记录最终使用版本)
- **Venv Path**: `/opt/venv` (Container internal)
- **Networking**: Service-name based (`postgres:5432`)
- **Persistence**: Named volume `pgdata`

### 17.2 挂载策略冻结（V1.3 新增）
- **Mount Strategy**: 方案A 并列挂载
    - `/app`     <- `backend/`   （源码）
    - `/data`    <- `data/`      （数据资产）
    - `/scripts` <- `scripts/`   （脚本）
    - `/infra`   <- `infra/`     （基础设施）
    - `/out`     <- `out/`       （产物）
- **PYTHONPATH**: `/app:/`（追加根目录 `/`，使 `import scripts.xxx` 能命中挂在 `/scripts` 的脚本目录）
- **DATA_DIR**: `/data`
- **脚本调用约定**:
    - 注入 `$env:PROJECT_ROOT`
    - `$f1` / `$f2` 使用 `$PWD` 绝对路径
    - 不使用 `--project-directory`
    - 所有 `docker compose` 调用用 `cmd /c` 包装
- **Healthcheck host 约定（V1.3 统一）**: 一律使用 `127.0.0.1`，禁止 `localhost`（规避 IPv6 解析差异）。
- **配置来源约定（V1.3 澄清）**: 容器内配置以 compose `environment` 注入为准；`.env` 仅供 compose 在宿主机读取，不挂载进容器。
- **根因归档**: Docker Desktop for Windows 嵌套 bind mount 误建行为，详见 §6 A4 与 §19 故障排查。
- **Windows 结构化输出约定**（源自 `docs/feedback_demo.md`）：
    - 所有 JSON/YAML 产物：**容器内 python 落盘 + `docker compose cp` 二进制拷贝回宿主机**；
    - 禁止 `docker-exec stdout` 传递 JSON/YAML 中文文本（规避 GBK 破坏性转码）；
    - 业务日志：落容器 `payload_run.log`，stdout 仅单行 JSON；
    - `tests/test_demo.py` 必须完整运行整个模块，禁止挑选用例。

### 17.3 版本锁定策略
- Python: 使用 `python:3.11-slim` 标签，自动获取 3.11.x 系列最新安全补丁
- uv: 固定为 0.12.6，使用官方镜像 `ghcr.io/astral-sh/uv:0.12.6`
- 如需完全锁定 Python 版本，可改用 `python:3.11.16-slim`（已验证可用）
---
## 18. 业务冻结清单 (Business Freeze)
| 冻结项 | 对应文档/文件 | Sprint 0 完成标准 |
| :--- | :--- | :--- |
| **Schematic IR Schema v1.0** | `docs/ir/schematic_ir_spec_v1.0.md` | 支持 Component 三语义字段（`intent` / `context` / `constraint`）+ `lib_name` 必填 + `Net.connected_pins` 格式 `{ref}.{pin_id}` + Pin 统一命名（`pin_id` / `name` / `number`）；Golden Case `--strict` 模式 100% 通过 |
| **Database Schema v1.0** | `infra/docker/initdb/002_schema.sql` | 10 张 P0 表 + 2 张 P1 表（`review_defect` / `rule_candidates`）字段固化；P1 表 Sprint 0 建表并强校验，业务深度使用留 Sprint 1 |
| **Golden Case Format v1.0** | `data/cases/README_CASE_B_DESIGN.md` | 包含 expert_reasoning 与 evidence 结构 |
| **Rule YAML Format v1.0** | `docs/rules/rule_yaml_spec_v1.0.md` | 9 字段冻结：`rule_id` / `rule_name` / `version` / `category` / `applicable_condition` / `check_logic` / `severity` / `rule_basis` / `suggestion` / `enabled`；`check_logic.function` 必须命中 `app/rules/registry.py` 白名单；`applicable_condition` 支持 `any_of` / `all_of` / scope-only |
| **Benchmark Format v1.0** | `docs/benchmark/metrics_definition_v1.0.md` | 明确 Precision/Recall/EAR/EVC 计算公式 |
| **RAG Seed Format v1.0（V1.3 补充）** | `app/rag/sources.py` / `app/rag/chunking.py` | frontmatter 5 必填字段（`source_type` / `source_title` / `source_section` / `part_numbers` / `related_rule_ids`）+ 5 种 `source_type` 枚举 + chunk 级 `source_section` + `is_continuation` + `RetrievalResult` 10 字段契约；切片策略 `chunk_size=800` + 语义优先 |
| **反馈↔候选引用语义 v1.0（V1.3 补充）** | `models.py` / `002_schema.sql` | `feedback_item.rule_candidate_ref` 为**软引用**（String 字符串，无 FK）；`rule_candidates.from_feedback_id` 为**硬外键**（FK → feedback_item.id）。Sprint 1 删除候选时需反向置空 `rule_candidate_ref` |
| **Feedback Schema v1.0（V1.3 补充）** | `app/schemas/feedback.py` | 6 类 `FeedbackType` 枚举 + `RuleCandidate` 11 字段契约（5 规则 YAML 必填 + 3 草稿可空 + 3 ORM 冗余）+ `FeedbackCreate` 外键约束（`review_result_id` / `review_defect_id`，禁 `task_id` / `defect_id` 字符串）+ `FeedbackOut` JOIN 回填 `task_id` / `defect_id` |
| **Rule Evolution 边界 v1.0（V1.3 补充）** | `app/services/rule_evolution_service.py` | 仅生成 `proposed` 草稿，**禁止直接合并正式 rules 库**；草稿 YAML 头部带 `AUTO-GENERATED DRAFT` 警告；`rule_candidates` 表**无 `task_id` 列**；`evidence_refs` 三要素校验（`source` / `section` / `reason`）|
| **Case-Review 关联策略 v1.0（V1.3 补充）** | `models.py` / `002_schema.sql` | 不建 `tasks` 表；`schematic_case` 承担 case 主数据；`review_result.schematic_case_id`（Sprint 1 新增）承担"评审-案例"关联；反馈链路 `feedback_item → review_result → schematic_case`；Sprint 0 保持现状（过渡实现） |
---
## 19. 环境故障排查 (Troubleshooting)
### 19.1 问题：backend目录下自动生成 data/scripts/infra/out 空目录
- **现象**：执行 `.\scripts\dev.ps1 start` 之后，`./backend/` 目录下多出 data、scripts、infra、out 文件夹，不符合项目目录规划。
- **根因**：旧配置使用嵌套bind mount（`./xxx:/app/xxx`），Docker Desktop for Windows 在父挂载 `/app` 存在时，会自动在宿主机源目录下创建子挂载目标目录。
- **修复方案**：采用方案A并列挂载，移除嵌套挂载，将挂载目标改为容器根路径 `/data`、`/scripts`、`/infra`、`/out`，不再挂载到 `/app` 内部。
- **清理旧残留目录命令（标准ASCII短横线）**
```powershell
Remove-Item -Recurse -Force ./backend/data
Remove-Item -Recurse -Force ./backend/scripts
Remove-Item -Recurse -Force ./backend/infra
Remove-Item -Recurse -Force ./backend/out
```
### 19.2 问题：docker compose 警告 `__file__ variable is not set`

- **现象**：执行 dev.ps1 stop/start 时打印 `The "__file__" variable is not set. Defaulting to a blank string.`
- **根因**：compose yaml 内部引用 `${__file__}`，PowerShell 环境没有注入该变量。
- **修复方案**：
  1. 检查 `docker-compose.yml` / `docker-compose.dev.yml`，移除所有 `${__file__}` 引用；
  2. 在 dev.ps1 脚本内预先注入 `PROJECT_ROOT` 环境变量，使用绝对路径挂载。

### 19.3 问题：PowerShell 识别不了 Remove‑Item 命令

- **现象**：复制粘贴命令提示 `无法将“Remove‑Item”项识别为 cmdlet`
- **根因**：复制内容里是**非断行长破折号 U+2011**，不是标准 ASCII 连字符 `-` (U+002D)。
- **修复方案**：手动重写命令，使用标准短横线；优先直接复制上方 19.1 给出的清理脚本。

### 19.4 问题：backend 容器 healthcheck 失败

- **现象**：容器反复重启，状态 unhealthy。
- **根因**：
  1. 后端服务未正常监听 8000 端口；
  2. start-period 时间不足，应用启动慢；
  3. 旧 healthcheck 使用 curl，镜像缺少 curl 工具。
- **修复方案**：
  1. Healthcheck 改用 python 内置 urllib，不再依赖 curl；
  2. 调整 `start-period=10s`；
  3. 查看日志：`.\scripts\dev.ps1 logs backend`

### 19.5 问题：容器内 import scripts.xxx 模块找不到

- **现象**：`ModuleNotFoundError` 导入 scripts 下脚本失败
- **根因**：scripts 挂载到容器 `/scripts`，但 PYTHONPATH 没有包含根目录 `/`
- **修复方案**：环境变量 `PYTHONPATH=/app:/`

### 19.6 通用排查步骤

1. 查看容器状态：`.\scripts\dev.ps1 status`
2. 查看容器原始挂载信息，验证挂载是否生效：
```powershell
docker inspect backend | Select-String "Mounts"
```
3. 进入 backend 容器交互式调试：
```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend bash
```

### 19.7 问题：删了某张表后重跑 `init_db.py` 无法恢复

- **现象**：手动 `DROP TABLE rule_candidates CASCADE` 后重跑 `init_db.py`，日志显示"检测到数据库已初始化，跳过执行"，表仍未恢复，且后续校验 `raise`。
- **根因**：
  1. `002_schema.sql` 作为**整体脚本**一次性 `conn.execute(text(sql_text))` 提交；遇到首个 `DuplicateObject`（如 `trigger "update_users_updated_at" for relation "users" already exists`）即抛异常，`run_init_sql()` 捕获后**跳过整个文件的后续所有语句**，包括 `CREATE TABLE IF NOT EXISTS rule_candidates`。
  2. 该行为**不会**因为 `CREATE TABLE IF NOT EXISTS` 本身幂等而改变——因为**整体跳过发生在语句级之前**。
- **修复方案**：
  - **短期（Sprint 0）**：彻底重建，触发 postgres initdb 全量执行：
    ```powershell
    # 1. 停服务 + 删 volume
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" down -v
    
    # 2. 重启（postgres 首次初始化 volume 时自动执行 /docker-entrypoint-initdb.d/）
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" up -d --build

    # 3. 校验
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend uv run python -m scripts.init_db
    ```
    ⚠️ **注意**：`dropdb` + `createdb` **不会**触发 initdb 重跑（initdb 只在容器首次初始化 volume 时执行一次），所以必须用 `down -v` 删 volume。
  - **长期（Sprint 1）**：引入 Alembic，使用 `alembic upgrade head` 进行增量迁移，不再依赖 `002_schema.sql` 整体重建。
  - **可选增强（Sprint 1）**：将 `002_schema.sql` 按分号拆分逐条执行，每条独立捕获 `DuplicateObject`，实现"单表补建"能力。注意需处理 `$$ ... $$` 函数体内的分号。  

---
## 20. 变更记录 Changelog
### V1.0
- 初始Sprint0规划，采用嵌套bind mount方案：`./data:/app/data`，`./scripts:/app/scripts`等。
- 采用curl作为backend健康检查工具。
- 业务Schema、IR、Benchmark指标初稿完成。

### V1.2
1. 修复P0级Healthcheck：替换curl为python内置urllib，移除容器curl依赖。
2. 补齐Phase B~L完整阶段流程，文档转为执行型落地指南。
3. 新增Phase L风险管理矩阵。
4. 明确5项Business Freeze与Environment Freeze边界。
5. 统一Docker Compose调用指令；限定前端仅为占位，不作为Sprint0阻塞项。
6. 工程规范调整：Python测试脚本统一放入`backend/tests`，scripts目录仅存放运维脚本。

### V1.3（当前定稿版本）
1. **挂载策略重大变更**：从嵌套bind mount升级为方案A并列挂载，解决Docker Desktop Windows下自动在backend目录生成data/scripts/infra/out目录的缺陷。
    - 容器挂载更新：`/app`、`/data`、`/scripts`、`/infra`、`/out`相互独立并列挂载
    - 更新`PYTHONPATH=/app:/`，解决scripts模块导入问题
    - 更新`DATA_DIR=/data`，所有业务代码路径对齐新挂载点
2. PowerShell脚本增强：
    - 注入`PROJECT_ROOT`环境变量，compose配置使用绝对路径挂载
    - docker compose调用使用`cmd /c`包装，规避PowerShell 5.1 stderr异常
    - 启动前预创建bind mount宿主机源目录
    - 统一替换U+2011非断行破折号为标准ASCII短横线`-`，解决命令复制报错
3. Compose配置清理：移除`${__file__}`变量引用，消除启动警告。
4. 所有业务脚本、benchmark、demo脚本内文件路径全部更新为容器内新路径 `/data`、`/out`。
5. 新增风险项：Docker Desktop嵌套挂载自动创建目录，写入风险管理矩阵。
6. 故障排查章节新增3条专项问题（目录自动生成、__file__警告、破折号字符问题）。
---

## 附录 A. V1.2 → V1.3 变更点索引

| # | 章节 | 变更摘要 |
| :--- | :--- | :--- |
| 1 | 文档头部 | 新增 V1.3 修订说明 |
| 2 | 1.2 | 迁移表追加"嵌套挂载 → 并列挂载"一行 |
| 3 | 2 | 目标追加"挂载策略定稿" |
| 4 | 3 | backend Mount 段补全 5 条并列挂载 |
| 5 | 5 | scripts/ 补全脚本清单；data/ 注释更新 |
| 6 | 6.A1 | 追加实际分支布局 |
| 7 | 6.A4 | docker-compose.dev.yml 挂载策略替换为方案A；根因归档 |
| 8 | 6.A5 | 脚本调用约定（PROJECT_ROOT / 绝对路径 -f / cmd /c / 源目录预创建） |
| 9 | 7~12 | 各 Phase 验证命令统一为 `"$PWD/..."` 绝对路径；路径从 `/app/xxx` 改 `/xxx` |
| 10 | 14 | Phase J 路径对齐说明 |
| 11 | 15.L | 风险矩阵追加"嵌套挂载误建目录"一行 |
| 12 | 16 | 命令表替换；明确 bootstrap / dev 分工 |
| 13 | 17 | 新增 17.2 挂载策略冻结子节 |
| 14 | 19 | 追加 4 条故障排查（误建目录 / import scripts / U+2011 / PowerShell 5.1） |
| 15 | 附录 A | 新增变更点索引（本节） |
| 16 | 8 | 补 P1 表校验策略 |
| 17 | 17.1 | 补 `psycopg` 版本 |
| 18 | 18 | Database Schema 行说明 P1 表；追加"反馈↔候选引用语义"行 |
| 19 | 19.7 | 新增"删表后无法重跑恢复"故障排查 |
| 20 | 15.K | 验收项 3 同步 P1 表 |
| 21 | 11 | Phase F：3 条 → 5 条规则；补双入口 / 安全设计 / 适用条件匹配 |
| 22 | 12 | Phase G：补严格 IR 校验；补 `case_class` 待改进 |
| 23 | 13 | Phase H：补完整流程 / 格式要求 / 切片策略 / 双模式 / 检索器契约 / `_as_dict` / 哨兵 |
| 24 | 15.K | 验收项 6 同步 5 条规则 |
| 25 | 17.1 | 补 `langchain-core` |
| 26 | 18 | 追加"RAG Seed Format v1.0"行 |
| 27 | 8 | IR-case 关联策略改为"跨模块待定项"（与 Phase I 一起确定，6 个待明确问题） |
| 28 | 14 | Phase I：6 类反馈触发规则 / 外键约束 / RuleCandidate 11 字段 / RuleEvolution 边界 / FeedbackService v1.1 / report.py TD-007 / 待解决耦合 / Feedback 依赖 |
| 29 | 14 | Phase J：双模式 / 幂等写入 / Mock ReviewReport 边界 / 6 步流程 / payload_run.log 三段式 / PS5.1 修复 |
| 30 | 18 | 追加"Feedback Schema v1.0"行；追加"Rule Evolution 边界 v1.0"行 |
| 31 | 8 | IR 与 case 关联策略改为"过渡实现 + Sprint 1 平替方案"；明确"用 `schematic_case` 平替 `tasks` 表" |
| 32 | 18 | 追加"Case-Review 关联策略 v1.0"行 |
| 33 | 8 | 追加"`tasks` 表的平替方案"（含平替对照表） |
| 34 | 8 | 追加"Sprint 1 实现文件清单"（8 处改动映射到具体文件） |
---

**文档版本**：V1.3
**最后更新**：2026-09-21
**变更依据**：Sprint 0 方案A 并列挂载落地与相关排障记录
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
         CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/healthz')" || exit 1
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
    1. `app/core/config.py`: 实现 Pydantic Settings 加载 `.env`。
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
- **目标**：实现确定性规则引擎与 3 条电源规则。
- **实现步骤**：
    1. `app/rules/engine.py`: 实现基于 YAML 的规则解析与匹配逻辑。
    2. `data/rules/`: 编写 `POWER_001` (去耦电容), `POWER_002` (反馈分压), `POWER_003` (降额)。
- **验证**：
    ```powershell
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
      exec backend uv run python -m tests.test_rules --case case001
    ```
**V1.3 修订**：规则引擎内部引用规则目录的路径已改为 `/data/rules`（详见 `app/rules/engine.py` 的 `RULES_DIR` 常量）。
- **标准**：Case001 至少命中 2 条规则。
---
## 12. Phase G · Benchmark (Rule Only Baseline)
- **目标**：以 Case001 为主建立规则引擎基线，评估 Precision, Recall, EAR, EVC 四项指标。
- **评测定义**:
    - **Ground Truth**: 使用 Golden Case 的 `expected_review.json`。
    - **Prediction**: 使用 Rule Engine 的执行输出。
    - **计算依据**: 匹配规则与指标公式以 `docs/benchmark/metrics_definition_v1.0.md` 为准。
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
---
## 13. Phase H · RAG Seed Knowledge
- **目标**：注入评审所需的 Datasheet 与应用笔记知识。
- **实现步骤**：
    1. `data/knowledge/`: 准备 5 段 Markdown 格式的种子知识。
    2. 注入向量库验证检索准确度。
---
## 14. Phase I-J · Feedback & Demo
- **Phase I**: 验证 6 类反馈（正确/误报/漏检等）的入库与闭环。
- **Phase J**: 执行 `.\scripts\demo_sprint0.ps1`，展示"IR -> 规则 -> 评审 -> 反馈"全链路。
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
3. [ ] **DB Schema v1.0**: 10 张业务表创建完成。
4. [ ] **IR Schema v1.0**: 冻结 Component/Pin/Net 规范。
5. [ ] **Case001 Calibration**: 专家校准纪要签字确认。
6. [ ] **3 Power Rules**: 规则 YAML 编写完成且在容器内可运行。
7. [ ] **Case001 Rule Hit**: 命中条数 ≥ 2。
8. [ ] **Benchmark Metrics**: 包含 Precision, Recall, EAR, EVC 四项核心指标。
9. [ ] **Feedback 6 Categories**: 支持全分类反馈录入。
10. [ ] **5 Business Freezes**: 见第 18 章节详细清单。
11. [ ] **Environment Freeze**: 见第 17 章节详细清单。
12. [ ] **Sprint 0 Demo PASS**: 全链路演示通过。

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
- **RAG 依赖（Phase‑H 新增）**：`langchain-openai==0.2.8`（仅用于 Embedding 调用，Sprint0 不使用 LangGraph）
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
| **Schematic IR Schema v1.0** | `docs/ir/schematic_ir_spec_v1.0.md` | 支持 Component 三语义字段解析 |
| **Database Schema v1.0** | `infra/docker/initdb/002_schema.sql` | 10 张核心业务表字段固化 |
| **Golden Case Format v1.0** | `data/cases/README_CASE_B_DESIGN.md` | 包含 expert_reasoning 与 evidence 结构 |
| **Rule YAML Format v1.0** | `docs/rules/rule_yaml_spec_v1.0.md` | 支持 applicable_condition 与 check_logic |
| **Benchmark Format v1.0** | `docs/benchmark/metrics_definition_v1.0.md` | 明确 Precision/Recall/EAR/EVC 计算公式 |
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
---

**文档版本**：V1.3
**最后更新**：2026-09-21
**变更依据**：Sprint 0 方案A 并列挂载落地与相关排障记录
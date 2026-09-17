# Sprint 0 开发落地规划
## Docker + uv 可复现开发环境 V1.2 最终版

## 0. 文档定位
本文件是基于「AI 原理图评审 MVP」项目架构设计的 **Sprint 0 顺序落地指南**。
**核心变更**：从原来的“宿主机 Python venv + pip”迁移为 **“Docker 全运行环境 + uv Python 依赖管理”**。
**V1.2 修订说明**：
1. 修复 P0 级 Docker Healthcheck（改用 Python urllib，移除 curl 依赖）。
2. 完整补齐 Phase B-L 业务阶段详情，确保“执行型文档”属性。
3. 补齐 Phase L 风险管理矩阵（Risk -> Impact -> Mitigation -> Acceptance）。
4. 明确 5 项 Business Freeze 具体内容与 Environment Freeze 边界。
5. 统一 Docker / Compose 执行指令规范，明确 Frontend Placeholder 验收边界。
6. 工程规范调整：全部Python测试脚本统一放置于 `backend/tests`，scripts目录仅保留运维脚本。

---

## 1. 上游基准与本次架构调整
### 1.1 上游基准
- [第一阶段工程实施方案 V1.2 增强版](file:///e:/projects/Schematic_Design_Review/.trae/documents/%E7%AC%AC%E4%B8%80%E9%98%B6%E6%AE%B5%E5%B7%A5%E7%A8%8B%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88_V1.2_%E5%A2%9E%E5%BC%BA%E7%89%88_plan.md)
- [项目实施方案_修订版_MVP闭环验证](file:///e:/projects/Schematic_Design_Review/.trae/documents/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88_%E4%BF%AE%E8%AE%A2%E7%89%88_MVP%E9%97%AD%E7%8E%AF%E9%AA%8C%E8%AF%81_plan.md)

### 1.2 原规划 → 新规划迁移说明
| 原规划特性 | 新规划 (V1.2 Docker+uv) | 处理方式 |
| :--- | :--- | :--- |
| **宿主机 Python** | **Backend Container (Python 3.11.x)** | 删除宿主机依赖 |
| **venv / pip** | **uv (pyproject.toml + uv.lock)** | 替换为现代依赖管理 |
| **Venv 位置** | **容器内 /opt/venv** | 隔离源码挂载冲突 |
| **Docker Compose** | **Dev/Prod 分离** | 明确职责边界 |
| **Healthcheck** | **Python urllib 实现** | 移除 curl，提高镜像纯净度 |
| **Bootstrap** | **全自动化 Healthz 验证** | 强化环境就绪检查 |

---

## 2. Sprint 0 最终目标
1.  **环境闭环**：开发者只需 Git + Docker Desktop，即可一键启动后端、数据库及前端占位。
2.  **数据闭环**：完成 Buck IR 验证，Case001 专家校准，3 条 Power 规则在容器内运行通过。
3.  **验收闭环**：在容器内执行 Rule Only Benchmark，建立规则引擎基线，评估指标，完成 Sprint 0 Demo。
4.  **架构冻结**：完成业务 Schema 与开发环境双重冻结。

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
         │     └── Mount: ./backend -> /app (Dev 模式)
         │
         ├── postgres (Service: postgres)
         │     ├── Image: pgvector/pgvector:pg15
         │     └── Volume: pgdata (Named Volume)
         │
         └── frontend (Service: frontend - Placeholder)
               └── 边界说明：Frontend Placeholder 仅作为占位，不属于 Sprint 0 核心验收阻塞项
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
├── scripts/                # 自动化脚本
│   ├── bootstrap.ps1       # 环境初始化
│   ├── dev.ps1             # 快速启动开发
│   └── demo_sprint0.ps1    # Sprint 0 演示
├── data/                   # 数据资产 (挂载到容器)
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
    - `docker-compose.dev.yml`: 挂载 `./backend:/app`，使用 `uvicorn --reload`。
    - `docker-compose.prod.yml`: 不挂载源码，使用 Immutable Image，禁用 `--reload`。

### A5. 强化一键式 Bootstrap
- **目标**：环境就绪自动化验证。
- **脚本逻辑**：
    1. 预检 Docker Engine。
    2. 复制 `.env.example` -> `.env`。
    3. 执行启动命令：
       ```powershell
       docker compose `
         -f infra/docker/docker-compose.yml `
         -f infra/docker/docker-compose.dev.yml `
         up -d --build
       ```
    4. 轮询检查 `postgres` 与 `backend` 的 Health 状态。
    5. 调用 `healthz` 接口，输出 `PASS` 及访问地址。

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
    docker compose `
      -f infra/docker/docker-compose.yml `
      -f infra/docker/docker-compose.dev.yml `
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
    docker compose `
      -f infra/docker/docker-compose.yml `
      -f infra/docker/docker-compose.dev.yml `
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
    docker compose `
      -f infra/docker/docker-compose.yml `
      -f infra/docker/docker-compose.dev.yml `
      exec backend uv run python -m app.ir.serializer --validate ./data/cases/case001
    ```

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
    docker compose `
      -f infra/docker/docker-compose.yml `
      -f infra/docker/docker-compose.dev.yml `
      exec backend uv run python -m tests.test_rules --case case001
    ```
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
    2. 运行脚本：
        ```powershell
        docker compose `
          -f infra/docker/docker-compose.yml `
          -f infra/docker/docker-compose.dev.yml `
          exec backend uv run python -m tests.test_benchmark
        ```
- **输出**：`out/bench_ruleonly_sprint0.csv`。
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
- **Phase J**: 执行 `.\scripts\demo_sprint0.ps1`，展示“IR -> 规则 -> 评审 -> 反馈”全链路。

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
| **RAG种子知识库依赖OpenAI Embedding外网服务** | ingest导入失败；CI执行消耗OpenAI token；网络不通导致Sprint0 Phase‑H无法验收 | 1.开发环境配置`.env`密钥；2.CI流水线必须mock OpenAIEmbeddings，禁止真实外网调用；3.密钥可独立配置`OPENAI_EMBEDDING_API_KEY` | 开发容器ingest可正常完成知识库导入；CI单元测试无OpenAI网络请求 |

---

## 16. Development Command Reference
所有指令推荐使用 `scripts/` 下的脚本或完整的 `docker compose` 命令：

| 任务 | 统一指令 (Host 执行) |
| :--- | :--- |
| **启动开发环境** | `.\scripts\dev.ps1` 或 `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml up -d` |
| **停止环境** | `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml down` |
| **查看后端日志** | `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml logs -f backend` |
| **初始化数据库** | `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m scripts.init_db` |
| **运行单元测试** | `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest` |
| **运行代码检查** | `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run ruff check .` |
| **运行 Benchmark** | `docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m tests.test_benchmark` |

---

## 17. 环境架构冻结 (Environment Freeze)
- **Python Runtime**: 3.11.x (Debian Slim)
    - 基础镜像：`python:3.11-slim`
    - 当前验证版本:Python 3.11.16 (通过python:3.11-slim 镜像获取)
    - 备注：使用 `python:3.11-slim` 标签可自动获取最新补丁版本，确保安全性。
- **Dependency Manager**: uv 0.12.6 (ghcr.io/astral-sh/uv:0.12.6)
- **【✅Phase‑H RAG依赖新增】**：`langchain‑openai==0.2.8`（仅用于Embedding调用，Sprint0不使用LangGraph）
- **Database**: PostgreSQL 15
- **Vector Extension**: pgvector (以 Sprint 0 实际构建验证结果记录最终使用版本)
- **Venv Path**: `/opt/venv` (Container internal)
- **Networking**: Service-name based (`postgres:5432`)
- **Persistence**: Named volume `pgdata`

**版本锁定策略**：
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
1. **Healthcheck 失败**: 检查 `backend` 是否在 8000 端口监听，且 `/api/v1/healthz` 返回 200。
2. **挂载冲突**: 确认宿主机没有同名 `.venv` 目录被映射进容器 `/app`（已通过 `/opt/venv` 隔离）。
3. **数据库连接**: 确保容器内 `DATABASE_URL` 使用 `postgres` 服务名而非 `localhost`。
4. **Python测试模块导入失败**：确认 `backend/tests/__init__.py` 空文件存在，tests目录被识别为Python包，支持`‑m tests.xxx`模块执行方式。


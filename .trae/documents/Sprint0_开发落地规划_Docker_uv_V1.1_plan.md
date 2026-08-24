# Sprint 0 开发落地规划
## Docker + uv 可复现开发环境 V1.1

## 0. 文档定位
本文件是基于「AI 原理图评审 MVP」项目架构设计的 **Sprint 0 顺序落地指南**。
**核心变更**：从原来的“宿主机 Python venv + pip”迁移为 **“Docker 全运行环境 + uv Python 依赖管理”**。
**V1.1 修订说明**：修复了 V1.0 中 Dev 模式源码挂载导致的环境冲突问题，强化了健康检查与生产分离，并完整恢复了原 Sprint 0 的 Phase B-L 业务阶段与验收标准。

---

## 1. 上游基准与本次架构调整
### 1.1 上游基准
- [第一阶段工程实施方案 V1.2 增强版](file:///e:/projects/Schematic_Design_Review/.trae/documents/%E7%AC%AC%E4%B8%80%E9%98%B6%E6%AE%B5%E5%B7%A5%E7%A8%8B%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88_V1.2_%E5%A2%9E%E5%BC%BA%E7%89%88_plan.md)
- [Sprint0_详细执行计划_V1.1_修订版](file:///e:/projects/Schematic_Design_Review/.trae/documents/Sprint0_%E8%AF%A6%E7%BB%86%E6%89%A7%E8%A1%8C%E8%AE%A1%E5%88%92_V1.1_%E4%BF%AE%E8%AE%A2%E7%89%88_plan.md)

### 1.2 原规划 → 新规划迁移说明
| 原规划特性 | 新规划 (V1.1 Docker+uv) | 处理方式 |
| :--- | :--- | :--- |
| **宿主机 Python** | **Backend Container (Python 3.11-slim)** | 删除宿主机依赖 |
| **venv / pip** | **uv (pyproject.toml + uv.lock)** | 替换为现代依赖管理 |
| **requirements.txt** | **pyproject.toml** | 迁移依赖声明 |
| **Venv 位置** | **宿主机 .venv** -> **容器内 /opt/venv** | 隔离源码挂载冲突 |
| **Docker Compose** | **单一文件** -> **Dev/Prod 分离** | 明确职责边界 |
| **Bootstrap** | **手动脚本** -> **一键式自动化脚本** | 强化环境就绪检查 |

---

## 2. Sprint 0 最终目标
1.  **环境闭环**：开发者只需 Git + Docker Desktop，即可一键启动后端、数据库及前端占位。
2.  **数据闭环**：完成 Buck IR 验证，Case001 专家校准，3 条 Power 规则在容器内运行通过。
3.  **验收闭环**：在容器内执行 Benchmark，生成 Precision/Recall 报表，完成 Sprint 0 Demo。

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
         │     ├── Runtime: Python 3.11 + uv (v0.4.0)
         │     ├── Venv Path: /opt/venv (隔离挂载)
         │     └── Mount: ./backend -> /app (Dev 模式)
         │
         ├── postgres (Service: postgres)
         │     ├── Image: pgvector/pgvector:pg15
         │     └── Mount: ./pgdata -> /var/lib/postgresql/data
         │
         └── frontend (Service: frontend - Placeholder)
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
**目标**：确立开发规范。
- **分支**：`main` (发布), `develop` (集成), `sprint0/base` (本 Sprint 集成分支)。
- **.gitignore**: 排除 `.venv`, `pgdata`, `out`, `.env` 等。

### A2. 迁移依赖到 pyproject.toml
**目标**：使用 uv 管理依赖。
`backend/pyproject.toml` 包含所有运行时及开发依赖（pytest, ruff 等）。

### A3. 编写工程级 Backend Dockerfile (Pin uv 0.4.0)
**目标**：修复 V1.0 虚拟环境冲突，固定 uv 版本。
```dockerfile
# 使用固定版本的 uv
FROM ghcr.io/astral-sh/uv:0.4.0 AS uv_bin

FROM python:3.11-slim
COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /app
# 将虚拟环境安装在 /opt/venv，避免被 Dev 模式源码挂载覆盖
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN useradd -m app && chown -R app:app /app && chown -R app:app /opt/venv
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### A4. 重新设计 Docker Compose (Dev/Prod 分离)
**目标**：Dev 模式挂载源码，Prod 模式镜像自包含。
- `docker-compose.yml`: 定义 `postgres` 和 `backend` 基础配置。
- `docker-compose.dev.yml`: 挂载 `./backend:/app`, 开启 `--reload`。
- `docker-compose.prod.yml`: 不挂载源码，移除 `--reload`。

### A5. 强化一键式 Bootstrap
**脚本**：`scripts/bootstrap.ps1`
**流程**：
1. 检查 Docker 状态。
2. 生成 `.env` (基于 `.env.example`)。
3. 执行 `docker compose build`。
4. 执行 `docker compose up -d`。
5. 循环等待 `postgres` 健康。
6. 循环等待 `backend` 健康。
7. 验证 `healthz` 接口。
8. 输出 `PASS` 及 `http://localhost:8000/docs`。

---

## 7. Phase B · Backend Framework (容器化执行)
### B1-B5. 核心框架代码
- `app/core/config.py`: Pydantic Settings。
- `app/core/logging.py`: JSON Logger。
- `app/core/errors.py`: 全局异常处理。
- `app/main.py`: FastAPI 入口，挂载 `api_router`。
- `app/domain/schemas/`: 统一枚举与基础模型。

---

## 8. Phase C · Database
### C1-C3. ORM & SQL
- `app/persistence/db.py`: SQLAlchemy Session 管理。
- `app/persistence/models.py`: 恢复 10 张业务表（projects, designs, tasks, defects 等）。
- `infra/docker/initdb/`: 001_pgvector.sql, 002_schema.sql。
### C4. 容器化初始化
命令：`docker compose exec backend uv run python -m scripts.init_db`。

---

## 9. Phase D · Schematic IR
### D1-D3. IR 标准与 Case001
- `app/ir/schema.py`: Component (含语义字段), Pin, Net, Attribute。
- `data/cases/case001_buck_converter_A/`: 真实 IR 数据。
- **验证**：在容器内运行 IR Validate 脚本，确保 0 Error。

---

## 10. Phase E · Golden Case
### E1-E5. 案例体系
- `README_CASE_B_DESIGN.md`: 案例规范。
- 10 个 Case 框架：6 个 A 类，4 个 B 类。
- **Case001 专家校准**：恢复校准纪要文档，确认基准。

---

## 11. Phase F · Rule System
### F1-F3. 规则库与引擎
- `data/rules/`: POWER_001/002/003 YAML。
- `app/rules/engine.py`: 匹配与执行。
- **Power 规则**：去耦电容检查、反馈分压比检查、器件选型检查。

---

## 12. Phase G · Benchmark
### G1-G2. 评测指标
- `app/services/benchmark_service.py`: 计算 Precision, Recall, FPR, FNR。
- `scripts/run_benchmark.py`: 容器内执行，输出 `out/bench_xxx.csv`。

---

## 13. Phase H-I · RAG & Feedback
- **Phase H**: 5 段 Markdown 种子知识挂载入库。
- **Phase I**: 6 类反馈（正确/误报/漏检等）入库验证。

---

## 14. Phase J · Sprint 0 Demo
**脚本**：`scripts/demo_sprint0.ps1`
**流程**：容器化执行所有 Phase，最终打印 PASS。

---

## 15. Phase K-L · 验收与风险
### K. Sprint 0 关键验收项 (恢复)
- [ ] Docker & Backend Healthy (200 OK)
- [ ] DB Schema v1.0 (10 Tables)
- [ ] IR Schema v1.0 冻结
- [ ] Case001 校准纪要签字
- [ ] 3 条 Power 规则命中 (Recall=1.0 for Case001)
- [ ] Benchmark 4 项指标齐全
- [ ] Feedback 6 类覆盖
- [ ] 5 项业务冻结文档完成
- [ ] **新增**：环境架构冻结完成

---

## 16. Development Command Reference
所有 Python 命令必须在 backend 容器中执行：
- **启动环境**: `.\scripts\dev.ps1`
- **初始化数据库**: `docker compose exec backend uv run python -m scripts.init_db`
- **运行测试**: `docker compose exec backend uv run pytest`
- **运行 Lint**: `docker compose exec backend uv run ruff check .`
- **运行 Benchmark**: `docker compose exec backend uv run python -m scripts.run_benchmark`

---

## 17. 环境架构冻结 (Environment Freeze)
- **Python**: 3.11.x (Debian Slim)
- **uv**: 0.4.0 (ghcr.io/astral-sh/uv:0.4.0)
- **Postgres**: 15 (pgvector/pgvector:pg15)
- **Venv Path**: `/opt/venv`
- **Network**: Service-name based (`postgres:5432`)
- **Persistence**: Named volume `pgdata`

---

## 18. 原规划 → 新规划迁移说明 (V1.1 修订版)
| 原规划               | 新规划                           | 处理方式    |
| ----------------- | ----------------------------- | ------- |
| Host Python 3.11  | Backend Container Python 3.11 | 删除宿主机依赖 |
| venv              | Container /opt/venv           | 隔离挂载冲突 |
| pip               | uv 0.4.0                      | 版本锚定    |
| requirements.txt  | pyproject.toml + uv.lock      | 现代依赖管理 |
| Dev Compose       | Dev (Mount) + Prod (Immutable)| 分离配置    |
| Healthcheck       | Backend /api/v1/healthz       | 新增稳定性保障 |

# Sprint 0 → Python Import → Dependency Matrix

| Sprint 0 能力 | 典型 Python Import / 用途 | pyproject 依赖 | 决策 |
|---|---|---|---|
| FastAPI API | `from fastapi import FastAPI` | fastapi | 保留 |
| ASGI 服务 | `import uvicorn` | uvicorn[standard] | 保留 |
| 配置模型 | `from pydantic import ...` | pydantic | 保留 |
| 环境配置 | `from pydantic_settings import ...` | pydantic-settings | 保留 |
| ORM / DB Model | `from sqlalchemy import ...` | sqlalchemy | 保留 |
| PostgreSQL Driver | `import psycopg` / SQLAlchemy PostgreSQL URL | psycopg[binary] | 保留；Sprint 0 不同时引入 asyncpg/psycopg2 |
| Vector 类型/操作 | `from pgvector...` | pgvector | 保留 |
| `.env` | `from dotenv import load_dotenv` | python-dotenv | 保留 |
| YAML 配置/规则 | `import yaml` | pyyaml | 保留 |
| Benchmark CSV / 表格处理 | `import pandas as pd` | pandas | 保留 |
| Golden Case / JSON Schema 校验 | `import jsonschema` | jsonschema | 保留 |
| HTTP 客户端/测试 | `import httpx` | httpx | 保留 |
| 测试 | `import pytest` | pytest | dev |
| Async 测试支持 | `import pytest_asyncio` | pytest-asyncio | dev；若实际测试全为同步，可后续删除 |
| 覆盖率 | pytest-cov plugin | pytest-cov | dev |
| Lint / format | `ruff` CLI | ruff | dev |

## 明确暂不纳入 Sprint 0

| 依赖 | 决策 | 原因 |
|---|---|---|
| asyncpg | 删除/不加入 | 当前计划未要求 Async SQLAlchemy；避免与 psycopg 重复 |
| psycopg2-binary | 删除/不加入 | psycopg 3 已作为 PostgreSQL Driver；避免双驱动 |
| alembic | 删除/不加入 | Sprint 0 以 `002_schema.sql` 为 DB Schema Source of Truth，不建立第二套 migration source |
| numpy | 删除/不加入 | 当前 Sprint 0 计划没有明确直接使用 |
| langgraph | 删除/延后 | AI Review 不属于 Sprint 0 Rule Only Benchmark 核心路径 |
| langchain-core | 删除/延后 | 同上 |
| openai | 删除/延后 | Sprint 0 不做完整 AI Review 效能评估 |
| jinja2 | 删除/延后 | 当前计划未形成明确 import 需求 |
| black | 删除 | Ruff 已承担 lint/format；避免重复工具 |
| mypy | 删除 | Sprint 0 验收未要求类型检查 |
| coverage | 删除 | pytest-cov 已提供覆盖率能力；无需重复直接依赖 |

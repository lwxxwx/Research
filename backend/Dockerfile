# ============================================================================
# Stage 1: uv binary - 固定版本构建工具
# ============================================================================
FROM ghcr.io/astral-sh/uv:0.12.6 AS uv_bin

# ============================================================================
# Stage 2: Runtime - 基础运行环境（轻量、安全、统一配置）
# ============================================================================
FROM python:3.11-slim AS runtime

# 全局安全/性能环境变量
# - uv 配置：硬链接跨阶段失效 → UV_LINK_MODE=copy
# - S2-H 修复：镜像源 + 超时 + 重试，缓解 files.pythonhosted.org 下载超时
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=5 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    APP_HOME=/app

# 最小化系统依赖：ca-certificates 用于 HTTPS 出站调用
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# ============================================================================
# Stage 3: Builder - 依赖构建阶段
# ============================================================================
FROM runtime AS builder

COPY --from=uv_bin /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./

# /opt/venv：完整安装（含 dev，本地开发跑 pytest 用）
RUN uv sync --frozen --python python3.11 --extra dev

# /opt/venv_prod：生产用（剔除 dev）—— 保留，供生产 compose 使用
RUN bash -c "UV_PROJECT_ENVIRONMENT=/opt/venv_prod uv sync --frozen --no-dev --python python3.11"

# 拷贝业务源码
COPY . .

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /opt/venv /opt/venv_prod ${APP_HOME}

# ============================================================================
# Stage 4: Final - 运行镜像
# ============================================================================
FROM runtime AS final

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN mkdir -p /home/appuser/.cache/uv /home/appuser/.cache/pip && chown -R appuser:appuser /home/appuser

# [S2-H 修复] 开发镜像拷含 dev 的 venv（pytest 可用）
# 生产构建若需精简，用单独的生产 compose 覆盖此行为或换 Dockerfile target。
COPY --from=builder /opt/venv /opt/venv

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /app ${APP_HOME}

RUN python -c "import sys; print(f'✅ Python venv ok: {sys.executable}')"
RUN python -c "import fastapi, sqlalchemy, pydantic; print('✅ Core dependencies verified')"
# [S2-H 修复] 构建期校验 pytest 存在，缺了直接失败
RUN python -c "import pytest; print(f'✅ pytest {pytest.__version__} present')"

RUN test -d ${APP_HOME} || (echo "❌ Application code missing in ${APP_HOME}!" && exit 1)

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/healthz')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
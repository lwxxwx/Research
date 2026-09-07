# ============================================================================
# Stage 1: uv binary - 固定版本构建工具
# ============================================================================
FROM ghcr.io/astral-sh/uv:0.12.6 AS uv_bin

# ============================================================================
# Stage 2: Runtime - 基础运行环境（轻量、安全、统一配置）
# ============================================================================
FROM python:3.11-slim AS runtime

# 全局安全/性能环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    	PYTHONUNBUFFERED=1 \
    	PYTHONHASHSEED=random \
    	# uv 配置（解决硬链接跨阶段失效问题）
    	UV_PROJECT_ENVIRONMENT=/opt/venv \
    	UV_LINK_MODE=copy \
    	# Python 路径优化
   	PATH="/opt/venv/bin:$PATH" \
    	PYTHONPATH=/app \
    	# 应用统一目录
    	APP_HOME=/app

# 最小化系统依赖：ca‑certificates 用于 HTTPS 出站调用，无则可删除
RUN apt-get update && apt-get install -y --no-install-recommends \
   	ca-certificates \
    	&& rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# ============================================================================
# Stage 3: Builder - 依赖构建阶段（分层缓存优化，预置权限）
# ============================================================================
FROM runtime AS builder

COPY --from=uv_bin /uv /usr/local/bin/uv

# 先拷贝依赖文件，最大化 docker layer 缓存命中率
COPY pyproject.toml uv.lock README.md ./
#COPY pyproject.toml uv.lock ./
# 严格锁定版本、剔除开发依赖、指定 Python 版本（生产最小化）
#RUN uv sync --frozen --no-dev --python python3.11
# ========== 1. /opt/venv：完整安装【包含dev依赖】，builder镜像本地使用（开发跑pytest） ==========
RUN uv sync --frozen --python python3.11 --extra dev

# 👉 为final生产镜像重新构建一份剔除dev的venv，放到/opt/venv_prod
#RUN uv sync --frozen --no-dev --python python3.11 --project . /opt/venv_prod
# ========== 2. /opt/venv_prod：生产用虚拟环境，剔除dev依赖 ==========
# 子shell，临时修改UV_PROJECT_ENVIRONMENT，输出到 /opt/venv_prod
RUN bash -c "UV_PROJECT_ENVIRONMENT=/opt/venv_prod uv sync --frozen --no-dev --python python3.11"

# 拷贝业务源码
COPY . .

# 预置用户权限（builder 阶段执行，避免 final 增加镜像层）
# 注意：chown -R 会遍历所有文件，但仅在 builder 中间镜像执行，不影响最终镜像大小
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    	&& chown -R appuser:appuser /opt/venv ${APP_HOME}

# ============================================================================
# Stage 4: Final - 生产运行镜像（极致精简，非 root 运行）
# ============================================================================
FROM runtime AS final

# 创建非 root 用户（final 阶段必须重新创建，因为用户信息不跨阶段继承）
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 【MODIFIED】新增：预创建uv/pip缓存目录，消除权限告警
RUN mkdir -p /home/appuser/.cache/uv /home/appuser/.cache/pip && chown -R appuser:appuser /home/appuser

# 拷贝构建产物
# ⚠️ 重要：COPY --from 的源路径必须使用字面常量，不能用变量
# 因为源路径在当前阶段解析，而非源阶段，避免未来 APP_HOME 变更导致拷贝失败
#COPY --from=builder /opt/venv /opt/venv
# ========== 3. /opt/venv_prod：生产用虚拟环境，剔除dev依赖 ==========
COPY --from=builder /opt/venv_prod /opt/venv # 使用不带dev依赖的虚拟环境
# 【MODIFIED】新增：把uv二进制从builder拷贝进final镜像
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /app ${APP_HOME}

# 验证虚拟环境完整性（轻量校验）
RUN python -c "import sys; print(f'✅ Python venv ok: {sys.executable}')"

# 核心依赖强校验（失败时自动输出 ImportError 堆栈，CI 日志清晰可见）
# ⚠️ 强校验模式：核心库缺失直接终止构建；项目快速迭代可改为告警不阻断模式
# 注意：项目核心依赖变更时，请同步更新下方 import 列表
RUN python -c "import fastapi, sqlalchemy, pydantic; print('✅ Core dependencies verified')"

# 防御性检查：确保源码目录存在
RUN test -d ${APP_HOME} || (echo "❌ Application code missing in ${APP_HOME}!" && exit 1)

# 可选：校验服务入口文件，项目路径变更需同步修改
# RUN test -f ${APP_HOME}/app/main.py || (echo "❌ Missing app/main.py entry file!" && exit 1)

# ⚠️ 如果业务需要运行期写入日志/缓存文件，请在builder阶段预先创建对应目录并赋权appuser
# 切换到非 root 用户（用户已在 final 阶段创建）
USER appuser

# 健康检查（纯 Python urllib 实现，不依赖 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/healthz')" || exit 1

EXPOSE 8000
# 启动命令（依托 PATH 环境变量，直接调用 uvicorn）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

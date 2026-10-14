"""Alembic 迁移执行入口。

Sprint 1 起，Schema 唯一事实来源为 Alembic migration。
连接串从环境变量 DATABASE_URL 读取，不写死在 alembic.ini。
"""
from logging.config import fileConfig
import os

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from app.persistence.models import Base

# Alembic Config 对象
config = context.config

# 日志配置（alembic.ini 中 [loggers] 段）
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 迁移目标 metadata：指向 ORM 的 Base.metadata
target_metadata = Base.metadata


def _get_url() -> str:
    """从环境变量读取数据库连接串。

    Sprint 0 §7 约定：容器内配置以 compose environment 注入为准，
    .env 仅供 compose 在宿主机读取，不挂载进容器。
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL 环境变量未设置，无法执行 Alembic 迁移。"
            "请确认 docker compose 已注入该变量。"
        )
    return url


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连数据库。"""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连数据库执行迁移。"""
    configuration = config.get_section(config.config_ini_section, {})
    # 关键：显式传 url，避免从 alembic.ini 读 sqlalchemy.url（已被注释掉）
    configuration["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
# tests/test_core.py
import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import Settings, settings
from app.core.errors import BusinessException
from app.core.logging import setup_logging

client = TestClient(app)


# ============================================================================
# 1. 健康检查
# ============================================================================

def test_healthz_endpoint():
    resp = client.get("/api/v1/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["env"] == settings.app_env
    assert body["version"] == "sprint0-v1.0"


# ============================================================================
# 2. 配置加载（冒烟）
# ============================================================================

def test_config_load():
    assert settings.database_url is not None
    assert settings.data_dir is not None
    assert settings.llm_provider is not None


def test_is_dev_property():
    # is_dev 应与 app_env 一致
    assert settings.is_dev == (settings.app_env.lower() == "dev")


# ============================================================================
# 3. 配置校验器（用 Settings(_env_file=None) 隔离 .env）
# ============================================================================

def _set_required_env(monkeypatch, **overrides):
    """设置所有必填环境变量，overrides 可覆盖单项"""
    base = {
        "APP_ENV": "dev",
        "SECRET_KEY": "x" * 32,
        "DATABASE_URL": "postgresql+psycopg://u:p@postgres:5432/db",
        "DATA_DIR": "/data",
        "LLM_PROVIDER": "mock",
        "LLM_API_KEY": "k",
    }
    base.update(overrides)
    for k, v in base.items():
        monkeypatch.setenv(k, v)


def test_secret_key_prod_rejects_placeholder(monkeypatch):
    _set_required_env(
        monkeypatch,
        APP_ENV="production",
        SECRET_KEY="please_replace_this_secret_key_xxxx",
    )
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_secret_key_prod_accepts_real_key(monkeypatch):
    _set_required_env(
        monkeypatch,
        APP_ENV="production",
        SECRET_KEY="a_real_production_secret_key_value",
    )
    s = Settings(_env_file=None)
    assert s.secret_key == "a_real_production_secret_key_value"


def test_database_url_rejects_localhost(monkeypatch):
    _set_required_env(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg://u:p@localhost:5432/db",
    )
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_database_url_rejects_127(monkeypatch):
    _set_required_env(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg://u:p@127.0.0.1:5432/db",
    )
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_database_url_accepts_postgres_host(monkeypatch):
    _set_required_env(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg://u:p@postgres:5432/db",
    )
    s = Settings(_env_file=None)
    assert "postgres" in s.database_url


# ============================================================================
# 4. BusinessException 本体
# ============================================================================

def test_business_exception_super_init():
    exc = BusinessException(400, "参数错误")
    assert exc.code == 400
    assert exc.message == "参数错误"
    # 验证 super().__init__(message) 生效
    assert str(exc) == "参数错误"
    assert exc.args == ("参数错误",)


# ============================================================================
# 5. 异常处理器
# ============================================================================

def _remove_route(path: str):
    app.router.routes = [
        r for r in app.router.routes
        if getattr(r, "path", None) != path
    ]


def test_business_exception_handler():
    path = "/__test__/biz-error"

    @app.get(path)
    async def _raise_biz():
        raise BusinessException(400, "业务出错")

    try:
        resp = client.get(path)
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == 400
        assert body["message"] == "业务出错"
    finally:
        _remove_route(path)


def test_unhandled_exception_handler():
    path = "/__test__/unhandled-error"

    @app.get(path)
    async def _raise_unhandled():
        raise RuntimeError("boom")

    try:
        # 关键：关闭 raise_server_exceptions，才能拿到 500 响应
        local_client = TestClient(app, raise_server_exceptions=False)
        resp = local_client.get(path)
        assert resp.status_code == 500
        assert resp.json() == {"error": 500, "message": "internal server error"}
    finally:
        _remove_route(path)


# ============================================================================
# 6. 日志配置
# ============================================================================

def test_setup_logging_no_duplicate_handler():
    root = logging.getLogger()

    setup_logging()
    first = len(root.handlers)

    setup_logging()
    second = len(root.handlers)

    # 调用两次，handler 数量不应增加
    assert second == first


def test_setup_logging_sets_level():
    setup_logging()
    root = logging.getLogger()
    assert root.level == logging.getLevelName(settings.log_level.upper())


def test_json_log_format_and_rename():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    from pythonjsonlogger import jsonlogger
    handler.setFormatter(jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        #    "message": "message",
        },
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    ))

    lg = logging.getLogger("test.json")
    lg.handlers = [handler]
    lg.setLevel(logging.INFO)
    lg.propagate = False

    lg.info("hello")

    data = json.loads(stream.getvalue().strip())
    assert data["level"] == "INFO"
    assert data["logger"] == "test.json"
    assert data["message"] == "hello"
    assert "timestamp" in data
    # 时区格式：+0800 / +0000 等
    assert data["timestamp"][-5] in "+-"
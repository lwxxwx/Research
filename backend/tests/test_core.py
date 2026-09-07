# tests/test_core.py
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_healthz_endpoint():
    resp = client.get("/api/v1/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_config_load():
    # 验证关键配置字段成功加载
    assert settings.database_url is not None
    assert settings.data_dir is not None
    assert settings.llm_provider is not None

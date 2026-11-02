# backend/tests/test_case_benchmark.py
"""F07 Case 管理测试。"""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.persistence.db import get_db_session
from app.persistence.models import CaseBenchmarkResult


@pytest.fixture()
def client():
    return TestClient(app)


def test_get_case_detail_200(client):
    """GET /cases/{case_id} 返回资产预览。"""
    resp = client.get("/api/v1/cases/case001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["case_id"] == "case001"
    assert data["info_yaml"] is not None
    assert data["expected_review_json"] is not None


def test_get_case_detail_404(client):
    """GET /cases/{case_id} 不存在 → 404。"""
    resp = client.get("/api/v1/cases/nonexistent_case")
    assert resp.status_code == 404


def test_run_benchmark_rule_only(client):
    """POST /cases/{case_id}/benchmark mode=rule_only 成功。"""
    resp = client.post(
        "/api/v1/cases/case001/benchmark",
        json={"mode": "rule_only"},
    )
    assert resp.status_code == 200, f"实际 {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["case_id"] == "case001"
    assert data["mode"] == "rule_only"
    assert data["status"] == "done"
    assert data["tp"] >= 0
    assert data["fp"] >= 0
    assert data["fn"] >= 0

    # 清理
    with get_db_session() as db:
        db.query(CaseBenchmarkResult).filter(
            CaseBenchmarkResult.id == data["id"]
        ).delete()
        db.commit()


def test_run_benchmark_invalid_mode(client):
    """POST /cases/{case_id}/benchmark mode 非法 → 422（Pydantic Literal 校验）。"""
    resp = client.post(
        "/api/v1/cases/case001/benchmark",
        json={"mode": "invalid_mode"},
    )
    assert resp.status_code == 422


def test_run_benchmark_nonexistent_case(client):
    """POST /cases/{case_id}/benchmark Case 不存在 → 400。"""
    resp = client.post(
        "/api/v1/cases/nonexistent_case/benchmark",
        json={"mode": "rule_only"},
    )
    assert resp.status_code == 400


def test_list_case_benchmarks(client):
    """GET /cases/{case_id}/benchmarks 返回历史结果。"""
    # 先跑一次 benchmark
    run_resp = client.post(
        "/api/v1/cases/case001/benchmark",
        json={"mode": "rule_only"},
    )
    assert run_resp.status_code == 200

    resp = client.get("/api/v1/cases/case001/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1

    # 清理
    with get_db_session() as db:
        db.query(CaseBenchmarkResult).filter(
            CaseBenchmarkResult.case_id == "case001"
        ).delete()
        db.commit()
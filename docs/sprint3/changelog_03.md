# ============================================================
# G3-D: rule_candidates 负面用例（R2 要求：无 header / 非法整数 / 角色 403）
# ============================================================

def test_rule_candidates_no_auth_401(client_no_auth):
    """无 X-User-Id 访问 /rule-candidates → 401。"""
    resp = client_no_auth.get("/api/v1/rule-candidates")
    assert resp.status_code == 401


def test_rule_candidates_invalid_uid_401(client_with_invalid_uid):
    """非法整数 X-User-Id 访问 /rule-candidates → 401。"""
    resp = client_with_invalid_uid.get("/api/v1/rule-candidates")
    assert resp.status_code == 401


def test_rule_candidates_engineer_403(client_as_engineer):
    """engineer 访问 /rule-candidates → 403。"""
    resp = client_as_engineer.get("/api/v1/rule-candidates")
    assert resp.status_code == 403


def test_rule_candidates_invalid_status_422(client, test_admin_user):
    """admin 传非法 status 调 /status → 422（G3-D 错误语义统一）。"""
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleCandidate

    cid = f"RC-DTEST-{uuid.uuid4().hex[:8]}"
    with get_db_session() as db:
        db.add(RuleCandidate(
            candidate_id=cid, title="test", description="test",
            status="proposed", proposed_yaml="rule_id: TEST\n",
        ))
        db.commit()

    try:
        resp = client.post(
            f"/api/v1/rule-candidates/{cid}/status",
            json={"status": "invalid_status"},
        )
        # 注意：UpdateCandidateStatusRequest 的 status 是 Literal["accepted","rejected"]，
        # Pydantic 校验会先拦（422），走不到 service 的 BusinessValueError。
        # 这条测试断言的是"非法 status → 422"，不管是 Pydantic 还是 BusinessValueError。
        assert resp.status_code == 422
    finally:
        with get_db_session() as db:
            db.query(RuleCandidate).filter(RuleCandidate.candidate_id == cid).delete()
            db.commit()
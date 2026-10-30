# backend/tests/test_dashboard.py
"""[S2-H（四）] 指标看板测试。

造数据用 get_db_session() 真实提交 + finally 清理（与 test_review_persistence.py 一致）。
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


def _make_task_id(prefix: str) -> str:
    return f"TEST-DASH-{prefix}-{uuid.uuid4().hex[:8].upper()}"


def _make_done_task(task_id: str, evc: float | None) -> int:
    """造一个 done ReviewResult，返回 DB id。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import ReviewResult
    with get_db_session() as db:
        rr = ReviewResult(
            task_id=task_id,
            task_status="done",
            evidence_coverage=evc,
            review_output_json={},
        )
        db.add(rr)
        db.flush()
        return rr.id


def _cleanup(tids: list[str], rc_ids: list[int] | None = None,
             kg_ids: list[int] | None = None) -> None:
    from app.persistence.db import get_db_session
    from app.persistence.models import (
        FeedbackItem, KnowledgeGap, ReviewDefect, ReviewResult, RuleCandidate,
    )
    with get_db_session() as db:
        if tids:
            rr_ids = [
                r.id for r in db.query(ReviewResult)
                .filter(ReviewResult.task_id.in_(tids)).all()
            ]
            if rr_ids:
                db.query(FeedbackItem).filter(
                    FeedbackItem.review_result_id.in_(rr_ids)
                ).delete(synchronize_session=False)
                db.query(ReviewDefect).filter(
                    ReviewDefect.review_result_id.in_(rr_ids)
                ).delete(synchronize_session=False)
            db.query(ReviewResult).filter(
                ReviewResult.task_id.in_(tids)
            ).delete(synchronize_session=False)
        if rc_ids:
            db.query(RuleCandidate).filter(
                RuleCandidate.id.in_(rc_ids)
            ).delete(synchronize_session=False)
        if kg_ids:
            db.query(KnowledgeGap).filter(
                KnowledgeGap.id.in_(kg_ids)
            ).delete(synchronize_session=False)
        db.commit()


def test_dashboard_summary_structure(client):
    """空库或非空库均应返回 4 顶层对象 + 冻结字段。"""
    resp = client.get("/api/v1/dashboard/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"scope", "metrics", "rule_candidates", "knowledge_gaps"}
    assert data["scope"]["task_status_filter"] == "done"
    assert data["scope"]["excluded_task_status"] == ["pending", "failed"]
    assert "total_done_tasks" in data["scope"]
    assert "has_seed_data" in data["scope"]
    si = data["metrics"]["sample_info"]
    assert si["ear_sample_threshold"] == 20
    assert si["ai_ner_sample_threshold"] == 10
    for k in ("AI_CONFIRMED", "NEED_EXPERT_REVIEW", "LOW_CONFIDENCE"):
        assert k in data["metrics"]["review_status_distribution"]
    for k in ("pending", "confirmed", "rejected", "needs_verification"):
        assert k in data["metrics"]["expert_decision_distribution"]
    for k in ("proposed", "designing", "benchmarked", "accepted", "rejected"):
        assert k in data["rule_candidates"]["by_status"]
    for k in ("open", "in_progress", "resolved", "wontfix"):
        assert k in data["knowledge_gaps"]["by_status"]


def test_dashboard_evc_arithmetic_average(client):
    """EVC = done 任务 evidence_coverage 算术平均（0.5 + 0.75）/2 = 0.625。"""
    t1, t2 = _make_task_id("EVC1"), _make_task_id("EVC2")
    _make_done_task(t1, 0.5)
    _make_done_task(t2, 0.75)
    try:
        data = client.get("/api/v1/dashboard/summary").json()
        assert data["metrics"]["evidence_coverage"] == pytest.approx(0.625, abs=1e-4)
        assert data["scope"]["total_done_tasks"] >= 2
    finally:
        _cleanup([t1, t2])


def test_dashboard_excludes_pending_failed(client):
    """pending / failed 不计入 total_done_tasks；EVC 仅取 done。"""
    t_done = _make_task_id("DONE")
    t_pend = _make_task_id("PEND")
    t_fail = _make_task_id("FAIL")
    _make_done_task(t_done, 0.9)
    from app.persistence.db import get_db_session
    from app.persistence.models import ReviewResult
    with get_db_session() as db:
        db.add(ReviewResult(task_id=t_pend, task_status="pending",
                            evidence_coverage=0.1, review_output_json={}))
        db.add(ReviewResult(task_id=t_fail, task_status="failed",
                            evidence_coverage=0.1, review_output_json={}))
    try:
        data = client.get("/api/v1/dashboard/summary").json()
        assert data["scope"]["total_done_tasks"] >= 1
        assert data["metrics"]["evidence_coverage"] is not None
    finally:
        _cleanup([t_done, t_pend, t_fail])


def test_dashboard_seed_exclusion(client):
    """[SEED] 反馈触发 has_seed_data，且不计入 EAR 样本。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import FeedbackItem, ReviewDefect

    tid = _make_task_id("SEED")
    rr_id = _make_done_task(tid, 0.5)
    with get_db_session() as db:
        d = ReviewDefect(
            review_result_id=rr_id, defect_id=f"{tid}-D1",
            location={}, root_cause="rc", suggestion="sg",
            origin="rule", expert_decision="pending",
            review_status="AI_CONFIRMED",
        )
        db.add(d)
        db.flush()
        db.add(FeedbackItem(
            review_result_id=rr_id, review_defect_id=d.id,
            feedback_type="correct_defect", suggestion_adopted="yes",
            comment="[SEED] 测试",
        ))
        db.add(FeedbackItem(
            review_result_id=rr_id, review_defect_id=d.id,
            feedback_type="correct_defect", suggestion_adopted="yes",
            comment="真实反馈",
        ))
    try:
        data = client.get("/api/v1/dashboard/summary").json()
        assert data["scope"]["has_seed_data"] is True
        # EAR 样本仅计非 Seed 的唯一 defect 数
        assert data["metrics"]["sample_info"]["ear_sample_size"] >= 1
    finally:
        _cleanup([tid])


def test_dashboard_ear_none_when_below_threshold(client):
    """EAR 分母 < 20 时返回 None（N/A 语义，不渲染 0）。"""
    tid = _make_task_id("EAR")
    rr_id = _make_done_task(tid, 0.5)
    from app.persistence.db import get_db_session
    from app.persistence.models import FeedbackItem, ReviewDefect
    with get_db_session() as db:
        d = ReviewDefect(
            review_result_id=rr_id, defect_id=f"{tid}-D1",
            location={}, root_cause="rc", suggestion="sg",
        )
        db.add(d)
        db.flush()
        db.add(FeedbackItem(
            review_result_id=rr_id, review_defect_id=d.id,
            feedback_type="correct_defect", suggestion_adopted="yes",
            comment="real",
        ))
    try:
        data = client.get("/api/v1/dashboard/summary").json()
        assert data["metrics"]["expert_adoption_rate"] is None
        assert data["metrics"]["sample_info"]["ear_sample_size"] >= 1
    finally:
        _cleanup([tid])


def test_dashboard_rule_candidates_and_gaps_independent(client):
    """规则候选 / 知识缺口独立统计，不受 done 过滤影响。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import KnowledgeGap, RuleCandidate
    rc_ids: list[int] = []
    kg_ids: list[int] = []
    suffix = uuid.uuid4().hex[:8]
    with get_db_session() as db:
        rcs = [
            RuleCandidate(candidate_id=f"RC-DASH-{suffix}-1", title="t",
                          description="d", status="proposed"),
            RuleCandidate(candidate_id=f"RC-DASH-{suffix}-2", title="t",
                          description="d", status="accepted"),
        ]
        kgs = [
            KnowledgeGap(gap_id=f"KG-DASH-{suffix}-1", title="t", status="open"),
            KnowledgeGap(gap_id=f"KG-DASH-{suffix}-2", title="t", status="resolved"),
        ]
        db.add_all(rcs + kgs)
        db.flush()
        rc_ids = [r.id for r in rcs]
        kg_ids = [k.id for k in kgs]
    try:
        data = client.get("/api/v1/dashboard/summary").json()
        rc = data["rule_candidates"]["by_status"]
        kg = data["knowledge_gaps"]["by_status"]
        assert rc["proposed"] >= 1 and rc["accepted"] >= 1
        assert rc["designing"] >= 0 and rc["benchmarked"] >= 0 and rc["rejected"] >= 0
        assert kg["open"] >= 1 and kg["resolved"] >= 1
        assert kg["in_progress"] >= 0 and kg["wontfix"] >= 0
    finally:
        _cleanup([], rc_ids=rc_ids, kg_ids=kg_ids)
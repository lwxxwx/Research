# backend/app/services/dashboard_service.py
"""指标看板聚合服务（S2-H（四）新增）。

【会话管理】
  与 review_tasks.py / task_service 同架构：本服务内部用 get_db_session() 自开会话，
  API 层不注入 Session。
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from app.persistence.db import get_db_session
from app.persistence.models import (
    FeedbackItem, KnowledgeGap, ReviewDefect, ReviewResult, RuleCandidate,
)
from app.services.benchmark_service import (
    compute_real_ai_ner, compute_real_ear, compute_sample_info,
)

_REVIEW_STATUS_KEYS = ["AI_CONFIRMED", "NEED_EXPERT_REVIEW", "LOW_CONFIDENCE"]
_EXPERT_DECISION_KEYS = ["pending", "confirmed", "rejected", "needs_verification"]
_RULE_CANDIDATE_KEYS = ["proposed", "designing", "benchmarked", "accepted", "rejected"]
_KNOWLEDGE_GAP_KEYS = ["open", "in_progress", "resolved", "wontfix"]


def _avg(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _is_seed(comment: Optional[str]) -> bool:
    return bool(comment) and comment.startswith("[SEED]")


def _count_by_status(rows: list[Any], allowed_keys: list[str]) -> dict[str, int]:
    counter = {k: 0 for k in allowed_keys}
    for r in rows:
        status = getattr(r, "status", None)
        if status in counter:
            counter[status] += 1
    return counter


def _empty_response(rc_rows, kg_rows) -> dict[str, Any]:
    return {
        "scope": {
            "task_status_filter": "done",
            "excluded_task_status": ["pending", "failed"],
            "total_done_tasks": 0,
            "has_seed_data": False,
        },
        "metrics": {
            "evidence_coverage": None,
            "ai_new_effective_rate": None,
            "expert_adoption_rate": None,
            "review_status_distribution": {k: 0 for k in _REVIEW_STATUS_KEYS},
            "expert_decision_distribution": {k: 0 for k in _EXPERT_DECISION_KEYS},
            "sample_info": {
                "ear_sample_size": 0, "ear_sample_threshold": 20,
                "ai_ner_sample_size": 0, "ai_ner_sample_threshold": 10,
            },
        },
        "rule_candidates": {
            "total": len(rc_rows),
            "by_status": _count_by_status(rc_rows, _RULE_CANDIDATE_KEYS),
        },
        "knowledge_gaps": {
            "total": len(kg_rows),
            "by_status": _count_by_status(kg_rows, _KNOWLEDGE_GAP_KEYS),
        },
    }


def get_dashboard_summary() -> dict[str, Any]:
    """聚合全局指标看板数据（内部自开 Session）。"""
    with get_db_session() as db:
        return _get_summary_with_db(db)


def _get_summary_with_db(db: Session) -> dict[str, Any]:
    rc_rows = db.query(RuleCandidate).all()
    kg_rows = db.query(KnowledgeGap).all()

    done_results = (
        db.query(ReviewResult).filter(ReviewResult.task_status == "done").all()
    )
    if not done_results:
        return _empty_response(rc_rows, kg_rows)

    done_ids = [r.id for r in done_results]
    defects = (
        db.query(ReviewDefect)
        .filter(ReviewDefect.review_result_id.in_(done_ids)).all()
    )
    feedbacks = (
        db.query(FeedbackItem)
        .filter(FeedbackItem.review_result_id.in_(done_ids)).all()
    )

    defect_db_to_biz = {d.id: d.defect_id for d in defects}

    defect_dicts = [
        {
            "defect_id": d.defect_id,
            "origin": d.origin,
            "expert_decision": d.expert_decision,
            "review_status": d.review_status,
        }
        for d in defects
    ]
    # 键名严格对齐 compute_real_ear 源码读取：comment / feedback_type /
    # defect_id / created_at / id / suggestion_adopted
    feedback_dicts = [
        {
            "defect_id": defect_db_to_biz.get(f.review_defect_id),
            "feedback_type": f.feedback_type,
            "suggestion_adopted": f.suggestion_adopted,
            "comment": f.comment,
            "created_at": f.created_at.isoformat() if f.created_at else None,
            "id": f.id,
        }
        for f in feedbacks
    ]

    evc_avg = _avg(
        [r.evidence_coverage for r in done_results if r.evidence_coverage is not None]
    )
    ear = compute_real_ear(feedback_dicts, [])
    ai_ner = compute_real_ai_ner(defect_dicts)
    sample_info = compute_sample_info(feedback_dicts, defect_dicts)

    rs_dist = {k: 0 for k in _REVIEW_STATUS_KEYS}
    for d in defects:
        if d.review_status in rs_dist:
            rs_dist[d.review_status] += 1

    ed_dist = {k: 0 for k in _EXPERT_DECISION_KEYS}
    for d in defects:
        if d.expert_decision in ed_dist:
            ed_dist[d.expert_decision] += 1

    has_seed = any(_is_seed(f.comment) for f in feedbacks)
    if not has_seed:
        orphan = (
            db.query(FeedbackItem)
            .join(ReviewDefect, FeedbackItem.review_defect_id == ReviewDefect.id)
            .filter(
                FeedbackItem.review_result_id.is_(None),
                ReviewDefect.review_result_id.in_(done_ids),
            ).all()
        )
        has_seed = any(_is_seed(f.comment) for f in orphan)

    return {
        "scope": {
            "task_status_filter": "done",
            "excluded_task_status": ["pending", "failed"],
            "total_done_tasks": len(done_results),
            "has_seed_data": has_seed,
        },
        "metrics": {
            "evidence_coverage": evc_avg,
            "ai_new_effective_rate": ai_ner,
            "expert_adoption_rate": ear,
            "review_status_distribution": rs_dist,
            "expert_decision_distribution": ed_dist,
            "sample_info": sample_info,
        },
        "rule_candidates": {
            "total": len(rc_rows),
            "by_status": _count_by_status(rc_rows, _RULE_CANDIDATE_KEYS),
        },
        "knowledge_gaps": {
            "total": len(kg_rows),
            "by_status": _count_by_status(kg_rows, _KNOWLEDGE_GAP_KEYS),
        },
    }
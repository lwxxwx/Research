# backend/app/services/knowledge_gap_service.py
"""知识缺口 Backlog 服务（F06 新增）。

职责：
  - list_gaps：列表 + 筛选（status / case_id / task_id）
  - get_gap：详情 + 来源追溯（source_feedback + all_feedbacks）
  - update_gap_status：状态流转 + resolved_at 维护
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessValueError
from app.persistence.models import (
    KnowledgeGap, GapFeedbackLink, FeedbackItem,
    ReviewDefect, ReviewResult,
)


_ALLOWED_STATUS = {"open", "in_progress", "resolved", "wontfix"}


def _gap_to_dict(gap: KnowledgeGap) -> dict[str, Any]:
    """基础字段（不含来源追溯）。"""
    return {
        "gap_id": gap.gap_id,
        "title": gap.title,
        "description": gap.description,
        "status": gap.status,
        "case_id": gap.case_id,
        "task_id": gap.task_id,
        "defect_id": gap.defect_id,
        "created_at": gap.created_at.isoformat() if gap.created_at else None,
        "updated_at": gap.updated_at.isoformat() if gap.updated_at else None,
        "resolved_at": gap.resolved_at.isoformat() if gap.resolved_at else None,
        "resolved_by": gap.resolved_by,
    }


def list_gaps(
    db: Session,
    status: str | None = None,
    case_id: str | None = None,
    task_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """列表 + 筛选 + 分页。"""
    if status is not None and status not in _ALLOWED_STATUS:
        raise BusinessValueError(f"status 必须为 {_ALLOWED_STATUS}，实际: {status!r}")
    
    q = db.query(KnowledgeGap)
    if status is not None:
        q = q.filter(KnowledgeGap.status == status)
    if case_id is not None:
        q = q.filter(KnowledgeGap.case_id == case_id)
    if task_id is not None:
        q = q.filter(KnowledgeGap.task_id == task_id)

    total = q.count()
    rows = q.order_by(KnowledgeGap.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [_gap_to_dict(g) for g in rows],
        "limit": limit,
        "offset": offset,
    }


def get_gap(db: Session, gap_id: str) -> dict[str, Any] | None:
    """详情 + 来源追溯（source_feedback + all_feedbacks）。"""
    gap = db.query(KnowledgeGap).filter(KnowledgeGap.gap_id == gap_id).first()
    if gap is None:
        return None

    base = _gap_to_dict(gap)

    # 主来源
    source_feedback = None
    if gap.source_feedback_id:
        fb = db.query(FeedbackItem).filter(
            FeedbackItem.id == gap.source_feedback_id
        ).first()
        if fb:
            source_feedback = _feedback_to_dict(db, fb)

    # 全部关联反馈（含主来源）
    links = db.query(GapFeedbackLink).filter(GapFeedbackLink.gap_id == gap.id).all()
    all_feedbacks = []
    for link in links:
        fb = db.query(FeedbackItem).filter(
            FeedbackItem.id == link.feedback_id
        ).first()
        if fb:
            all_feedbacks.append(_feedback_to_dict(db, fb))

    base["source_feedback"] = source_feedback
    base["all_feedbacks"] = all_feedbacks
    return base


def _feedback_to_dict(db: Session, fb: FeedbackItem) -> dict[str, Any]:
    """反馈 + 反查 task_id / defect_id。"""
    defect_biz_id = None
    if fb.review_defect_id:
        d = db.query(ReviewDefect).filter(ReviewDefect.id == fb.review_defect_id).first()
        if d:
            defect_biz_id = d.defect_id
    task_id_val = None
    if fb.review_result_id:
        rr = db.query(ReviewResult).filter(ReviewResult.id == fb.review_result_id).first()
        if rr:
            task_id_val = rr.task_id
    return {
        "feedback_id": fb.id,
        "feedback_type": fb.feedback_type,
        "expert_suggestion": fb.expert_suggestion,
        "created_at": fb.created_at.isoformat() if fb.created_at else None,
        "task_id": task_id_val,
        "defect_id": defect_biz_id,
    }


def update_gap_status(
    db: Session,
    gap_id: str,
    status: str,
    resolved_by: int | None = None,
) -> dict[str, Any] | None:
    """状态流转 + resolved_at 维护。"""
    if status not in _ALLOWED_STATUS:
        raise BusinessValueError(f"status 必须为 {_ALLOWED_STATUS}，实际: {status!r}")  

    gap = db.query(KnowledgeGap).filter(KnowledgeGap.gap_id == gap_id).first()
    if gap is None:
        return None

    gap.status = status
    if status == "resolved":
        gap.resolved_at = datetime.now(timezone.utc)
        gap.resolved_by = resolved_by
    else:
        gap.resolved_at = None
        gap.resolved_by = None

    db.commit()
    db.refresh(gap)
    return _gap_to_dict(gap)
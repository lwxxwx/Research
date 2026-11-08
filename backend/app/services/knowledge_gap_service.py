# backend/app/services/knowledge_gap_service.py
"""知识缺口 Backlog 服务（F06 新增）。

职责：
  - list_gaps：列表 + 筛选（status / case_id / task_id）
  - get_gap：详情 + 来源追溯（source_feedback + all_feedbacks）
  - update_gap_status：状态流转 + resolved_at 维护
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessValueError
from app.core.ids import make_prefixed_id
from app.persistence.models import (
    KnowledgeGap, GapFeedbackLink, FeedbackItem,
    ReviewDefect, ReviewResult, SchematicCase, FeedbackItem,  # ← 补 SchematicCase
)


logger = logging.getLogger(__name__)              # ← 补 logger


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
    """列表 + 筛选 + 分页。

    D27：proposed_by 运行时 JOIN feedback_item 反查，无 DB 物理列；
    source_feedback_id 为 NULL 或关联 feedback 已软删除时返回 None。
    """
    if status is not None and status not in _ALLOWED_STATUS:
        raise BusinessValueError(f"status 必须为 {_ALLOWED_STATUS}，实际: {status!r}")

    base_q = db.query(KnowledgeGap)
    if status is not None:
        base_q = base_q.filter(KnowledgeGap.status == status)
    if case_id is not None:
        base_q = base_q.filter(KnowledgeGap.case_id == case_id)
    if task_id is not None:
        base_q = base_q.filter(KnowledgeGap.task_id == task_id)

    total = base_q.count()

    # 分页 + outerjoin 带出 proposed_by 所需字段（一次查询）
    rows = (
        base_q
        .outerjoin(FeedbackItem, KnowledgeGap.source_feedback_id == FeedbackItem.id)
        .add_columns(FeedbackItem.created_by, FeedbackItem.is_deleted)
        .order_by(KnowledgeGap.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = []
    for gap, fb_created_by, fb_is_deleted in rows:
        item = _gap_to_dict(gap)
        item["proposed_by"] = (
            fb_created_by if (fb_created_by is not None and not fb_is_deleted) else None
        )
        items.append(item)

    return {
        "total": total,
        "items": items,
        "limit": limit,
        "offset": offset,
    }


def get_gap(db: Session, gap_id: str) -> dict[str, Any] | None:
    """详情 + 来源追溯（source_feedback + all_feedbacks）。

    D27：proposed_by 由 source_feedback 的 created_by 反查；
    source_feedback_id 为 NULL 或关联 feedback 已软删除时返回 None。
    """
    gap = db.query(KnowledgeGap).filter(KnowledgeGap.gap_id == gap_id).first()
    if gap is None:
        return None

    base = _gap_to_dict(gap)

    # 主来源（顺带取 created_by / is_deleted，用于 proposed_by）
    source_feedback = None
    proposed_by = None
    if gap.source_feedback_id:
        fb = db.query(FeedbackItem).filter(
            FeedbackItem.id == gap.source_feedback_id
        ).first()
        if fb:
            source_feedback = _feedback_to_dict(db, fb)
            if not fb.is_deleted:
                proposed_by = fb.created_by
    base["proposed_by"] = proposed_by

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

# ============================================================
# [S3-G5] 从 feedback_item 生成 knowledge_gap（D12–D20）
# ============================================================

GAP_STATUS_OPEN = "open"
GAP_STATUS_IN_PROGRESS = "in_progress"
GAP_STATUS_RESOLVED = "resolved"
GAP_STATUS_WONTFIX = "wontfix"

GAP_TITLE_MAX = 256

_GAP_ID_COLLISION_RETRIES = 3


def _make_gap_id() -> str:
    """生成 gap_id，格式 `GAP-YYYYMMDD-XXXXXX`（6 位随机后缀，ASCII `-`）。"""
    return make_prefixed_id("GAP", random_len=6)


def _summarize_suggestion_diff(diff_json: list | None) -> str:
    """D13 降级第 2 级：suggestion_diff 摘要（JSON 序列化后截断 100 字符）。"""
    if not diff_json:
        return ""
    try:
        return json.dumps(diff_json, ensure_ascii=False)[:100]
    except Exception:
        return ""


def _resolve_gap_context(
    db: Session, fb: FeedbackItem
) -> tuple[str | None, str | None, str | None]:
    """D14：从 feedback_item 反查 task_id / case_id / defect_id（单次查询）。

    返回 (case_id, task_id, defect_id)，反查不到返回 None + logger.warning。
    """
    case_id: str | None = None
    task_id: str | None = None
    defect_id: str | None = None

    if fb.review_result_id is not None:
        row = (
            db.query(ReviewResult, SchematicCase.case_id)
            .outerjoin(
                SchematicCase,
                ReviewResult.schematic_case_id == SchematicCase.id,
            )
            .filter(ReviewResult.id == fb.review_result_id)
            .first()
        )
        if row is not None:
            rr, sc_case_id = row
            task_id = rr.task_id
            case_id = sc_case_id
            if case_id is None:
                logger.warning(
                    "create_gap_from_feedback: case_id 反查失败（review_result_id=%s）",
                    fb.review_result_id,
                )
        else:
            logger.warning(
                "create_gap_from_feedback: review_result_id=%s 不存在",
                fb.review_result_id,
            )

    if fb.review_defect_id is not None:
        rd = (
            db.query(ReviewDefect)
            .filter(ReviewDefect.id == fb.review_defect_id)
            .first()
        )
        if rd is not None:
            defect_id = rd.defect_id
        else:
            logger.warning(
                "create_gap_from_feedback: review_defect_id=%s 不存在",
                fb.review_defect_id,
            )

    return case_id, task_id, defect_id


def create_gap_from_feedback(
    db: Session,
    feedback_id: int,
    *,
    case_id: str | None = None,
    reviewed_by: int | None = None,
) -> str | None:
    """从 knowledge_gap 反馈生成知识缺口 + gap_feedback_link。

    幂等：已关联过直接返回已有 gap_id。
    原子性：**只 flush 不 commit**，由调用方统一 commit。
    """
    # 1. 幂等检查
    existing_link = (
        db.query(GapFeedbackLink)
        .filter(GapFeedbackLink.feedback_id == feedback_id)
        .first()
    )
    if existing_link is not None:
        existing_gap = (
            db.query(KnowledgeGap)
            .filter(KnowledgeGap.id == existing_link.gap_id)
            .first()
        )
        return existing_gap.gap_id if existing_gap else None

    # 2. 查 feedback_item
    fb = db.query(FeedbackItem).filter(FeedbackItem.id == feedback_id).first()
    if fb is None:
        raise ValueError(f"feedback_item id={feedback_id} not found")

    # 3. 反查 task_id / case_id / defect_id
    resolved_case_id, task_id, defect_id = _resolve_gap_context(db, fb)
    final_case_id = case_id or resolved_case_id

    # 4. title 降级链
    title_raw = (
        (fb.expert_suggestion or "").strip()
        or _summarize_suggestion_diff(fb.suggestion_diff_json)
        or f"知识缺口{(' ' + final_case_id) if final_case_id else ''}"
    )
    title = title_raw.split("\n")[0].strip()[:GAP_TITLE_MAX] or "知识缺口"
    description = title_raw

    # 5. gap_id 生成（碰撞重试 3 次）
    gap_id: str | None = None
    for _ in range(_GAP_ID_COLLISION_RETRIES):
        candidate = _make_gap_id()
        if (
            db.query(KnowledgeGap)
            .filter(KnowledgeGap.gap_id == candidate)
            .first()
            is None
        ):
            gap_id = candidate
            break
    if gap_id is None:
        raise RuntimeError(
            f"生成 gap_id 碰撞 {_GAP_ID_COLLISION_RETRIES} 次，feedback_id={feedback_id}"
        )

    # 6. 写 gap → flush → 写 link（不 commit）
    gap = KnowledgeGap(
        gap_id=gap_id,
        source_feedback_id=feedback_id,
        case_id=final_case_id,
        task_id=task_id,
        defect_id=defect_id,
        title=title,
        description=description,
        status=GAP_STATUS_OPEN,
    )
    db.add(gap)
    db.flush()

    db.add(GapFeedbackLink(gap_id=gap.id, feedback_id=feedback_id))
    db.flush()

    logger.info(
        "create_gap_from_feedback: gap_id=%s feedback_id=%s case_id=%s task_id=%s defect_id=%s",
        gap_id, feedback_id, final_case_id, task_id, defect_id,
    )
    return gap_id
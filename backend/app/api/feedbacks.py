# backend/app/api/feedbacks.py
"""反馈路由。
Sprint 3 G3 鉴权：
  - POST /feedbacks → reviewer / admin（不开放 engineer）
审计字段注入（G3‑C）：
  - feedback_item.created_by 由后端从 current_user.id 注入。
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import CurrentUser, ROLE_ADMIN, ROLE_REVIEWER, require_roles
from app.api.schemas import FeedbackCreateAPI, FeedbackResponse
from app.persistence.db import get_db_session
from app.services import feedback_service, task_service
from app.schemas.feedback import FeedbackCreate, FeedbackType
from app.persistence.models import ReviewResult, ReviewDefect, FeedbackItem  # 新增导入


router = APIRouter(prefix="/feedbacks", tags=["feedbacks"])

@router.post("", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackCreateAPI,
    current: CurrentUser = Depends(require_roles(ROLE_REVIEWER, ROLE_ADMIN)),
) -> FeedbackResponse:
    """提交专家反馈。reviewer / admin 可用。
    created_by 后端注入（G3‑C 会改 feedback_service.submit_feedback 签名）。
    """
    with get_db_session() as db:
        # 使用已有对外入口：自动做task_id/defect_id翻译成外键，内部调用FeedbackService
        out = task_service.submit_task_feedback(
            task_id=body.task_id,
            defect_id=body.defect_id,
            feedback_type=FeedbackType(body.feedback_type),
            expert_suggestion=body.expert_suggestion,
            suggestion_adopted=body.suggestion_adopted,
            suggestion_diff=body.suggestion_diff,
            rule_candidate=body.rule_candidate,
            created_by=current.id,
        )

    return FeedbackResponse(
        id=out.id,
        task_id=out.task_id,
        defect_id=out.defect_id,
        review_result_id=out.review_result_id,
        review_defect_id=out.review_defect_id,
        feedback_type=out.feedback_type.value,
        expert_suggestion=out.expert_suggestion,
        suggestion_diff=[d.model_dump() for d in out.suggestion_diff],
        rule_candidate_ref=out.rule_candidate_ref,
        knowledge_gap_ref=out.knowledge_gap_ref,    # ← 加这行
        created_by=out.created_by,
        created_at=out.created_at,
    )




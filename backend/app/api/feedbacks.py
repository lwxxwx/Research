# backend/app/api/feedbacks.py
"""反馈路由（G4 新增）。

【决策 6：FeedbackCreateAPI，传业务标识】
  - API body 含 task_id + defect_id + 6 类反馈字段
  - 端点内翻译为 service.submit_task_feedback 的参数
  - service 抛 FeedbackNotFoundError（继承 ValueError）→ 404

【约束 1：except 顺序】
  FeedbackNotFoundError 继承 ValueError，必须先于 ValueError 捕获。
  当前只有 FeedbackNotFoundError，若将来加 except ValueError，必须放后面。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import FeedbackCreateAPI, FeedbackResponse
from app.schemas.feedback import FeedbackType
from app.services import task_service
from app.services.feedback_service import FeedbackNotFoundError

router = APIRouter(prefix="/feedbacks", tags=["feedbacks"])


@router.post("", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackCreateAPI) -> FeedbackResponse:
    """提交专家反馈。

    【状态码】
      - task_id / defect_id 不存在 → 404（FeedbackNotFoundError）
      - 成功 → 200
    """
    try:
        fb_out = task_service.submit_task_feedback(
            task_id=body.task_id,
            defect_id=body.defect_id,
            # 显式转 FeedbackType，不依赖 Pydantic 隐式转换
            feedback_type=FeedbackType(body.feedback_type),
            expert_suggestion=body.expert_suggestion,
            suggestion_adopted=body.suggestion_adopted,
            suggestion_diff=body.suggestion_diff,
            rule_candidate=body.rule_candidate,
            created_by=body.created_by,
        )
    except FeedbackNotFoundError as exc:
        # 约束 1：FeedbackNotFoundError 先于 ValueError
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dumped = fb_out.model_dump()
    return FeedbackResponse(
        id=dumped["id"],
        task_id=dumped.get("task_id"),
        defect_id=dumped.get("defect_id"),
        review_result_id=dumped.get("review_result_id"),
        review_defect_id=dumped.get("review_defect_id"),
        feedback_type=dumped["feedback_type"],
        expert_suggestion=dumped.get("expert_suggestion"),
        suggestion_diff=dumped.get("suggestion_diff") or [],
        rule_candidate_ref=dumped.get("rule_candidate_ref"),
        created_by=dumped.get("created_by"),
        # 约束 3：直接传 datetime，FastAPI 自动序列化
        created_at=dumped["created_at"],
    )
# backend/app/api/rule_candidates.py
"""规则候选审核路由。

4 端点：
  GET  /api/v1/rule-candidates                     列表（支持 status 过滤）
  GET  /api/v1/rule-candidates/{candidate_id}      详情（含 proposed_yaml）
  POST /api/v1/rule-candidates/{candidate_id}/status   批准 / 驳回
  POST /api/v1/rule-candidates/{candidate_id}/publish  发布到正式规则库（F05）

【关键约束（S2-G 冻结）】
  - accepted 只改状态，不写 data/rules/（人工合入）
  - rejected 必须提供 rejected_reason
  - 不改 status 以外的任何字段

【F05 发布约束】
  - 批准（accepted）与发布（publish）分离，批准不等于发布
  - 方案 A：发布失败返回 200 + publish_status="publish_failed"，状态落库
  - 业务冲突（未批准 / 已发布）→ 409；校验失败（YAML / 规则 ID）→ 422
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    PublishRuleRequest,
    RuleCandidateDetail,
    RuleCandidateListResponse,
    RuleCandidateListItem,
    RulePublishResponse,
    UpdateCandidateStatusRequest,
    UpdateCandidateStatusResponse,
)
from app.persistence.db import get_db_session
from app.services.rule_evolution_service import (
    RuleEvolutionService,
    RulePublishConflictError,
    RulePublishValidationError,
)

router = APIRouter(prefix="/rule-candidates", tags=["rule_candidates"])

_svc = RuleEvolutionService()


# ============================================================
# GET /rule-candidates
# ============================================================
@router.get("", response_model=RuleCandidateListResponse)
def list_candidates(
    status: Literal["proposed", "designing", "benchmarked", "accepted", "rejected"] | None = Query(default=None),
) -> RuleCandidateListResponse:
    """查询候选规则列表（可按 status 过滤）。"""
    with get_db_session() as db:
        items = _svc.get_all_candidates(db, status=status)

    return RuleCandidateListResponse(
        total=len(items),
        items=[RuleCandidateListItem(**it) for it in items],
    )


# ============================================================
# GET /rule-candidates/{candidate_id}
# ============================================================
@router.get("/{candidate_id}", response_model=RuleCandidateDetail)
def get_candidate(candidate_id: str) -> RuleCandidateDetail:
    """查询候选规则详情（含 proposed_yaml 和 benchmark_result）。"""
    with get_db_session() as db:
        detail = _svc.get_candidate(db, candidate_id)

    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"rule_candidate candidate_id={candidate_id} 不存在",
        )

    return RuleCandidateDetail(**detail)


# ============================================================
# POST /rule-candidates/{candidate_id}/status
# ============================================================
@router.post(
    "/{candidate_id}/status",
    response_model=UpdateCandidateStatusResponse,
)
def update_candidate_status(
    candidate_id: str,
    body: UpdateCandidateStatusRequest,
) -> UpdateCandidateStatusResponse:
    """更新候选规则状态（accepted / rejected）。

    - accepted：批准，只改状态，不写 data/rules/
    - rejected：驳回，必须提供 rejected_reason
    """
    if body.status == "rejected" and not body.rejected_reason:
        raise HTTPException(
            status_code=422,
            detail="驳回必须提供 rejected_reason",
        )

    try:
        with get_db_session() as db:
            result = _svc.update_candidate_status(
                db,
                candidate_id=candidate_id,
                status=body.status,
                rejected_reason=body.rejected_reason,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return UpdateCandidateStatusResponse(**result)


# ============================================================
# POST /rule-candidates/{candidate_id}/publish（F05）
# ============================================================
@router.post(
    "/{candidate_id}/publish",
    response_model=RulePublishResponse,
)
def publish_rule_candidate(
    candidate_id: str,
    body: PublishRuleRequest,
) -> RulePublishResponse:
    """发布候选到正式规则库。

    状态机：accepted → publishing → published | publish_failed
    批准（accepted）与发布（publish）分离，批准不等于发布。

    方案 A：发布失败返回 200 + publish_status="publish_failed"，
    状态已落库，可重试（用新的 idempotency_key）。
    """
    try:
        with get_db_session() as db:
            result = _svc.publish_rule_candidate(
                db=db,
                candidate_id=candidate_id,
                idempotency_key=body.idempotency_key,
            )
    except RulePublishConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RulePublishValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RulePublishResponse(**result)
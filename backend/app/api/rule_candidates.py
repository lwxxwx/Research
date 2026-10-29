# backend/app/api/rule_candidates.py
"""规则候选审核路由（S2-G 新增）。

3 端点：
  GET  /api/v1/rule-candidates                     列表（支持 status 过滤）
  GET  /api/v1/rule-candidates/{candidate_id}      详情（含 proposed_yaml）
  POST /api/v1/rule-candidates/{candidate_id}/status   批准 / 驳回

【关键约束（S2-G 冻结）】
  - accepted 只改状态，不写 data/rules/（人工合入）
  - rejected 必须提供 rejected_reason
  - 不改 status 以外的任何字段
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    RuleCandidateDetail,
    RuleCandidateListResponse,
    RuleCandidateListItem,
    UpdateCandidateStatusRequest,
    UpdateCandidateStatusResponse,
)
from app.persistence.db import get_db_session
from app.services.rule_evolution_service import RuleEvolutionService

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
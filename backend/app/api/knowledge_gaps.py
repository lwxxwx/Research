"""知识缺口 Backlog 路由（F06 新增）。"""
from __future__ import annotations
from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session


from app.api.schemas import (
    KnowledgeGapListResponse,
    KnowledgeGapDetail,
    UpdateGapStatusRequest,
)
from app.persistence.db import get_db_session

from app.api.deps import (
    CurrentUser,
    ROLE_ADMIN,
    ROLE_RULE_ADMIN,
    require_roles,
)
from app.services import knowledge_gap_service

router = APIRouter(prefix="/knowledge-gaps", tags=["knowledge_gaps"])


def _get_db():
    with get_db_session() as db:
        yield db


@router.get("", response_model=KnowledgeGapListResponse)
def list_knowledge_gaps(
    status: Literal["open", "in_progress", "resolved", "wontfix"] | None = Query(default=None),
    case_id: str | None = Query(default=None),
    task_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=0),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(_get_db),
    current: CurrentUser = Depends(require_roles(ROLE_RULE_ADMIN, ROLE_ADMIN)),
) -> KnowledgeGapListResponse:
    try:
        result = knowledge_gap_service.list_gaps(
            db, status=status, case_id=case_id, task_id=task_id,
            limit=limit, offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return KnowledgeGapListResponse(**result)


@router.get("/{gap_id}", response_model=KnowledgeGapDetail)
def get_knowledge_gap(
    gap_id: str,
    db: Session = Depends(_get_db),
    current: CurrentUser = Depends(require_roles(ROLE_RULE_ADMIN, ROLE_ADMIN)),
) -> KnowledgeGapDetail:
    result = knowledge_gap_service.get_gap(db, gap_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"gap_id={gap_id} 不存在")
    return KnowledgeGapDetail(**result)


@router.patch("/{gap_id}", response_model=KnowledgeGapDetail)
def update_knowledge_gap(
    gap_id: str,
    body: UpdateGapStatusRequest,
    db: Session = Depends(_get_db),
    current: CurrentUser = Depends(require_roles(ROLE_RULE_ADMIN, ROLE_ADMIN)),
) -> KnowledgeGapDetail:
    try:
        result = knowledge_gap_service.update_gap_status(
            db, gap_id, status=body.status, resolved_by=body.resolved_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f"gap_id={gap_id} 不存在")
    # PATCH 返回详情（含来源追溯）
    full = knowledge_gap_service.get_gap(db, gap_id)
    return KnowledgeGapDetail(**full)
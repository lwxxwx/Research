# backend/app/api/cases.py
"""Case 列表路由（G4 新增）。"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import CaseListResponse
from app.services import task_service

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("", response_model=CaseListResponse)
def list_cases() -> CaseListResponse:
    """列出所有 Case 元信息。"""
    result = task_service.list_cases()
    return CaseListResponse(**result)
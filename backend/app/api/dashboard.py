# backend/app/api/dashboard.py
"""指标看板路由。

Sprint 3 G3 鉴权：
  - GET /dashboard/summary → admin / rule_admin / reviewer
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import (
    CurrentUser,
    ROLE_ADMIN,
    ROLE_REVIEWER,
    ROLE_RULE_ADMIN,
    require_roles,
)
from app.api.schemas import DashboardSummaryResponse
from app.services import dashboard_service


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_summary(
    current: CurrentUser = Depends(
        require_roles(ROLE_ADMIN, ROLE_RULE_ADMIN, ROLE_REVIEWER)
    ),
) -> DashboardSummaryResponse:
    """全局指标看板。admin / rule_admin / reviewer 可用。"""
    return DashboardSummaryResponse(**dashboard_service.get_dashboard_summary())
# backend/app/api/dashboard.py
"""[S2-H（四）] 指标看板 API。"""
from fastapi import APIRouter

from app.api.schemas import DashboardSummaryResponse
from app.services.dashboard_service import get_dashboard_summary

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_summary() -> DashboardSummaryResponse:
    data = get_dashboard_summary()
    return DashboardSummaryResponse(**data)
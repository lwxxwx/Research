# backend/app/api/healthz.py
"""GET /api/v1/healthz（G4 从 main.py 迁移）。

【响应体契约（Sprint 0 冻结）】
  {status: "ok", env: <settings.app_env>, version: <app.version>}
  与 main.py 原内联版完全一致，不改字段。
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import settings

router = APIRouter(tags=["healthz"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """健康检查。路径和响应体与阶段 A 的 main.py 内联版本一致。"""
    return HealthResponse(
        status="ok",
        env=settings.app_env,
        version="sprint0-v1.0",
    )
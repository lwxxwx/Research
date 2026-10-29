# backend/app/api/__init__.py
"""API 路由汇总（G4 新增）。"""
from __future__ import annotations

from fastapi import APIRouter

from . import cases, feedbacks, healthz, review_tasks
# ===== [S2-G] 规则候选 =====
from . import rule_candidates
# ===== [S2-G END] =====

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(healthz.router)
api_router.include_router(review_tasks.router)
api_router.include_router(cases.router)
api_router.include_router(feedbacks.router)
# ===== [S2-G] 规则候选 =====
api_router.include_router(rule_candidates.router)
# ===== [S2-G END] =====

__all__ = ["api_router"]
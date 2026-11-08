# backend/app/api/__init__.py
"""API 路由汇总（G4 新增）。"""
from __future__ import annotations

from fastapi import APIRouter

from . import cases, feedbacks, healthz, review_tasks
# ===== [S2-G] 规则候选 =====
from . import rule_candidates
# ===== [S2-G END] =====
# ===== [S2-H] 知识库 =====
from . import knowledge_docs
# ===== [S2-H END] =====
# ===== [S2-H（四）] 指标看板 =====
from . import dashboard
# ===== [S2-H（四） END] =====
# ===== [F06] 知识缺口 Backlog =====
from . import knowledge_gaps
# ===== [F06 END] =====
# ===== [S3-G2] Auth / 用户管理 =====
from . import auth
# ===== [S3-G2 END] =====



api_router = APIRouter(prefix="/api/v1")

api_router.include_router(healthz.router)
api_router.include_router(review_tasks.router)
api_router.include_router(cases.router)
api_router.include_router(feedbacks.router)
# ===== [S2-G] 规则候选 =====
api_router.include_router(rule_candidates.router)
# ===== [S2-G END] =====
# ===== [S2-H] 知识库 =====
api_router.include_router(knowledge_docs.router)
# ===== [S2-H END] =====
# ===== [S2-H（四）] 指标看板 =====
api_router.include_router(dashboard.router)
# ===== [S2-H（四） END] =====
# ===== [F06] 知识缺口 Backlog =====
api_router.include_router(knowledge_gaps.router)
# ===== [F06 END] =====
# ===== [S3-G2] Auth / 用户管理 =====
api_router.include_router(auth.router)
# ===== [S3-G2 END] =====

__all__ = ["api_router"]
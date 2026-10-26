# backend/app/api/__init__.py
"""API 路由汇总（G4 新增）。

挂载 7 个端点，统一 prefix="/api/v1"。

【为什么用顶层 import 而不是 build_api_router()】
  - main.py 用 `from app.api import api_router`（顶层版）
  - 保持与 main.py 一致，不引入 build_api_router() 包装
  - 前提：app/api/ 下所有子模块（含 feedbacks.py）必须存在
  - 若将来子模块动态加载（如插件机制），再改 build_api_router()
"""
from __future__ import annotations

from fastapi import APIRouter

from . import cases, feedbacks, healthz, review_tasks

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(healthz.router)
api_router.include_router(review_tasks.router)
api_router.include_router(cases.router)
api_router.include_router(feedbacks.router)

__all__ = ["api_router"]
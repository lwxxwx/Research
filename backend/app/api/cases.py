# backend/app/api/cases.py
"""Case 列表 / 详情 / Benchmark 路由。

Sprint 3 G3 鉴权：
  - 所有端点要求登录（get_current_user）。
  - Case 数据范围过滤（engineer 仅自己创建的 case）留到 G3-B Service 层实现。

端点：
  GET  /api/v1/cases                                列出所有 Case 元信息
  GET  /api/v1/cases/{case_id}                      Case 详情 + 资产预览
  POST /api/v1/cases/{case_id}/benchmark            跑单 Case Benchmark
  GET  /api/v1/cases/{case_id}/benchmarks           历史结果查询
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import CurrentUser, get_current_user
from app.api.schemas import (
    CaseBenchmarkListResponse,
    CaseBenchmarkItem,
    CaseDetailResponse,
    CaseListResponse,
    RunCaseBenchmarkRequest,
)
from app.persistence.db import get_db_session
from app.services import case_service, task_service

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("", response_model=CaseListResponse)
def list_cases(
    current: CurrentUser = Depends(get_current_user),
) -> CaseListResponse:
    """列出所有 Case 元信息。

    Sprint 3 G3-A：仅校验登录；对象级过滤（若 case 有 created_by）在 G3-B 追加。
    """
    result = task_service.list_cases()
    return CaseListResponse(**result)


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case(
    case_id: str,
    current: CurrentUser = Depends(get_current_user),
) -> CaseDetailResponse:
    """Case 详情 + 资产预览。不存在 → 404。"""
    detail = case_service.get_case_detail(case_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} 不存在")
    return CaseDetailResponse(**detail)


@router.post("/{case_id}/benchmark", response_model=CaseBenchmarkItem)
def run_case_benchmark(
    case_id: str,
    body: RunCaseBenchmarkRequest,
    current: CurrentUser = Depends(get_current_user),
) -> CaseBenchmarkItem:
    """跑单 Case Benchmark。

    - rule_only：调 run_single_case，纯规则评测
    - full_review：调 run_single_case_full_review，走 6 节点 Graph（LLM 耗时 20–30s）

    结果落库 case_benchmark_result，返回新记录。
    """
    try:
        with get_db_session() as db:
            result = case_service.run_case_benchmark(
                db, case_id=case_id, mode=body.mode,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CaseBenchmarkItem(**result)


@router.get("/{case_id}/benchmarks", response_model=CaseBenchmarkListResponse)
def list_case_benchmarks(
    case_id: str,
    limit: int = Query(default=20, ge=0),
    offset: int = Query(default=0, ge=0),
    current: CurrentUser = Depends(get_current_user),
) -> CaseBenchmarkListResponse:
    """历史 Benchmark 结果查询（按 created_at 倒序）。"""
    with get_db_session() as db:
        result = case_service.list_case_benchmarks(
            db, case_id=case_id, limit=limit, offset=offset,
        )
    return CaseBenchmarkListResponse(**result)
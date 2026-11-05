# backend/app/api/review_tasks.py
"""评审任务路由。

Sprint 3 G3 鉴权（按 Sprint3_详细执行计划_V1.0.md §9 权限矩阵）：
  - POST /review/tasks            → engineer / admin
  - POST /review/tasks/batch      → engineer / admin
  - GET  /review/tasks            → admin / engineer / reviewer（数据范围见 G3-B）
  - GET  /{id}                    → admin / engineer / reviewer（数据范围见 G3-B）
  - GET  /{id}/defects            → admin / engineer / reviewer（数据范围见 G3-B）
  - GET  /{id}/export             → admin / engineer / reviewer（数据范围见 G3-B）
  - PATCH /{id}/defects/{defect_id} → reviewer / admin

审计字段注入（G3-C）：
  - created_by / expert_decision_by 由后端从 current_user.id 注入，前端不可传。
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.api.deps import (
    CurrentUser,
    ROLE_ADMIN,
    ROLE_ENGINEER,
    ROLE_REVIEWER,
    get_current_user,
    require_roles,
)
from app.api.schemas import (
    BatchReviewRequest,
    BatchReviewResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    DefectsResponse,
    TaskDetailResponse,
    TaskListResponse,
    UpdateExpertDecisionRequest,
    ExpertDecisionResponse,
)
from app.persistence.db import get_db_session
from app.services import report_service, task_service


router = APIRouter(prefix="/review/tasks", tags=["review"])


# ============================================================
# POST /review/tasks —— 创建单个评审任务
# ============================================================
@router.post("", response_model=CreateTaskResponse)
def create_review_task(
    body: CreateTaskRequest,
    current: CurrentUser = Depends(require_roles(ROLE_ENGINEER, ROLE_ADMIN)),
) -> CreateTaskResponse:
    """创建评审任务。engineer / admin 可用。"""
    task_id = task_service.create_task(body.case_id, created_by=current.id)
    # TODO(G3-C): 升级 task_service.create_task 返回完整 dict 后改回 **resp
    # 现在回查一次 get_task 拿真实状态，避免硬编码污染 G3-B IDOR 测试
    detail = task_service.get_task(task_id)
    if detail is None:
        # Graph 在 feedback_save 前失败 → review_result 未落库
        return CreateTaskResponse(task_id=task_id, status="failed", defects_count=0)
    return CreateTaskResponse(
        task_id=detail["task_id"],
        status=detail["status"],
        defects_count=detail["defects_count"],
    )



# ============================================================
# POST /review/tasks/batch —— 批量创建（先占位，G3-C 补归属）
# ============================================================
@router.post("/batch", response_model=BatchReviewResponse)
def create_review_tasks_batch(
    body: BatchReviewRequest,
    current: CurrentUser = Depends(require_roles(ROLE_ENGINEER, ROLE_ADMIN)),
) -> BatchReviewResponse:
    """批量创建评审任务（上限 5 个）。"""
    items = []
    for cid in body.case_ids:
        try:
            task_id = task_service.create_task(cid, created_by=current.id)
            items.append({
                "case_id": cid,
                "task_id": task_id,                   # ← 直接用 str
                "status": "success",
                "error": None,
            })
        except Exception as exc:
            items.append({
                "case_id": cid,
                "task_id": None,
                "status": "failed",
                "error": str(exc),
            })
    return BatchReviewResponse(items=items)


# ============================================================
# GET /review/tasks —— 任务列表
# ============================================================
@router.get("", response_model=TaskListResponse)
def list_review_tasks(
    current: CurrentUser = Depends(
        require_roles(ROLE_ENGINEER, ROLE_REVIEWER, ROLE_ADMIN)   # ← 补 REVIEWER
    ),
    status: Optional[str] = Query(None, description="pending / done / failed"),   # ← 补
    case_class: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    limit: int = Query(20, ge=0),
    offset: int = Query(0, ge=0),
):
    result = task_service.list_tasks_filtered(
        status=status,             # ← 补透传
        case_class=case_class,
        review_status=review_status,
        limit=limit,
        offset=offset,
    )
    return result



# ============================================================
# GET /review/tasks/{task_id} —— 任务详情
# ============================================================
@router.get("/{task_id}", response_model=TaskDetailResponse)
def get_review_task(
    task_id: str,
    current: CurrentUser = Depends(
        require_roles(ROLE_ENGINEER, ROLE_REVIEWER, ROLE_ADMIN)
    ),
) -> TaskDetailResponse:
    """任务详情。G3-A：只校验角色；数据范围过滤 G3-B。"""
    detail = task_service.get_task(task_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")
    return TaskDetailResponse(**detail)


# ============================================================
# GET /review/tasks/{task_id}/defects —— 缺陷列表
# ============================================================
@router.get("/{task_id}/defects", response_model=DefectsResponse)
def get_review_task_defects(
    task_id: str,
    current: CurrentUser = Depends(require_roles(ROLE_ENGINEER, ROLE_ADMIN)),
) -> DefectsResponse:
    """获取评审任务缺陷列表"""
    # ----新增开始----
    task_info = task_service.get_task(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")
    # ----新增结束----
    items = task_service.get_task_defects(task_id)
    return DefectsResponse(items=items)



# ============================================================
# PATCH /review/tasks/{task_id}/defects/{defect_id} —— 专家判定
# ============================================================
@router.patch(
    "/{task_id}/defects/{defect_id}",
    response_model=ExpertDecisionResponse,
)
def patch_expert_decision(
    task_id: str,
    defect_id: str,
    body: UpdateExpertDecisionRequest,
    current: CurrentUser = Depends(require_roles(ROLE_REVIEWER, ROLE_ADMIN)),
) -> ExpertDecisionResponse:
    """专家判定。reviewer / admin 可用。

    expert_decision_by 后端注入（G3-C 会改 task_service.update_expert_decision）。
    """
    result = task_service.update_expert_decision(
        task_id=task_id,
        defect_id=defect_id,
        expert_decision=body.expert_decision,
        updated_by=current.id,  # ← 后端注入
    )
    return ExpertDecisionResponse(**result)


# ============================================================
# GET /review/tasks/{task_id}/export —— 导出报告
# ============================================================
@router.get("/{task_id}/export", response_class=PlainTextResponse)
def export_review_report(
    task_id: str,
    current: CurrentUser = Depends(
        require_roles(ROLE_ENGINEER, ROLE_REVIEWER, ROLE_ADMIN)
    ),
) -> PlainTextResponse:
    """导出 Markdown 报告。G3-A：只校验角色；数据范围过滤 G3-B。"""
    try:
        md = report_service.generate_markdown_report(task_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PlainTextResponse(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{task_id}_report.md"'
        },
    )
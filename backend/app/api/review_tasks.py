# backend/app/api/review_tasks.py
"""评审任务路由（G4 新增）。

4 端点：
  POST   /api/v1/review/tasks             创建任务（同步跑 Graph）
  GET    /api/v1/review/tasks/{task_id}   任务详情
  GET    /api/v1/review/tasks             任务列表（筛选 + 分页）
  GET    /api/v1/review/tasks/{task_id}/defects  任务缺陷明细

【状态码语义（G4 约束 2）】
  - 参数错误 → 422（入参校验 + service ValueError 定点转换）
  - 资源不存在 → 404（task_id 不存在或未完成落库）
  - 成功查询但结果为空 → 200（空列表 / 空 defects）
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    CreateTaskRequest,
    CreateTaskResponse,
    DefectsResponse,
    ExpertDecisionResponse,          # ← S2-B 新增
    TaskDetailResponse,
    TaskListResponse,
    UpdateExpertDecisionRequest,     # ← S2-B 新增
)
from app.services import task_service
from app.services.feedback_service import FeedbackNotFoundError  # ← S2-B 新增

router = APIRouter(prefix="/review/tasks", tags=["review_tasks"])

# 约束 2：get_task 返回 None 有两种可能（task_id 从未创建 / 任务未落库），
# detail 合并描述，避免前端误判为"task_id 不存在"。
_NOT_FOUND_DETAIL = (
    "task_id={task_id} 不存在或任务未完成落库"
    "（review_result 无记录，可能是 Graph 在 feedback_save 前失败）"
)


# ============================================================
# POST /review/tasks
# ============================================================
@router.post("", response_model=CreateTaskResponse)
def create_review_task(body: CreateTaskRequest) -> CreateTaskResponse:
    """创建并运行一次评审任务。

    【同步阻塞说明（G4 决策 2）】
      - create_task 内部跑完整 Graph（LLM 调用约 20 秒），是同步阻塞
      - FastAPI 的同步 def 端点自动跑在线程池，不阻塞其他请求
      - 不改 create_task 的同步语义
    """
    task_id = task_service.create_task(body.case_id)

    # 决策 5：创建后调 get_task 拿 status + defects_count
    task = task_service.get_task(task_id)
    if task is None:
        # 理论上不应该发生（create_task 刚落库），但防御性处理
        raise HTTPException(
            status_code=500,
            detail=f"任务已创建但查询失败: task_id={task_id}",
        )

    return CreateTaskResponse(
        task_id=task["task_id"],
        status=task["status"],
        defects_count=task["defects_count"],
    )


# ============================================================
# GET /review/tasks/{task_id}
# ============================================================
@router.get("/{task_id}", response_model=TaskDetailResponse)
def get_review_task(task_id: str) -> TaskDetailResponse:
    """任务详情。task 不存在或未落库 → 404。"""
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=_NOT_FOUND_DETAIL.format(task_id=task_id),
        )
    return TaskDetailResponse(**task)


# ============================================================
# GET /review/tasks
# ============================================================
@router.get("", response_model=TaskListResponse)
def list_review_tasks(
    case_class: Literal["A", "B"] | None = Query(default=None),
    review_status: Literal[
        "AI_CONFIRMED", "NEED_EXPERT_REVIEW", "LOW_CONFIDENCE"
    ] | None = Query(default=None),
    limit: int = Query(default=20, ge=0),
    offset: int = Query(default=0, ge=0),
) -> TaskListResponse:
    """任务列表（筛选 + 分页）。

    【决策 4：B + A】
      - 入参层用 Literal + Query(ge=0) 校验，非法直接 422
      - service 层 ValueError 作为双保险，端点内 try/except 转 422
    """
    try:
        result = task_service.list_tasks_filtered(
            case_class=case_class,
            review_status=review_status,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        # 双保险：入参层已校验，正常不会到这里
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TaskListResponse(**result)


# ============================================================
# GET /review/tasks/{task_id}/defects
# ============================================================
@router.get("/{task_id}/defects", response_model=DefectsResponse)
def get_review_task_defects(task_id: str) -> DefectsResponse:
    """任务缺陷明细。

    【检查点（你确认的实现要求）】
      1. 先 get_task() 判断任务是否存在
      2. None → 404
      3. 存在 → get_task_defects()，返回列表（可能空，200）
    这样保留"任务存在但无缺陷 → 200 + []"与"任务不存在 → 404"的语义区分。
    """
    # 第 1 步：先检查任务存在性
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=_NOT_FOUND_DETAIL.format(task_id=task_id),
        )

    # 第 2 步：任务存在，查缺陷（可能空列表，200）
    defects = task_service.get_task_defects(task_id)
    return DefectsResponse(items=defects)


# ============================================================
# PATCH /review/tasks/{task_id}/defects/{defect_id}（S2-B）
# ============================================================
@router.patch(
    "/{task_id}/defects/{defect_id}",
    response_model=ExpertDecisionResponse,
)
def patch_expert_decision(
    task_id: str,
    defect_id: str,
    body: UpdateExpertDecisionRequest,
) -> ExpertDecisionResponse:
    """更新单条缺陷的专家最终判定。

    【状态码语义】
      - 200：更新成功，返回 3 字段 + 审计
      - 404：task 不存在 / defect 不存在 / defect 不属于 task
      - 422：expert_decision 枚举非法（由 Literal 保证）+ Service ValueError 兜底

    【约束】
      - 不改 review_status（AI 质量状态）
      - 不隐式创建 feedback_item（过程记录）
      - expert_decision_by 允许 None（Sprint 2 无用户系统）
    """
    try:
        result = task_service.update_expert_decision(
            task_id=task_id,
            defect_id=defect_id,
            expert_decision=body.expert_decision,
            updated_by=body.expert_decision_by,
        )
    except FeedbackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        # Service 层枚举兜底（正常不会到，入参 Literal 已校验）
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ExpertDecisionResponse(**result)
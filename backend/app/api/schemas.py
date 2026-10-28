# backend/app/api/schemas.py
"""
API 层 Pydantic schemas（G4 新增）。

【与 service 层 schemas 的区别】
  - service 层（app/schemas/）是业务模型
  - 本文件是 API 契约模型：只定义"HTTP 请求体 / 响应体"
  - 不复用 service 层 schemas，避免耦合（如 FeedbackCreate 含外键，API 层不暴露）
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ===== 复用 service 层输入模型（选项 A）=====
# SuggestionDiff / RuleCandidate 是稳定的业务输入契约，不是 DB 外键实现细节，
# API 层复用它们是合理的，避免字段漂移。
from app.schemas.feedback import RuleCandidate, SuggestionDiff


# ============================================================
# 健康检查
# ============================================================
class HealthResponse(BaseModel):
    """GET /api/v1/healthz 响应体（对齐 Sprint 0 契约：status/env/version）。"""
    status: Literal["ok"] = "ok"
    env: str
    version: str

# ============================================================
# 评审任务
# ============================================================
class CreateTaskRequest(BaseModel):
    """POST /api/v1/review/tasks 请求体。"""
    case_id: str = Field(..., min_length=1, description="Case ID，如 case001")


class CreateTaskResponse(BaseModel):
    """POST /api/v1/review/tasks 响应体。"""
    task_id: str
    status: Literal["done", "failed", "pending"]
    defects_count: int


class TaskDetailResponse(BaseModel):
    """GET /api/v1/review/tasks/{task_id} 响应体（透出 get_task 的返回）。"""
    task_id: str
    status: Literal["done", "failed", "pending"]
    case_id: str | None
    report: dict[str, Any] | None
    errors: list[str]
    created_at: str | None
    defects_count: int


class TaskListItem(BaseModel):
    """GET /api/v1/review/tasks 列表项（透出 list_tasks_filtered 的 item）。"""
    task_id: str
    case_id: str | None
    case_class: str | None
    defects_count: int
    review_status: str
    created_at: str | None


class TaskListResponse(BaseModel):
    """GET /api/v1/review/tasks 响应体。"""
    total: int
    items: list[TaskListItem]
    errors: list[str]


class DefectsResponse(BaseModel):
    """GET /api/v1/review/tasks/{task_id}/defects 响应体。

    service.get_task_defects 返回 list[dict]，API 层用 list[dict] 透传，
    不重复定义 16 字段（避免与 service 层 schema 漂移）。
    """
    items: list[dict[str, Any]]


# ============================================================
# Case 列表
# ============================================================
class CaseItem(BaseModel):
    """GET /api/v1/cases 列表项（透出 list_cases 的 item）。"""
    case_id: str
    case_class: str
    title: str | None = None
    description: str | None = None
    calibration_status: str | None = None
    tags: list[str] = Field(default_factory=list)


class CaseListResponse(BaseModel):
    """GET /api/v1/cases 响应体。"""
    total: int
    items: list[CaseItem]
    errors: list[str]


# ============================================================
# 反馈
# ============================================================
class FeedbackCreateAPI(BaseModel):
    """POST /api/v1/feedbacks 请求体。

    【与 service 层 FeedbackCreate 的区别】
      - service 层含 review_result_id / review_defect_id 外键
      - API 层用业务标识 task_id + defect_id，端点内翻译为外键
      - suggestion_diff / rule_candidate 复用 service 层模型（选项 A）
    """
    task_id: str = Field(..., min_length=1)
    defect_id: str = Field(..., min_length=1)
    feedback_type: Literal[
        "correct_defect",
        "false_positive",
        "false_negative",
        "suggestion_update",
        "knowledge_gap",
        "new_rule_candidate",
    ]
    expert_suggestion: str | None = None
    suggestion_adopted: Literal["yes", "partial", "no"] | None = None
    suggestion_diff: list[SuggestionDiff] | None = None
    rule_candidate: RuleCandidate | None = None
    created_by: int | None = None


class FeedbackResponse(BaseModel):
    """POST /api/v1/feedbacks 响应体（对齐 FeedbackOut）。

    【约束 3】created_at 用 datetime 类型，由 FastAPI 自动序列化为 ISO 字符串，
    不手动 .isoformat()。
    """
    id: int
    task_id: str | None = None
    defect_id: str | None = None
    review_result_id: int | None = None
    review_defect_id: int | None = None
    feedback_type: str
    expert_suggestion: str | None = None
    suggestion_diff: list[dict[str, Any]] = Field(default_factory=list)
    rule_candidate_ref: str | None = None
    created_by: int | None = None
    created_at: datetime

# ============================================================
# S2-B：专家判定 PATCH
# ============================================================
ExpertDecisionLiteral = Literal["pending", "confirmed", "rejected", "needs_verification"]


class UpdateExpertDecisionRequest(BaseModel):
    """PATCH /api/v1/review/tasks/{task_id}/defects/{defect_id} 请求体。"""
    expert_decision: ExpertDecisionLiteral
    expert_decision_by: int | None = None


class ExpertDecisionResponse(BaseModel):
    """PATCH 响应体：返回更新后的 3 字段 + 审计信息。"""
    defect_id: str
    expert_decision: ExpertDecisionLiteral
    expert_decision_updated_at: datetime
    expert_decision_by: int | None = None
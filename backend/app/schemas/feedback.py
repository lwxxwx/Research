# backend/app/schemas/feedback.py
from enum import StrEnum
from pydantic import BaseModel
from typing import Optional, List


class FeedbackType(StrEnum):
    """V1.2 6类反馈枚举，DB feedback_item.feedback_type 对齐"""
    CORRECT_DEFECT = "correct_defect"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    SUGGESTION_UPDATE = "suggestion_update"
    NEW_RULE_CANDIDATE = "new_rule_candidate"
    KNOWLEDGE_GAP = "knowledge_gap"


class SuggestionDiff(BaseModel):
    """suggestion_update场景：新旧建议对比条目"""
    old: str
    new: str


class FeedbackCreate(BaseModel):
    """
    提交反馈入参模型（API/CLI共用）
    ⚠️重要：不再直接传task_id/defect_id字符串！
    改为数据库外键：review_result_id / review_defect_id
    created_by: 用户表users.id（int主键，不是用户名字符串），可为None
    """
    review_result_id: int
    review_defect_id: int
    feedback_type: FeedbackType
    expert_suggestion: Optional[str] = None
    suggestion_adopted: Optional[str] = None
    suggestion_diff: Optional[List[SuggestionDiff]] = None
    rule_candidate: Optional[dict] = None
    created_by: Optional[int] = None


class FeedbackOut(BaseModel):
    """
    反馈查询输出模型
    task_id / defect_id 通过join关联查询回填（业务视图字段，非feedback_item真实列）
    """
    id: int
    review_result_id: Optional[int]
    review_defect_id: Optional[int]
    task_id: Optional[str]
    defect_id: Optional[str]
    feedback_type: FeedbackType
    expert_suggestion: Optional[str]
    suggestion_diff: List[SuggestionDiff]
    rule_candidate_ref: Optional[str]
    created_by: Optional[int]
    created_at: str

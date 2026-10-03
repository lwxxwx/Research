# backend/app/schemas/report.py
from enum import StrEnum


class ReviewStatus(StrEnum):
    """V1.2 Defect三态评审状态，review_defect.review_status DB字段对齐"""
    AI_CONFIRMED = "AI_CONFIRMED"
    NEED_EXPERT_REVIEW = "NEED_EXPERT_REVIEW"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

"""
注意：Sprint‑0 **不迁移存量Defect 10字段Pydantic**，仍然保留在原有模块；
TD‑007 Sprint‑1迭代，把ir/rag/rules/report全部Pydantic收拢到app/schemas/
本文件Sprint‑0只存放ReviewStatus枚举。
"""

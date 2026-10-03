# backend/app/schemas/feedback.py
"""
反馈模块 Pydantic schema

v1.1 变更：
- FeedbackOut.suggestion_diff 加 Field(default_factory=list)
- FeedbackOut.created_at 改为 datetime
- Optional[X] = None 统一改为 X | None = None
- 所有模型加 model_config（extra=forbid + str_strip_whitespace）
- FeedbackCreate.suggestion_diff 加 min_length=1

v1.2 变更（RuleCandidate 字段集对齐）：
- ★ RuleCandidate 从 5 字段扩展为 11 字段，字段来源：
    - rule_format.md（Rule YAML Format v1.0，Business Freeze）：规则 YAML 字段
    - models.py 的 RuleCandidate ORM：rule_candidates 表冗余字段
    - rule_evolution_service._build_proposed_rule_yaml 读取的 key
- ★ 字段改名：
    - rule_text  → rule_name （对齐 rule_format.md）
    - rationale  → rule_basis（对齐 rule_format.md）
- ★ 新增字段：
    - category（从可选改必填，对齐 rule_format.md §1）
    - applicable_condition / check_logic / suggestion（规则 YAML 必填，草稿可空）
    - title / description / evidence_refs（ORM 冗余字段）
- ★ 排除字段（系统生成，专家不填）：
    - version / enabled：草稿系统固定
    - candidate_id / from_feedback_id / case_id / task_id
      / proposed_yaml / status：系统字段
"""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# ============================================================
# 枚举：反馈类型
# ============================================================
class FeedbackType(StrEnum):
    """V1.2 6类反馈枚举，DB feedback_item.feedback_type 对齐"""
    CORRECT_DEFECT = "correct_defect"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    SUGGESTION_UPDATE = "suggestion_update"
    NEW_RULE_CANDIDATE = "new_rule_candidate"
    KNOWLEDGE_GAP = "knowledge_gap"


# ============================================================
# 嵌套模型：建议对比条目
# ============================================================
class SuggestionDiff(BaseModel):
    """suggestion_update场景：新旧建议对比条目"""
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    old: str
    new: str


# ============================================================
# ★ v1.2：RuleCandidate 字段集对齐（11 字段）
# ------------------------------------------------------------
# 字段来源：
#   1. rule_format.md（Rule YAML Format v1.0，Business Freeze）
#      规则 YAML 字段：rule_id / rule_name / category / severity /
#                     applicable_condition / check_logic / rule_basis / suggestion
#      （version / enabled 系统生成，排除）
#   2. models.py 的 RuleCandidate ORM（rule_candidates 表冗余字段）
#      title / description / evidence_refs
#      （candidate_id / from_feedback_id / case_id / task_id /
#        proposed_yaml / status 系统字段，排除）
#
# 用途：
#   - FeedbackCreate.rule_candidate 的强类型
#   - model_dump() 后直接喂给 rule_evolution_service.generate_candidate_from_feedback
#   - _build_proposed_rule_yaml 读前 8 个字段生成 YAML 草稿
#   - ORM 构造读 title / description / severity / evidence_refs
# ============================================================
class RuleCandidate(BaseModel):
    """
    false_negative / new_rule_candidate 场景：专家提交的规则候选

    字段对齐 rule_format.md（Rule YAML Format v1.0）和
    models.py 的 RuleCandidate ORM。

    字段说明：
      - rule_id:              规则唯一标识（如 "POWER_001"）
      - rule_name:            规则中文名
      - category:             规则分类（power/clock/interface/memory/thermal/esd/timing/layout/parametric/other）
      - severity:             low/medium/high/critical
      - rule_basis:           规则依据（Datasheet/规范章节）
      - applicable_condition: 规则匹配条件（专家可给简版或省略，草稿由系统填占位）
      - check_logic:          检查逻辑 {function, params}（专家可省略，草稿由系统填占位）
      - suggestion:           建议模板（专家可给初稿，正式版再精修）
      - title:                rule_candidates.title 字段（ORM 冗余，用于 DB 查询展示）
      - description:          rule_candidates.description 字段（ORM 冗余）
      - evidence_refs:        证据列表（V1.2 §3.2 三要素：source/section/reason）

    排除字段（系统生成）：
      - version / enabled：草稿固定 "0.1-proposed" / true
      - candidate_id / from_feedback_id / case_id / task_id
        / proposed_yaml / status：系统字段
    """
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    # --- 规则 YAML 必填字段（rule_format.md §1）---
    rule_id: str
    rule_name: str                          # ★ 改名：原 rule_text
    category: str                           # ★ 改必填：原 Optional
    severity: str
    rule_basis: str                         # ★ 改名：原 rationale

    # --- 规则 YAML 必填字段，但草稿阶段专家可省略（系统填占位）---
    applicable_condition: dict | None = None
    check_logic: dict | None = None
    suggestion: str | None = None

    # --- rule_candidates ORM 冗余字段 ---
    title: str | None = None
    description: str | None = None
    evidence_refs: list[dict] | None = None


# ============================================================
# 入参模型：提交反馈
# ============================================================
class FeedbackCreate(BaseModel):
    """
    提交反馈入参模型（API/CLI共用）

    ⚠️ 重要：不再直接传 task_id/defect_id 字符串！
       改为数据库外键：review_result_id / review_defect_id
    created_by: 用户表 users.id（int 主键，不是用户名字符串），可为 None
    """
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    review_result_id: int
    review_defect_id: int
    feedback_type: FeedbackType

    expert_suggestion: str | None = None
    suggestion_adopted: str | None = None

    suggestion_diff: list[SuggestionDiff] | None = Field(
        default=None,
        min_length=1,
        description="suggestion_update 场景的新旧建议对比列表，非空时至少 1 条",
    )

    rule_candidate: RuleCandidate | None = None

    created_by: int | None = None


# ============================================================
# 出参模型：查询反馈
# ============================================================
class FeedbackOut(BaseModel):
    """
    反馈查询输出模型

    task_id / defect_id 通过 join 关联查询回填（业务视图字段，
    非 feedback_item 真实列）
    """
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    id: int
    review_result_id: int | None = None
    review_defect_id: int | None = None

    # 业务视图字段（join 回填，非 DB 列）
    task_id: str | None = None
    defect_id: str | None = None

    feedback_type: FeedbackType

    expert_suggestion: str | None = None

    suggestion_diff: list[SuggestionDiff] = Field(
        default_factory=list,
        description="新旧建议对比列表；DB 为 NULL 时返回空列表",
    )

    rule_candidate_ref: str | None = None
    created_by: int | None = None

    created_at: datetime

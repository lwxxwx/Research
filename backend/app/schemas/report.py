# backend/app/schemas/report.py
"""
报告/缺陷 Pydantic Schema

【Sprint 0 原有内容】
  - ReviewStatus 三态枚举（保持不变）

【Sprint 1 阶段 C 新增】TD-007 收拢 report 部分
  - Evidence：单条证据，type=rag_ref 需 source/section/reason（各 >=10）；
              type=ir_ref 需 detail（>=5）
  - Defect：10 字段（对齐 models.py::ReviewDefect）+ review_status/origin/rule_id
            + expert_decision（Sprint 1 §23.1）
  - ReviewReportSummary / ReviewStatusDistribution：报告摘要
  - ReviewReport：task_id / report_id / defects[] / summary / errors[]

字段唯一事实来源：
  - Defect 10 字段  ← app/persistence/models.py::ReviewDefect
  - Evidence 结构   ← tests/test_case_b_validation.py + benchmark_service._evidence_is_complete
  - review_status   ← 本文件 ReviewStatus
  - origin 取值     ← models.py::ReviewDefect.origin（rule / ai_discovered / hybrid）
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# 保留：ReviewStatus（Sprint 0 已有，不改）
# ============================================================
class ReviewStatus(StrEnum):
    """V1.2 Defect三态评审状态，review_defect.review_status DB字段对齐"""
    AI_CONFIRMED = "AI_CONFIRMED"
    NEED_EXPERT_REVIEW = "NEED_EXPERT_REVIEW"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


# ============================================================
# 保留：Sprint 0 文件尾说明（不改）
# ============================================================
"""
注意：Sprint‑0 **不迁移存量Defect 10字段Pydantic**，仍然保留在原有模块；
TD‑007 Sprint‑1迭代，把ir/rag/rules/report全部Pydantic收拢到app/schemas/
本文件Sprint‑0只存放ReviewStatus枚举。
"""


# ============================================================
# ===== [S1-C 新增] DiscoveryOrigin =====
# ------------------------------------------------------------
# 与 models.py::ReviewDefect.origin 对齐（默认 'rule'）
# 不引入 discovery_source，统一用 origin（tests/test_case_b_validation.py 已固化）
# ============================================================
DiscoveryOrigin = Literal["rule", "ai_discovered", "hybrid"]


# ============================================================
# ===== [S1-C 新增] Evidence =====
# ------------------------------------------------------------
# 结构来源：
#   - tests/test_case_b_validation.py::test_defect_evidence_three_elements
#     · type=rag_ref → source / section / reason 均 str 且 len >= 10
#     · type=ir_ref  → detail 为 str 且 len >= 5
#   - app/services/benchmark_service.py::_evidence_is_complete 同口径
#
# 设计取舍：
#   - type 必填，Literal 限定两值
#   - source_type 可选（test_demo 带，test_ir_schema 示例不带）
#   - 用 model_validator 做「按 type 条件必填」，Defect.evidence 可统一承载两种引用
# ============================================================
EVIDENCE_TEXT_MIN_LEN = 10      # rag_ref 三要素最小长度（对齐 benchmark_service）
EVIDENCE_IR_REF_MIN_LEN = 5     # ir_ref.detail 最小长度


class Evidence(BaseModel):
    """
    单条证据。type 决定哪些字段必填：

    - type="rag_ref"：source / section / reason 必填，各长度 >= 10
    - type="ir_ref" ：detail 必填，长度 >= 5

    source_type 为可选扩展（datasheet / reference_design / application_note 等），
    对应 app.rag.sources.DocumentType；不强制，避免与 test_ir_schema 简化示例冲突。
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["rag_ref", "ir_ref"]

    # rag_ref 专用（type=rag_ref 时必填）
    source: str | None = None
    section: str | None = None
    reason: str | None = None

    # ir_ref 专用（type=ir_ref 时必填）
    detail: str | None = None

    # 可选扩展
    source_type: str | None = None

    @model_validator(mode="after")
    def _check_required_by_type(self) -> "Evidence":
        if self.type == "rag_ref":
            missing = [
                name for name in ("source", "section", "reason")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    f"rag_ref 缺必填字段: {missing}"
                    f"（三要素 source/section/reason 均必填）"
                )
            # ===== [S1-B 修正] 长度不足从 Schema 硬拒绝改为 Parser 业务降级 =====
            # 原因：LLM 偶发输出短 section（如 "p5"），Schema 硬拒绝导致整份报告
            #      校验失败、task_status=failed，3 个 Case 全部作废。
            #      符合 Sprint 1 §5.3「不合格不丢弃，只降级」。
            # 修法：Schema 只校验非空（上方 missing 分支已做），长度不足交给
            #      Parser 的 _evidence_is_valid 判定 → review_status 降级。
            # --- 原代码（保留，已注释） ---
            # too_short = [
            #     name for name in ("source", "section", "reason")
            #     if len(getattr(self, name)) < EVIDENCE_TEXT_MIN_LEN
            # ]
            # if too_short:
            #     raise ValueError(
            #         f"rag_ref 字段长度不足 {EVIDENCE_TEXT_MIN_LEN}: {too_short}"
            #     )
            # --- 原代码（保留，已注释） ---
            # ===== [S1-B 修正 END] =====
        elif self.type == "ir_ref":
            if not self.detail:
                raise ValueError("ir_ref 缺必填字段: detail")
            # ===== [S1-B 修正] 长度不足交给 Parser 降级，Schema 只校验非空 =====
            # --- 原代码（保留，已注释） ---
            # if len(self.detail) < EVIDENCE_IR_REF_MIN_LEN:
            #     raise ValueError(
            #         f"ir_ref.detail 长度不足 {EVIDENCE_IR_REF_MIN_LEN}"
            #         f"（实际 {len(self.detail)}）"
            #     )
            # --- 原代码（保留，已注释） ---
            # ===== [S1-B 修正 END] =====
        return self


# ============================================================
# ===== [S1-C 新增] ExpertDecision + Defect =====
# ------------------------------------------------------------
# 字段对齐 models.py::ReviewDefect：
#   defect_id / category / location / component / net / risk /
#   evidence / root_cause / suggestion / confidence   ← 10 字段
#   review_status / origin / rule_id                  ← 固有追踪
#   expert_decision                                    ← Sprint 1 §23.1
#
# 注意：
#   - 不引入 discovery_source，统一用 origin
#   - expert_decision 默认 pending，与 DB server_default 一致
#   - location 用 dict 承载（DB 为 JSONB），不强制内部结构，
#     避免与 test_demo（coords 为 dict）和 test_db_schema（coords 为 list）冲突
# ============================================================
ExpertDecision = Literal["pending", "confirmed", "rejected", "needs_verification"]


class Defect(BaseModel):
    """单条缺陷，10 字段 + 追踪字段 + 专家判定"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- 10 字段（对齐 models.py::ReviewDefect）---
    defect_id: str
    category: str = "other"
    location: dict[str, Any]
    component: str | None = None
    net: str | None = None
    risk: str = "medium"
    evidence: list[Evidence] = Field(default_factory=list)
    root_cause: str
    suggestion: str
    confidence: float = 0.5

    # --- 固有追踪字段 ---
    review_status: ReviewStatus = ReviewStatus.AI_CONFIRMED
    origin: DiscoveryOrigin = "rule"
    rule_id: str | None = None

    # --- Sprint 1 专家判定（§23.1）---
    expert_decision: ExpertDecision = "pending"

    # ===== [S1-B 新增] provenance 标记 =====
    # 用途：区分"LLM 原始输出"和"代码补齐（从 RuleResult 构造）"。
    #   - synthesized=False：LLM 输出的缺陷（默认）
    #   - synthesized=True ：Merge 层从 RuleResult 补齐的缺陷（LLM 未输出）
    # 说明：不改变 10 字段口径，仅作为溯源标记，供 get_task / 诊断 / 前端使用。
    # ===== [S1-B 新增 END] =====
    synthesized: bool = False

    @field_validator("confidence")
    @classmethod
    def _check_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence 必须在 [0,1]，实际 {v}")
        return v

    @field_validator("risk")
    @classmethod
    def _check_risk(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "critical"}
        if v not in allowed:
            raise ValueError(f"risk 非法: {v}，允许 {sorted(allowed)}")
        return v


# ============================================================
# ===== [S1-C 新增] ReviewReportSummary / ReviewStatusDistribution =====
# ------------------------------------------------------------
# 对应 test_demo.py mock_report["summary"] + V1.2 §5.1
#   evidence_coverage            ← §8.1 EVC
#   review_status_distribution   ← §5.1 三态分布
# ============================================================
class ReviewStatusDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    AI_CONFIRMED: int = 0
    NEED_EXPERT_REVIEW: int = 0
    LOW_CONFIDENCE: int = 0


class ReviewReportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_coverage: float | None = None
    review_status_distribution: ReviewStatusDistribution = Field(
        default_factory=ReviewStatusDistribution
    )

    @field_validator("evidence_coverage")
    @classmethod
    def _check_evc_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"evidence_coverage 必须在 [0,1] 或 None，实际 {v}"
            )
        return v


# ============================================================
# ===== [S1-C 新增] ReviewReport =====
# ------------------------------------------------------------
# 对应 test_demo.py mock_report 顶层 + models.py::ReviewResult.review_output_json
#   task_id / report_id / defects[] / summary / errors[]
#
# errors[]：LangGraph 节点异常收集（Sprint 1 §5.3 异常容错）
# ============================================================
class ReviewReport(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str
    report_id: str
    defects: list[Defect] = Field(default_factory=list)
    summary: ReviewReportSummary = Field(default_factory=ReviewReportSummary)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_defect_id_unique(self) -> "ReviewReport":
        ids = [d.defect_id for d in self.defects]
        dup = {x for x in ids if ids.count(x) > 1}
        if dup:
            raise ValueError(f"defect_id 重复: {sorted(dup)}")
        return self

    def recount_review_status(self) -> None:
        """
        依据 defects 实际 review_status 重算 summary.review_status_distribution。
        供 OutputParser 在完成状态判定后调用，避免手写计数漂移。
        """
        counter = {
            "AI_CONFIRMED": 0,
            "NEED_EXPERT_REVIEW": 0,
            "LOW_CONFIDENCE": 0,
        }
        for d in self.defects:
            counter[d.review_status.value] += 1
        self.summary.review_status_distribution = ReviewStatusDistribution(**counter)


# ============================================================
# ===== [S1-C 新增] __all__ =====
# ============================================================
__all__ = [
    # Sprint 0 原有
    "ReviewStatus",
    # Sprint 1 阶段 C 新增
    "DiscoveryOrigin",
    "Evidence",
    "Defect",
    "ExpertDecision",
    "ReviewStatusDistribution",
    "ReviewReportSummary",
    "ReviewReport",
    "EVIDENCE_TEXT_MIN_LEN",
    "EVIDENCE_IR_REF_MIN_LEN",
]
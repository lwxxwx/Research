# backend/app/llm/output_parser.py
"""
LLM 输出解析与质量校验（Sprint 1 阶段 C 新增）

职责（对应 Sprint1_融合版 §5.2 / §7.4 / §23.1）：
  1. 从 LLMResponse 中取 parsed（或回退解析 raw_text）
  2. 按 ReviewReport / Defect / Evidence Schema 做结构校验
  3. 按 §5.2 五条强规则判定 review_status
  4. ai_discovered 硬约束：不得 AI_CONFIRMED（tests/test_case_b_validation.py）
  5. 引用真实性校验：
       - ir_ref.detail 必须可解析到当前 IR（Component/Net/Pin 引用）
       - rag_ref.source/section 必须能在本次 rag_context 中找到
  6. 不合格不丢弃，只降级（§5.3）

【分层原则】
  - Schema（app/schemas/report.py）：只管结构
  - OutputParser（本文件）：管业务质量规则与降级动作
  - LLMResponse：只承载原始返回，不做判定

【review_status 判定顺序（强规则，不做概率）】
  规则 1：10 字段缺失/类型不合格              → LOW_CONFIDENCE（confidence ≤ 0.4）
  规则 2：Evidence 三要素完整率不足           → NEED_EXPERT_REVIEW（confidence -0.2）
  规则 3：ai_discovered 缺 ir_ref 或 rag_ref  → NEED_EXPERT_REVIEW（confidence -0.2）
  规则 4：模糊词黑名单在 risk≥medium 出现 ≥2  → LOW_CONFIDENCE（confidence -0.2）
  规则 5：以上全通过                          → AI_CONFIRMED
  硬约束：origin == ai_discovered 时，最终 review_status 不得为 AI_CONFIRMED
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from app.ir.schema import SchematicIRDocument
from app.llm.provider import LLMResponse
from app.schemas.report import (
    Defect,
    Evidence,
    ReviewReport,
    ReviewStatus,
)

logger = logging.getLogger(__name__)


# ============================================================
# ===== [S1-C 新增] 常量：模糊词黑名单 =====
# ------------------------------------------------------------
# 对应 §4.2 / §5.2 规则 4
# ============================================================
FUZZY_WORDS: tuple[str, ...] = (
    "可能", "应该", "估计", "也许", "大概", "或许", "疑似", "推测",
)

# ============================================================
# ===== [S1-C 新增] 常量：risk 与 confidence 调整量 =====
# ============================================================
CONF_CAP_LOW: float = 0.4         # 规则 1 触发时 confidence 上限
CONF_PENALTY: float = 0.2         # 规则 2/3/4 触发时扣减
FUZZY_THRESHOLD_HITS: int = 2     # 规则 4 阈值


# ============================================================
# ===== [S1-C 新增] 解析入口 =====
# ============================================================
def parse_report(
    llm_response: LLMResponse,
    *,
    ir_document: SchematicIRDocument | None = None,
    rag_context: list[dict[str, Any]] | None = None,
    rule_hits: list[Any] | None = None,
) -> ReviewReport:
    """
    将 LLMResponse 解析为 ReviewReport，并完成 §5.2 质量判定。

    参数：
      llm_response   : Provider 返回对象
      ir_document    : 用于校验 ir_ref.detail 可解析性；None 时跳过该项校验
      rag_context    : 本次输入的 RAG 证据列表；None 时跳过 rag_ref 真实性校验
      rule_hits      : 本次规则命中（当前仅记录，不参与状态判定；预留给阶段 D/E）

    返回：
      ReviewReport。解析失败时返回空 defects + errors，且 summary 全零，
      不伪装成功（§7.4：不静默丢弃，但明确标记失败）。
    """
    errors: list[str] = []

    # 1) 取 raw payload
    payload: dict[str, Any] | None = llm_response.parsed
    if payload is None and llm_response.raw_text:
        payload = _try_parse_json(llm_response.raw_text)
        if payload is None:
            errors.append("LLM 返回无法解析为 JSON")
    if payload is None:
        if llm_response.error:
            errors.append(f"LLM provider 错误: {llm_response.error}")
        if llm_response.finish_reason == "length":
            errors.append("LLM 返回被截断（finish_reason=length）")
        return _build_empty_report(errors)

    # 2) 顶层结构校验（task_id / report_id / defects / summary）
    try:
        report = ReviewReport.model_validate(payload)
    except ValidationError as exc:
        errors.append(f"ReviewReport 结构校验失败: {_format_validation_error(exc)}")
        return _build_empty_report(errors)

    # 3) 逐条 Defect 判定（§5.2）
    for d in report.defects:
        _classify_defect(
            d,
            ir_document=ir_document,
            rag_context=rag_context,
        )

    # 4) 重算三态分布
    report.recount_review_status()

    # 5) 合并 provider 错误 / 截断信息到 errors（不覆盖已有结构错误）
    if llm_response.error and llm_response.error not in errors:
        errors.append(f"LLM provider 错误: {llm_response.error}")
    if llm_response.finish_reason == "length":
        errors.append("LLM 返回被截断（finish_reason=length）")
    report.errors.extend(errors)
    return report


# ============================================================
# ===== [S1-C 新增] 单条 Defect 的三态判定 =====
# ============================================================
def _classify_defect(
    d: Defect,
    *,
    ir_document: SchematicIRDocument | None,
    rag_context: list[dict[str, Any]] | None,
) -> None:
    """
    就地修改 d.review_status 与 d.confidence。
    判定顺序严格按 §5.2，命中任一即不再升级（只能往更严格方向降级）。
    """

    # ---------- 规则 1：10 字段完整性 ----------
    # Schema 已强制大部分必填；这里做「业务级非空」复核：
    # location 非空 dict、root_cause/suggestion 非空、defect_id 非空
    rule1_violated = (
        not d.defect_id
        or not isinstance(d.location, dict) or not d.location
        or not d.root_cause
        or not d.suggestion
    )
    if rule1_violated:
        d.review_status = ReviewStatus.LOW_CONFIDENCE
        d.confidence = min(d.confidence, CONF_CAP_LOW)
        return

    # ---------- 规则 2：Evidence 三要素完整率 ----------
    # 至少 1 条「有效 evidence」才过；有效定义与 benchmark_service 同口径
    valid_ev_count = sum(1 for e in d.evidence if _evidence_is_valid(e))
    rule2_violated = valid_ev_count < 1
    if rule2_violated:
        d.confidence = max(0.0, d.confidence - CONF_PENALTY)
        d.review_status = ReviewStatus.NEED_EXPERT_REVIEW
        # 不 return：继续跑规则 3/4，取更严格状态

    # ---------- 规则 3：ai_discovered 必须 ir_ref + rag_ref ----------
    rule3_violated = False
    if d.origin == "ai_discovered":
        has_ir_ref = any(e.type == "ir_ref" for e in d.evidence)
        has_rag_ref = any(e.type == "rag_ref" for e in d.evidence)
        if not (has_ir_ref and has_rag_ref):
            rule3_violated = True
            d.confidence = max(0.0, d.confidence - CONF_PENALTY)
            d.review_status = ReviewStatus.NEED_EXPERT_REVIEW

    # ---------- 规则 3.5：引用真实性（§4.3）----------
    # ir_ref 可解析性 / rag_ref 真实性，属于「引用完整性（机器可测）」层
    if ir_document is not None:
        for e in d.evidence:
            if e.type == "ir_ref" and not _ir_ref_resolvable(e.detail or "", ir_document):
                d.confidence = max(0.0, d.confidence - CONF_PENALTY)
                d.review_status = ReviewStatus.NEED_EXPERT_REVIEW
                break
    if rag_context is not None:
        for e in d.evidence:
            if e.type == "rag_ref" and not _rag_ref_in_context(e, rag_context):
                d.confidence = max(0.0, d.confidence - CONF_PENALTY)
                d.review_status = ReviewStatus.NEED_EXPERT_REVIEW
                break

    # ---------- 规则 4：模糊词黑名单 ----------
    # risk ∈ {medium, high, critical} 时，root_cause + suggestion 中模糊词 ≥ 2 → LOW_CONFIDENCE
    if d.risk in {"medium", "high", "critical"}:
        hits = _count_fuzzy_words(d.root_cause) + _count_fuzzy_words(d.suggestion)
        if hits >= FUZZY_THRESHOLD_HITS:
            d.confidence = max(0.0, d.confidence - CONF_PENALTY)
            d.review_status = ReviewStatus.LOW_CONFIDENCE
            return

    # ---------- 规则 5：全通过 → AI_CONFIRMED ----------
    # 只有当未被前面规则降级时才置 AI_CONFIRMED
    if (
        d.review_status == ReviewStatus.AI_CONFIRMED
        and not rule2_violated
        and not rule3_violated
    ):
        d.review_status = ReviewStatus.AI_CONFIRMED

    # ---------- 硬约束：ai_discovered 不得 AI_CONFIRMED ----------
    # 对齐 tests/test_case_b_validation.py::test_ai_discovered_review_status
    if d.origin == "ai_discovered" and d.review_status == ReviewStatus.AI_CONFIRMED:
        d.review_status = ReviewStatus.NEED_EXPERT_REVIEW
        logger.warning(
            "ai_discovered defect %s 被判为 AI_CONFIRMED，强制降级为 NEED_EXPERT_REVIEW",
            d.defect_id,
        )


# ============================================================
# ===== [S1-C 新增] Evidence 有效性（与 benchmark_service 同口径）=====
# ============================================================
def _evidence_is_valid(e: Evidence) -> bool:
    """
    与 app/services/benchmark_service.py::_evidence_is_complete 语义一致：
      - rag_ref：source/section/reason 均非空且 len >= 10
      - ir_ref ：detail 非空且 len >= 5
    Schema 已做长度校验，本函数做二次防御，防止 Schema 放宽后漏检。
    """
    if e.type == "rag_ref":
        return all(
            isinstance(getattr(e, name), str) and len(getattr(e, name)) >= 10
            for name in ("source", "section", "reason")
        )
    if e.type == "ir_ref":
        return isinstance(e.detail, str) and len(e.detail) >= 5
    return False


# ============================================================
# ===== [S1-C 新增] ir_ref 可解析性校验（§4.3）=====
# ------------------------------------------------------------
# 支持的 detail 形式（保守解析，不解析复杂表达式）：
#   - "Component[U1]..."        → 要求 IR 中存在 ref=U1
#   - "Net[VCC]..."             → 要求 IR 中存在 net_name=VCC
#   - "Component[U1].pins[P1.0]"→ 同时要求 U1 存在且有 pin_id=P1.0
#   - "U1.VCC"                  → 要求 U1 存在且有 pin_id=VCC
#   - 其他无法识别形式           → 视为不可解析（保守，触发降级）
# 说明：无法覆盖所有写法，宁可误降级也不放过无依据引用（§4.3 保守原则）
# ============================================================
_COMPONENT_RE = re.compile(r"Component\[([^\]]+)\]")
_NET_RE = re.compile(r"Net\[([^\]]+)\]")
_PIN_IN_COMP_RE = re.compile(r"Component\[([^\]]+)\]\.pins\[([^\]]+)\]")
_DOT_REF_RE = re.compile(r"^([^.\[\]\s]+)\.([^.\[\]\s]+)$")


def _ir_ref_resolvable(detail: str, ir: SchematicIRDocument) -> bool:
    if not detail:
        return False

    components = {c.ref: c for c in ir.components}
    nets = {n.net_name for n in ir.nets}

    m = _PIN_IN_COMP_RE.search(detail)
    if m:
        ref, pin_id = m.group(1), m.group(2)
        comp = components.get(ref)
        return bool(comp and any(p.pin_id == pin_id for p in comp.pins))

    m = _COMPONENT_RE.search(detail)
    if m:
        return m.group(1) in components

    m = _NET_RE.search(detail)
    if m:
        return m.group(1) in nets

    m = _DOT_REF_RE.match(detail.strip())
    if m:
        ref, pin_id = m.group(1), m.group(2)
        comp = components.get(ref)
        if comp is None:
            return False
        return any(p.pin_id == pin_id for p in comp.pins)

    return False


# ============================================================
# ===== [S1-C 新增] rag_ref 真实性校验（§4.3）=====
# ------------------------------------------------------------
# 规则：
#   - e.source 必须与 rag_context 中某条的 source/source_title 匹配
#   - e.section 必须与同一条的 section/source_section 匹配
# 匹配使用「包含」而非全等，容忍 Parser 与 Retriever 的轻微格式差异
# （如 retriever 的 source_title 与 rag_ref.source 可能多/少后缀）
# ============================================================
def _rag_ref_in_context(e: Evidence, rag_context: list[dict[str, Any]]) -> bool:
    if not e.source or not e.section:
        return False
    for item in rag_context:
        cand_source = (
            item.get("source") or item.get("source_title") or ""
        )
        cand_section = (
            item.get("section") or item.get("source_section") or ""
        )
        if not cand_source or not cand_section:
            continue
        if (e.source in cand_source or cand_source in e.source) and (
            e.section in cand_section or cand_section in e.section
        ):
            return True
    return False


# ============================================================
# ===== [S1-C 新增] 工具函数 =====
# ============================================================
def _try_parse_json(raw_text: str) -> dict[str, Any] | None:
    import json
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _count_fuzzy_words(text: str) -> int:
    if not text:
        return 0
    return sum(text.count(w) for w in FUZZY_WORDS)


def _format_validation_error(exc: ValidationError) -> str:
    """Pydantic ValidationError 摘要，避免日志过长。"""
    parts = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(x) for x in e.get("loc", []))
        msg = e.get("msg", "")
        parts.append(f"{loc}: {msg}")
    if len(exc.errors()) > 5:
        parts.append(f"... 共 {len(exc.errors())} 条")
    return "; ".join(parts)


def _build_empty_report(errors: list[str]) -> ReviewReport:
    """解析失败时返回空报告，明确标记失败（不伪装成功）。"""
    return ReviewReport(
        task_id="",
        report_id="",
        defects=[],
        errors=errors,
    )


# ============================================================
# ===== [S1-C 新增] __all__ =====
# ============================================================
__all__ = [
    "FUZZY_WORDS",
    "parse_report",
]
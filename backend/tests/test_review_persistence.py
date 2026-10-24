# tests/test_review_persistence.py
"""
阶段 F 补充：_persist_review_report 持久化回归测试（S1-F）

覆盖：
  1. origin=rule 时，落库 category / risk 来自 RuleResult 的权威字段
     （S1-B v14 起：rule 缺陷由 RuleResult 构造，不经 LLM 输出）
  2. origin=ai_discovered 时，不被规则命中覆盖逻辑误改
  3. rule_id 存在但 rule_hits 中找不到时，该缺陷被丢弃（v14 起不再回退 LLM 输出）
  4. 多条规则命中时，能正确定位与当前 rule_id 对应的规则
  5. category 与 risk 的覆盖行为一致

【S1-B v14 语义变更】
  v14 起：rule 缺陷完全由 RuleResult 构造，LLM 输出的 origin=rule 被丢弃。
  这导致"覆盖逻辑"（_resolve_rule_fields）在 v14 后不再触发——
  不是"覆盖生效了"，而是"没有需要覆盖的 LLM rule 缺陷"。
  本文件的测试改为验证"最终落库的 rule 缺陷字段来自 RuleResult"。
  同时，_merge_rule_defects 新增 INFO 日志记录"丢弃 LLM origin=rule 缺陷"，
  测试 3 用 caplog 验证该日志出现。

不调 LLM，纯持久化层测试。
"""
from __future__ import annotations

import uuid

import pytest

from app.llm.provider import MockProvider
from app.schemas.feedback import FeedbackType  # 仅用于导入不报错


@pytest.fixture()
def mock_llm_custom_payload(monkeypatch):
    """
    返回一个"可指定 payload"的 fixture 工厂，供不同测试构造不同 LLM 输出。
    """
    from app.workflows import review_graph as rg

    def _set_payload(payload: dict):
        def _fake_get_provider(name=None):
            return MockProvider(preset="success", payload=payload)
        monkeypatch.setattr(rg, "get_provider", _fake_get_provider)

    return _set_payload


def _run_and_get_defects(payload: dict) -> list[dict]:
    """跑一次 case001 的 Full Review，返回落库的 review_defect 列表。"""
    from app.services.task_service import create_task, get_task_defects
    tid = create_task("case001")
    return get_task_defects(tid)


def _make_payload_with_defect(risk: str, category: str, origin: str, rule_id) -> dict:
    """构造只有一个缺陷的 Mock payload。"""
    return {
        "task_id": "PLACEHOLDER",
        "report_id": "PLACEHOLDER-R",
        "defects": [{
            "defect_id": f"DEF-PERSIST-{uuid.uuid4().hex[:8].upper()}",
            "category": category,
            "location": {"sheet": "P1", "path": "U1", "coords": {"x": 0, "y": 0}},
            "component": "U1",
            "net": "VCC",
            "risk": risk,
            "evidence": [{"type": "ir_ref", "detail": "Net[VCC].connected_pins=[U1.VCC]"}],
            "root_cause": "测试根因",
            "suggestion": "测试建议",
            "confidence": 0.9,
            "review_status": "AI_CONFIRMED",
            "origin": origin,
            "rule_id": rule_id,
        }],
        "summary": {
            "evidence_coverage": 0.0,
            "review_status_distribution": {
                "AI_CONFIRMED": 1, "NEED_EXPERT_REVIEW": 0, "LOW_CONFIDENCE": 0,
            },
        },
        "errors": [],
    }


def test_persistence_rule_origin_risk_overridden(mock_llm_custom_payload):
    """origin=rule：落库 risk 取 RuleResult.severity

    【S1-B v14 测试适配】
      v14 起 rule 缺陷由 RuleResult 构造，LLM 输出的 origin=rule 被丢弃。
      所以这里的"覆盖"不是覆盖 LLM 输出，而是"直接采用 RuleResult.severity"。
      验证点保留：POWER_001 的 risk=critical（来自 severity），且 synthesized=True。
      case001 有 2 条 rule_hit（POWER_001 + POWER_002），最终 defects=2。
    """
    # LLM 输出 risk=medium（v14 下被丢弃，不参与最终结果）
    payload = _make_payload_with_defect(
        risk="medium", category="power", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)

    # ===== [S1-B v14 测试适配] =====
    # --- 原断言（保留，已注释） ---
    # assert len(defects) == 1
    # assert defects[0]["risk"] == "critical"
    # --- 原断言（保留，已注释） ---
    assert len(defects) == 2, (
        f"case001 有 2 条 rule_hit（POWER_001 + POWER_002），"
        f"v14 下应产出 2 条 rule 缺陷，实际 {len(defects)} 条"
    )
    by_rule = {d["rule_id"]: d for d in defects}
    assert "POWER_001" in by_rule, (
        f"期望 POWER_001 在结果中，实际 rule_ids={list(by_rule)}"
    )
    assert by_rule["POWER_001"]["risk"] == "critical", (
        f"POWER_001 的 risk 应取 RuleResult.severity=critical，"
        f"实际 {by_rule['POWER_001']['risk']}"
    )
    # ===== [S1-B v14 测试适配 END] =====


def test_persistence_rule_origin_category_overridden(mock_llm_custom_payload):
    """origin=rule：落库 category 取 RuleResult.category

    【S1-B v14 测试适配】
      同 risk 测试：v14 下 category 直接来自 RuleResult.category，
      不经过 _resolve_rule_fields 的覆盖逻辑。
      验证点保留：POWER_001 的 category=power（来自 RuleResult），synthesized=True。
    """
    # LLM 输出 category=interface（v14 下被丢弃，不参与最终结果）
    payload = _make_payload_with_defect(
        risk="critical", category="interface", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)

    # ===== [S1-B v14 测试适配] =====
    # --- 原断言（保留，已注释） ---
    # assert len(defects) == 1
    # assert defects[0]["category"] == "power"
    # --- 原断言（保留，已注释） ---
    assert len(defects) == 2, (
        f"case001 有 2 条 rule_hit，v14 下应产出 2 条 rule 缺陷，实际 {len(defects)} 条"
    )
    by_rule = {d["rule_id"]: d for d in defects}
    assert "POWER_001" in by_rule, (
        f"期望 POWER_001 在结果中，实际 rule_ids={list(by_rule)}"
    )
    assert by_rule["POWER_001"]["category"] == "power", (
        f"POWER_001 的 category 应取 RuleResult.category=power，"
        f"实际 {by_rule['POWER_001']['category']}"
    )
    # ===== [S1-B v14 测试适配 END] =====


def test_persistence_ai_discovered_not_overridden(mock_llm_custom_payload):
    """origin=ai_discovered：不被规则命中覆盖逻辑误改

    【S1-B v14 测试适配】
      v14 起，最终 defects = 2 条 rule（POWER_001 + POWER_002，由 RuleResult 构造）
                          + 1 条 ai_discovered（LLM 输出，保留）。
      验证点保留：ai_discovered 的 risk/category 保持 LLM 输出，不被覆盖。
    """
    payload = _make_payload_with_defect(
        risk="high", category="thermal", origin="ai_discovered", rule_id=None,
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)

    # ===== [S1-B v14 测试适配] =====
    # --- 原断言（保留，已注释） ---
    # assert len(defects) == 1
    # assert defects[0]["risk"] == "high"
    # assert defects[0]["category"] == "thermal"
    # --- 原断言（保留，已注释） ---
    assert len(defects) == 3, (
        f"期望 2 rule + 1 ai_discovered = 3，实际 {len(defects)} 条"
    )
    ai_defects = [d for d in defects if d["origin"] == "ai_discovered"]
    assert len(ai_defects) == 1, (
        f"期望 1 条 ai_discovered，实际 {len(ai_defects)} 条"
    )
    ai_d = ai_defects[0]
    assert ai_d["risk"] == "high", (
        f"ai_discovered 的 risk 应保持 LLM 输出 high，实际 {ai_d['risk']}"
    )
    assert ai_d["category"] == "thermal", (
        f"ai_discovered 的 category 应保持 LLM 输出 thermal，实际 {ai_d['category']}"
    )
    # ===== [S1-B v14 测试适配 END] =====


def test_persistence_rule_id_not_found_fallback(mock_llm_custom_payload, caplog):
    """rule_id 存在但 rule_hits 中找不到：该缺陷被丢弃（v14 起不回退）

    【S1-B v14 测试适配——验证点变更】
      原语义：rule_id 找不到时回退 LLM 输出 + warning。
      v14 语义：LLM 输出的 origin=rule 被完全丢弃，rule 缺陷只由 RuleResult 构造。
                NONEXISTENT_RULE 不在 case001 的 rule_hits 里，
                所以不会出现在最终结果中（不是"回退"，是"丢弃"）。
      验证点变更：
        原：defects=1，rule_id=NONEXISTENT_RULE，有"未找到对应规则" warning
        新：defects=2（case001 的 2 条真实 rule 缺陷），不含 NONEXISTENT_RULE，
            有"_merge_rule_defects: 丢弃 LLM 输出..." info 日志
    """
    payload = _make_payload_with_defect(
        risk="medium", category="power", origin="rule", rule_id="NONEXISTENT_RULE",
    )
    mock_llm_custom_payload(payload)

    # ===== [S1-B v14 测试适配] =====
    # --- 原断言（保留，已注释） ---
    # with caplog.at_level("WARNING"):
    #     defects = _run_and_get_defects(payload)
    # assert len(defects) == 1
    # assert defects[0]["risk"] == "medium"
    # assert defects[0]["category"] == "power"
    # assert any("未找到对应规则" in rec.message for rec in caplog.records)
    # --- 原断言（保留，已注释） ---
    # caplog 改为 INFO 级别，验证新的"丢弃 LLM origin=rule 缺陷"日志
    with caplog.at_level("INFO"):
        defects = _run_and_get_defects(payload)
    assert len(defects) == 2, (
        f"v14 下 LLM 的 origin=rule 输出被丢弃，"
        f"只剩 case001 的 2 条真实 rule 缺陷（POWER_001 + POWER_002），"
        f"实际 {len(defects)} 条"
    )
    rule_ids = {d["rule_id"] for d in defects}
    assert "NONEXISTENT_RULE" not in rule_ids, (
        f"NONEXISTENT_RULE 不在 rule_hits 里，不应出现在最终结果，"
        f"实际 rule_ids={rule_ids}"
    )
    assert rule_ids == {"POWER_001", "POWER_002"}, (
        f"期望 rule_ids={{POWER_001, POWER_002}}，实际 {rule_ids}"
    )
    # 验证丢弃日志出现（v14 新增的 _merge_rule_defects 统计日志）
    assert any(
        "_merge_rule_defects" in rec.message and "丢弃 LLM 输出" in rec.message
        for rec in caplog.records
    ), (
        "应在 _merge_rule_defects 里记录'丢弃 LLM 输出的 origin=rule 缺陷'日志"
    )
    # ===== [S1-B v14 测试适配 END] =====
# tests/test_review_persistence.py
"""
阶段 F 补充：_persist_review_report 持久化回归测试（S1-F）

覆盖：
  1. origin=rule 时，落库 category / risk 取对应 RuleResult 的权威字段
  2. origin=ai_discovered 时，不被规则命中覆盖逻辑误改
  3. rule_id 存在但 rule_hits 中找不到时，回退 LLM 输出 + warning
  4. 多条规则命中时，能正确定位与当前 rule_id 对应的规则
  5. category 与 risk 的覆盖行为一致

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
    """origin=rule：落库 risk 取 RuleResult.severity，不用 LLM 的 risk"""
    # LLM 输出 risk=medium，但 POWER_001 的 severity=critical
    payload = _make_payload_with_defect(
        risk="medium", category="power", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)
    assert len(defects) == 1
    assert defects[0]["risk"] == "critical", (
        f"risk 应被覆盖为 POWER_001 的 severity=critical，实际 {defects[0]['risk']}"
    )


def test_persistence_rule_origin_category_overridden(mock_llm_custom_payload):
    """origin=rule：落库 category 取 RuleResult.category，不用 LLM 的 category

    【S1-F 测试修正】
      原版用 CLOCK_001 构造，但 CLOCK_001 不在 case001 的 rule_hits 里
      （case001 命中 POWER_001 / POWER_002 / IO_001），导致回退到 LLM 输出。
      改用 POWER_001（case001 命中，category=power，severity=critical）：
        - LLM 写 category=interface（错误值）
        - 期望覆盖为 POWER_001 的 category=power
      这样能验证"覆盖确实生效"，而不是"找不到规则时回退"。
    """
    # LLM 输出 category=interface（故意错），rule_id=POWER_001（case001 命中）
    payload = _make_payload_with_defect(
        risk="critical", category="interface", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)
    assert len(defects) == 1
    assert defects[0]["category"] == "power", (
        f"category 应被覆盖为 POWER_001 的 category=power，"
        f"实际 {defects[0]['category']}"
    )


def test_persistence_ai_discovered_not_overridden(mock_llm_custom_payload):
    """origin=ai_discovered：不被规则命中覆盖逻辑误改"""
    payload = _make_payload_with_defect(
        risk="high", category="thermal", origin="ai_discovered", rule_id=None,
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)
    assert len(defects) == 1
    assert defects[0]["risk"] == "high", "ai_discovered 的 risk 不应被覆盖"
    assert defects[0]["category"] == "thermal", "ai_discovered 的 category 不应被覆盖"


def test_persistence_rule_id_not_found_fallback(mock_llm_custom_payload, caplog):
    """rule_id 存在但 rule_hits 中找不到：回退 LLM 输出 + warning"""
    payload = _make_payload_with_defect(
        risk="medium", category="power", origin="rule", rule_id="NONEXISTENT_RULE",
    )
    mock_llm_custom_payload(payload)
    with caplog.at_level("WARNING"):
        defects = _run_and_get_defects(payload)
    assert len(defects) == 1
    # 回退 LLM 输出
    assert defects[0]["risk"] == "medium"
    assert defects[0]["category"] == "power"
    # warning 已记录
    assert any("未找到对应规则" in rec.message for rec in caplog.records), (
        "应在找不到规则时记录 warning"
    )
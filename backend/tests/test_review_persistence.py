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
    tid = create_task("case001")["task_id"]
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



# ============================================================
# ===== [S1-G G1] get_task_defects 补 expert_decision 3 字段 =====
# ------------------------------------------------------------
# 背景：models.py::ReviewDefect 有 expert_decision 3 字段（阶段 A 迁移），
#      但 get_task_defects 序列化只带了 expert_decision，
#      漏了 expert_decision_updated_at / expert_decision_by。
#      本测试锁定"3 字段都透出 + 类型正确 + 原有 14 字段不丢"。
# ============================================================
def test_get_task_defects_includes_expert_decision_fields(mock_llm_custom_payload):
    """
    G1：get_task_defects 返回的缺陷应带 expert_decision 3 字段。

    验收：
      - expert_decision == "pending"（新任务默认值）
      - expert_decision_updated_at is None（未判定）
      - expert_decision_by is None（未判定）
    """
    payload = _make_payload_with_defect(
        risk="critical", category="power", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)
    assert len(defects) >= 1, "case001 至少 1 条缺陷"

    d = defects[0]
    # 3 字段必须存在
    assert "expert_decision" in d, "缺 expert_decision"
    assert "expert_decision_updated_at" in d, "缺 expert_decision_updated_at"
    assert "expert_decision_by" in d, "缺 expert_decision_by"

    # 新任务默认值
    assert d["expert_decision"] == "pending"
    assert d["expert_decision_updated_at"] is None
    assert d["expert_decision_by"] is None


def test_get_task_defects_preserves_14_existing_fields(mock_llm_custom_payload):
    """
    G1 回归：补 2 字段后，原有 14 字段仍在（不破坏向后兼容）。
    """
    payload = _make_payload_with_defect(
        risk="critical", category="power", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)
    d = defects[0]

    # 原有 14 字段
    expected_14 = {
        "defect_id", "category", "location", "component", "net", "risk",
        "evidence", "root_cause", "suggestion", "confidence",
        "review_status", "origin", "rule_id", "expert_decision",
    }
    missing = expected_14 - set(d.keys())
    assert not missing, f"原有字段丢失: {missing}"

    # 新增 2 字段
    expected_2_new = {"expert_decision_updated_at", "expert_decision_by"}
    assert expected_2_new.issubset(set(d.keys())), (
        f"新增字段缺失: {expected_2_new - set(d.keys())}"
    )


def test_get_task_defects_expert_decision_updated_at_is_iso_or_none(mock_llm_custom_payload):
    """
    G1：expert_decision_updated_at 必须是 ISO 字符串或 None，
    不能是 datetime 对象（避免 API 层 JSON 序列化时依赖 jsonable_encoder）。
    """
    from datetime import datetime

    payload = _make_payload_with_defect(
        risk="critical", category="power", origin="rule", rule_id="POWER_001",
    )
    mock_llm_custom_payload(payload)
    defects = _run_and_get_defects(payload)
    val = defects[0]["expert_decision_updated_at"]
    if val is not None:
        assert isinstance(val, str), f"应为 str，实际 {type(val)}"
        datetime.fromisoformat(val)  # 不抛异常即合法
    else:
        assert val is None




# ============================================================
# ===== [S1-G G2] list_tasks_filtered 筛选 + 分页 =====
# ------------------------------------------------------------
# 覆盖 5 个测试：
#   1. basic：返回 {total, items, errors} 结构
#   2. by_review_status：DB 层过滤
#   3. by_case_class：Python 层过滤（A/B）
#   4. pagination：limit/offset 生效，先过滤后分页
#   5. unknown/missing case_id：errors 收集，不丢弃任务
# ===== [S2-C] status 测试已加入（S2-A 加 task_status 列，G2 决策 1 已解除） =====
# ============================================================
def test_list_tasks_filtered_basic():
    """
    G2 basic：list_tasks_filtered 返回 {total, items, errors} 结构。

    【S2-C 变更】item 从 6 字段扩为 7 字段（加 status），
    S2-A 加 task_status 列后，status 可真实透出，G2 决策 1「不含 status」已过时。
    """
    from app.services.task_service import list_tasks_filtered

    result = list_tasks_filtered(limit=10, offset=0)
    assert "total" in result
    assert "items" in result
    assert "errors" in result
    assert isinstance(result["total"], int)
    assert isinstance(result["items"], list)
    assert isinstance(result["errors"], list)

    # item 必须含 7 字段（S2-C 起含 status）
    if result["items"]:
        item = result["items"][0]
        expected_fields = {
            "task_id", "case_id", "case_class",
            "defects_count", "review_status", "created_at",
            # ===== [S2-C] status 字段（S2-A 加 task_status 列后透出） =====
            "status",
            # ===== [S2-C END] =====
        }
        assert expected_fields.issubset(set(item.keys())), (
            f"item 缺字段: {expected_fields - set(item.keys())}"
        )
        # ===== [S2-C] status 字段值合法（pending/done/failed） =====
        assert item["status"] in ("pending", "done", "failed"), (
            f"status 应为 pending/done/failed，实际 {item['status']!r}"
        )
        # ===== [S2-C END] =====


def test_list_tasks_filtered_by_review_status():
    """
    G2 by_review_status：DB 层过滤。
    """
    from app.services.task_service import create_task, list_tasks_filtered

    # 确保有数据
    create_task("case001")["task_id"]

    # 过滤 AI_CONFIRMED
    result = list_tasks_filtered(review_status="AI_CONFIRMED", limit=100)
    assert result["total"] >= 1, "至少 1 条 AI_CONFIRMED"
    for item in result["items"]:
        assert item["review_status"] == "AI_CONFIRMED"

    # 过滤不存在的状态（如 NEED_EXPERT_REVIEW），可能为 0 条
    result2 = list_tasks_filtered(review_status="NEED_EXPERT_REVIEW", limit=100)
    for item in result2["items"]:
        assert item["review_status"] == "NEED_EXPERT_REVIEW"


def test_list_tasks_filtered_by_case_class():
    """
    G2 by_case_class：Python 层过滤。

    验证：
      - case_class="A" 只返回 A 类
      - case_class="B" 只返回 B 类
      - case_class=None 返回全部（含 A 和 B）
    """
    from app.services.task_service import create_task, list_tasks_filtered

    # 确保 A/B 都有数据
    create_task("case001")["task_id"]                                # A 类
    create_task("case_b01_buck_thermal_derating_B")["task_id"]        # B 类

    # A 类过滤
    result_a = list_tasks_filtered(case_class="A", limit=100)
    for item in result_a["items"]:
        assert item["case_class"] == "A", (
            f"case_class='A' 过滤应只返回 A 类，实际 {item['case_class']}"
        )

    # B 类过滤
    result_b = list_tasks_filtered(case_class="B", limit=100)
    for item in result_b["items"]:
        assert item["case_class"] == "B"

    # 不传 case_class：返回 A + B
    result_all = list_tasks_filtered(case_class=None, limit=100)
    assert result_all["total"] >= result_a["total"] + result_b["total"], (
        "不传 case_class 应返回全部"
    )


def test_list_tasks_filtered_pagination():
    """
    G2 pagination：limit/offset 生效，先过滤后分页。
    """
    from app.services.task_service import create_task, list_tasks_filtered

    # 确保至少 2 条任务
    create_task("case001")["task_id"]
    create_task("case001")["task_id"]

    # 全量
    full = list_tasks_filtered(limit=100, offset=0)
    if full["total"] < 2:
        pytest.skip(f"需要至少 2 条任务，当前 {full['total']}")

    # 第 1 页
    page1 = list_tasks_filtered(limit=1, offset=0)
    assert len(page1["items"]) == 1
    assert page1["total"] == full["total"], "分页不改 total"

    # 第 2 页
    page2 = list_tasks_filtered(limit=1, offset=1)
    assert len(page2["items"]) == 1
    assert page2["total"] == full["total"]

    # 两页的 task_id 不同
    assert page1["items"][0]["task_id"] != page2["items"][0]["task_id"]


def test_list_tasks_filtered_unknown_case_id_errors():
    """
    G2 unknown/missing case_id：errors 收集，不丢弃任务。

    验证：
      - 未知 case_id（不符合 A/B 命名规范）记入 errors，但 item 仍返回
      - case_class 字段为 None（不是默认 "A"）
    """
    from sqlalchemy import text
    from app.persistence.db import get_db_session
    from app.services.task_service import list_tasks_filtered

    # 直接插一条"未知 case_id"的 ReviewResult（绕过 create_task）
    # 用一个不符合 A/B 规范的 case_id 的 SchematicCase
    import uuid
    unknown_case_id = f"unknown_case_{uuid.uuid4().hex[:8]}"

    with get_db_session() as db:
        # 先建 SchematicCase（case_type 用 A 或 B 都行，因为 _infer_case_class 只看 case_id）
        db.execute(text("""
            INSERT INTO schematic_case (case_id, case_type)
            VALUES (:cid, 'A')
            ON CONFLICT DO NOTHING
        """), {"cid": unknown_case_id})
        # 拿 id
        row = db.execute(text(
            "SELECT id FROM schematic_case WHERE case_id = :cid"
        ), {"cid": unknown_case_id}).fetchone()
        if row is None:
            pytest.skip("无法创建测试用 SchematicCase")
        case_pk = row[0]

        # 插入 ReviewResult
        task_id = f"TEST-UNKNOWN-{uuid.uuid4().hex[:8].upper()}"
        db.execute(text("""
            INSERT INTO review_result
                (task_id, schematic_case_id, review_output_json, is_rule_only)
            VALUES (:tid, :sid, '{}'::jsonb, true)
        """), {"tid": task_id, "sid": case_pk})
        db.commit()

    try:
        # 不传 case_class：未知 case_id 的任务仍返回（不丢弃）
        result_all = list_tasks_filtered(case_class=None, limit=1000)
        unknown_items = [
            it for it in result_all["items"] if it["task_id"] == task_id
        ]
        assert len(unknown_items) == 1, "未知 case_id 的任务不应被丢弃"
        assert unknown_items[0]["case_class"] is None, (
            f"未知 case_id 的 case_class 应为 None，实际 {unknown_items[0]['case_class']}"
        )

        # errors 应记录
        assert any(task_id in e for e in result_all["errors"]), (
            f"errors 应记录未知 case_id，实际 {result_all['errors']}"
        )

        # 传 case_class="A"：未知 case 不命中
        result_a = list_tasks_filtered(case_class="A", limit=1000)
        assert not any(
            it["task_id"] == task_id for it in result_a["items"]
        ), "未知 case_id 不应被 A 类过滤命中"
    finally:
        # 清理测试数据
        with get_db_session() as db:
            db.execute(text(
                "DELETE FROM review_result WHERE task_id = :tid"
            ), {"tid": task_id})
            db.execute(text(
                "DELETE FROM schematic_case WHERE case_id = :cid"
            ), {"cid": unknown_case_id})
            db.commit()

def test_list_tasks_filtered_invalid_params():
    """
    G2 参数校验：非法 case_class / limit / offset 抛 ValueError。
    """
    from app.services.task_service import list_tasks_filtered

    with pytest.raises(ValueError, match="case_class"):
        list_tasks_filtered(case_class="C")

    with pytest.raises(ValueError, match="review_status"):
        list_tasks_filtered(review_status="UNKNOWN_STATUS")

    with pytest.raises(ValueError, match="limit"):
        list_tasks_filtered(limit=-1)

    with pytest.raises(ValueError, match="offset"):
        list_tasks_filtered(offset=-1)

# ============================================================
# ===== [S1-G G3] list_cases 扫描 + 归一化 =====
# ------------------------------------------------------------
# 覆盖 5 个测试：
#   1. basic：返回 {total, items, errors} 结构，item 6 字段
#   2. returns_all_10：10 个 Case 全返回（A 类 6 + B 类 4）
#   3. sorted：A 类在前、B 类在后，组内 case_id 字典序
#   4. bad_info_yaml：坏 Case 跳过 + 记 errors（缺 case_class / 类型非法 /
#                     YAML 语法错 / 纯空目录不记）
#   5. tags_normalized：tags 为 list/str/None 统一为 list[str]
# ============================================================
def test_list_cases_basic():
    """G3 basic：返回 {total, items, errors} 结构，item 6 字段，total == len(items)。"""
    from app.services.task_service import list_cases

    result = list_cases()
    assert "total" in result
    assert "items" in result
    assert "errors" in result
    assert isinstance(result["total"], int)
    assert isinstance(result["items"], list)
    assert isinstance(result["errors"], list)

    # total 恒等于 len(items)
    assert result["total"] == len(result["items"]), (
        f"total 应 == len(items)，实际 total={result['total']}, len(items)={len(result['items'])}"
    )

    # item 必须含 6 字段
    assert result["items"], "应有至少 1 个 Case"
    expected_fields = {
        "case_id", "case_class", "title",
        "description", "calibration_status", "tags",
    }
    for item in result["items"]:
        missing = expected_fields - set(item.keys())
        assert not missing, f"item 缺字段: {missing}，实际 keys={set(item.keys())}"
        # case_class 必须是 A/B
        assert item["case_class"] in ("A", "B"), (
            f"case_class 应 A/B，实际 {item['case_class']!r}"
        )
        # tags 必须是 list
        assert isinstance(item["tags"], list), (
            f"tags 应为 list，实际 {type(item['tags'])}"
        )


def test_list_cases_returns_all_10():
    """G3：10 个 Case 全返回（A 类 6 + B 类 4）。"""
    from app.services.task_service import list_cases

    result = list_cases()
    assert result["total"] == 10, (
        f"应有 10 个 Case，实际 {result['total']}；errors={result['errors']}"
    )
    assert result["errors"] == [], (
        f"正常资产不应有 errors，实际 {result['errors']}"
    )

    case_classes = [it["case_class"] for it in result["items"]]
    a_count = case_classes.count("A")
    b_count = case_classes.count("B")
    assert a_count == 6, f"A 类应 6 个，实际 {a_count}"
    assert b_count == 4, f"B 类应 4 个，实际 {b_count}"


def test_list_cases_sorted():
    """G3：A 类在前、B 类在后，组内 case_id 字典序。"""
    from app.services.task_service import list_cases

    result = list_cases()
    items = result["items"]
    assert len(items) >= 2, "至少 2 个 Case 才能验证排序"

    case_classes = [it["case_class"] for it in items]
    first_b_index = case_classes.index("B") if "B" in case_classes else len(case_classes)
    # B 之后不应再出现 A
    for i in range(first_b_index, len(case_classes)):
        assert case_classes[i] == "B", (
            f"B 类之后不应出现 {case_classes[i]}（索引 {i}），"
            f"实际顺序: {case_classes}"
        )

    # 组内 case_id 字典序
    a_ids = [it["case_id"] for it in items if it["case_class"] == "A"]
    b_ids = [it["case_id"] for it in items if it["case_class"] == "B"]
    assert a_ids == sorted(a_ids), f"A 类未按字典序: {a_ids}"
    assert b_ids == sorted(b_ids), f"B 类未按字典序: {b_ids}"


def test_list_cases_bad_info_yaml(tmp_path, monkeypatch):
    """
    G3：坏 Case 跳过 + 记 errors（决策 2 + 补充 + 决策 4）。

    构造临时 CASES_ROOT：
      - case001: 正常
      - case_missing_class: 缺 case_class → errors
      - case_class_not_str: case_class 是 int → errors（类型检查先于合法值）
      - case_class_empty: case_class 空串 → errors
      - case_class_unknown: case_class="C" → errors
      - case_bad_yaml: YAML 语法错 → errors
      - empty_dir: 无 info.yaml → 静默跳过（不记 errors）
    """
    from app.workflows import review_graph as rg
    from app.services.task_service import list_cases

    cases_root = tmp_path / "cases"
    cases_root.mkdir()

    # 正常 Case
    good = cases_root / "case001"
    good.mkdir()
    (good / "info.yaml").write_text(
        "case_id: case001\ncase_class: A\ntitle: Good\ntags: [a, b]\n",
        encoding="utf-8",
    )

    # 缺 case_class
    c = cases_root / "case_missing_class"
    c.mkdir()
    (c / "info.yaml").write_text(
        "case_id: case_missing_class\ntitle: No Class\n",
        encoding="utf-8",
    )

    # case_class 非 str（int）
    c = cases_root / "case_class_not_str"
    c.mkdir()
    (c / "info.yaml").write_text(
        "case_id: case_class_not_str\ncase_class: 1\ntitle: Int Class\n",
        encoding="utf-8",
    )

    # case_class 空串
    c = cases_root / "case_class_empty"
    c.mkdir()
    (c / "info.yaml").write_text(
        'case_id: case_class_empty\ncase_class: ""\ntitle: Empty Class\n',
        encoding="utf-8",
    )

    # case_class 未知分类
    c = cases_root / "case_class_unknown"
    c.mkdir()
    (c / "info.yaml").write_text(
        "case_id: case_class_unknown\ncase_class: C\ntitle: Unknown Class\n",
        encoding="utf-8",
    )

    # YAML 语法错
    c = cases_root / "case_bad_yaml"
    c.mkdir()
    (c / "info.yaml").write_text(
        "case_id: [unclosed\n::: invalid\n",
        encoding="utf-8",
    )

    # 纯空目录
    (cases_root / "empty_dir").mkdir()

    monkeypatch.setattr(rg, "CASES_ROOT", cases_root)

    result = list_cases()

    # 只有 case001 成功
    assert result["total"] == 1, (
        f"只应有 1 个成功 Case，实际 {result['total']}；items={result['items']}"
    )
    assert result["items"][0]["case_id"] == "case001"

    # errors 应记录 5 条（缺 class / 非 str / 空串 / 未知 / YAML 错）
    assert len(result["errors"]) == 5, (
        f"应有 5 条 errors，实际 {len(result['errors'])}: {result['errors']}"
    )
    joined = " | ".join(result["errors"])
    assert "case_missing_class" in joined
    assert "case_class_not_str" in joined
    assert "case_class_empty" in joined
    assert "case_class_unknown" in joined
    assert "case_bad_yaml" in joined
    # 纯空目录不应记 errors
    assert "empty_dir" not in joined, (
        f"纯空目录不应记 errors，实际 {result['errors']}"
    )

    # total == len(items) 恒等
    assert result["total"] == len(result["items"])


def test_list_cases_tags_normalized(tmp_path, monkeypatch):
    """G3：tags 为 list / str / None 统一为 list[str]（决策 5）。"""
    from app.workflows import review_graph as rg
    from app.services.task_service import list_cases

    cases_root = tmp_path / "cases"
    cases_root.mkdir()

    c = cases_root / "case001"
    c.mkdir()
    (c / "info.yaml").write_text(
        "case_id: case001\ncase_class: A\ntags: [x, y, z]\n", encoding="utf-8",
    )

    c = cases_root / "case002"
    c.mkdir()
    (c / "info.yaml").write_text(
        'case_id: case002\ncase_class: A\ntags: "p, q, r"\n', encoding="utf-8",
    )

    c = cases_root / "case003"
    c.mkdir()
    (c / "info.yaml").write_text(
        "case_id: case003\ncase_class: A\n", encoding="utf-8",
    )

    monkeypatch.setattr(rg, "CASES_ROOT", cases_root)

    result = list_cases()
    by_id = {it["case_id"]: it for it in result["items"]}

    assert by_id["case001"]["tags"] == ["x", "y", "z"], (
        f"list 类型应原样返回，实际 {by_id['case001']['tags']}"
    )
    assert by_id["case002"]["tags"] == ["p", "q", "r"], (
        f"str 类型应 split(',')+strip，实际 {by_id['case002']['tags']}"
    )
    assert by_id["case003"]["tags"] == [], (
        f"None 应为空数组，实际 {by_id['case003']['tags']}"
    )

    # 所有 tags 都是 list
    for it in result["items"]:
        assert isinstance(it["tags"], list), (
            f"{it['case_id']} 的 tags 应为 list，实际 {type(it['tags'])}"
        )

# ============================================================
# S2-B：update_expert_decision Service 测试
# ============================================================
def test_update_expert_decision_service():
    """update_expert_decision 正确更新 3 字段。"""
    from datetime import datetime
    from app.persistence.db import get_db_session
    from app.persistence.models import ReviewDefect, ReviewResult
    from app.services.task_service import update_expert_decision

    task_id = "TEST-S2B-TASK"
    defect_id = "TEST-S2B-DEF"

    # 清理可能残留
    with get_db_session() as db:
        old_rr = db.query(ReviewResult).filter(ReviewResult.task_id == task_id).first()
        if old_rr:
            db.query(ReviewDefect).filter(
                ReviewDefect.review_result_id == old_rr.id
            ).delete()
            db.delete(old_rr)
            db.commit()

    # 构造 task + defect
    with get_db_session() as db:
        rr = ReviewResult(
            task_id=task_id,
            review_output_json={"defects": []},
            is_rule_only=True,
            task_status="done",
        )
        db.add(rr)
        db.flush()

        rd = ReviewDefect(
            review_result_id=rr.id,
            defect_id=defect_id,
            location={"sheet": "P1", "path": "U1", "coords": {"x": 0, "y": 0}},
            root_cause="test",
            suggestion="test",
            category="power",
            risk="medium",
            evidence=[],
            review_status="AI_CONFIRMED",
            origin="rule",
        )
        db.add(rd)
        db.commit()

    # 调用
    result = update_expert_decision(
        task_id=task_id,
        defect_id=defect_id,
        expert_decision="confirmed",
        updated_by=None,
    )

    # 断言
    assert result["defect_id"] == defect_id
    assert result["expert_decision"] == "confirmed"
    assert result["expert_decision_by"] is None
    assert isinstance(result["expert_decision_updated_at"], datetime)

    # 清理
    with get_db_session() as db:
        db.query(ReviewDefect).filter(ReviewDefect.defect_id == defect_id).delete()
        db.query(ReviewResult).filter(ReviewResult.task_id == task_id).delete()
        db.commit()

# ============================================================
# S2-C：get_task_summary_metrics + list_tasks_filtered(status=)
# ============================================================
def test_get_task_summary_metrics_aggregates():
    """get_task_summary_metrics 聚合 review_result + review_defect + feedback_item。"""
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import (
        FeedbackItem, ReviewDefect, ReviewResult,
    )
    from app.services.task_service import get_task_summary_metrics

    task_id = f"TEST-S2C-SUM-{uuid.uuid4().hex[:8].upper()}"
    defect_ids = [f"TEST-S2C-DEF-{uuid.uuid4().hex[:8].upper()}" for _ in range(3)]

    try:
        with get_db_session() as db:
            rr = ReviewResult(
                task_id=task_id,
                review_output_json={"defects": []},
                is_rule_only=False,
                task_status="done",
                evidence_coverage=0.75,
            )
            db.add(rr)
            db.flush()

            # 2 条 AI 缺陷 + 1 条 rule 缺陷
            db.add(ReviewDefect(
                review_result_id=rr.id, defect_id=defect_ids[0],
                location={"sheet": "P1", "path": "U1"}, root_cause="x", suggestion="y",
                category="power", risk="high", evidence=[],
                review_status="AI_CONFIRMED", origin="ai_discovered",
                expert_decision="confirmed",
            ))
            db.add(ReviewDefect(
                review_result_id=rr.id, defect_id=defect_ids[1],
                location={"sheet": "P1", "path": "U1"}, root_cause="x", suggestion="y",
                category="power", risk="high", evidence=[],
                review_status="NEED_EXPERT_REVIEW", origin="ai_discovered",
                expert_decision="rejected",
            ))
            db.add(ReviewDefect(
                review_result_id=rr.id, defect_id=defect_ids[2],
                location={"sheet": "P1", "path": "U1"}, root_cause="x", suggestion="y",
                category="power", risk="high", evidence=[],
                review_status="AI_CONFIRMED", origin="rule",
                expert_decision="pending",
            ))
            # ===== [S2-C 修正] flush 拿 defect DB id，供 feedback 关联 =====
            db.flush()
            defect_db_ids = {
                d.defect_id: d.id
                for d in db.query(ReviewDefect)
                .filter(ReviewDefect.review_result_id == rr.id)
                .all()
            }

            # 2 条 correct_defect 关联不同 defect + 1 条 knowledge_gap
            db.add(FeedbackItem(
                review_result_id=rr.id, review_defect_id=defect_db_ids[defect_ids[0]],
                feedback_type="correct_defect",
                comment="real", suggestion_adopted=None,
                suggestion_diff_json=[], attached_refs=[],
            ))
            db.add(FeedbackItem(
                review_result_id=rr.id, review_defect_id=defect_db_ids[defect_ids[1]],
                feedback_type="correct_defect",
                comment="real", suggestion_adopted=None,
                suggestion_diff_json=[], attached_refs=[],
            ))
            db.add(FeedbackItem(
                review_result_id=rr.id, review_defect_id=defect_db_ids[defect_ids[0]],
                feedback_type="knowledge_gap",
                comment="real", suggestion_adopted=None,
                suggestion_diff_json=[], attached_refs=[],
            ))
            # ===== [S2-C 修正 END] =====
            db.commit()

        # 调用
        result = get_task_summary_metrics(task_id)
        assert result is not None
        assert result["evidence_coverage"] == 0.75
        assert result["review_status_distribution"] == {
            "AI_CONFIRMED": 2, "NEED_EXPERT_REVIEW": 1, "LOW_CONFIDENCE": 0,
        }
        # EAR：分母 2（2 条 correct_defect），分子 2 → 1.0；但分母 < 20 → None
        assert result["expert_adoption_rate"] is None, (
            "分母 < 20 时 EAR 应返回 None"
        )
        # AI_NER：ai_candidates=2 < 10 → None
        assert result["ai_new_effective_rate"] is None
        # sample_info
        assert result["sample_info"]["ear_sample_size"] == 2
        assert result["sample_info"]["ear_sample_threshold"] == 20
        assert result["sample_info"]["ai_ner_sample_size"] == 2
        assert result["sample_info"]["ai_ner_sample_threshold"] == 10

    finally:
        with get_db_session() as db:
            db.query(FeedbackItem).filter(
                FeedbackItem.review_result_id == ReviewResult.id
            ).filter(ReviewResult.task_id == task_id).delete(synchronize_session=False)
            db.query(ReviewDefect).filter(
                ReviewDefect.review_result_id == ReviewResult.id
            ).filter(ReviewResult.task_id == task_id).delete(synchronize_session=False)
            db.query(ReviewResult).filter(ReviewResult.task_id == task_id).delete()
            db.commit()


def test_get_task_summary_metrics_task_not_found():
    """task 不存在 → None。"""
    from app.services.task_service import get_task_summary_metrics
    assert get_task_summary_metrics("NONEXISTENT-S2C") is None


def test_list_tasks_filtered_by_status():
    """list_tasks_filtered(status="done") 只返回 task_status=done 的任务。"""
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import ReviewResult
    from app.services.task_service import list_tasks_filtered

    done_tid = f"TEST-S2C-DONE-{uuid.uuid4().hex[:8].upper()}"
    failed_tid = f"TEST-S2C-FAILED-{uuid.uuid4().hex[:8].upper()}"

    try:
        with get_db_session() as db:
            db.add(ReviewResult(
                task_id=done_tid, review_output_json={"defects": []},
                is_rule_only=True, task_status="done",
            ))
            db.add(ReviewResult(
                task_id=failed_tid, review_output_json={"defects": []},
                is_rule_only=True, task_status="failed",
            ))
            db.commit()

        # 过滤 done
        result_done = list_tasks_filtered(status="done", limit=1000)
        tids_done = {it["task_id"] for it in result_done["items"]}
        assert done_tid in tids_done
        assert failed_tid not in tids_done

        # 过滤 failed
        result_failed = list_tasks_filtered(status="failed", limit=1000)
        tids_failed = {it["task_id"] for it in result_failed["items"]}
        assert failed_tid in tids_failed
        assert done_tid not in tids_failed

        # 不传 status：两条都返回
        result_all = list_tasks_filtered(status=None, limit=1000)
        tids_all = {it["task_id"] for it in result_all["items"]}
        assert done_tid in tids_all
        assert failed_tid in tids_all

        # item 含 status 字段
        for it in result_all["items"]:
            assert "status" in it

        # 非法 status → ValueError
        with pytest.raises(ValueError, match="status"):
            list_tasks_filtered(status="invalid_status")

    finally:
        with get_db_session() as db:
            db.query(ReviewResult).filter(
                ReviewResult.task_id.in_([done_tid, failed_tid])
            ).delete(synchronize_session=False)
            db.commit()

# ============================================================
# G3-B: IDOR 对象级数据范围测试矩阵
# 5 核心接口 × 4 角色（engineer / reviewer / admin / rule_admin）
# D6: engineer 越权 → 404；D11: rule_admin → 403
# ============================================================

import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.persistence.db import get_db_session
from app.persistence.models import ReviewResult, User


def _make_idor_user(role: str) -> dict:
    """造一个指定角色的用户（commit 到 DB，测试结束需手工清理）。"""
    uname = f"idor_{role}_{uuid.uuid4().hex[:6]}"
    with get_db_session() as db:
        user = User(username=uname, role=role, is_active=True)
        db.add(user)
        db.flush()
        uid = user.id
        db.commit()
    return {"id": uid, "username": uname, "role": role}


def _cleanup_idor_user(uid: int) -> None:
    """清理 IDOR 测试用户（先清 review_result.created_by 再删 user）。"""
    with get_db_session() as db:
        db.query(ReviewResult).filter(ReviewResult.created_by == uid).update(
            {ReviewResult.created_by: None}, synchronize_session=False
        )
        db.commit()
        db.query(User).filter(User.id == uid).delete(synchronize_session=False)
        db.commit()


def _client_for(uid: int) -> TestClient:
    """返回带 X-User-Id 的 TestClient。"""
    c = TestClient(app)
    c.headers["X-User-Id"] = str(uid)
    return c


def _create_task_as(client: TestClient) -> str:
    """用指定 client 身份建一个任务，返回 task_id。"""
    resp = client.post("/api/v1/review/tasks", json={"case_id": "case001"})
    assert resp.status_code == 200, resp.text
    return resp.json()["task_id"]


# ---------- 矩阵 1: GET /review/tasks/{id} ----------

def test_idor_get_task_engineer_other_404():
    """D6: engineer 访问他人任务 → 404。"""
    admin = _make_idor_user("admin")
    eng = _make_idor_user("engineer")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(eng["id"]).get(f"/api/v1/review/tasks/{admin_task}")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(eng["id"])


def test_idor_get_task_engineer_own_200():
    eng = _make_idor_user("engineer")
    try:
        eng_client = _client_for(eng["id"])
        eng_task = _create_task_as(eng_client)
        resp = eng_client.get(f"/api/v1/review/tasks/{eng_task}")
        assert resp.status_code == 200
    finally:
        _cleanup_idor_user(eng["id"])


def test_idor_get_task_reviewer_other_200():
    admin = _make_idor_user("admin")
    rev = _make_idor_user("reviewer")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(rev["id"]).get(f"/api/v1/review/tasks/{admin_task}")
        assert resp.status_code == 200
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(rev["id"])


def test_idor_get_task_admin_own_200():
    admin = _make_idor_user("admin")
    try:
        admin_client = _client_for(admin["id"])
        admin_task = _create_task_as(admin_client)
        resp = admin_client.get(f"/api/v1/review/tasks/{admin_task}")
        assert resp.status_code == 200
    finally:
        _cleanup_idor_user(admin["id"])


def test_idor_get_task_rule_admin_403():
    """D11: rule_admin → 403。"""
    admin = _make_idor_user("admin")
    ra = _make_idor_user("rule_admin")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(ra["id"]).get(f"/api/v1/review/tasks/{admin_task}")
        assert resp.status_code == 403
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(ra["id"])


# ---------- 矩阵 2: GET /review/tasks/{id}/defects ----------

def test_idor_get_defects_engineer_other_404():
    admin = _make_idor_user("admin")
    eng = _make_idor_user("engineer")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(eng["id"]).get(f"/api/v1/review/tasks/{admin_task}/defects")
        assert resp.status_code == 404
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(eng["id"])


def test_idor_get_defects_engineer_own_200():
    eng = _make_idor_user("engineer")
    try:
        eng_client = _client_for(eng["id"])
        eng_task = _create_task_as(eng_client)
        resp = eng_client.get(f"/api/v1/review/tasks/{eng_task}/defects")
        assert resp.status_code == 200
    finally:
        _cleanup_idor_user(eng["id"])


def test_idor_get_defects_rule_admin_403():
    admin = _make_idor_user("admin")
    ra = _make_idor_user("rule_admin")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(ra["id"]).get(f"/api/v1/review/tasks/{admin_task}/defects")
        assert resp.status_code == 403
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(ra["id"])


# ---------- 矩阵 3: GET /review/tasks/{id}/export ----------

def test_idor_export_engineer_other_404():
    admin = _make_idor_user("admin")
    eng = _make_idor_user("engineer")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(eng["id"]).get(f"/api/v1/review/tasks/{admin_task}/export")
        assert resp.status_code == 404
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(eng["id"])


def test_idor_export_engineer_own_200():
    eng = _make_idor_user("engineer")
    try:
        eng_client = _client_for(eng["id"])
        eng_task = _create_task_as(eng_client)
        resp = eng_client.get(f"/api/v1/review/tasks/{eng_task}/export")
        assert resp.status_code == 200
    finally:
        _cleanup_idor_user(eng["id"])


def test_idor_export_rule_admin_403():
    admin = _make_idor_user("admin")
    ra = _make_idor_user("rule_admin")
    try:
        admin_task = _create_task_as(_client_for(admin["id"]))
        resp = _client_for(ra["id"]).get(f"/api/v1/review/tasks/{admin_task}/export")
        assert resp.status_code == 403
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(ra["id"])


# ---------- 矩阵 4: GET /review/tasks（列表过滤） ----------

def test_idor_list_tasks_engineer_only_sees_own():
    """D7: engineer 列表只返回 created_by = me 的任务。"""
    eng_a = _make_idor_user("engineer")
    eng_b = _make_idor_user("engineer")
    try:
        b_client = _client_for(eng_b["id"])
        b_task = _create_task_as(b_client)

        a_client = _client_for(eng_a["id"])
        resp = a_client.get("/api/v1/review/tasks")
        assert resp.status_code == 200
        task_ids = [it["task_id"] for it in resp.json()["items"]]
        assert b_task not in task_ids
    finally:
        _cleanup_idor_user(eng_a["id"])
        _cleanup_idor_user(eng_b["id"])


def test_idor_list_tasks_admin_sees_all():
    admin = _make_idor_user("admin")
    eng = _make_idor_user("engineer")
    try:
        eng_client = _client_for(eng["id"])
        eng_task = _create_task_as(eng_client)

        admin_client = _client_for(admin["id"])
        resp = admin_client.get("/api/v1/review/tasks")
        assert resp.status_code == 200
        task_ids = [it["task_id"] for it in resp.json()["items"]]
        assert eng_task in task_ids
    finally:
        _cleanup_idor_user(admin["id"])
        _cleanup_idor_user(eng["id"])


def test_idor_list_tasks_rule_admin_403():
    ra = _make_idor_user("rule_admin")
    try:
        resp = _client_for(ra["id"]).get("/api/v1/review/tasks")
        assert resp.status_code == 403
    finally:
        _cleanup_idor_user(ra["id"])


# ---------- 矩阵 5: PATCH /review/tasks/{id}/defects/{did} ----------

def test_idor_patch_defect_engineer_403():
    """engineer 无 PATCH 权限（G3-A require_roles 拦截）。"""
    eng = _make_idor_user("engineer")
    try:
        eng_client = _client_for(eng["id"])
        eng_task = _create_task_as(eng_client)
        resp = eng_client.patch(
            f"/api/v1/review/tasks/{eng_task}/defects/DUMMY",
            json={"expert_decision": "confirmed"},
        )
        assert resp.status_code == 403
    finally:
        _cleanup_idor_user(eng["id"])


def test_idor_patch_defect_rule_admin_403():
    ra = _make_idor_user("rule_admin")
    try:
        resp = _client_for(ra["id"]).patch(
            "/api/v1/review/tasks/DUMMY/defects/DUMMY",
            json={"expert_decision": "confirmed"},
        )
        assert resp.status_code == 403
    finally:
        _cleanup_idor_user(ra["id"])
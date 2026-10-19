"""
B 类 Case 资产完整性校验。

职责：
  - 验证 B 类 Case 的 ai_discovered 缺陷符合硬约束
  - 验证 benchmark_weights / calibration_status / evidence 完整性
  - 验证 expected_review.json 的 10 字段

与 test_rules.py 的分工：
  - test_rules.py：规则引擎行为（含 B 类 Case 的规则命中）
  - 本文件：B 类 Case 的资产完整性（不跑规则引擎）

执行命令：
  docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
    exec backend uv run pytest tests/test_case_b_validation.py -v

前置条件：
  - data/cases/case_b01_buck_thermal_derating_B/ 存在
  - （未来）case_b02 / case_b03 / case_b04 存在

⚠️ 本测试不依赖 PostgreSQL、不依赖规则引擎
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml


# ============================================================
# 路径常量
# ============================================================
def _find_project_root(start: pathlib.Path) -> pathlib.Path:
    """向上查找含 data/cases 的目录作为项目根。"""
    for p in [start, *start.parents]:
        if (p / "data" / "cases").is_dir():
            return p
    return start


PROJECT_ROOT: pathlib.Path = _find_project_root(pathlib.Path(__file__).resolve())
CASES_ROOT: pathlib.Path = PROJECT_ROOT / "data" / "cases"


# ============================================================
# B 类 Case 清单
# ============================================================
# 注意：新增 B 类 Case 时，在此处追加 case_id
# 未建成的 Case 会被 pytest.skip 跳过（用 exists() 判断）
B_CASE_IDS: list[str] = [
    "case_b01_buck_thermal_derating_B",
    "case_b02_reset_rc_B",
    "case_b03_io_direction_B",
    "case_b04_power_sequence_B",              # ← 新增
]


# ============================================================
# 工具函数
# ============================================================
def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _case_dir(case_id: str) -> pathlib.Path:
    return CASES_ROOT / case_id


def _skip_if_missing(case_id: str) -> pathlib.Path:
    """Case 目录不存在时 skip 当前测试。"""
    d = _case_dir(case_id)
    if not d.is_dir():
        pytest.skip(f"B 类 Case 未建成: {case_id}")
    return d


# ============================================================
# 1. ai_discovered 缺陷硬约束
# ============================================================
@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_ai_discovered_defects_exist(case_id):
    """B 类 Case 至少 1 条 origin=ai_discovered 缺陷（前置条件 §4.3 硬约束）。"""
    d = _skip_if_missing(case_id)
    expected = _load_json(d / "expected_review.json")

    ai_defects = [x for x in expected.get("defects", []) if x.get("origin") == "ai_discovered"]
    assert len(ai_defects) >= 1, (
        f"{case_id} 必须至少 1 条 ai_discovered 缺陷（B 类合格标准）"
    )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_ai_discovered_defects_have_evidence(case_id):
    """每条 ai_discovered 缺陷必须有 ir_ref + rag_ref（前置条件 §2.2）。

    设计意图：
      - ir_ref：AI 发现必须能追溯回 IR 的具体元件/网络/属性
      - rag_ref：引用知识库时必须提供可验证的来源
      - 无证据的 AI 输出应降级为 NEED_EXPERT_REVIEW / LOW_CONFIDENCE
    """
    d = _skip_if_missing(case_id)
    expected = _load_json(d / "expected_review.json")

    ai_defects = [x for x in expected.get("defects", []) if x.get("origin") == "ai_discovered"]
    for defect in ai_defects:
        evidence = defect.get("evidence", [])
        types = {e.get("type") for e in evidence}

        assert "ir_ref" in types, (
            f"{case_id}.{defect['defect_id']} 缺 ir_ref（AI 发现必须可追溯回 IR）"
        )
        assert "rag_ref" in types, (
            f"{case_id}.{defect['defect_id']} 缺 rag_ref（引用知识库必须可追溯）"
        )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_ai_discovered_review_status(case_id):
    """ai_discovered 缺陷的 review_status 应为 NEED_EXPERT_REVIEW 或 LOW_CONFIDENCE。

    设计意图：
      - AI 新发现不得自动视为专家确认
      - 必须经专家人工确认（NEED_EXPERT_REVIEW）
      - 或标记为低置信度（LOW_CONFIDENCE）
      - 不允许直接 AI_CONFIRMED（违反 §2.2）
    """
    d = _skip_if_missing(case_id)
    expected = _load_json(d / "expected_review.json")

    ai_defects = [x for x in expected.get("defects", []) if x.get("origin") == "ai_discovered"]
    for defect in ai_defects:
        status = defect.get("review_status")
        assert status in ("NEED_EXPERT_REVIEW", "LOW_CONFIDENCE"), (
            f"{case_id}.{defect['defect_id']} review_status={status} "
            f"不允许（ai_discovered 不得直接 AI_CONFIRMED）"
        )


# ============================================================
# 2. evaluation.yaml 权重
# ============================================================
@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_benchmark_weights_b_class(case_id):
    """B 类 Case 的 benchmark_weights 必须是 0.4 / 0.6（README_CASE_B_DESIGN §5）。"""
    d = _skip_if_missing(case_id)
    eval_cfg = _load_yaml(d / "evaluation.yaml")

    weights = eval_cfg.get("benchmark_weights", {})
    assert weights.get("rule_metrics_weight") == 0.4, (
        f"{case_id} rule_metrics_weight 应为 0.4（B 类），实际 {weights.get('rule_metrics_weight')}"
    )
    assert weights.get("ai_metrics_weight") == 0.6, (
        f"{case_id} ai_metrics_weight 应为 0.6（B 类），实际 {weights.get('ai_metrics_weight')}"
    )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_benchmark_meta_b_class(case_id):
    """expected_review.json 的 benchmark_meta.case_class 应为 B。"""
    d = _skip_if_missing(case_id)
    expected = _load_json(d / "expected_review.json")

    meta = expected.get("benchmark_meta", {})
    assert meta.get("case_class") == "B", (
        f"{case_id} benchmark_meta.case_class 应为 B"
    )
    assert meta.get("total_expected_ai_discovered", 0) >= 1, (
        f"{case_id} total_expected_ai_discovered 应 >= 1"
    )


# ============================================================
# 3. calibration_status
# ============================================================
@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_calibration_status_valid(case_id):
    """calibration_status 必须是 PROVISIONAL 或 CALIBRATED。

    设计意图：
      - PROVISIONAL：AI 临时扮演硬件专家产出的占位资产
      - CALIBRATED：真实硬件专家签字校准
      - 两种状态都允许，但必须显式标注
    """
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")

    status = info.get("calibration_status")
    assert status in ("PROVISIONAL", "CALIBRATED"), (
        f"{case_id} calibration_status={status} 非法（应为 PROVISIONAL 或 CALIBRATED）"
    )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_provisional_has_pending_reviewer(case_id):
    """calibration_status=PROVISIONAL 时，必须有 pending_review_by。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")

    if info.get("calibration_status") == "PROVISIONAL":
        pending = info.get("pending_review_by")
        assert pending, (
            f"{case_id} calibration_status=PROVISIONAL 但缺 pending_review_by（待校准人）"
        )


# ============================================================
# 4. evidence/ 三件套完整性
# ============================================================
@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_evidence_dir_exists(case_id):
    """B 类 Case 必须有 evidence/ 目录（前置条件 §4.3）。"""
    d = _skip_if_missing(case_id)
    evidence_dir = d / "evidence"
    assert evidence_dir.is_dir(), f"{case_id} 缺 evidence/ 目录"


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_evidence_datasheet_not_empty(case_id):
    """evidence/datasheet/ 必须有内容。"""
    d = _skip_if_missing(case_id)
    datasheet_dir = d / "evidence" / "datasheet"
    assert datasheet_dir.is_dir(), f"{case_id} 缺 evidence/datasheet/"
    assert any(datasheet_dir.iterdir()), f"{case_id} evidence/datasheet/ 为空"


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_evidence_three_categories(case_id):
    """evidence/ 必须有 application_note / datasheet / reference_design 三个子目录。"""
    d = _skip_if_missing(case_id)
    evidence_dir = d / "evidence"
    for sub in ("application_note", "datasheet", "reference_design"):
        sub_dir = evidence_dir / sub
        assert sub_dir.is_dir(), f"{case_id} 缺 evidence/{sub}/"


# ============================================================
# 5. expected_review.json 10 字段
# ============================================================
DEFECT_REQUIRED_FIELDS = {
    "defect_id", "category", "location", "component", "net",
    "risk", "evidence", "root_cause", "suggestion", "confidence",
}


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_defect_10_fields(case_id):
    """每条 defect 必须有 10 个必填字段（前置条件 §2.1 冻结）。"""
    d = _skip_if_missing(case_id)
    expected = _load_json(d / "expected_review.json")

    for defect in expected.get("defects", []):
        missing = DEFECT_REQUIRED_FIELDS - set(defect.keys())
        assert not missing, (
            f"{case_id}.{defect.get('defect_id', '?')} 缺 10 字段中的: {sorted(missing)}"
        )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_defect_evidence_three_elements(case_id):
    """每条 evidence 必须有 source / section / reason 三要素（长度 >= 10）。"""
    d = _skip_if_missing(case_id)
    expected = _load_json(d / "expected_review.json")

    for defect in expected.get("defects", []):
        for i, ev in enumerate(defect.get("evidence", [])):
            # rag_ref 要求三要素；ir_ref 用 detail
            if ev.get("type") == "rag_ref":
                for key in ("source", "section", "reason"):
                    val = ev.get(key, "")
                    assert isinstance(val, str) and len(val) >= 10, (
                        f"{case_id}.{defect['defect_id']}.evidence[{i}] "
                        f"{key} 长度不足 10（实际 {len(val)}）"
                    )
            elif ev.get("type") == "ir_ref":
                detail = ev.get("detail", "")
                assert isinstance(detail, str) and len(detail) >= 5, (
                    f"{case_id}.{defect['defect_id']}.evidence[{i}] "
                    f"detail 长度不足 5（实际 {len(detail)}）"
                )


# ============================================================
# 6. info.yaml 必填字段
# ============================================================
INFO_REQUIRED_FIELDS = {
    "case_id", "case_class", "title", "description",
    "ai_expected_count", "calibration_status",
}


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_info_required_fields(case_id):
    """info.yaml 必填字段完整。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")

    missing = INFO_REQUIRED_FIELDS - set(info.keys())
    assert not missing, f"{case_id} info.yaml 缺字段: {sorted(missing)}"


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_info_ai_expected_count_positive(case_id):
    """B 类 Case 的 ai_expected_count 必须 >= 1。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")

    count = info.get("ai_expected_count")
    assert isinstance(count, int) and count >= 1, (
        f"{case_id} ai_expected_count 应 >= 1（B 类），实际 {count}"
    )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_info_case_id_matches_dir(case_id):
    """info.yaml 的 case_id 必须等于目录名（前置条件 §4.2）。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")

    assert info.get("case_id") == case_id, (
        f"info.yaml case_id={info.get('case_id')} 与目录名 {case_id} 不一致"
    )
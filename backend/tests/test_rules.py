"""
Phase-F 规则引擎测试（pytest + CLI 入口）

覆盖内容：
    - 规则 YAML 加载（恰好 5 条：POWER_001/002/003 + IO_001/002）
    - 规则必填字段完整性（对齐 Rule YAML Format v1.0）
    - POWER_001 在 case001 命中 VCC 去耦缺失
    - POWER_002 在 case001 命中 RST 拓扑异常
    - POWER_003 在 case001 不命中（Sprint0 预留）
    - IO_001 在 case001 命中（U1 的 32 个 IO 引脚全悬空，汇总一条）
    - IO_002 在 case001 不命中（全为 BIDIRECTIONAL，无输出冲突）
    - execute_defect_rules 排除 low 级提示（IO_001）
    - RuleResult 关键字段非空 + evidence_ir_refs 结构化
    - Sprint0 K7 锚点：case001 Rule Hit >= 2（仅计缺陷，不含 low 提示）

执行命令：
    # 全部测试（pytest，CI 使用）
    docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
      exec backend uv run pytest tests/test_rules.py -v

    # 单 case CLI 验证（demo 使用）
    docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
      exec backend uv run python -m tests.test_rules --case case001

前置条件：
    1. 容器已启动
    2. data/cases/case001/schematic_ir.json 已就位
    3. data/rules/POWER_00{1,2,3}.yaml + IO_00{1,2}.yaml 已就位

⚠️ 本测试不依赖 PostgreSQL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pytest

# ⚠️ V1.3 修改：import 块扩充（新测试 test_io002_bidirectional_counted_when_flag_false 需要）
# [原逻辑 - 保留注释，便于对照回滚]
# from app.ir.schema import SchematicIRDocument
# from app.ir.serializer import load_ir
# from app.rules.engine import (
#     RuleResult,
#     execute_all_rules,
#     execute_defect_rules,
#     load_rule_definitions,
# )
from app.ir.schema import (
    Component,
    Net,
    Pin,
    PinDirection,
    PinType,
    SchematicIRDocument,
)
from app.ir.serializer import load_ir
from app.rules.builtin.circuit_checks import check_io_direction
from app.rules.engine import (
    RuleResult,
    execute_all_rules,
    execute_defect_rules,
    load_rule_definitions,
)


# ============================================================
# 路径常量
# 向上查找含 data/cases 的目录作为项目根，兼容容器与本地
# ============================================================
def _find_project_root(start: Path) -> Path:
    """向上查找含 data/cases 的目录作为项目根。"""
    for p in [start, *start.parents]:
        if (p / "data" / "cases").is_dir():
            return p
    return start  # fallback：找不到时退回起点


PROJECT_ROOT: Path = _find_project_root(Path(__file__).resolve())
CASES_ROOT: Path = PROJECT_ROOT / "data" / "cases"
RULES_ROOT: Path = PROJECT_ROOT / "data" / "rules"

CASE001_DIR: Path = CASES_ROOT / "case001"
CASE001_IR: Path = CASE001_DIR / "schematic_ir.json"


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def case001_ir() -> SchematicIRDocument:
    """case001 的 IR 文档（模块级缓存，只加载一次）。"""
    assert CASE001_IR.exists(), f"IR not found: {CASE001_IR}"
    return load_ir(CASE001_IR)


@pytest.fixture(scope="module")
def case001_results(case001_ir) -> List[RuleResult]:
    """case001 跑全部规则的命中结果（模块级缓存，含 low 级提示）。"""
    assert RULES_ROOT.exists(), f"rules dir not found: {RULES_ROOT}"
    return execute_all_rules(case001_ir, RULES_ROOT)


@pytest.fixture(scope="module")
def case001_defects(case001_ir) -> List[RuleResult]:
    """case001 只含缺陷（severity ∈ {medium, high, critical}）的命中结果。"""
    assert RULES_ROOT.exists(), f"rules dir not found: {RULES_ROOT}"
    return execute_defect_rules(case001_ir, RULES_ROOT)


# ============================================================
# 规则加载测试
# ============================================================

def test_rules_loaded_exactly_five():
    """Phase F + IO 扩展：加载恰好 5 条规则 POWER_001/002/003 + IO_001/002。"""
    rules = load_rule_definitions(RULES_ROOT)
    rule_ids = sorted(r.rule_id for r in rules)
    assert rule_ids == ["IO_001", "IO_002", "POWER_001", "POWER_002", "POWER_003"], (
        f"应加载 5 条规则，实际 {rule_ids}"
    )


def test_rules_all_enabled_by_default():
    """所有规则默认 enabled=True。"""
    rules = load_rule_definitions(RULES_ROOT)
    for r in rules:
        assert r.enabled is True, f"{r.rule_id} enabled 应为 True"


def test_rules_required_fields_present():
    """规则必填字段非空（对齐 Rule YAML Format v1.0）。"""
    rules = load_rule_definitions(RULES_ROOT)
    for r in rules:
        assert r.rule_id, "rule_id 为空"
        assert r.rule_name, f"{r.rule_id} rule_name 为空"
        assert r.version, f"{r.rule_id} version 为空"
        assert r.category, f"{r.rule_id} category 为空"
        assert r.severity in {"low", "medium", "high", "critical"}, (
            f"{r.rule_id} severity 非法: {r.severity}"
        )
        assert r.rule_basis, f"{r.rule_id} rule_basis 为空"
        assert r.suggestion, f"{r.rule_id} suggestion 为空"
        assert "function" in r.check_logic or "func" in r.check_logic, (
            f"{r.rule_id} check_logic 缺 function/func"
        )


# ============================================================
# Sprint0 K7 锚点（仅计缺陷，不含 low 提示）
# ============================================================

def test_case001_defect_hit_at_least_2(case001_defects):
    """Sprint0 K7：case001 至少命中 2 条缺陷（severity ∈ {medium,high,critical}）。"""
    assert len(case001_defects) >= 2, (
        f"case001 Rule Hit = {len(case001_defects)} < 2 (Sprint0 K7)"
    )


def test_case001_hits_power001_and_power002(case001_defects):
    """case001 必须命中 POWER_001 与 POWER_002。"""
    hit_ids = {r.rule_id for r in case001_defects}
    assert "POWER_001" in hit_ids, f"POWER_001 未命中；hit_ids={sorted(hit_ids)}"
    assert "POWER_002" in hit_ids, f"POWER_002 未命中；hit_ids={sorted(hit_ids)}"


# ============================================================
# RuleResult 字段结构
# ============================================================

def test_case001_rule_result_schema(case001_results):
    """RuleResult 关键字段非空；evidence_ir_refs 必须结构化非空。"""
    assert case001_results, "case001 应至少命中 1 条规则"
    for r in case001_results:
        assert r.rule_id, f"rule_id 为空: {r}"
        assert r.rule_name, f"rule_name 为空: {r.rule_id}"
        assert r.version, f"version 为空: {r.rule_id}"
        assert r.category, f"category 为空: {r.rule_id}"
        assert r.severity in {"low", "medium", "high", "critical"}, (
            f"{r.rule_id} severity 非法: {r.severity}"
        )
        assert r.hit_message, f"hit_message 为空: {r.rule_id}"
        assert r.rule_suggestion, f"rule_suggestion 为空: {r.rule_id}"
        assert r.rule_basis, f"rule_basis 为空: {r.rule_id}"
        assert r.origin == "rule", f"{r.rule_id} origin 应为 'rule'，实际 {r.origin}"
        assert r.evidence_ir_refs, f"{r.rule_id} evidence_ir_refs 为空"


# ============================================================
# 单条规则定位精度（POWER 系列）
# ============================================================

def test_power001_hits_vcc_net(case001_results):
    """POWER_001 定位应落在 VCC 网络，severity=critical，category=power。"""
    power001 = [r for r in case001_results if r.rule_id == "POWER_001"]
    assert len(power001) == 1, f"POWER_001 命中数应为 1，实际 {len(power001)}"
    hit = power001[0]
    assert hit.net == "VCC", f"POWER_001 net 应为 VCC，实际 {hit.net}"
    assert hit.severity == "critical"
    assert hit.category == "power"


def test_power002_hits_rst_net(case001_results):
    """POWER_002 定位应落在 RST 网络，severity=medium，category=timing。"""
    power002 = [r for r in case001_results if r.rule_id == "POWER_002"]
    assert len(power002) == 1, f"POWER_002 命中数应为 1，实际 {len(power002)}"
    hit = power002[0]
    assert hit.net == "RST", f"POWER_002 net 应为 RST，实际 {hit.net}"
    assert hit.severity == "medium"
    assert hit.category == "timing"


def test_power003_no_hits_on_case001(case001_results):
    """POWER_003（降额）Sprint0 预留，case001 不应命中。"""
    power003 = [r for r in case001_results if r.rule_id == "POWER_003"]
    assert power003 == [], (
        f"POWER_003 在 case001 上不应命中，实际命中 {len(power003)} 条"
    )


# ============================================================
# IO 规则测试（Sprint0 扩展）
# ============================================================

def test_io001_hits_case001(case001_results):
    """IO_001 应在 case001 上命中（U1 有 32 个 IO 悬空，汇总 1 条）。"""
    io001 = [r for r in case001_results if r.rule_id == "IO_001"]
    assert len(io001) == 1, f"IO_001 命中数应为 1（汇总），实际 {len(io001)}"
    hit = io001[0]
    assert hit.component == "U1", f"IO_001 component 应为 U1，实际 {hit.component}"
    assert "32" in hit.hit_message, (
        f"IO_001 message 应包含 32，实际 {hit.hit_message}"
    )


def test_io001_severity_is_low(case001_results):
    """IO_001 severity 应为 low（提示级，不改规范）。"""
    io001 = [r for r in case001_results if r.rule_id == "IO_001"]
    assert len(io001) == 1
    assert io001[0].severity == "low"
    assert io001[0].category == "interface"


def test_io002_no_hits_on_case001(case001_results):
    """IO_002 在 case001 上不应命中（全部 BIDIRECTIONAL，无输出冲突）。"""
    io002 = [r for r in case001_results if r.rule_id == "IO_002"]
    assert io002 == [], f"IO_002 不应命中，实际 {len(io002)} 条"


def test_execute_defect_rules_excludes_low(case001_defects):
    """execute_defect_rules 只返回 medium/high/critical，不含 low。"""
    for r in case001_defects:
        assert r.severity in {"medium", "high", "critical"}, (
            f"{r.rule_id} severity={r.severity} 不应出现在 defect 结果里"
        )
    # IO_001（low）不应出现
    assert all(r.rule_id != "IO_001" for r in case001_defects), (
        "IO_001（low）不应出现在 execute_defect_rules 结果里"
    )


# ============================================================
# applicable_condition 匹配测试
# ============================================================

def test_applicable_condition_scope_schematic_passes(case001_ir):
    """scope=schematic 的规则对 case001 应全部 applicable。"""
    rules = load_rule_definitions(RULES_ROOT)
    for r in rules:
        if r.applicable_condition.get("scope") == "schematic":
            assert r.enabled, f"{r.rule_id} 应 enabled"


# ============================================================
# IO_002 ignore_bidirectional 参数生效测试（V1.3 新增）
# ============================================================

def test_io002_bidirectional_counted_when_flag_false():
    """IO_002 规则：ignore_bidirectional=false 时，BIDIRECTIONAL 引脚也计入输出冲突。"""
    # ⚠️ V1.3 修改：import 已上移到文件顶部（避免 I001）
    # [原逻辑 - 保留注释，便于对照回滚]
    # from app.ir.schema import (
    #     Component, Pin, Net, SchematicIRDocument,
    #     PinDirection, PinType,
    # )
    # from app.rules.builtin.circuit_checks import check_io_direction

    # 构造：两个 BIDIRECTIONAL IO 引脚同网
    comp1 = Component(
        ref="U1", lib_name="TEST",
        pins=[Pin(pin_id="p1", name="p1",
                  direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO)],
        intent="test", context="test",
    )
    comp2 = Component(
        ref="U2", lib_name="TEST",
        pins=[Pin(pin_id="p1", name="p1",
                  direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO)],
        intent="test", context="test",
    )
    doc = SchematicIRDocument(
        case_id="test", title="T", description="T",
        components=[comp1, comp2],
        nets=[Net(net_name="BUS", connected_pins=["U1.p1", "U2.p1"])],
    )

    # ignore_bidirectional=false → 应命中
    hits_lenient = check_io_direction(doc, {"ignore_bidirectional": False})
    assert len(hits_lenient) == 1, "false 模式下应检出 2 个 BIDIRECTIONAL 冲突"

    # ignore_bidirectional=true（默认）→ 不应命中
    hits_strict = check_io_direction(doc, {"ignore_bidirectional": True})
    assert len(hits_strict) == 0, "true 模式下 BIDIRECTIONAL 应被忽略"


# ============================================================
# CLI 入口（demo / 手动验证）
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase F 单 case 规则测试入口（Sprint0）"
    )
    parser.add_argument("--case", required=True, help="case id, e.g. case001")
    parser.add_argument(
        "--data-root",
        default=str(CASES_ROOT),
        help=f"cases 根目录（默认 {CASES_ROOT}）",
    )
    parser.add_argument(
        "--rule-root",
        default=str(RULES_ROOT),
        help=f"rules 根目录（默认 {RULES_ROOT}）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="（保留选项）当前 load_ir 不支持 strict；此选项不影响加载行为",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    cases_root = Path(args.data_root)
    rules_root = Path(args.rule_root)

    ir_path = cases_root / args.case / "schematic_ir.json"
    print(f"[RuleEngine] cases_root={cases_root}")
    print(f"[RuleEngine] rules_root={rules_root}")
    print(f"[RuleEngine] ir_path={ir_path}")

    if not ir_path.exists():
        print(f"[FAIL] IR not found: {ir_path}", file=sys.stderr)
        return 1

    # 注：load_ir 当前不支持 strict 参数；
    # --strict 选项保留是为了向后兼容 CLI，当前不影响行为。
    ir = load_ir(ir_path)

    rules = load_rule_definitions(rules_root)
    print(f"[RuleEngine] case={args.case}")
    print(f"[RuleEngine] rules_loaded={[r.rule_id for r in rules]}")

    results = execute_all_rules(ir, rules_root)
    defects = execute_defect_rules(ir, rules_root)

    print(f"[RuleEngine] applicable_hits={len(results)}")
    for r in results:
        print(
            f"  - {r.rule_id} | {r.severity:8s} | "
            f"net={r.net or '-':10s} | {r.hit_message}"
        )

    if args.case == "case001":
        if len(defects) < 2:
            print(
                f"[FAIL] case001 Rule Hit = {len(defects)} < 2 (Sprint0 K7)",
                file=sys.stderr,
            )
            return 1
        hit_ids = sorted({r.rule_id for r in defects})
        print(
            f"[PASS] case001 Rule Hit = {len(defects)} "
            f"(>=2, Sprint0 K7, defects only), hit_ids={hit_ids}"
        )
        return 0

    print(f"[OK] case={args.case} Rule Hit (defects) = {len(defects)}")
    return 0


# ⚠️ V1.3 修改：删除重复的 test_io002_bidirectional_counted_when_flag_false
# （该测试已上移到 _build_arg_parser() 之前，此处重复定义触发 F811）
# [原逻辑 - 保留注释，便于对照回滚]
# def test_io002_bidirectional_counted_when_flag_false():
#     """IO_002 规则：ignore_bidirectional=false 时，BIDIRECTIONAL 引脚也计入输出冲突"""
#     from app.ir.schema import (
#         Component, Pin, Net, SchematicIRDocument,
#         PinDirection, PinType,
#     )
#     from app.rules.builtin.circuit_checks import check_io_direction
#
#     # 构造：两个 BIDIRECTIONAL IO 引脚同网
#     comp1 = Component(
#         ref="U1", lib_name="TEST",
#         pins=[Pin(pin_id="p1", name="p1",
#                   direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO)],
#         intent="test", context="test",
#     )
#     comp2 = Component(
#         ref="U2", lib_name="TEST",
#         pins=[Pin(pin_id="p1", name="p1",
#                   direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO)],
#         intent="test", context="test",
#     )
#     doc = SchematicIRDocument(
#         case_id="test", title="T", description="T",
#         components=[comp1, comp2],
#         nets=[Net(net_name="BUS", connected_pins=["U1.p1", "U2.p1"])],
#     )
#
#     # ignore_bidirectional=false → 应命中
#     hits_lenient = check_io_direction(doc, {"ignore_bidirectional": False})
#     assert len(hits_lenient) == 1, "false 模式下应检出 2 个 BIDIRECTIONAL 冲突"
#
#     # ignore_bidirectional=true（默认）→ 不应命中
#     hits_strict = check_io_direction(doc, {"ignore_bidirectional": True})
#     assert len(hits_strict) == 0, "true 模式下 BIDIRECTIONAL 应被忽略"


if __name__ == "__main__":
    raise SystemExit(main())

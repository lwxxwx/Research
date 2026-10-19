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

def test_rules_loaded_exactly_ten():
    """Sprint1：加载恰好 10 条规则。"""
    rules = load_rule_definitions(RULES_ROOT)
    rule_ids = sorted(r.rule_id for r in rules)
    assert rule_ids == [
        "CLOCK_001", "CLOCK_002",
        "CTRL_001", "CTRL_002",
        "IO_001", "IO_002", "IO_003",
        "POWER_001", "POWER_002", "POWER_003",
    ], f"应加载 10 条规则，实际 {rule_ids}"


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

# ============================================================
# Sprint1 新增规则的测试（占位：实现后逐步补全）
# ============================================================

def test_clock001_placeholder():
    """CLOCK_001 占位实现：当前返回空列表，待 Sprint1-B3 完善。"""
    rules = load_rule_definitions(RULES_ROOT)
    rule_ids = {r.rule_id for r in rules}
    assert "CLOCK_001" in rule_ids


def test_clock002_placeholder():
    """CLOCK_002 占位实现：当前返回空列表。"""
    rules = load_rule_definitions(RULES_ROOT)
    rule_ids = {r.rule_id for r in rules}
    assert "CLOCK_002" in rule_ids


def test_ctrl002_placeholder():
    """CTRL_002 占位实现：当前返回空列表。"""
    rules = load_rule_definitions(RULES_ROOT)
    rule_ids = {r.rule_id for r in rules}
    assert "CTRL_002" in rule_ids


def test_power002_reset_polarity_high_still_hits_case001(case001_defects):
    """POWER_002 通用化后，8051（reset_polarity=high）仍应在 case001 命中。"""
    power002 = [r for r in case001_defects if r.rule_id == "POWER_002"]
    assert len(power002) == 1


def test_power003_no_hits_without_power_attributes(case001_results):
    """POWER_003 在无功耗属性的 case001 上不命中（占位实现的行为）。"""
    power003 = [r for r in case001_results if r.rule_id == "POWER_003"]
    assert power003 == []

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

# ============================================================
# CLOCK_001 测试（case003）
# ============================================================

CASE003_DIR: Path = CASES_ROOT / "case003"
CASE003_IR: Path = CASE003_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case003_results() -> List[RuleResult]:
    """case003 跑全部规则的命中结果（模块级缓存）。"""
    if not CASE003_IR.exists():
        pytest.skip(f"case003 IR not found: {CASE003_IR}")
    ir = load_ir(CASE003_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_clock001_hits_case003(case003_results):
    """CLOCK_001 应在 case003 命中（C1=10pF 偏离 22pF）。

    设计意图（对齐 CLOCK_001 实现注释）：
      - case003 的 C1=10pF，C2=22pF；期望负载电容 22pF ± 5pF
      - 10pF 偏离 → 命中
      - 命中项应定位到 C1，而不是 C2（C2 正确）
    """
    clock001 = [r for r in case003_results if r.rule_id == "CLOCK_001"]
    assert len(clock001) == 1, f"CLOCK_001 命中数应为 1，实际 {len(clock001)}"
    hit = clock001[0]
    assert "C1" in hit.component_refs, (
        f"CLOCK_001 component_refs 应含 C1（值错的那个），实际 {hit.component_refs}"
    )
    assert "C2" not in hit.component_refs, (
        f"CLOCK_001 component_refs 不应含 C2（值正确的那个），实际 {hit.component_refs}"
    )
    assert hit.severity == "medium"
    assert hit.category == "clock"


def test_clock001_no_hits_on_case001(case001_results):
    """CLOCK_001 在 case001 上不命中（C1/C2 都是 22pF，正确）。

    设计意图：
      - case001 的负载电容 C1=C2=22pF，都在 22pF ± 5pF 内
      - 作为 CLOCK_001 的负例，证明规则不会误报
    """
    clock001 = [r for r in case001_results if r.rule_id == "CLOCK_001"]
    assert clock001 == [], f"CLOCK_001 不应在 case001 命中，实际 {len(clock001)} 条"


def test_clock002_no_hits_on_case003(case003_results):
    """CLOCK_002 在 case003 上不命中（电容数量够，只是值错）。

    设计意图（对齐 CLOCK_001/CLOCK_002 分工）：
      - CLOCK_001 = 值错（数量够但值不符）
      - CLOCK_002 = 缺项（数量不足或拓扑不完整）
      - case003 是"值错"场景，CLOCK_002 不应命中
    """
    clock002 = [r for r in case003_results if r.rule_id == "CLOCK_002"]
    assert clock002 == [], f"CLOCK_002 不应在 case003 命中，实际 {len(clock002)} 条"


def test_power001_no_hits_on_case003(case003_results):
    """POWER_001 在 case003 上不命中（C4 提供去耦电容）。

    设计意图：
      - case003 显式加了 C4=0.1μF 跨接 VCC-GND
      - 作为 POWER_001 的负例，隔离 case003 的缺陷范围
    """
    power001 = [r for r in case003_results if r.rule_id == "POWER_001"]
    assert power001 == [], f"POWER_001 不应在 case003 命中，实际 {len(power001)} 条"


def test_power002_no_hits_on_case003(case003_results):
    """POWER_002 在 case003 上不命中（R2 提供 RST-VCC 上拉路径）。

    设计意图：
      - case003 显式加了 R2 跨接 RST-VCC，修正了 case001 的复位拓扑缺陷
      - 作为 POWER_002 的负例，隔离 case003 的缺陷范围
    """
    power002 = [r for r in case003_results if r.rule_id == "POWER_002"]
    assert power002 == [], f"POWER_002 不应在 case003 命中，实际 {len(power002)} 条"

# ============================================================
# CTRL_001 测试（case005）
# ============================================================

CASE005_DIR: Path = CASES_ROOT / "case005"
CASE005_IR: Path = CASE005_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case005_results() -> List[RuleResult]:
    """case005 跑全部规则的命中结果（模块级缓存）。"""
    if not CASE005_IR.exists():
        pytest.skip(f"case005 IR not found: {CASE005_IR}")
    ir = load_ir(CASE005_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_ctrl001_hits_case005(case005_results):
    """CTRL_001 应在 case005 命中（U2.CE 悬空）。

    设计意图（对齐 CTRL_001 实现注释）：
      - 采用方案 A（每悬空引脚一条 hit）：定位精确
      - case005 故意让 U2.CE 悬空，其他使能脚（OE/WE）已连接
      - 命中项应只定位到 U2.CE 这一条
    """
    ctrl001 = [r for r in case005_results if r.rule_id == "CTRL_001"]
    assert len(ctrl001) == 1, f"CTRL_001 命中数应为 1，实际 {len(ctrl001)}"
    hit = ctrl001[0]
    assert "U2" in hit.component_refs, (
        f"CTRL_001 component_refs 应含 U2，实际 {hit.component_refs}"
    )
    assert "CE" in hit.hit_message, (
        f"CTRL_001 message 应含 CE，实际 {hit.hit_message}"
    )
    assert hit.severity == "high"
    assert hit.category == "interface"


def test_ctrl001_no_hits_on_case001(case001_results):
    """CTRL_001 在 case001 上不命中（无使能脚元件）。

    设计意图：
      - case001 只有 MCU + 晶振 + 复位电路，无外部使能脚器件
      - 作为 CTRL_001 的负例，证明规则不会误报
    """
    ctrl001 = [r for r in case001_results if r.rule_id == "CTRL_001"]
    assert ctrl001 == [], f"CTRL_001 不应在 case001 命中，实际 {len(ctrl001)} 条"


def test_ctrl001_no_hits_on_case003(case003_results):
    """CTRL_001 在 case003 上不命中（无使能脚元件）。"""
    ctrl001 = [r for r in case003_results if r.rule_id == "CTRL_001"]
    assert ctrl001 == [], f"CTRL_001 不应在 case003 命中，实际 {len(ctrl001)} 条"


def test_power001_no_hits_on_case005(case005_results):
    """POWER_001 在 case005 上不命中（C4 去耦）。"""
    power001 = [r for r in case005_results if r.rule_id == "POWER_001"]
    assert power001 == [], f"POWER_001 不应在 case005 命中，实际 {len(power001)} 条"


def test_power002_no_hits_on_case005(case005_results):
    """POWER_002 在 case005 上不命中（R2 上拉修正）。"""
    power002 = [r for r in case005_results if r.rule_id == "POWER_002"]
    assert power002 == [], f"POWER_002 不应在 case005 命中，实际 {len(power002)} 条"


def test_clock001_no_hits_on_case005(case005_results):
    """CLOCK_001 在 case005 上不命中（C1/C2 都 22pF 正确）。"""
    clock001 = [r for r in case005_results if r.rule_id == "CLOCK_001"]
    assert clock001 == [], f"CLOCK_001 不应在 case005 命中，实际 {len(clock001)} 条"

# ============================================================
# CLOCK_002 + CTRL_002 测试（case006）
# ============================================================

CASE006_DIR: Path = CASES_ROOT / "case006"
CASE006_IR: Path = CASE006_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case006_results() -> List[RuleResult]:
    """case006 跑全部规则的命中结果（模块级缓存）。"""
    if not CASE006_IR.exists():
        pytest.skip(f"case006 IR not found: {CASE006_IR}")
    ir = load_ir(CASE006_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_clock002_hits_case006(case006_results):
    """CLOCK_002 应在 case006 命中（XTAL_OSC 只有 C1，缺 C2）。

    设计意图（对齐 CLOCK_002 实现注释）：
      - CLOCK_002 检查晶振电路完整性（缺项）
      - case006 的 XTAL_OSC 只有 1 个负载电容（要求 2）
      - 命中项应定位到 Y1，net 为 XTAL_OSC
    """
    clock002 = [r for r in case006_results if r.rule_id == "CLOCK_002"]
    assert len(clock002) == 1, f"CLOCK_002 命中数应为 1，实际 {len(clock002)}"
    hit = clock002[0]
    assert "Y1" in hit.component_refs, (
        f"CLOCK_002 component_refs 应含 Y1，实际 {hit.component_refs}"
    )
    assert hit.severity == "high"
    assert hit.category == "clock"


def test_clock001_no_hits_on_case006(case006_results):
    """CLOCK_001 在 case006 上不命中（C1=22pF 值正确）。

    设计意图（对齐 CLOCK_001/CLOCK_002 分工）：
      - case006 是"缺项"场景，不是"值错"
      - CLOCK_001 只查"数量够但值不符"的情况
      - C1=22pF 正确 → CLOCK_001 不命中
    """
    clock001 = [r for r in case006_results if r.rule_id == "CLOCK_001"]
    assert clock001 == [], f"CLOCK_001 不应在 case006 命中，实际 {len(clock001)} 条"


def test_ctrl002_hits_case006(case006_results):
    """CTRL_002 应在 case006 命中（C3=1nF 时间常数不足）。

    设计意图（对齐 CTRL_002 实现注释）：
      - CTRL_002 检查复位 RC 时间常数 ≥ min_reset_high_ms
      - case006 的 C3=1nF，R1=10kΩ → τ=10μs << 2ms
      - 命中项应定位到 C3 / R1
    """
    ctrl002 = [r for r in case006_results if r.rule_id == "CTRL_002"]
    assert len(ctrl002) == 1, f"CTRL_002 命中数应为 1，实际 {len(ctrl002)}"
    hit = ctrl002[0]
    assert "C3" in hit.component_refs or "R1" in hit.component_refs, (
        f"CTRL_002 component_refs 应含 C3 或 R1，实际 {hit.component_refs}"
    )
    assert hit.severity == "high"
    assert hit.category == "timing"


def test_power001_no_hits_on_case006(case006_results):
    """POWER_001 在 case006 上不命中（C4 去耦）。"""
    power001 = [r for r in case006_results if r.rule_id == "POWER_001"]
    assert power001 == [], f"POWER_001 不应在 case006 命中，实际 {len(power001)} 条"


def test_power002_no_hits_on_case006(case006_results):
    """POWER_002 在 case006 上不命中（R2 上拉修正）。"""
    power002 = [r for r in case006_results if r.rule_id == "POWER_002"]
    assert power002 == [], f"POWER_002 不应在 case006 命中，实际 {len(power002)} 条"


def test_clock002_no_hits_on_case001(case001_results):
    """CLOCK_002 在 case001 上不命中（C1/C2 都有，完整性满足）。"""
    clock002 = [r for r in case001_results if r.rule_id == "CLOCK_002"]
    assert clock002 == [], f"CLOCK_002 不应在 case001 命中，实际 {len(clock002)} 条"


def test_clock002_no_hits_on_case003(case003_results):
    """CLOCK_002 在 case003 上不命中（C1/C2 都有，只是值错）。"""
    clock002 = [r for r in case003_results if r.rule_id == "CLOCK_002"]
    assert clock002 == [], f"CLOCK_002 不应在 case003 命中，实际 {len(clock002)} 条"

# ============================================================
# IO_003 测试（case004）
# ============================================================

CASE004_DIR: Path = CASES_ROOT / "case004"
CASE004_IR: Path = CASE004_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case004_results() -> List[RuleResult]:
    """case004 跑全部规则的命中结果（模块级缓存）。"""
    if not CASE004_IR.exists():
        pytest.skip(f"case004 IR not found: {CASE004_IR}")
    ir = load_ir(CASE004_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_io003_hits_case004(case004_results):
    """IO_003 应在 case004 命中（MCU 与外部器件输出冲突）。

    设计意图（对齐 IO_003 实现注释）：
      - case004 的 CTRL_WR 网络上有 MCU 输出（U1.P3_6）+ 外部器件输出（U2.OUT）
      - IO_003 应命中且定位到 CTRL_WR
      - 与 IO_002 分工：IO_003 检查 MCU-vs-外设；IO_002 检查 MCU 内部 IO 冲突
    """
    io003 = [r for r in case004_results if r.rule_id == "IO_003"]
    assert len(io003) == 1, f"IO_003 命中数应为 1，实际 {len(io003)}"
    hit = io003[0]
    assert "CTRL_WR" in hit.net_refs, (
        f"IO_003 net_refs 应含 CTRL_WR，实际 {hit.net_refs}"
    )
    assert "U1" in hit.component_refs
    assert "U2" in hit.component_refs
    assert hit.severity == "critical"
    assert hit.category == "interface"


def test_io002_hits_case004(case004_results):
    """IO_002 应在 case004 命中（同 MCU 两个 output 同网）。

    设计意图：
      - case004 的 BUS_CONFLICT 网络上有 U1.P1_0 + U1.P1_1（都是 output）
      - IO_002 默认 ignore_bidirectional=true，只统计 output
      - 两个 output 同网 → 命中
    """
    io002 = [r for r in case004_results if r.rule_id == "IO_002"]
    assert len(io002) == 1, f"IO_002 命中数应为 1，实际 {len(io002)}"
    hit = io002[0]
    assert "BUS_CONFLICT" in hit.net_refs, (
        f"IO_002 net_refs 应含 BUS_CONFLICT，实际 {hit.net_refs}"
    )
    assert hit.severity == "critical"


def test_io002_no_hits_on_ctrl_wr(case004_results):
    """IO_002 不应在 CTRL_WR 命中（只有 1 个 output，其他是 bidirectional）。

    设计意图：
      - CTRL_WR 网络上 U1.P3_6 是 bidirectional（默认忽略），U2.OUT 是 output
      - 只有 1 个 output → IO_002 不命中
      - 但 IO_003 命中（MCU vs 外设）→ 分工清晰
    """
    io002 = [r for r in case004_results if r.rule_id == "IO_002"]
    for h in io002:
        assert "CTRL_WR" not in h.net_refs, (
            f"IO_002 不应在 CTRL_WR 命中，实际 {h.net_refs}"
        )


def test_io003_no_hits_on_case001(case001_results):
    """IO_003 在 case001 上不命中（无 MCU-外设输出冲突）。

    设计意图：
      - case001 只有 MCU + 晶振 + 复位电路，无外部器件输出
      - 作为 IO_003 的负例
    """
    io003 = [r for r in case001_results if r.rule_id == "IO_003"]
    assert io003 == [], f"IO_003 不应在 case001 命中，实际 {len(io003)} 条"


def test_io003_no_hits_on_case005(case005_results):
    """IO_003 在 case005 上不命中（SRAM 数据/地址线方向正确）。

    设计意图：
      - case005 的 DATA_D0-D3 上，U1.P0_x 是 bidirectional，U2.Dx 是 bidirectional
      - 两者都是 bidirectional，不是 output → 不命中
      - 未来 case005 若加外部输出引脚会命中，但当前不涉及
    """
    io003 = [r for r in case005_results if r.rule_id == "IO_003"]
    assert io003 == [], f"IO_003 不应在 case005 命中，实际 {len(io003)} 条"


def test_power001_no_hits_on_case004(case004_results):
    """POWER_001 在 case004 上不命中（C4 去耦）。"""
    power001 = [r for r in case004_results if r.rule_id == "POWER_001"]
    assert power001 == [], f"POWER_001 不应在 case004 命中，实际 {len(power001)} 条"


def test_clock001_no_hits_on_case004(case004_results):
    """CLOCK_001 在 case004 上不命中（C1/C2 都 22pF）。"""
    clock001 = [r for r in case004_results if r.rule_id == "CLOCK_001"]
    assert clock001 == [], f"CLOCK_001 不应在 case004 命中，实际 {len(clock001)} 条"

# ============================================================
# POWER_003 测试（case002）
# ============================================================

CASE002_DIR: Path = CASES_ROOT / "case002"
CASE002_IR: Path = CASE002_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case002_results() -> List[RuleResult]:
    """case002 跑全部规则的命中结果（模块级缓存）。"""
    if not CASE002_IR.exists():
        pytest.skip(f"case002 IR not found: {CASE002_IR}")
    ir = load_ir(CASE002_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_power003_hits_case002(case002_results):
    """POWER_003 应在 case002 命中（U2 LDO 功耗超降额）。

    设计意图（对齐 POWER_003 实现注释）：
      - case002 的 LDO U2 实际功耗 3.5W > 阈值 0.8W
      - 功耗数据通过 Component.attributes 承载
      - 命中项应定位到 U2
    """
    power003 = [r for r in case002_results if r.rule_id == "POWER_003"]
    assert len(power003) == 1, f"POWER_003 命中数应为 1，实际 {len(power003)}"
    hit = power003[0]
    assert "U2" in hit.component_refs, (
        f"POWER_003 component_refs 应含 U2，实际 {hit.component_refs}"
    )
    assert hit.severity == "high"
    assert hit.category == "power"
    assert "降额" in hit.hit_message or "超过" in hit.hit_message


def test_power001_no_hits_on_case002(case002_results):
    """POWER_001 在 case002 上不命中（VCC 有 C4 去耦）。"""
    power001 = [r for r in case002_results if r.rule_id == "POWER_001"]
    assert power001 == [], f"POWER_001 不应在 case002 命中，实际 {len(power001)} 条"


def test_power002_no_hits_on_case002(case002_results):
    """POWER_002 在 case002 上不命中（R2 上拉修正）。"""
    power002 = [r for r in case002_results if r.rule_id == "POWER_002"]
    assert power002 == [], f"POWER_002 不应在 case002 命中，实际 {len(power002)} 条"


def test_clock001_no_hits_on_case002(case002_results):
    """CLOCK_001 在 case002 上不命中（C1/C2 都 22pF）。"""
    clock001 = [r for r in case002_results if r.rule_id == "CLOCK_001"]
    assert clock001 == [], f"CLOCK_001 不应在 case002 命中，实际 {len(clock001)} 条"


def test_clock002_no_hits_on_case002(case002_results):
    """CLOCK_002 在 case002 上不命中（C1/C2 都存在）。"""
    clock002 = [r for r in case002_results if r.rule_id == "CLOCK_002"]
    assert clock002 == [], f"CLOCK_002 不应在 case002 命中，实际 {len(clock002)} 条"


def test_power003_no_hits_on_case001(case001_results):
    """POWER_003 在 case001 上不命中（无功耗属性的器件）。

    设计意图：
      - case001 只有 MCU + 晶振 + 复位电路，无带功耗属性的器件
      - 缺属性 → 跳过（不误报）
      - 作为 POWER_003 的负例
    """
    power003 = [r for r in case001_results if r.rule_id == "POWER_003"]
    assert power003 == [], f"POWER_003 不应在 case001 命中，实际 {len(power003)} 条"

# ============================================================
# B 类 Case 规则命中（case_b01）
# ============================================================
#
# 说明：
#   - 本节只验证"规则命中/不命中"（与 A 类 Case 同构）
#   - B 类特有的资产完整性校验（ai_discovered 硬约束 / 权重 / evidence）
#     在 test_case_b_validation.py 中
#   - 引用 case_b01 的 IR，跑现有 10 条规则，断言命中结果

CASE_B01_DIR: Path = CASES_ROOT / "case_b01_buck_thermal_derating_B"
CASE_B01_IR: Path = CASE_B01_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case_b01_results() -> List[RuleResult]:
    """case_b01 跑全部规则的命中结果（模块级缓存）。"""
    if not CASE_B01_IR.exists():
        pytest.skip(f"case_b01 IR not found: {CASE_B01_IR}")
    ir = load_ir(CASE_B01_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_case_b01_hits_power001(case_b01_results):
    """case_b01 应命中 POWER_001（VCC_5V 缺 0.1μF 陶瓷去耦）。

    设计意图（对齐 POWER_001 修正后的逻辑）：
      - case_b01 的 VCC_5V 网络上只有 C_out=220μF 电解
      - 220μF 偏离 0.1μF 太远 → 不算去耦
      - POWER_001 应命中 VCC_5V
    """
    power001 = [r for r in case_b01_results if r.rule_id == "POWER_001"]
    assert len(power001) == 1, f"POWER_001 命中数应为 1，实际 {len(power001)}"
    hit = power001[0]
    assert "VCC_5V" in hit.net_refs, (
        f"POWER_001 net_refs 应含 VCC_5V，实际 {hit.net_refs}"
    )
    assert hit.severity == "critical"


def test_case_b01_no_hits_power003(case_b01_results):
    """case_b01 不应命中 POWER_003（IR 无功耗属性，AI-only 前提）。

    设计意图：
      - U2.attributes 无 power_dissipation_actual / thermal_resistance
      - POWER_003 规则无法触发
      - 热降额缺陷 100% 属于 AI/RAG 推理
    """
    power003 = [r for r in case_b01_results if r.rule_id == "POWER_003"]
    assert power003 == [], (
        f"POWER_003 不应在 case_b01 命中（IR 无功耗属性），实际 {len(power003)}"
    )


def test_case_b01_hits_io001_low(case_b01_results):
    """case_b01 应命中 IO_001（low 级，不计入 defect）。"""
    io001 = [r for r in case_b01_results if r.rule_id == "IO_001"]
    assert len(io001) == 1
    assert io001[0].severity == "low"

# ============================================================
# B 类 Case 规则命中（case_b02）
# ============================================================

CASE_B02_DIR: Path = CASES_ROOT / "case_b02_reset_rc_B"
CASE_B02_IR: Path = CASE_B02_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case_b02_results() -> List[RuleResult]:
    """case_b02 跑全部规则的命中结果。"""
    if not CASE_B02_IR.exists():
        pytest.skip(f"case_b02 IR not found: {CASE_B02_IR}")
    ir = load_ir(CASE_B02_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_case_b02_no_hits_ctrl002(case_b02_results):
    """case_b02 不应命中 CTRL_002（标称 τ=2.2ms ≥ 2ms）。

    设计意图（纯 AI-only 前提）：
      - CTRL_002 只检查标称 τ，不看容差裕量
      - 标称 τ=22kΩ×100nF=2.2ms ≥ 2ms → 不命中
      - 最坏工况 τ_min=1.782ms 属 AI 发现范畴
    """
    ctrl002 = [r for r in case_b02_results if r.rule_id == "CTRL_002"]
    assert ctrl002 == [], (
        f"CTRL_002 不应命中（标称 τ=2.2ms ≥ 2ms），实际 {len(ctrl002)}"
    )


def test_case_b02_no_hits_power001(case_b02_results):
    """case_b02 不应命中 POWER_001（C4=0.1μF 正确）。"""
    power001 = [r for r in case_b02_results if r.rule_id == "POWER_001"]
    assert power001 == []


def test_case_b02_no_hits_power002(case_b02_results):
    """case_b02 不应命中 POWER_002（R2 上拉正确）。"""
    power002 = [r for r in case_b02_results if r.rule_id == "POWER_002"]
    assert power002 == []


def test_case_b02_hits_io001_low(case_b02_results):
    """case_b02 应命中 IO_001（low 级，不计入 defect）。"""
    io001 = [r for r in case_b02_results if r.rule_id == "IO_001"]
    assert len(io001) == 1
    assert io001[0].severity == "low"

# ============================================================
# B 类 Case 规则命中（case_b03）
# ============================================================

CASE_B03_DIR: Path = CASES_ROOT / "case_b03_io_direction_B"
CASE_B03_IR: Path = CASE_B03_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case_b03_results() -> List[RuleResult]:
    """case_b03 跑全部规则的命中结果。"""
    if not CASE_B03_IR.exists():
        pytest.skip(f"case_b03 IR not found: {CASE_B03_IR}")
    ir = load_ir(CASE_B03_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_case_b03_hits_io002(case_b03_results):
    """case_b03 应命中 IO_002（BUS_CONFLICT 两个 output 同网）。"""
    io002 = [r for r in case_b03_results if r.rule_id == "IO_002"]
    assert len(io002) == 1
    assert "BUS_CONFLICT" in io002[0].net_refs


def test_case_b03_no_hits_io003(case_b03_results):
    """case_b03 不应命中 IO_003（SDA/SCL 是 bidirectional，不是 output）。

    设计意图（纯 AI-only 前提）：
      - IO_003 的 external_outputs 要求 direction==output
      - U2.SDA 是 bidirectional，U2.SCL 是 input
      - 所以 IO_003 不命中
    """
    io003 = [r for r in case_b03_results if r.rule_id == "IO_003"]
    assert io003 == [], (
        f"IO_003 不应命中（SDA/SCL 是 bidirectional），实际 {len(io003)}"
    )


def test_case_b03_hits_io001_low(case_b03_results):
    """case_b03 应命中 IO_001（low 级，不计入 defect）。"""
    io001 = [r for r in case_b03_results if r.rule_id == "IO_001"]
    assert len(io001) == 1
    assert io001[0].severity == "low"

# ============================================================
# B 类 Case 规则命中（case_b04）
# ============================================================

CASE_B04_DIR: Path = CASES_ROOT / "case_b04_power_sequence_B"
CASE_B04_IR: Path = CASE_B04_DIR / "schematic_ir.json"


@pytest.fixture(scope="module")
def case_b04_results() -> List[RuleResult]:
    """case_b04 跑全部规则的命中结果。"""
    if not CASE_B04_IR.exists():
        pytest.skip(f"case_b04 IR not found: {CASE_B04_IR}")
    ir = load_ir(CASE_B04_IR)
    return execute_all_rules(ir, RULES_ROOT)


def test_case_b04_hits_power001_vcc3v3(case_b04_results):
    """case_b04 应命中 POWER_001（VCC_3V3 缺 0.1μF 陶瓷去耦）。"""
    power001 = [r for r in case_b04_results if r.rule_id == "POWER_001"]
    assert len(power001) == 1
    assert "VCC_3V3" in power001[0].net_refs


def test_case_b04_no_hits_power001_vcc5v(case_b04_results):
    """case_b04 的 VCC_5V 有 C3=0.1μF，不应重复命中。

    设计意图：
      - VCC_5V 有 C3=0.1μF 陶瓷去耦
      - VCC_3V3 无去耦（命中）
      - POWER_001 只命中 VCC_3V3，不命中 VCC_5V
    """
    power001 = [r for r in case_b04_results if r.rule_id == "POWER_001"]
    for h in power001:
        assert "VCC_5V" not in h.net_refs, (
            f"POWER_001 不应命中 VCC_5V（已有 C3=0.1μF），实际 {h.net_refs}"
        )


def test_case_b04_no_hits_io003(case_b04_results):
    """case_b04 不应命中 IO_003（SPI_MISO 的 U1.P1_5 是 input）。

    设计意图（纯 AI-only 前提）：
      - U1.P1_5 改为 direction=input（SPI MISO 输入）
      - IO_003 的 mcu_outputs 要求 output/bidirectional
      - input 不计入 → IO_003 不命中
    """
    io003 = [r for r in case_b04_results if r.rule_id == "IO_003"]
    assert io003 == [], (
        f"IO_003 不应命中（U1.P1_5 是 input），实际 {len(io003)}"
    )


def test_case_b04_hits_io001_low(case_b04_results):
    """case_b04 应命中 IO_001（low 级，不计入 defect）。"""
    io001 = [r for r in case_b04_results if r.rule_id == "IO_001"]
    assert len(io001) >= 1
    for h in io001:
        assert h.severity == "low"

if __name__ == "__main__":
    raise SystemExit(main())

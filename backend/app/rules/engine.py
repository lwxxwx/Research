"""
app/rules/engine.py
Sprint0 Phase F：规则引擎。

职责：
  1. 加载 data/rules/*.yaml 规则定义
  2. 按 applicable_condition 匹配适用规则
  3. 调用内置 check 函数，把 RuleHitItem 包装为 RuleResult
  4. 输出结构对齐 evaluation.yaml 断言与 report_defects 落库字段

输出 RuleResult 字段设计（对齐 expected_review.json Defect 10 字段子集）：
  rule_id / rule_name / version / category / severity
  component / net / hit_message / rule_suggestion / rule_basis
  evidence_ir_refs / origin

v1.0.1 变更：
  - 新增 execute_defect_rules()：仅返回 severity ∈ {medium, high, critical} 的命中，
    用于 K7 断言 / Benchmark 等"只看缺陷"的场景；
    execute_all_rules() 保持"返回所有命中"的语义不变。
  - 修复 _cond_matches 的 attribute_exists 分支：
    Attribute 是 Pydantic 对象而非 dict，改用 hasattr(a, "key") 判定。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.ir.schema import SchematicIRDocument
from app.rules.registry import get_builtin_check_func


# ---------------------------------------------------------------------------
# 规则定义模型（对应 data/rules/*.yaml）
# ---------------------------------------------------------------------------
class RuleDefinition(BaseModel):
    rule_id: str
    rule_name: str
    version: str = "1.0"
    category: str = "other"
    applicable_condition: Dict[str, Any] = Field(default_factory=dict)
    check_logic: Dict[str, Any]
    severity: str
    rule_basis: str
    suggestion: str
    enabled: bool = True


# ---------------------------------------------------------------------------
# 引擎输出模型
# ---------------------------------------------------------------------------
class RuleResult(BaseModel):
    rule_id: str
    rule_name: str
    version: str
    category: str
    severity: str
    component: Optional[str] = None
    net: Optional[str] = None
    component_refs: List[str] = Field(default_factory=list)
    net_refs: List[str] = Field(default_factory=list)
    hit_message: str
    rule_suggestion: str
    rule_basis: str
    evidence_ir_refs: List[str] = Field(default_factory=list)
    origin: str = "rule"


# ---------------------------------------------------------------------------
# 规则加载
# ---------------------------------------------------------------------------
def load_rule_definitions(rules_dir: Path) -> List[RuleDefinition]:
    """加载目录下全部 *.yaml 规则文件。单个文件失败不影响其他文件，仅打印警告。"""
    rules: List[RuleDefinition] = []
    for f in sorted(rules_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            rules.append(RuleDefinition(**raw))
        except (ValidationError, yaml.YAMLError, TypeError) as e:
            # 显式带文件名，便于排查
            print(f"[RuleEngine][WARN] failed to load {f.name}: {e}")
    return rules


# ---------------------------------------------------------------------------
# applicable_condition 匹配
# ---------------------------------------------------------------------------
def _cond_matches(cond: Dict[str, Any], ir: SchematicIRDocument) -> bool:
    """单个条件块匹配。支持 component_type_in / component_ref_in / net_name_in
    / net_is_power / attribute_exists / always。
    """
    if not cond or cond.get("always") is True:
        return True

    if "component_type_in" in cond:
        types = set(cond["component_type_in"])
        if not any(c.lib_name in types for c in ir.components):
            return False

    if "component_ref_in" in cond:
        refs = set(cond["component_ref_in"])
        if not any(c.ref in refs for c in ir.components):
            return False

    if "net_name_in" in cond:
        names = set(cond["net_name_in"])
        if not any(n.net_name in names for n in ir.nets):
            return False

    if "net_is_power" in cond:
        want = bool(cond["net_is_power"])
        if not any(n.is_power == want for n in ir.nets):
            return False

    if "attribute_exists" in cond:
        keys = set(cond["attribute_exists"])
        found = False
        for c in ir.components:
            for a in c.attributes:
                # 修复：Attribute 是 Pydantic 对象，用 .key 而非 dict.get("name")
                if hasattr(a, "key") and a.key in keys:
                    found = True
                    break
            if found:
                break
        if not found:
            return False

    return True


def match_applicable(rule: RuleDefinition, ir: SchematicIRDocument) -> bool:
    """
    适用条件匹配：
      - 未声明 any_of/all_of → 视为 scope 级规则，只要 enabled=True 即适用
      - any_of → 任一条件块匹配即适用
      - all_of → 全部条件块匹配才适用
      - 其他未知键 → 视为 always（MVP 宽松策略），但会打印 warning
    """
    cond = rule.applicable_condition or {}

    if "any_of" in cond:
        return any(_cond_matches(c, ir) for c in cond["any_of"])
    if "all_of" in cond:
        return all(_cond_matches(c, ir) for c in cond["all_of"])

    # scope-only 规则（如 case001 的 POWER_001/002/003）
    known_keys = {"scope"}
    unknown = set(cond.keys()) - known_keys
    if unknown:
        print(f"[RuleEngine][WARN] rule {rule.rule_id} unknown applicable_condition keys: {unknown}")
    return True


# ---------------------------------------------------------------------------
# 单规则执行
# ---------------------------------------------------------------------------
def run_single_rule(rule: RuleDefinition, ir: SchematicIRDocument) -> List[RuleResult]:
    results: List[RuleResult] = []
    if not rule.enabled:
        return results
    if not match_applicable(rule, ir):
        return results

    func_ref = rule.check_logic.get("function") or rule.check_logic.get("func")
    if not func_ref:
        print(f"[RuleEngine][WARN] rule {rule.rule_id} missing check_logic.function")
        return results
    params = rule.check_logic.get("params", {})

    try:
        check_func = get_builtin_check_func(func_ref)
    except KeyError as e:
        print(f"[RuleEngine][WARN] {e}")
        return results

    hit_items = check_func(ir, params)

    for hit in hit_items:
        primary_comp = hit.component_refs[0] if hit.component_refs else None
        primary_net = hit.net_refs[0] if hit.net_refs else None
        results.append(RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            version=rule.version,
            category=rule.category,
            severity=rule.severity,
            component=primary_comp,
            net=primary_net,
            component_refs=hit.component_refs,
            net_refs=hit.net_refs,
            hit_message=hit.message,
            rule_suggestion=rule.suggestion,
            rule_basis=rule.rule_basis,
            evidence_ir_refs=hit.evidence_ir_refs,
            origin="rule",
        ))
    return results


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def execute_all_rules(
    ir: SchematicIRDocument,
    rules_dir: Path,
    rule_ids: Optional[List[str]] = None,
) -> List[RuleResult]:
    """加载全部规则，批量执行，返回所有 RuleResult（含 low 级提示）。"""
    rules = load_rule_definitions(rules_dir)
    if rule_ids:
        rules = [r for r in rules if r.rule_id in rule_ids]

    all_hits: List[RuleResult] = []
    for r in rules:
        all_hits.extend(run_single_rule(r, ir))
    return all_hits


def execute_defect_rules(
    ir: SchematicIRDocument,
    rules_dir: Path,
    rule_ids: Optional[List[str]] = None,
) -> List[RuleResult]:
    """
    只返回缺陷（severity ∈ {medium, high, critical}）的命中结果。

    用途：
      - Sprint0 K7 断言：case001 Rule Hit ≥ 2
      - Benchmark / 报告 / 落库：只处理真正的缺陷，不含 low 级提示

    说明：
      execute_all_rules() 保持"返回所有命中"的语义不变；
      本函数是其"只取缺陷"的便捷入口，两者可共存。
    """
    all_hits = execute_all_rules(ir, rules_dir, rule_ids)
    return [r for r in all_hits if r.severity in {"medium", "high", "critical"}]

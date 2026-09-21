"""
benchmark_service.py
Phase G · Rule-Only Baseline 评测服务 · Sprint0 (v2 优化版)
=============================================================================
Metrics: Rule Precision / Recall / FP Rate / FN Rate / EVC
         + 三态计数 (AI_CONFIRMED_N / NEED_EXPERT_REVIEW_N / LOW_CONFIDENCE_N)
         + EAR / AI_NER 占位（Sprint1 实装）

Note:
    - Phase-F 规则引擎为函数式 API: execute_all_rules，无 RuleEngine 类
    - GT expected_review.json 的 defect 顶层**无 rule_id**，
      rule_id 埋在 evidence[type=rule_hit].rule_id；匹配时做「弱校验」
    - ir_ref 是否豁免三要素由 IR_REF_EXEMPT 控制，默认 False（严格对齐 V1.2 §3.2）

对齐规范:
    - Sprint0_详细执行计划_V1.2 §12 Phase G
    - 第一阶段工程实施方案_V1.2 §8 (Benchmark)
    - docs/benchmark/metrics_definition_v1.0.md
=============================================================================
"""
import csv
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Optional, List

from app.ir import load_ir, SchematicIRDocument, IRSchemaValidator

#BENCH_OUT_DIR = "/app/out"
#RULES_DIR = pathlib.Path("/app/data/rules")
BENCH_OUT_DIR = os.getenv("BENCH_OUT_DIR", "/out")
RULES_DIR = pathlib.Path(os.getenv("RULES_DIR", "/data/rules"))

# === 规范开关 ===
# ir_ref 是否豁免 source/section 要求（仅要求 detail >= 10）。
# 默认 False：与 V1.2 §3.2 严格对齐，case001 EVC = 0.6667。
# 若架构组后续澄清 ir_ref 应豁免，改为 True 即可，无需改其他代码。
IR_REF_EXEMPT = False

# === 三态判定阈值（对齐 V1.2 §5.2 简化版，Rule-Only 场景） ===
EVIDENCE_MIN_LEN = 10


# ==================== 数据结构 ====================
@dataclass
class CaseBenchResult:
    case_id: str
    case_class: str
    total_gt_defects: int
    total_pred_defects: int
    tp: int
    fp: int
    fn: int
    rule_precision: Optional[float]
    rule_recall: Optional[float]
    rule_fp_rate: Optional[float]
    rule_fn_rate: Optional[float]
    evc: float
    ear: Optional[float]
    ai_ner: float
    # 三态计数（V1.2 §8.2 对齐）
    ai_confirmed_n: int = 0
    need_expert_review_n: int = 0
    low_confidence_n: int = 0
    # Overall 预留（Sprint0 单 case 时填 None → CSV 写 NaN）
    overall_precision: Optional[float] = None
    overall_fp_rate: Optional[float] = None
    overall_fn_rate: Optional[float] = None
    notes: str = ""


# ==================== Evidence 完整性 ====================
def _evidence_is_complete(ev: dict[str, Any]) -> bool:
    """
    判断单条 evidence 是否「三要素完整」（V1.2 §3.2）：
        source / section / reason 均非空且长度 >= 10
    特例：
        - ir_ref 且 IR_REF_EXEMPT=True 时，只校验 detail >= 10
        - 其他 type 严格三要素
    """
    ev_type = ev.get("type", "")
    if ev_type == "ir_ref" and IR_REF_EXEMPT:
        detail = ev.get("detail", "")
        return len(detail) >= EVIDENCE_MIN_LEN

    src = ev.get("source", "") or ""
    sec = ev.get("section", "") or ""
    rea = ev.get("reason", "") or ""
    return (
        len(src) >= EVIDENCE_MIN_LEN
        and len(sec) >= EVIDENCE_MIN_LEN
        and len(rea) >= EVIDENCE_MIN_LEN
    )


def calc_defect_evc(defect: dict[str, Any]) -> float:
    """单条 Defect 的 Evidence 完整率（逐条 evidence 维度，V1.2 §3.2）"""
    ev_list = defect.get("evidence", [])
    if not ev_list:
        return 0.0
    ok_cnt = sum(1 for e in ev_list if _evidence_is_complete(e))
    return float(ok_cnt) / len(ev_list)


def calc_case_evc(gt_defects: list[dict[str, Any]]) -> float:
    """Case 级 EVC：对 GT expected_review 中全部 defects 取算术平均"""
    if not gt_defects:
        return 0.0
    return sum(calc_defect_evc(d) for d in gt_defects) / len(gt_defects)


# ==================== 三态判定（Rule-Only 简化版） ====================
def _classify_review_status(gt_defect: dict[str, Any]) -> str:
    """
    Rule-Only 场景的 review_status 简化判定：
        - 无 evidence           → LOW_CONFIDENCE
        - 有完整 evidence       → AI_CONFIRMED
        - 有 evidence 但不完整  → NEED_EXPERT_REVIEW
    （Sprint1 接入 AI 输出后升级为 V1.2 §5.2 五条强规则）
    """
    ev_list = gt_defect.get("evidence", [])
    if not ev_list:
        return "LOW_CONFIDENCE"
    if any(_evidence_is_complete(e) for e in ev_list):
        return "AI_CONFIRMED"
    return "NEED_EXPERT_REVIEW"


def _count_review_status(gt_defects: list[dict[str, Any]]) -> tuple[int, int, int]:
    c = {"AI_CONFIRMED": 0, "NEED_EXPERT_REVIEW": 0, "LOW_CONFIDENCE": 0}
    for d in gt_defects:
        c[_classify_review_status(d)] += 1
    return c["AI_CONFIRMED"], c["NEED_EXPERT_REVIEW"], c["LOW_CONFIDENCE"]


# ==================== 匹配（含 rule_id 弱校验） ====================
def _extract_gt_rule_id(gt_def: dict[str, Any]) -> Optional[str]:
    """
    从 GT defect 的 evidence 中提取 rule_id（通常埋于 type=rule_hit 条目）。
    取不到返回 None，此时匹配退化为「component ∩ net ∩ risk」。
    """
    for ev in gt_def.get("evidence", []):
        if ev.get("type") == "rule_hit" and ev.get("rule_id"):
            return ev["rule_id"]
    return None


def _set_of(v: Any) -> set:
    """把可能为 None / str / list 的字段统一成 set"""
    if v is None:
        return set()
    if isinstance(v, str):
        return {v} if v else set()
    if isinstance(v, (list, tuple, set)):
        return {x for x in v if x}
    return {v}


def match_defect(pred_def: dict[str, Any], gt_def: dict[str, Any]) -> bool:
    """
    A 类 Rule-Only 缺陷匹配（V1.2 metrics_definition v1.0 §2）：

    主条件（必须全部满足）：
        1) 元件集合存在交集：pred = {component} ∪ component_refs；gt = {component}
        2) 网络集合存在交集：pred = {net} ∪ net_refs；        gt = {net}
        3) 风险等级等价：pred.severity == gt.risk

    弱校验（M1 优化）：
        4) 若 pred.rule_id 与 GT 可提取的 rule_id 都存在，则必须相等；
           否则跳过该条件（保持与 v1 行为兼容）。

    注意：
        - GT defect 顶层**无 rule_id**，只有 evidence 内可能带 rule_hit.rule_id
        - 不使用 rule_id 作为主匹配键（V1.2 明确要求）
    """
    # 1) component 交集
    pred_comp = _set_of(pred_def.get("component")) | _set_of(pred_def.get("component_refs"))
    gt_comp = _set_of(gt_def.get("component"))
    if pred_comp.isdisjoint(gt_comp):
        return False

    # 2) net 交集
    pred_net = _set_of(pred_def.get("net")) | _set_of(pred_def.get("net_refs"))
    gt_net = _set_of(gt_def.get("net"))
    if pred_net.isdisjoint(gt_net):
        return False

    # 3) 风险等级等价（pred.severity ↔ gt.risk 同义字段）
    if pred_def.get("severity") != gt_def.get("risk"):
        return False

    # 4) rule_id 弱校验
    pred_rule = pred_def.get("rule_id")
    gt_rule = _extract_gt_rule_id(gt_def)
    if pred_rule and gt_rule and pred_rule != gt_rule:
        return False

    return True


# ==================== 单 case 评测 ====================
def run_single_case(case_root: str) -> CaseBenchResult:
    """
    对单个 golden case 执行 Rule-Only 评测。

    贪心匹配策略（M2 显式说明）：
        - 外层遍历 GT，内层遍历未占用的 pred
        - 一个 GT 命中一个 pred 后立即 break：GT 唯一占用
        - pred 通过 tp_set_pred_idx 保证唯一占用
        - 若多条 pred 命中同一 GT，仅第一条计 TP，其余计入 FP
          （反映「规则重复命中」，是有意义的误报信号）
    """
    gt_path = os.path.join(case_root, "expected_review.json")
    ir_path = pathlib.Path(os.path.join(case_root, "schematic_ir.json"))

    # 加载 GT
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_json = json.load(f)
    gt_defects: list[dict[str, Any]] = gt_json.get("defects", [])
    case_id = gt_json.get("case_id", "unknown")

    # 加载并严格校验 IR
    ir_doc: SchematicIRDocument = load_ir(ir_path)
    validate_result = IRSchemaValidator.validate(ir_doc, strict_mode=True)
    if not validate_result.is_valid:
        raise RuntimeError(f"IR校验失败 case={case_id}, errors={validate_result.errors}")

    # 调用 Phase-F 函数式规则引擎
    try:
        from app.rules.engine import execute_all_rules, RuleResult
    except ImportError as e:
        raise NotImplementedError(
            f"[Phase-F规则引擎导入失败] 原始异常: {e}"
        ) from e

    rule_results: List[RuleResult] = execute_all_rules(ir_doc, RULES_DIR)
    pred_defects: List[dict[str, Any]] = [r.model_dump() for r in rule_results]

    case_class = "A"
    total_gt = len(gt_defects)
    total_pred = len(pred_defects)

    # 贪心匹配
    tp_set_gt_idx: set[int] = set()
    tp_set_pred_idx: set[int] = set()
    for gt_idx, gt_d in enumerate(gt_defects):
        for pred_idx, pred_d in enumerate(pred_defects):
            if pred_idx in tp_set_pred_idx:
                continue
            if match_defect(pred_d, gt_d):
                tp_set_gt_idx.add(gt_idx)
                tp_set_pred_idx.add(pred_idx)
                break

    tp = len(tp_set_gt_idx)
    fp = total_pred - len(tp_set_pred_idx)
    fn = total_gt - len(tp_set_gt_idx)

    # 规则指标（除零保护 → None → CSV 写 NaN）
    if tp + fp > 0:
        rule_precision: Optional[float] = tp / (tp + fp)
        rule_fp_rate: Optional[float] = fp / (tp + fp)
    else:
        rule_precision = None
        rule_fp_rate = None

    if tp + fn > 0:
        rule_recall: Optional[float] = tp / (tp + fn)
        rule_fn_rate: Optional[float] = fn / (tp + fn)
    else:
        rule_recall = None
        rule_fn_rate = None

    # EVC（严格 V1.2 §3.2）
    evc = calc_case_evc(gt_defects)

    # 三态计数
    ai_conf, need_rev, low_conf = _count_review_status(gt_defects)

    # Sprint0 占位
    ear: Optional[float] = None   # 无反馈数据
    ai_ner: float = 0.0           # A 类 case ai_expected_count=0

    return CaseBenchResult(
        case_id=case_id,
        case_class=case_class,
        total_gt_defects=total_gt,
        total_pred_defects=total_pred,
        tp=tp, fp=fp, fn=fn,
        rule_precision=rule_precision,
        rule_recall=rule_recall,
        rule_fp_rate=rule_fp_rate,
        rule_fn_rate=rule_fn_rate,
        evc=evc,
        ear=ear,
        ai_ner=ai_ner,
        ai_confirmed_n=ai_conf,
        need_expert_review_n=need_rev,
        low_confidence_n=low_conf,
        overall_precision=None,   # Sprint1 多 case 时计算
        overall_fp_rate=None,
        overall_fn_rate=None,
        notes="sprint0_baseline,rule_only",
    )


# ==================== CSV 输出 ====================
def _fmt(v: Optional[float], ndigits: int = 4) -> Any:
    """None → 'NaN'；float → 四舍五入；其他 → 原值"""
    if v is None:
        return "NaN"
    if isinstance(v, float):
        return round(v, ndigits)
    return v


def write_csv(all_result: list[CaseBenchResult], out_file: str) -> None:
    """
    输出 Rule-Only benchmark CSV。
    字段与 V1.2 §8.2 的映射关系见 docs/benchmark/metrics_definition_v1.0.md §5。
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    header = [
        "case_id", "case_class",
        "total_gt_defects", "total_pred_defects",
        "tp", "fp", "fn",
        "Rule_Precision", "Rule_Recall", "Rule_FP_Rate", "Rule_FN_Rate",
        "EVC", "EAR", "AI_NER",
        "AI_CONFIRMED_N", "NEED_EXPERT_REVIEW_N", "LOW_CONFIDENCE_N",
        "Overall_Precision", "Overall_FP_Rate", "Overall_FN_Rate",
        "notes",
    ]
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in all_result:
            writer.writerow({
                "case_id": r.case_id,
                "case_class": r.case_class,
                "total_gt_defects": r.total_gt_defects,
                "total_pred_defects": r.total_pred_defects,
                "tp": r.tp, "fp": r.fp, "fn": r.fn,
                "Rule_Precision": _fmt(r.rule_precision),
                "Rule_Recall": _fmt(r.rule_recall),
                "Rule_FP_Rate": _fmt(r.rule_fp_rate),
                "Rule_FN_Rate": _fmt(r.rule_fn_rate),
                "EVC": _fmt(r.evc),
                "EAR": _fmt(r.ear),
                "AI_NER": _fmt(r.ai_ner),
                "AI_CONFIRMED_N": r.ai_confirmed_n,
                "NEED_EXPERT_REVIEW_N": r.need_expert_review_n,
                "LOW_CONFIDENCE_N": r.low_confidence_n,
                "Overall_Precision": _fmt(r.overall_precision),
                "Overall_FP_Rate": _fmt(r.overall_fp_rate),
                "Overall_FN_Rate": _fmt(r.overall_fn_rate),
                "notes": r.notes,
            })


# ==================== Console Summary ====================
def print_console_summary(res: CaseBenchResult) -> None:
    """控制台摘要，供 CI 日志解析"""
    print("=" * 72)
    print(f"Benchmark Case: {res.case_id} | class={res.case_class} | notes={res.notes}")
    print(f"GT defects={res.total_gt_defects}, Pred defects={res.total_pred_defects}")
    print(f"TP={res.tp}, FP={res.fp}, FN={res.fn}")
    print(f"Rule_Precision={_fmt(res.rule_precision)}")
    print(f"Rule_Recall   ={_fmt(res.rule_recall)}")
    print(f"Rule_FP_Rate  ={_fmt(res.rule_fp_rate)}")
    print(f"Rule_FN_Rate  ={_fmt(res.rule_fn_rate)}")
    print(f"EVC (GT avg)  ={_fmt(res.evc)}")
    print(f"EAR           ={_fmt(res.ear)}")
    print(f"AI_NER        ={_fmt(res.ai_ner)}")
    print(f"review_status: AI_CONFIRMED={res.ai_confirmed_n} "
          f"NEED_EXPERT_REVIEW={res.need_expert_review_n} "
          f"LOW_CONFIDENCE={res.low_confidence_n}")
    print("=" * 72)
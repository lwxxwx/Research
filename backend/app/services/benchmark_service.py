"""
benchmark_service.py
Phase G · Rule-Only Baseline 评测服务 · Sprint0 (v2 优化版)
=============================================================================
Metrics: Rule Precision / Recall / FP Rate / FN Rate / EVC
         + 三态计数 (AI_CONFIRMED_N / NEED_EXPERT_REVIEW_N / LOW_CONFIDENCE_N)
         + EAR / AI_NER 占位（Sprint1 实装）

Note:
    - Phase-F 规则引擎为函数式 API: execute_all_rules，无 RuleEngine 类
    - V1.3：Benchmark 只取 defect 级命中（severity ∈ {medium, high, critical}），
      low 级提示（如 IO_001）不参与 Precision / Recall 计算。
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
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional

from app.ir import IRSchemaValidator, SchematicIRDocument, load_ir

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

# ==================== 数据结构（S1-F 冻结） ====================
# 【S1-F 变更】CaseBenchResult 重命名为 BenchmarkResult，统一 Rule Only 与 Full Review。
#
# 字段设计原则：
#   1. 兼容两种模式：Rule Only 与 Full Review 共用同一结构
#   2. 不适用的字段用明确的空值（None / 0 / ""），不省略字段
#   3. 真实指标与模拟指标分列，口径由 metric_source 标记
#   4. 旧列（rule_precision / evc / ear / ai_ner）口径不变，新增字段不改变旧列计算
#
# 【重要边界】
#   ai_ner_simulated / ear_simulated 只说明 "Pred 与 GT 金标的一致性"，
#   不是真实 AI 确认率，也不是真实专家采纳率。
#   - 不写 review_defect.expert_decision
#   - 不写 review_result.ai_new_effective_rate / expert_adoption_rate
#   - 正式 AI_NER / EAR 等真实反馈数据积累后再算，列在 ai_ner / ear 旧列
# ============================================================================

# --- mode 枚举 ---
MODE_RULE_ONLY = "rule_only"
MODE_FULL_REVIEW = "full_review"

# --- status 枚举 ---
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# --- metric_source 枚举 ---
# rule_only_baseline      : Rule Only 基线，无 AI 参与，无模拟指标
# golden_label_simulated  : Full Review 金标模拟，ai_ner_simulated / ear_simulated 有值
METRIC_SOURCE_RULE_ONLY = "rule_only_baseline"
METRIC_SOURCE_GOLDEN_SIM = "golden_label_simulated"


@dataclass
class BenchmarkResult:
    """
    单 Case Benchmark 结果（S1-F 冻结）。

    兼容 Rule Only 与 Full Review 两种模式，字段说明：

    【标识】
      case_id         : Case 标识
      mode            : "rule_only" / "full_review"
      status          : "done" / "failed"
      error           : 失败时的错误摘要；成功为 None

    【GT / Pred 计数】
      total_gt_defects : expected_review.json 的 defects 数
      total_pred_defects : Pred 的 defects 数（Rule Only = 规则命中；Full Review = 报告缺陷）

    【匹配结果】
      tp / fp / fn : 贪心匹配结果（benchmark_service.match_defect 口径）

    【真实指标】
      rule_precision / rule_recall / rule_fp_rate / rule_fn_rate
        : 规则指标，分母为 0 时 None
      evc : GT 侧 Evidence 三要素完整率（S1-F 冻结：口径不变，与 Rule Only 一致）
            分母 = total_gt_defects
      ear : 真实 EAR；Full Review 无真实反馈时 None
      ai_ner : 真实 AI_NER；Full Review 无真实反馈时 None

    【三态计数】
      ai_confirmed_n / need_expert_review_n / low_confidence_n
        : Pred 缺陷按 review_status 三态计数

    【模拟指标（仅 Full Review 有值）】
      ai_ner_simulated : 金标模拟 AI_NER
                         = N_ai_effective_sim / (N_rule_effective_sim + N_ai_effective_sim)
                         N_ai_effective_sim = Pred 中 origin=ai_discovered 且与 GT 匹配的缺陷数
                         N_rule_effective_sim = Pred 中 origin=rule 且与 GT 匹配的缺陷数
                         分母为 0 时 None
      ear_simulated    : 金标模拟 EAR（实际语义：gold-label match precision）
                         = len(tp_set_pred) / len(pred_defects)
                         即 "Pred 中有多少比例被 GT 金标认可"，分母为 0 时 None
                         【重要】不是真实专家采纳率；真实 EAR 见 ear 旧列。
                         字段名暂保留以兼容已冻结设计；若后续允许改名，
                         gold_match_precision 更准确。
      evc_pred         : Pred 侧 Evidence 三要素完整率（S1-F 新增）
                         分母 = total_pred_defects（只算 Pred 自己的 evidence，不混入 GT 未发现的缺陷）
                         Rule Only 模式下 Pred 无 evidence → None

    【口径标记】
      metric_source : "rule_only_baseline" / "golden_label_simulated"
                      调用方据此判断 ai_ner_simulated / ear_simulated 是否可信

    【Full Review 专用】
      task_id : Full Review 的 task_id；Rule Only 为 None

    【元信息】
      case_class : "A" / "B"
      notes      : 备注
    """
    # --- 标识 ---
    case_id: str
    mode: str
    status: str
    error: Optional[str]

    # --- GT / Pred 计数 ---
    total_gt_defects: int
    total_pred_defects: int

    # --- 匹配结果 ---
    tp: int
    fp: int
    fn: int

    # --- 真实指标 ---
    rule_precision: Optional[float]
    rule_recall: Optional[float]
    rule_fp_rate: Optional[float]
    rule_fn_rate: Optional[float]
    evc: Optional[float]                  # GT 侧，口径不变
    ear: Optional[float]                  # 真实 EAR；Full Review 无反馈时 None
    ai_ner: Optional[float]               # 真实 AI_NER；Full Review 无反馈时 None

    # --- 三态计数 ---
    ai_confirmed_n: int = 0
    need_expert_review_n: int = 0
    low_confidence_n: int = 0

    # --- 模拟指标（仅 Full Review 有值） ---
    ai_ner_simulated: Optional[float] = None
    ear_simulated: Optional[float] = None
    evc_pred: Optional[float] = None      # Pred 侧，S1-F 新增

    # --- 口径标记 ---
    metric_source: str = METRIC_SOURCE_RULE_ONLY

    # --- Full Review 专用 ---
    task_id: Optional[str] = None

    # --- 元信息 ---
    case_class: str = "A"
    notes: str = ""


# --- 向后兼容别名（S1-F：旧名暂时保留，避免一次性改动所有引用） ---
# 说明：新代码统一用 BenchmarkResult；旧代码可继续用 CaseBenchResult。
#      阶段 G 写 API 时统一改为 BenchmarkResult，届时删除别名。
CaseBenchResult = BenchmarkResult


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
    """
    把可能为 None / str / list 的字段统一成 set。

    【S1-F 修正】LLM 可能输出 "-" / "" 作为字段占位符，
    视为"无值"，返回空集。这影响 match_defect_full_review 的降级路径判断。
    """
    if v is None:
        return set()
    if isinstance(v, str):
        # "-" / "" 视为无值
        return {v} if v and v.strip() not in ("", "-") else set()
    if isinstance(v, (list, tuple, set)):
        return {x for x in v if x and str(x).strip() not in ("", "-")}
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


# ============================================================
# ===== [S1-B 新增] risk 容忍匹配 =====
# ------------------------------------------------------------
# 背景：GT 的 risk 是专家标定的严重程度，LLM 可能判得更严重
#      （如 GT=high，LLM=critical）。这是"更保守的判断"，不应算漏检。
# 规则 B（S1-B 冻结）：
#   - pred_risk 不得低于 gt_risk（判轻了算漏检）
#   - pred_risk 最多比 gt_risk 高 1 级（判重了算"更保守的命中"）
#   - GT=low 时要求严格一致（low 不计入统计，任何高于 low 都算误判）
#   - 未知 risk 值 → 不匹配
# ============================================================
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _risk_matches(pred_risk: str | None, gt_risk: str | None) -> bool:
    """
    判定 pred_risk 与 gt_risk 是否匹配（S1-B 规则 B）。

    - pred_risk 不得低于 gt_risk
    - pred_risk 最多比 gt_risk 高 1 级
    - GT=low 时严格一致
    - 未知 risk 值不匹配
    """
    if not pred_risk or not gt_risk:
        return False
    pred_level = _RISK_ORDER.get(pred_risk)
    gt_level = _RISK_ORDER.get(gt_risk)
    if pred_level is None or gt_level is None:
        return False  # 未知 risk 值
    if gt_risk == "low":
        return pred_risk == "low"  # GT=low 时严格一致
    if pred_level < gt_level:
        return False  # pred 判轻了
    if pred_level - gt_level > 1:
        return False  # pred 判重了超过 1 级
    return True


def match_defect_full_review(
    pred_def: dict[str, Any],
    gt_def: dict[str, Any],
) -> bool:
    """
    Full Review 模式的缺陷匹配（S1-F）。

    【匹配优先级（S1-F 冻结）】
      1. 严格匹配（pred_net 非空）：
         net 交集 AND risk 容忍匹配 AND (rule_id 一致 OR category 一致)
      2. 降级匹配（pred_net 为空，受限例外）：
         risk 容忍匹配 AND (rule_id 一致 OR (category 一致 AND component 交集))

    【risk 容忍匹配（S1-B 补充）】
      - pred_risk 不得低于 gt_risk
      - pred_risk 最多比 gt_risk 高 1 级
      - GT=low 时严格一致
      - 未知 risk 值不匹配
      详见 _risk_matches 注释。

    【为什么 pred_net 非空时不能降级】
      pred 有明确网络但与 GT 网络不一致，说明定位不同，不能忽略差异。
      降级路径只对"器件级缺陷（无网络）"开放，如 POWER_003 降额检查。

    【为什么降级路径用 category + component 双锚点】
      category 是宽泛大类（如 power），单独不足以证明语义等价。
      加 component 交集作为辅助锚点，避免"同 category 不同器件"被误判为 TP。
      对 case002：Pred U2 / GT U2，component 交集，通过。
      对 case_b04：Pred U3 / GT-1 U2，component 不交集，B 判据拒绝（更保守）。

    【与 match_defect 的区别】
      match_defect             : Rule Only 用，component 和 net 都必须交集
      match_defect_full_review : Full Review 用，net 必须交集（或 pred_net 为空时降级），
                                 component 在严格路径不要求、降级路径要求交集

    :param pred_def: Pred 缺陷 dict
    :param gt_def: GT 缺陷 dict
    :return: 是否匹配
    """
    # --- 公共字段提取 ---
    pred_net = _set_of(pred_def.get("net")) | _set_of(pred_def.get("net_refs"))
    gt_net = _set_of(gt_def.get("net"))

    pred_risk = pred_def.get("risk") or pred_def.get("severity")
    gt_risk = gt_def.get("risk")

    pred_rule = pred_def.get("rule_id")
    gt_rule = _extract_gt_rule_id(gt_def)

    pred_cat = pred_def.get("category")
    gt_cat = gt_def.get("category")

    pred_comp = _set_of(pred_def.get("component")) | _set_of(pred_def.get("component_refs"))
    gt_comp = _set_of(gt_def.get("component"))

    # --- 1. 严格匹配：pred_net 非空 ---
    if pred_net:
        # 1.1 net 必须交集
        if not (pred_net & gt_net):
            return False
        # 1.2 risk 容忍匹配（S1-B 规则 B）
        # --- 原代码（保留，已注释） ---
        # if pred_risk != gt_risk:
        #     return False
        # --- 原代码（保留，已注释） ---
        if not _risk_matches(pred_risk, gt_risk):
            return False
        # 1.3 语义同一：rule_id 一致 OR category 一致
        if pred_rule and gt_rule and pred_rule == gt_rule:
            return True
        if pred_cat and gt_cat and pred_cat == gt_cat:
            return True
        return False

    # --- 2. 降级匹配：pred_net 为空（受限例外） ---
    # 说明：pred_net 为空代表"器件级缺陷，不绑定网络"（如 POWER_003 降额检查）。
    #      GT 可能给了网络（代表"缺陷涉及的网络"），语义上不冲突。
    #      risk 容忍匹配；语义同一需要 rule_id 一致，或 category + component 双锚点。
    # --- 原代码（保留，已注释） ---
    # if pred_risk != gt_risk:
    #     return False
    # --- 原代码（保留，已注释） ---
    if not _risk_matches(pred_risk, gt_risk):
        return False

    # 2.1 rule_id 一致（最强锚点）
    if pred_rule and gt_rule and pred_rule == gt_rule:
        return True

    # 2.2 category 一致 AND component 交集（双锚点）
    if pred_cat and gt_cat and pred_cat == gt_cat and (pred_comp & gt_comp):
        return True

    return False


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
    # ⚠️ V1.3 修改：改用 execute_defect_rules（只返回 severity ∈ {medium, high, critical}）。
    # 原因：Rule_Precision 定义是"缺陷检出准确率"，low 级提示（如 IO_001）不是缺陷，
    #      不应计入 FP 拉低 Precision。与方案 §11 V1.3 补充"Benchmark / Report 必须用
    #      execute_defect_rules()"对齐。
    # [原逻辑 - 保留注释，便于对照回滚]
    # try:
    #     from app.rules.engine import execute_all_rules, RuleResult
    # except ImportError as e:
    #     raise NotImplementedError(
    #         f"[Phase-F规则引擎导入失败] 原始异常: {e}"
    #     ) from e
    # rule_results: List[RuleResult] = execute_all_rules(ir_doc, RULES_DIR)
    try:
        from app.rules.engine import RuleResult, execute_defect_rules
    except ImportError as e:
        raise NotImplementedError(
            f"[Phase-F规则引擎导入失败] 原始异常: {e}"
        ) from e

    rule_results: List[RuleResult] = execute_defect_rules(ir_doc, RULES_DIR)
    # =====以上是 [/V1.3 修改] =====
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

    # ===== [S1-F] 返回值升级为 BenchmarkResult =====
    # 变更点：
    #   1. 显式填 mode / status / metric_source，供批量执行和 CSV 分流
    #   2. evc_pred = None：Rule Only 的 Pred 是 RuleResult，
    #      无 evidence 三要素结构，不强行计算（S1-F 冻结决策 A）
    #   3. task_id = None：Rule Only 不走 create_task
    #   4. ear / ai_ner 保持原值（None / 0.0），不填模拟值
    #      模拟值只在 Full Review 的 ai_ner_simulated / ear_simulated 列
    return BenchmarkResult(
        # --- 标识 ---
        case_id=case_id,
        mode=MODE_RULE_ONLY,
        status=STATUS_DONE,
        error=None,
        # --- GT / Pred 计数 ---
        total_gt_defects=total_gt,
        total_pred_defects=total_pred,
        # --- 匹配结果 ---
        tp=tp, fp=fp, fn=fn,
        # --- 真实指标（口径不变） ---
        rule_precision=rule_precision,
        rule_recall=rule_recall,
        rule_fp_rate=rule_fp_rate,
        rule_fn_rate=rule_fn_rate,
        evc=evc,
        ear=ear,          # Rule Only: None（无反馈）
        ai_ner=ai_ner,    # Rule Only: 0.0（A 类 case ai_expected_count=0）
        # --- 三态计数 ---
        ai_confirmed_n=ai_conf,
        need_expert_review_n=need_rev,
        low_confidence_n=low_conf,
        # --- 模拟指标（Rule Only 不适用） ---
        ai_ner_simulated=None,
        ear_simulated=None,
        evc_pred=None,    # S1-F 决策 A：Rule Only 无 Pred evidence 三要素
        # --- 口径标记 ---
        metric_source=METRIC_SOURCE_RULE_ONLY,
        # --- Full Review 专用 ---
        task_id=None,
        # --- 元信息 ---
        case_class=case_class,
        notes="sprint0_baseline,rule_only",
    )    


# ==================== 辅助函数（S1-F） ====================
def _infer_case_class(case_id: str) -> str:
    """
    从 case_id 推断 case_class。

    【S1-F 决策 B】从目录名推断，理由：
      1. 目录名是 Case 命名规范的唯一事实来源
         （阶段 B 的 tests/test_case_naming.py 已校验）
      2. A 类：case001–case006（case<3位数字>）
      3. B 类：case_b01–case_b04（case_b<2位数字>_<描述>_B）
      4. 不依赖 expected_review.json::benchmark_meta.case_class，
         因为 A 类 Case 可能没有该字段
    """
    if case_id.startswith("case_b"):
        return "B"
    return "A"


def _evidence_is_complete_pred(ev: dict[str, Any]) -> bool:
    """
    Pred 侧 Evidence 完整性判定（S1-F 新增）。

    【与 _evidence_is_complete 的区别】
      _evidence_is_complete       : GT 侧 EVC 用，严格三要素（Sprint 0 口径不变）
      _evidence_is_complete_pred  : Pred 侧 evc_pred 用，按 type 分流：
        - rag_ref : source / section / reason 均 >= 10（同 GT 侧）
        - ir_ref  : detail >= 5（不要求三要素）

    【理由】
      Pred 的 ir_ref 是 "IR 结构引用"，只提供 detail（如
      "Net[VCC].connected_pins=[...]"），不提供 source/section/reason。
      要求 ir_ref 有三要素会误判所有 ir_ref 为不完整，导致 evc_pred=0.0。
    """
    ev_type = ev.get("type", "")
    if ev_type == "ir_ref":
        detail = ev.get("detail", "") or ""
        return len(detail) >= 5
    src = ev.get("source", "") or ""
    sec = ev.get("section", "") or ""
    rea = ev.get("reason", "") or ""
    return (
        len(src) >= EVIDENCE_MIN_LEN
        and len(sec) >= EVIDENCE_MIN_LEN
        and len(rea) >= EVIDENCE_MIN_LEN
    )


def _compute_evc_pred(pred_defects: list[dict[str, Any]]) -> Optional[float]:
    """
    计算 Pred 侧 Evidence 三要素完整率（S1-F 新增）。

    【重要边界】
      只算 Pred 自己的 evidence，不混入 GT 中未发现的缺陷。
      分母 = total_pred_defects（Pred 的缺陷总数）
      分子 = Pred 中"至少 1 条 evidence 三要素完整"的缺陷数

    【与 evc 的区别】
      evc      : GT 侧，分母 = total_gt_defects，口径不变
      evc_pred : Pred 侧，分母 = total_pred_defects，只算 Pred 的 evidence

    :param pred_defects: Pred 缺陷列表（dict）
    :return: Pred 侧 EVC；Pred 为空时 None
    """
    if not pred_defects:
        return None
    ok = 0
    for d in pred_defects:
        ev_list = d.get("evidence", [])
        # ===== [S1-F 修正] 用 Pred 侧判定：ir_ref 只需 detail，rag_ref 才要三要素 =====
        if any(_evidence_is_complete_pred(e) for e in ev_list):
        # ===== [S1-F 修正 END] =====
        #if any(_evidence_is_complete(e) for e in ev_list):
            ok += 1
    return ok / len(pred_defects)


# ==================== 模拟指标（S1-F 第 5 步） ====================
# 【重要边界】
#   ai_ner_simulated / ear_simulated 只说明 "Pred 与 GT 金标的一致性"，
#   不是真实 AI 确认率，也不是真实专家采纳率。
#   - 不写 review_defect.expert_decision
#   - 不写 review_result.ai_new_effective_rate / expert_adoption_rate
#   - 正式 AI_NER / EAR 等真实反馈数据积累后再算，列在 ai_ner / ear 旧列
#
# 【命名说明】
#   ear_simulated 的实际语义是 "gold-label match precision"（金标匹配精确率），
#   不是真实专家采纳率。字段名暂保留以兼容已冻结的阶段 F 设计；
#   若后续允许调整命名，gold_match_precision 更准确。
# ============================================================================

def compute_ai_ner_simulated(
    pred_defects: list[dict[str, Any]],
    tp_set_pred: set[int],
) -> Optional[float]:
    """
    金标模拟 AI_NER（S1-F 第 5 步）。

    【公式】
      AI_NER_SIM = N_ai_effective_sim / (N_rule_effective_sim + N_ai_effective_sim)

      N_ai_effective_sim   : Pred 中 origin=ai_discovered 且与 GT 匹配的缺陷数
      N_rule_effective_sim : Pred 中 origin=rule 且与 GT 匹配的缺陷数
      分母为 0 时返回 None

    【重要边界】
      只说明 "Pred 的 AI 发现与 GT 金标的一致性"，
      不是真实 AI 确认率。不写 review_defect.expert_decision。

    【不重新匹配】
      匹配结果由调用方传入（tp_set_pred 是已匹配的 Pred 索引集合），
      本函数只做计数，不重复 O(n²) 匹配。

    :param pred_defects: Pred 缺陷列表
    :param tp_set_pred: 已匹配的 Pred 索引集合
    :return: AI_NER_SIM；分母为 0 时 None
    """
    n_ai_effective = 0
    n_rule_effective = 0
    for idx in tp_set_pred:
        if idx < 0 or idx >= len(pred_defects):
            continue
        origin = pred_defects[idx].get("origin", "rule")
        if origin == "ai_discovered":
            n_ai_effective += 1
        elif origin == "rule":
            n_rule_effective += 1
        # hybrid 暂不计入（Sprint 1 的 GT 金标无 hybrid）
    denom = n_rule_effective + n_ai_effective
    if denom == 0:
        return None
    return n_ai_effective / denom


def compute_ear_simulated(
    pred_defects: list[dict[str, Any]],
    tp_set_pred: set[int],
) -> Optional[float]:
    """
    金标模拟 EAR（S1-F 第 5 步）。

    【公式】
      EAR_SIM = len(tp_set_pred) / len(pred_defects)

    【语义（重要）】
      EAR_SIM 的实际语义是 "gold-label match precision"（金标匹配精确率）：
        - 分子：Pred 中与 GT 匹配的缺陷数
        - 分母：Pred 全部缺陷数
      即 "Pred 中有多少比例被 GT 金标认可"。

      **不是真实专家采纳率**。真实 EAR 需要专家对每条缺陷提交
      correct_defect / suggestion_update 等反馈后才能计算，列在 ear 旧列。

    【命名说明】
      ear_simulated 字段名暂保留以兼容已冻结的阶段 F 设计；
      若后续允许调整命名，gold_match_precision 更准确。

    【不重新匹配】
      匹配结果由调用方传入，本函数只做计数。

    :param pred_defects: Pred 缺陷列表
    :param tp_set_pred: 已匹配的 Pred 索引集合
    :return: EAR_SIM；Pred 为空时 None
    """
    if not pred_defects:
        return None
    return len(tp_set_pred) / len(pred_defects)


# ==================== Full Review 单 Case（S1-F 第 3 步） ====================
def run_single_case_full_review(
    case_id: str,
    cases_root: str | Path = "/data/cases",
) -> BenchmarkResult:
    """
    对单个 Case 执行 Full Review 评测（S1-F 新增）。

    【流程】
      1. create_task(case_id) → task_id（走 6 节点 Graph）
      2. get_task(task_id) → 任务状态 + 报告
      3. get_task_defects(task_id) → Pred 缺陷列表
      4. 加载 GT（expected_review.json）
      5. 贪心匹配 Pred vs GT（benchmark_service.match_defect 同口径）
      6. 计算指标：rule_* / evc / evc_pred / 三态计数
      7. 返回 BenchmarkResult（mode=full_review）

    【Pred 来源】
      review_result.review_output_json.defects 是原始报告，
      但直接用 get_task_defects 拿 review_defect 表数据更可靠：
        - defect_id 已带 task_id 前缀（S1-E 修正）
        - evidence 已按 Pydantic 序列化为 dict
        - review_status / origin 字段已落库
      两者内容一致，但 get_task_defects 是"已落库的权威数据"。

    【GT 来源】
      data/cases/<case_id>/expected_review.json::defects

    【匹配口径】
      复用 match_defect（component ∩ net ∩ risk + rule_id 弱校验），
      与 Rule Only 完全一致，不写第二套匹配规则。

    【模拟指标】
      ai_ner_simulated / ear_simulated 在第 5 步填充，本函数先留 None。

    【失败处理】
      - create_task 失败 → 返回 status=failed + error
      - get_task 返回 None → 返回 status=failed + error
      - 不抛异常，保证批量执行不中断

    :param case_id: Case 标识（如 "case001" / "case_b01_buck_thermal_derating_B"）
    :param cases_root: Case 根目录，默认 /data/cases
    :return: BenchmarkResult
    """
    from app.services.task_service import create_task, get_task, get_task_defects

    case_class = _infer_case_class(case_id)
    cases_root_p = Path(cases_root)
    gt_path = cases_root_p / case_id / "expected_review.json"

    # --- 加载 GT ---
    if not gt_path.exists():
        return BenchmarkResult(
            case_id=case_id,
            mode=MODE_FULL_REVIEW,
            status=STATUS_FAILED,
            error=f"GT 文件不存在: {gt_path}",
            total_gt_defects=0,
            total_pred_defects=0,
            tp=0, fp=0, fn=0,
            rule_precision=None, rule_recall=None,
            rule_fp_rate=None, rule_fn_rate=None,
            evc=None, ear=None, ai_ner=None,
            metric_source=METRIC_SOURCE_GOLDEN_SIM,
            case_class=case_class,
            notes="full_review,failed,gt_missing",
        )

    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)
        gt_defects: list[dict[str, Any]] = gt_json.get("defects", [])
    except Exception as exc:
        return BenchmarkResult(
            case_id=case_id,
            mode=MODE_FULL_REVIEW,
            status=STATUS_FAILED,
            error=f"GT 加载失败: {type(exc).__name__}: {exc}",
            total_gt_defects=0,
            total_pred_defects=0,
            tp=0, fp=0, fn=0,
            rule_precision=None, rule_recall=None,
            rule_fp_rate=None, rule_fn_rate=None,
            evc=None, ear=None, ai_ner=None,
            metric_source=METRIC_SOURCE_GOLDEN_SIM,
            case_class=case_class,
            notes="full_review,failed,gt_load_error",
        )

    # --- 跑 Full Review（create_task 走 6 节点 Graph） ---
    try:
        task_id = create_task(case_id)
    except Exception as exc:
        return BenchmarkResult(
            case_id=case_id,
            mode=MODE_FULL_REVIEW,
            status=STATUS_FAILED,
            error=f"create_task 异常: {type(exc).__name__}: {exc}",
            total_gt_defects=len(gt_defects),
            total_pred_defects=0,
            tp=0, fp=0, fn=0,
            rule_precision=None, rule_recall=None,
            rule_fp_rate=None, rule_fn_rate=None,
            evc=calc_case_evc(gt_defects),   # GT 侧仍可算
            ear=None, ai_ner=None,
            metric_source=METRIC_SOURCE_GOLDEN_SIM,
            case_class=case_class,
            notes="full_review,failed,create_task_error",
        )

    task_info = get_task(task_id)
    if task_info is None:
        return BenchmarkResult(
            case_id=case_id,
            mode=MODE_FULL_REVIEW,
            status=STATUS_FAILED,
            error=f"get_task 返回 None（task_id={task_id}），review_result 未落库",
            total_gt_defects=len(gt_defects),
            total_pred_defects=0,
            tp=0, fp=0, fn=0,
            rule_precision=None, rule_recall=None,
            rule_fp_rate=None, rule_fn_rate=None,
            evc=calc_case_evc(gt_defects),
            ear=None, ai_ner=None,
            metric_source=METRIC_SOURCE_GOLDEN_SIM,
            task_id=task_id,
            case_class=case_class,
            notes="full_review,failed,no_review_result",
        )

    if task_info["status"] == "failed":
        return BenchmarkResult(
            case_id=case_id,
            mode=MODE_FULL_REVIEW,
            status=STATUS_FAILED,
            error=f"任务失败: {task_info.get('errors', [])}",
            total_gt_defects=len(gt_defects),
            total_pred_defects=0,
            tp=0, fp=0, fn=0,
            rule_precision=None, rule_recall=None,
            rule_fp_rate=None, rule_fn_rate=None,
            evc=calc_case_evc(gt_defects),
            ear=None, ai_ner=None,
            metric_source=METRIC_SOURCE_GOLDEN_SIM,
            task_id=task_id,
            case_class=case_class,
            notes="full_review,failed,task_failed",
        )

    # --- 取 Pred 缺陷 ---
    pred_defects = get_task_defects(task_id)

    # ===== [S1-F 修正] 字段映射：review_defect.risk -> severity =====
    # 原因：match_defect 用 pred["severity"] 与 gt["risk"] 比较（Sprint 0 约定：
    #      RuleResult.severity 对应 GT 的 risk）。
    #      Full Review 的 Pred 来自 review_defect 表，字段名是 risk，没有 severity。
    #      为复用同一套 match_defect，把 risk 映射为 severity。
    #      注意：只在副本上映射，不改 pred_defects 原数据。
    #pred_defects_for_match = [
    #    {**d, "severity": d.get("risk")} for d in pred_defects
    #]
    # ===== [S1-F 修正 END] =====

    # ===== [S1-F 修正] match_defect_full_review 内部已处理 risk/severity 兼容，
    # 不再手动映射 =====
    pred_defects_for_match = pred_defects
    # ===== [S1-F 修正 END] =====    

    # --- 贪心匹配（与 Rule Only 同口径） ---
    tp_set_gt: set[int] = set()
    tp_set_pred: set[int] = set()
    for gt_idx, gt_d in enumerate(gt_defects):
        for pred_idx, pred_d in enumerate(pred_defects_for_match):
            if pred_idx in tp_set_pred:
                continue
            if match_defect_full_review(pred_d, gt_d):
                tp_set_gt.add(gt_idx)
                tp_set_pred.add(pred_idx)
                break

    total_gt = len(gt_defects)
    total_pred = len(pred_defects)
    tp = len(tp_set_gt)
    fp = total_pred - len(tp_set_pred)
    fn = total_gt - len(tp_set_gt)

    # --- 规则指标（同 Rule Only 口径） ---
    rule_precision = tp / (tp + fp) if (tp + fp) > 0 else None
    rule_recall = tp / (tp + fn) if (tp + fn) > 0 else None
    rule_fp_rate = fp / (tp + fp) if (tp + fp) > 0 else None
    rule_fn_rate = fn / (tp + fn) if (tp + fn) > 0 else None

    # --- EVC：GT 侧（口径不变） ---
    evc = calc_case_evc(gt_defects)

    # --- EVC_PRED：Pred 侧（S1-F 新增） ---
    evc_pred = _compute_evc_pred(pred_defects)

    # --- 三态计数（Pred 侧） ---
    status_counts = {"AI_CONFIRMED": 0, "NEED_EXPERT_REVIEW": 0, "LOW_CONFIDENCE": 0}
    for d in pred_defects:
        s = d.get("review_status", "AI_CONFIRMED")
        if s in status_counts:
            status_counts[s] += 1

    # ===== [S1-F 第 5 步] 模拟指标：金标匹配口径 =====
    # 重要边界：只说明 "Pred 与 GT 金标的一致性"，
    #   不是真实 AI 确认率 / 专家采纳率；
    #   不写 review_defect.expert_decision，
    #   不写 review_result.ai_new_effective_rate / expert_adoption_rate。
    ai_ner_simulated = compute_ai_ner_simulated(pred_defects, tp_set_pred)
    ear_simulated = compute_ear_simulated(pred_defects, tp_set_pred)
    # ===== [S1-F 第 5 步 END] =====

    return BenchmarkResult(
        # --- 标识 ---
        case_id=case_id,
        mode=MODE_FULL_REVIEW,
        status=STATUS_DONE,
        error=None,
        # --- GT / Pred 计数 ---
        total_gt_defects=total_gt,
        total_pred_defects=total_pred,
        # --- 匹配结果 ---
        tp=tp, fp=fp, fn=fn,
        # --- 真实指标 ---
        rule_precision=rule_precision,
        rule_recall=rule_recall,
        rule_fp_rate=rule_fp_rate,
        rule_fn_rate=rule_fn_rate,
        evc=evc,
        ear=None,        # 无真实反馈，留空
        ai_ner=None,     # 无真实反馈，留空
        # --- 三态计数 ---
        ai_confirmed_n=status_counts["AI_CONFIRMED"],
        need_expert_review_n=status_counts["NEED_EXPERT_REVIEW"],
        low_confidence_n=status_counts["LOW_CONFIDENCE"],
        # --- 模拟指标（第 5 步填充） ---
        ai_ner_simulated=ai_ner_simulated,
        ear_simulated=ear_simulated,
        evc_pred=evc_pred,
        # --- 口径标记 ---
        metric_source=METRIC_SOURCE_GOLDEN_SIM,
        # --- Full Review 专用 ---
        task_id=task_id,
        # --- 元信息 ---
        case_class=case_class,
        notes="full_review,golden_label_simulated",
    )


# ==================== 批量执行（S1-F 第 4 步） ====================
def run_all_benchmark(
    case_ids: list[str],
    mode: str,
    cases_root: str | Path = "/data/cases",
) -> list[BenchmarkResult]:
    """
    批量执行 Benchmark（S1-F 新增）。

    【职责边界】
      只负责调度、收集、失败保留；不做汇总。
      汇总由独立的 summarize_benchmark 做（第 6 步），
      避免"执行"和"汇总"混在一个函数里。

    【按 mode 分流】
      MODE_RULE_ONLY  : 对每个 case_id 调 run_single_case(cases_root/case_id)
      MODE_FULL_REVIEW: 对每个 case_id 调 run_single_case_full_review(case_id, cases_root)

    【失败保留】
      单个 case 抛异常时，构造 BenchmarkResult(status=failed, error=...)，
      继续跑下一个，不中断整个批次。
      理由：Benchmark 的目的是"收集所有 Case 的表现"，
           一个 Case 失败不应丢失其他 Case 的结果。

    【返回】
      list[BenchmarkResult]，按输入 case_ids 顺序。

    :param case_ids: Case 标识列表
    :param mode: MODE_RULE_ONLY / MODE_FULL_REVIEW
    :param cases_root: Case 根目录
    :return: list[BenchmarkResult]
    """
    if mode not in (MODE_RULE_ONLY, MODE_FULL_REVIEW):
        raise ValueError(
            f"不支持的 mode={mode}，可选值：{MODE_RULE_ONLY} / {MODE_FULL_REVIEW}"
        )

    cases_root_p = Path(cases_root)
    results: list[BenchmarkResult] = []

    for case_id in case_ids:
        try:
            if mode == MODE_RULE_ONLY:
                case_root = str(cases_root_p / case_id)
                res = run_single_case(case_root)
            else:  # MODE_FULL_REVIEW
                res = run_single_case_full_review(case_id, str(cases_root_p))
        except Exception as exc:
            # 失败保留：构造 failed 结果，不中断批次
            case_class = _infer_case_class(case_id)
            res = BenchmarkResult(
                case_id=case_id,
                mode=mode,
                status=STATUS_FAILED,
                error=f"批量执行异常: {type(exc).__name__}: {exc}",
                total_gt_defects=0,
                total_pred_defects=0,
                tp=0, fp=0, fn=0,
                rule_precision=None, rule_recall=None,
                rule_fp_rate=None, rule_fn_rate=None,
                evc=None, ear=None, ai_ner=None,
                metric_source=(
                    METRIC_SOURCE_RULE_ONLY
                    if mode == MODE_RULE_ONLY
                    else METRIC_SOURCE_GOLDEN_SIM
                ),
                case_class=case_class,
                notes=f"{mode},batch_exception",
            )
        results.append(res)

    return results


# ==================== CSV 输出 ====================
def _fmt(v: Optional[float], ndigits: int = 4) -> Any:
    """None → 'NaN'；float → 四舍五入；其他 → 原值"""
    if v is None:
        return "NaN"
    if isinstance(v, float):
        return round(v, ndigits)
    return v


# ==================== 汇总（S1-F 第 6 步） ====================
def summarize_benchmark(results: list[BenchmarkResult]) -> dict[str, Any]:
    """
    汇总批量 Benchmark 结果，算全局指标（S1-F 第 6 步）。

    【与 write_csv 的职责边界】
      write_csv  : 输出 per-case CSV + 汇总行（调用本函数）
      summarize_benchmark : 只算汇总，不写文件；可被 API / CLI 复用

    【汇总口径】
      - 只统计 status=done 的结果（failed 不计入分母）
      - 全局 Precision = sum(tp) / (sum(tp) + sum(fp))
      - 全局 Recall    = sum(tp) / (sum(tp) + sum(fn))
      - 全局 EVC       = 各 Case evc 的算术平均（仅计 evc 非 None 的 Case）
      - 全局 EAR       = 各 Case ear 的算术平均（仅计 ear 非 None 的 Case）
      - 全局 AI_NER    = 各 Case ai_ner 的算术平均（仅计非 None）
      - 全局 EVC_PRED  = 各 Case evc_pred 的算术平均（仅计非 None）
      - 全局 AI_NER_SIMULATED = 各 Case ai_ner_simulated 的算术平均（仅计非 None）
      - 全局 EAR_SIMULATED    = 各 Case ear_simulated 的算术平均（仅计非 None）
      - 分母为 0 时返回 None

    :param results: list[BenchmarkResult]
    :return: dict，含全局指标
    """
    done = [r for r in results if r.status == STATUS_DONE]

    total_tp = sum(r.tp for r in done)
    total_fp = sum(r.fp for r in done)
    total_fn = sum(r.fn for r in done)

    def _safe_div(num: int | float, den: int | float) -> Optional[float]:
        return num / den if den > 0 else None

    def _avg(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    global_precision = _safe_div(total_tp, total_tp + total_fp)
    global_recall = _safe_div(total_tp, total_tp + total_fn)

    evc_values = [r.evc for r in done if r.evc is not None]
    ear_values = [r.ear for r in done if r.ear is not None]
    ai_ner_values = [r.ai_ner for r in done if r.ai_ner is not None]
    evc_pred_values = [r.evc_pred for r in done if r.evc_pred is not None]
    ai_ner_sim_values = [r.ai_ner_simulated for r in done if r.ai_ner_simulated is not None]
    ear_sim_values = [r.ear_simulated for r in done if r.ear_simulated is not None]

    return {
        "total_cases": len(results),
        "done_cases": len(done),
        "failed_cases": len(results) - len(done),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_evc": _avg(evc_values),
        "global_ear": _avg(ear_values),                 # 真实 EAR（Full Review 无反馈时 None）
        "global_ai_ner": _avg(ai_ner_values),           # 真实 AI_NER（同上）
        "global_evc_pred": _avg(evc_pred_values),
        "global_ai_ner_simulated": _avg(ai_ner_sim_values),
        "global_ear_simulated": _avg(ear_sim_values),   # gold-label match precision 平均
    }


def write_csv(all_result: list[CaseBenchResult], out_file: str) -> None:
    """
    输出 benchmark CSV（S1-F 第 6 步扩展）。

    【列布局（S1-F 冻结）】
      旧列在前（Sprint 0 兼容，位置不变）：
        case_id, case_class, total_gt_defects, total_pred_defects,
        tp, fp, fn, Rule_Precision, Rule_Recall, Rule_FP_Rate, Rule_FN_Rate,
        EVC, EAR, AI_NER, AI_CONFIRMED_N, NEED_EXPERT_REVIEW_N, LOW_CONFIDENCE_N,
        notes
      新列在后（S1-F 新增）：
        EVC_COMPLETE_RATE, EVC_PRED, AI_NER_SIMULATED, EAR_SIMULATED,
        RUN_MODE, METRIC_SOURCE

    【旧列口径】
      完全不变：
        EVC     : GT 侧 EVC（口径不变）
        EAR     : 真实 EAR（Full Review 无反馈时 NaN）
        AI_NER  : 真实 AI_NER（Full Review 无反馈时 NaN）

    【新列口径】
      EVC_COMPLETE_RATE : 与 EVC 同值（Sprint 1 §9.4 的 evd_complete_rate 别名）
      EVC_PRED          : Pred 侧 EVC（S1-F 新增）
      AI_NER_SIMULATED  : 模拟 AI_NER（金标匹配，不冒充真实值）
      EAR_SIMULATED     : 模拟 EAR（gold-label match precision，不是专家采纳率）
      RUN_MODE          : rule_only / full_review
      METRIC_SOURCE     : rule_only_baseline / golden_label_simulated

    【汇总行】
      CSV 末尾追加一行 case_id=__SUMMARY__ 的汇总行：
        - 旧列里的规则指标 / 计数用全局汇总值
        - 新列里的模拟指标用全局平均
        - 便于解析时区分 per-case 行与汇总行
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # ===== [S1-F 第 6 步] 列布局：旧列在前（位置不变）+ 新列在后 =====
    header = [
        # --- 旧列（Sprint 0 兼容，位置不变） ---
        "case_id", "case_class",
        "total_gt_defects", "total_pred_defects",
        "tp", "fp", "fn",
        "Rule_Precision", "Rule_Recall", "Rule_FP_Rate", "Rule_FN_Rate",
        "EVC", "EAR", "AI_NER",
        "AI_CONFIRMED_N", "NEED_EXPERT_REVIEW_N", "LOW_CONFIDENCE_N",
        "notes",
        # --- 新列（S1-F 新增） ---
        "EVC_COMPLETE_RATE",
        "EVC_PRED",
        "AI_NER_SIMULATED",
        "EAR_SIMULATED",
        "RUN_MODE",
        "METRIC_SOURCE",
    ]

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        # --- per-case 行 ---
        for r in all_result:
            writer.writerow({
                # --- 旧列 ---
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
                "notes": r.notes,
                # --- 新列 ---
                "EVC_COMPLETE_RATE": _fmt(r.evc),
                "EVC_PRED": _fmt(r.evc_pred),
                "AI_NER_SIMULATED": _fmt(r.ai_ner_simulated),
                "EAR_SIMULATED": _fmt(r.ear_simulated),
                "RUN_MODE": r.mode,
                "METRIC_SOURCE": r.metric_source,
            })

        # --- 汇总行（S1-F 第 6 步） ---
        summary = summarize_benchmark(all_result)
        writer.writerow({
            # --- 旧列 ---
            "case_id": "__SUMMARY__",
            "case_class": "",
            "total_gt_defects": "",          # 跨 Case 合计无意义，留空
            "total_pred_defects": "",
            "tp": summary["total_tp"],
            "fp": summary["total_fp"],
            "fn": summary["total_fn"],
            "Rule_Precision": _fmt(summary["global_precision"]),
            "Rule_Recall": _fmt(summary["global_recall"]),
            "Rule_FP_Rate": _fmt(
                summary["total_fp"] / (summary["total_tp"] + summary["total_fp"])
                if (summary["total_tp"] + summary["total_fp"]) > 0 else None
            ),
            "Rule_FN_Rate": _fmt(
                summary["total_fn"] / (summary["total_tp"] + summary["total_fn"])
                if (summary["total_tp"] + summary["total_fn"]) > 0 else None
            ),
            "EVC": _fmt(summary["global_evc"]),
            "EAR": _fmt(summary["global_ear"]),
            "AI_NER": _fmt(summary["global_ai_ner"]),
            "AI_CONFIRMED_N": sum(r.ai_confirmed_n for r in all_result),
            "NEED_EXPERT_REVIEW_N": sum(r.need_expert_review_n for r in all_result),
            "LOW_CONFIDENCE_N": sum(r.low_confidence_n for r in all_result),
            "notes": f"summary, done={summary['done_cases']}/{summary['total_cases']}",
            # --- 新列 ---
            "EVC_COMPLETE_RATE": _fmt(summary["global_evc"]),
            "EVC_PRED": _fmt(summary["global_evc_pred"]),
            "AI_NER_SIMULATED": _fmt(summary["global_ai_ner_simulated"]),
            "EAR_SIMULATED": _fmt(summary["global_ear_simulated"]),
            "RUN_MODE": "",                  # 汇总行跨 mode，留空
            "METRIC_SOURCE": "",             # 同上
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
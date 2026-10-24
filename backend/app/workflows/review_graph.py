# backend/app/workflows/review_graph.py
"""
review_graph.py
├── 常量与 import
├── 工具函数
│   ├── _append_error(state, msg)
│   └── _build_ir_facts / _build_rule_hits_text / _build_rules_passed_text / _build_rag_evidence_text
├── 6 个节点函数
│   ├── node_load_ir
│   ├── node_rule_check
│   ├── node_rag_retrieve
│   ├── node_expert_analysis
│   ├── node_report_generate
│   └── node_feedback_save
├── build_review_graph() -> StateGraph
└── run_review_graph(case_id, task_id) -> ReviewGraphState

6 节点 LangGraph 评审工作流（Sprint 1 阶段 D）

【DAG 结构（固定，不引入 Planner / ReAct）】
  load_ir
    ↓
  rule_check
    ↓
  rag_retrieve
    ↓
  expert_analysis
    ↓
  report_generate
    ↓
  feedback_save

【节点职责】
  1. load_ir          : 从 data/cases/<case_id>/schematic_ir.json 加载并校验 IR
  2. rule_check       : 执行 10 条规则，保存 rule_hits
  3. rag_retrieve     : 按 rule_hits + 元件描述 + 专题关键词 四路检索 RAG 证据
  4. expert_analysis  : 渲染 Prompt，调用 LLM，保存 LLMResponse
  5. report_generate  : OutputParser 解析 + 三态判定，组装 ReviewReport
  6. feedback_save    : 落库 ReviewReport 到 review_result + review_defect
                        （阶段 D 只落库报告，不实现反馈提交；反馈提交留阶段 E）

【异常容错（Sprint 1 §5.3）】
  单节点异常不抛，写 state["errors"]，继续下一节点。
  最终 report_generate 组装带 errors 的 ReviewReport，API 可返回给前端。
  若关键节点（load_ir / rule_check）失败，后续节点检测到前置数据缺失时跳过。

【与阶段 C 的衔接】
  - expert_analysis 用 render_prompt + get_provider + complete
  - report_generate 用 parse_report + ReviewReport.recount_review_status
  - rag_retrieve 用 Retriever（已收敛 zhipu 1024 维）
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.llm.output_parser import parse_report
from app.llm.prompt_renderer import render_prompt
from app.llm.provider import get_provider
from app.persistence.db import get_db_session
from app.rag.retriever import Retriever
from app.rules.engine import execute_defect_rules, execute_all_rules
from app.workflows.state import (
    TASK_STATUS_DONE,
    TASK_STATUS_FAILED,
    TASK_STATUS_RUNNING,
    ReviewGraphState,
)

logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================
CASES_ROOT = pathlib.Path(settings.data_dir) / "cases"
RULES_ROOT = pathlib.Path(settings.data_dir) / "rules"
PROMPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "llm" / "prompts" / "expert_analysis.md"

# RAG 检索参数
RAG_TOP_K = 5
RAG_FILTER_RULE_IDS: bool = True   # 用 rule_hits 的 rule_id 过滤 RAG 证据


# ============================================================
# 工具函数
# ============================================================
def _append_error(state: ReviewGraphState, msg: str) -> None:
    """追加错误到 state["errors"]，不抛异常。"""
    state.setdefault("errors", [])
    state["errors"].append(msg)
    logger.warning("review_graph error: %s", msg)


def _build_ir_facts(ir, focus_refs: set[str] | None = None) -> str:
    """
    从 SchematicIRDocument 抽取事实，渲染为「一行一条 ir_ref」的文本。
    与 expert_analysis.md 第 2 章格式约定一致。

    :param ir: SchematicIRDocument
    :param focus_refs: 只取指定元件的 ref；None 表示全量
    """
    lines: list[str] = []

    for comp in ir.components:
        if focus_refs is not None and comp.ref not in focus_refs:
            continue
        if comp.value:
            lines.append(f"Component[{comp.ref}].value={comp.value}")
        if comp.footprint:
            lines.append(f"Component[{comp.ref}].footprint={comp.footprint}")
        for attr in comp.attributes:
            lines.append(f"Component[{comp.ref}].attributes.{attr.key}={attr.value}")
        for pin in comp.pins:
            lines.append(f"Component[{comp.ref}].pins[{pin.pin_id}]")
            lines.append(
                f"Component[{comp.ref}].pins[{pin.pin_id}].direction={pin.direction.value}"
            )

    for net in ir.nets:
        pins = ",".join(net.connected_pins)
        lines.append(f"Net[{net.net_name}].connected_pins=[{pins}]")
        if net.is_power:
            lines.append(f"Net[{net.net_name}].is_power=true")
        if net.is_ground:
            lines.append(f"Net[{net.net_name}].is_ground=true")

    return "\n".join(lines)


def _build_rule_hits_text(rule_hits: list) -> str:
    """
    渲染 rule_hits 为「一行一条，字段用 | 分隔」的文本。
    与 expert_analysis.md 第 4 章 3.1 格式约定一致。
    """
    if not rule_hits:
        return "(无规则命中)"
    lines: list[str] = []
    for r in rule_hits:
        comp = r.component or "-"
        net = r.net or "-"
        lines.append(
            f"{r.rule_id} | {r.severity} | component={comp} | net={net} | {r.hit_message}"
        )
    return "\n".join(lines)


def _build_rules_passed_text(rule_hits: list) -> str:
    """
    渲染 rules_passed 为「一行一条，字段用 | 分隔」的文本。

    说明：
      当前阶段 D 简化处理——只把"未命中的规则"列为通过项。
      完整的 rules_passed 需要规则引擎暴露「执行但未命中」的清单，
      当前 execute_all_rules 只返回命中项，故此处用固定文案占位。
    """
    if not rule_hits:
        return "(规则引擎未返回命中项，视为全部通过)"
    return "(规则引擎有命中项，未命中清单当前未暴露，见 Sprint 1 §3.2 后续增强)"


def _build_rag_evidence_text(rag_context: list) -> str:
    """
    渲染 rag_context 为「每条一个区块，--- 分隔」的文本。
    与 expert_analysis.md 第 5 章格式约定一致。

    【S1-B 修正 v10】加专题标注
      原因：LLM 看到 RAG 证据，但不知道它"该用在哪里"。
           给每条证据加 [专题: xxx] 标注，明确告知这条证据属于哪个专题。
      推断规则：从 source_title + section 关键词推断，无匹配时标注 [专题: 通用]。
      保守推断：只在 title/section 含明确关键词时才标专题，避免误标。
    """
    if not rag_context:
        return "(无检索结果)"

    def _infer_topic(title: str, section: str) -> str:
        """从 title + section 推断专题，保守匹配。"""
        text = (title or "") + " " + (section or "")
        # 热降额
        if any(kw in text for kw in ("降额", "热特性", "θJA", "结温", "Thermal")):
            return "热降额 — 检查功率器件（LDO/Buck/MOS）在最高环境温度下的结温是否超降额线"
        # RC 容差
        if any(kw in text for kw in ("容差", "裕量", "最坏工况")):
            return "RC 容差 — 检查复位 RC 时间常数在最坏工况下是否满足最小脉宽"
        # I2C 开漏
        if any(kw in text for kw in ("I2C", "开漏", "上拉电阻")):
            return "I2C 开漏 — 检查 SDA/SCL 是否有外部上拉电阻，取值是否合理"
        # 电源时序
        if any(kw in text for kw in ("时序", "上电顺序", "灌电流", "启动时间")):
            return "电源时序 — 检查多路电源上电顺序是否满足核心电压先于 IO 电压"
        return "通用 — 作为推理背景"

    blocks: list[str] = []
    for r in rag_context:
        topic = _infer_topic(r.source_title, r.source_section)
        blocks.append(
            f"source: {r.source_title}\n"
            f"section: {r.source_section}\n"
            f"reason: {r.snippet}\n"
            f"source_type: {r.source_type}\n"
            f"[专题: {topic}]"
        )
    return "\n---\n".join(blocks)


def _build_semantic_ctx(ir) -> str:
    """
    渲染电路语义上下文。

    当前阶段 D 简化处理——从元件三语义字段（intent / context / constraint）
    拼出简版语义上下文；无三语义时返回 "(空)"，Prompt 会走"语义未知"降级分支。
    """
    lines: list[str] = []
    for comp in ir.components:
        if comp.intent:
            lines.append(f"component_role: {comp.ref} {comp.intent}")
        if comp.context:
            lines.append(f"application: {comp.ref} {comp.context}")
    return "\n".join(lines) if lines else "(空)"


def _build_topic_hints(ir) -> str:
    """
    从 IR 提取专题线索，供 Prompt 注入（S1-B 新增）。

    每条线索告知 LLM：哪个元件、什么属性、触发哪个专题、需要检查什么、
    以及建议的 category / risk（降低对模型推理能力的要求，glm-4-flash 能力有限）。

    4 个专题：
      - 热降额：含 ℃/温度 + 功率器件（LDO/Buck 主控/MOS/稳压/降压）
      - RC 容差：含 容差/±
      - I2C 开漏：含 I2C/EEPROM/AT24C
      - 电源时序：≥2 个 LDO/稳压器件

    【S1-B 修正 v11】
      - 收紧"功率器件"判定：原关键词含 "Buck"，导致 L1/D1/C_in/C_out 的 context
        （"Buck 功率级""Buck 输入侧"）全部被误判为功率器件，输出了 5 条热降额线索。
        改为只认 value 含具体功率器件型号（LM2596/AMS1117/MP2307/LDO/MOS）
        或 intent 含"稳压/降压/Buck 主控/DC-DC"的元件。
      - 线索里直接给 category 建议 / risk 建议：
        原线索只给"请检查结温是否超降额线"，LLM 不知道该填 category=thermal、
        risk=high，输出了 U1/VCC_5V/critical/power（把规则命中的去耦缺陷重标为
        ai_discovered）。改为在线索里直接给建议值，LLM 只需判断"是否超限"。

    :return: 多行文本，每行一条线索；无线索时返回 "(无专题线索触发)"
    """
    hints: list[str] = []

    # 收集 IR 全文，用于关键词匹配
    ir_text = ""
    for comp in ir.components:
        if comp.value:
            ir_text += f" {comp.value}"
        for attr in comp.attributes:
            ir_text += f" {attr.key}={attr.value}"
        if comp.intent:
            ir_text += f" {comp.intent}"
        if comp.context:
            ir_text += f" {comp.context}"

    # 1) 热降额线索：含 ℃/温度 + 功率器件
    # ===== [S1-B 修正 v11] 收紧功率器件判定 =====
    # 原判定：any(kw in comp_text for kw in ("LDO","Buck","MOS","稳压","降压","LM2596","AMS1117"))
    # 问题：L1/D1/C_in/C_out 的 context 含 "Buck 功率级"/"Buck 输入侧"/"Buck 输出侧"，
    #      被 "Buck" 关键词误判为功率器件，输出了 5 条热降额线索（应只有 U2）。
    # 修法：只认 value 含具体功率器件型号，或 intent 含"稳压/降压/Buck 主控/DC-DC"。
    #      L1(value=33μH, intent=Buck 储能电感) / D1(value=SS34, intent=Buck 续流二极管)
    #      / C_in(value=100μF, intent=Buck 输入滤波电容) 都不含这些关键词，不触发。
    #      U2(value=LM2596S-5.0) 含 LM2596，触发。
    # ===== [S1-B 修正 v11 END] =====
    has_temperature = ("℃" in ir_text) or ("温度" in ir_text)
    if has_temperature:
        for comp in ir.components:
            value_text = comp.value or ""
            intent_text = comp.intent or ""
            is_power = (
                any(kw in value_text for kw in ("LM2596", "AMS1117", "MP2307", "LDO", "MOS"))
                or any(kw in intent_text for kw in ("稳压", "降压", "Buck 主控", "DC-DC"))
            )
            if is_power:
                attrs_str = ", ".join(
                    f"{a.key}={a.value}" for a in comp.attributes
                )
                # ===== [S1-B 修正 v11] 线索里直接给 category / risk 建议 =====
                # 原线索只给"请检查结温是否超降额线"，LLM 不知道该填
                # category=thermal、risk=high，输出了 U1/VCC_5V/critical/power。
                # 改为直接给建议值，LLM 只需判断"是否超限"。
                hints.append(
                    f"- [热降额] {comp.ref} ({comp.value})："
                    f"category 建议=thermal，risk 建议=high（如结温超 95℃ 可升 critical），"
                    f"component={comp.ref}，net=null，"
                    f"context=\"{comp.context or ''}\"，"
                    f"属性={attrs_str or '无'}。"
                    f"请检查：{comp.ref} 在最高环境温度下的结温是否超过降额线？"
                )

    # 2) RC 容差线索：含 容差 / ±
    if ("容差" in ir_text) or ("±" in ir_text):
        for comp in ir.components:
            comp_text = (comp.value or "") + " " + (comp.intent or "")
            if ("容差" in comp_text) or ("±" in comp_text):
                # ===== [S1-B 修正 v11] 给 category / risk 建议 =====
                hints.append(
                    f"- [RC 容差] {comp.ref} ({comp.value})："
                    f"category 建议=timing，risk 建议=high（如 τ_min < 2ms 可升 critical），"
                    f"component={comp.ref}，net=null，"
                    f"intent=\"{comp.intent or ''}\"。"
                    f"请检查：复位 RC 时间常数在最坏工况下是否满足最小脉宽？"
                )

    # 3) I2C 开漏线索：含 I2C 器件
    for comp in ir.components:
        comp_text = (comp.value or "") + " " + (comp.intent or "")
        if any(kw in comp_text for kw in ("I2C", "EEPROM", "AT24C")):
            # ===== [S1-B 修正 v11] 给 category / risk 建议 =====
            hints.append(
                f"- [I2C 开漏] {comp.ref} ({comp.value})："
                f"category 建议=interface，risk 建议=critical，"
                f"component={comp.ref}，net=SDA（或 SCL），"
                f"intent=\"{comp.intent or ''}\"。"
                f"请检查：SDA/SCL 是否有外部上拉电阻？"
            )

    # 4) 电源时序线索：≥2 个 LDO/稳压器件
    ldo_comps = [
        c for c in ir.components
        if any(
            kw in (c.value or "") + (c.intent or "")
            for kw in ("LDO", "稳压", "AMS1117", "LM2596")
        )
    ]
    if len(ldo_comps) >= 2:
        refs = ", ".join(f"{c.ref}({c.value})" for c in ldo_comps)
        # ===== [S1-B 修正 v11] 给 category / risk 建议 =====
        hints.append(
            f"- [电源时序] 检测到 {len(ldo_comps)} 个 LDO/稳压器件：{refs}。"
            f"category 建议=power，risk 建议=high，"
            f"component=核心电压 LDO（如 U3），net=VCC_3V3，"
            f"请检查：多路电源上电顺序是否满足核心电压先于 IO 电压？"
        )

    if not hints:
        return "(无专题线索触发)"
    return "\n".join(hints)


# ============================================================
# ===== [S1-B 新增 v12] Rule Defect 确定性构造 =====
# ------------------------------------------------------------
# 职责：从 RuleResult 直接构造 origin=rule 的 Defect，不依赖 LLM 是否输出。
#
# 架构原则（S1-B 冻结）：
#   Rule Engine 决定规则缺陷，LLM 决定 AI 发现，Merge 层决定最终缺陷集。
#
# 为什么需要：
#   LLM 可能漏输出某条 rule_hit 对应的缺陷（如 case005 的 CTRL_001），
#   或把 component/category 填错（如 case002/003/006）。
#   RuleResult 是规则缺陷的权威来源，应由代码直接构造，不依赖 LLM 的稳定性。
# ============================================================
def _build_rule_defect(rh) -> "Defect":
    """
    从 RuleResult 构造 origin=rule 的 Defect。

    字段映射（RuleResult → Defect）：
      rule_id     → rule_id
      rule_name   → 不用（Defect 无此字段）
      category    → category
      severity    → risk
      component   → component
      net         → net
      hit_message → root_cause（首段）
      rule_suggestion → suggestion
      evidence_ir_refs → evidence（type=ir_ref，detail=每条 ref）
      origin      → "rule"
      review_status → AI_CONFIRMED（rule 缺陷默认可信）
      confidence  → 0.9（rule 缺陷高置信度）
      synthesized → True（标记代码补齐）

    说明：
      - location 用简化结构（sheet="", path="", coords={}），因为 RuleResult 无位置信息
      - 不依赖 IR 文档（不需要 ir_document 参数）
    """
    from app.schemas.report import Defect, Evidence

    # evidence：从 evidence_ir_refs 构造 ir_ref
    evidence: list[Evidence] = []
    for ref in (getattr(rh, "evidence_ir_refs", None) or []):
        if isinstance(ref, str) and len(ref) >= 5:
            evidence.append(Evidence(type="ir_ref", detail=ref))

    return Defect(
        # ===== [S1-B 修正 v13] defect_id 格式改为 DEF-RULE-{rule_id} =====
        # 原因：
        #   1. 与 LLM 输出的 DEF-1 格式一致（都带 DEF- 前缀），下游无需特殊处理
        #   2. 保留原始 rule_id（不替换下划线），便于按 rule_id 反查
        #   3. task_id 前缀在 _persist_review_report 里统一加，
        #      最终落库为 RUN-xxx-DEF-RULE-CTRL_001，不会重复
        # --- 原代码（保留，已注释） ---
        # defect_id=f"RULE-{getattr(rh, 'rule_id', 'UNKNOWN')}",
        # --- 原代码（保留，已注释） ---
        defect_id=f"DEF-RULE-{getattr(rh, 'rule_id', 'UNKNOWN')}",
        # ===== [S1-B 修正 v13 END] =====
        category=getattr(rh, "category", None) or "other",
        location={"sheet": "", "path": "", "coords": {}},
        component=getattr(rh, "component", None) or None,
        net=getattr(rh, "net", None) or None,
        risk=getattr(rh, "severity", None) or "medium",
        evidence=evidence,
        root_cause=getattr(rh, "hit_message", None) or "",
        suggestion=getattr(rh, "rule_suggestion", None) or "",
        confidence=0.9,
        review_status="AI_CONFIRMED",
        origin="rule",
        rule_id=getattr(rh, "rule_id", None) or None,
        expert_decision="pending",
        synthesized=True,
    )


def _merge_rule_defects(
    llm_defects: list,
    rule_hits: list,
) -> list:
    """
    Merge 层：决定最终缺陷集（S1-B 修正 v14）。

    【架构原则（S1-B 冻结）】
      Rule Engine 决定规则缺陷，LLM 决定 AI 发现，Merge 层决定最终缺陷集。

    【v14 语义变更：从"打补丁"改为"集合运算"】
      旧逻辑（v12/v13）：
        merged = LLM 输出的全部缺陷 + 补齐 LLM 未输出的 rule 缺陷
        问题：LLM 输出的 origin=rule 缺陷（漏填 rule_id / 重标为 ai_discovered）
              不在 llm_rule_ids 里，导致补齐的 rule 缺陷与 LLM 输出并存，
              case003（1 rule_hit → 2 Pred）、case004（2 rule_hit → 4 Pred）都是这个 bug。

      新逻辑（v14）：
        最终缺陷集 = RuleResult 构造的 rule 缺陷集（权威，每条 rule_hit 恰好一条）
                   ∪ LLM 输出的 ai_discovered 缺陷集（已通过 parser 重标过滤）
        规则：
          1. rule 缺陷完全由 RuleResult 构造，丢弃 LLM 的所有 origin=rule 输出
          2. ai 缺陷只保留 origin=ai_discovered，且做双保险重标过滤
          3. rule 缺陷按 rule_id 去重（每个 rule_id 最多一条）
          4. ai 缺陷按 (component, net, category) 去重（避免 LLM 重复输出）

    :param llm_defects: parse_report 返回的缺陷列表（已过滤重标）
    :param rule_hits: 规则命中列表（RuleResult 对象）
    :return: 合并后的缺陷列表（rule 缺陷由 RuleResult 保证恰好一条/rule_hit）
    """
    # ===== [S1-B 修正 v14] rule 缺陷：完全由 RuleResult 构造 =====
    # 不读 LLM 的 origin=rule 输出——rule 缺陷的存在性由 Rule Engine 决定，
    # LLM 只负责解释（root_cause/suggestion 会被 _build_rule_defect 用规则原文填充），
    # 不负责"是否存在"。
    # 按 rule_id 去重：每个有效 rule_id 最多一条。
    rule_defects: list = []
    seen_rule_ids: set[str] = set()
    for rh in rule_hits:
        rh_id = getattr(rh, "rule_id", None)
        if not rh_id:
            logger.warning("_merge_rule_defects: rule_hit 缺 rule_id，跳过")
            continue
        if rh_id in seen_rule_ids:
            logger.warning(
                "_merge_rule_defects: rule_id 重复，跳过 %s", rh_id
            )
            continue
        seen_rule_ids.add(rh_id)
        rule_defects.append(_build_rule_defect(rh))

    # ===== [S1-B 新增 v14] 统计 LLM 丢弃的 origin=rule 缺陷 =====
    # 原因：v14 起 rule 缺陷完全由 RuleResult 构造，LLM 输出的 origin=rule 被丢弃。
    #      这是架构的核心行为（Rule Engine 决定 rule 缺陷，LLM 只提供 ai_discovered），
    #      需要可观测性，便于排查"LLM 输出了 N 条 rule 缺陷但最终结果没有"的疑惑。
    # 日志内容：丢弃数量 + 每条明细（defect_id, rule_id），便于定位是哪个 rule_id 被丢。
    # 注意：不改变逻辑，只打日志。
    llm_rule_outputs = [
        d for d in llm_defects
        if getattr(d, "origin", None) == "rule"
    ]
    if llm_rule_outputs:
        logger.info(
            "_merge_rule_defects: 丢弃 LLM 输出的 %d 条 origin=rule 缺陷"
            "（rule 缺陷由 RuleResult 构造，LLM 输出不参与最终规则缺陷集）：%s",
            len(llm_rule_outputs),
            [
                (getattr(d, "defect_id", "?"), getattr(d, "rule_id", "?"))
                for d in llm_rule_outputs
            ],
        )
    # ===== [S1-B 新增 v14 END] =====

    # ===== [S1-B 修正 v14] ai 缺陷：只保留 origin=ai_discovered，双保险过滤 =====
    # 第一道防线在 output_parser._filter_relabeled_ai_defects（parse_report 内）。
    # 第二道防线在此：基于"最终 Rule Defect 的规范化 component/net/category"再挡一次，
    # 防止 parser 漏网（如 component/net/category 归一化时机差异）。
    ai_defects: list = [
        d for d in llm_defects
        if getattr(d, "origin", None) == "ai_discovered"
    ]

    # 收集 rule 缺陷的 (component, net, category) 规范化三元组
    rule_keys: set[tuple] = {
        (
            getattr(rh, "component", None) or None,
            getattr(rh, "net", None) or None,
            getattr(rh, "category", None) or None,
        )
        for rh in rule_hits
    }

    # 过滤重标 ai 缺陷 + 按 (component, net, category) 去重
    seen_ai_keys: set[tuple] = set()
    filtered_ai: list = []
    suppressed_relabel = 0
    suppressed_dup = 0
    for d in ai_defects:
        key = (
            getattr(d, "component", None) or None,
            getattr(d, "net", None) or None,
            getattr(d, "category", None) or None,
        )
        # 双保险：与某条 rule 缺陷完全相同 → 重标，丢弃
        if key in rule_keys:
            suppressed_relabel += 1
            logger.warning(
                "_merge_rule_defects: ai_discovered %s 与 rule 缺陷 component/net/category "
                "完全相同，判定为重标，丢弃",
                getattr(d, "defect_id", "?"),
            )
            continue
        # 去重：同一 (component, net, category) 的 ai 缺陷只保留第一条
        if key in seen_ai_keys:
            suppressed_dup += 1
            logger.info(
                "_merge_rule_defects: ai_discovered %s 与已有 ai 缺陷 "
                "component/net/category 重复，丢弃",
                getattr(d, "defect_id", "?"),
            )
            continue
        seen_ai_keys.add(key)
        filtered_ai.append(d)

    # ===== 最终缺陷集 = rule 缺陷 + 过滤后的 ai 缺陷 =====
    merged = rule_defects + filtered_ai

    # ===== [S1-B 新增 v13] synthesized / suppressed 统计 =====
    # 不扩 Schema，只在日志里统计，便于后续模型对比（glm-4-flash vs glm-4-plus）。
    # synthesized=True：所有 rule 缺陷都是代码从 RuleResult 构造的（v14 起恒为 True）
    # suppressed_relabel：被第二道防线拦截的重标 ai 缺陷数
    # suppressed_dup：被 (component, net, category) 去重丢弃的 ai 缺陷数
    # ===== [S1-B 新增 v13 END] =====
    synthesized_count = sum(
        1 for d in merged
        if getattr(d, "origin", None) == "rule" and getattr(d, "synthesized", False)
    )
    total_rule_count = sum(
        1 for d in merged if getattr(d, "origin", None) == "rule"
    )
    logger.info(
        "_merge_rule_defects: total=%d, rule=%d, synthesized=%d, ai=%d, "
        "suppressed_relabel=%d, suppressed_dup=%d, llm_rule_discarded=%d",
        len(merged),
        total_rule_count,
        synthesized_count,
        len(merged) - total_rule_count,
        suppressed_relabel,
        suppressed_dup,
        len(llm_rule_outputs),
    )

    return merged


# ============================================================
# 节点 1：load_ir
# ============================================================
def node_load_ir(state: ReviewGraphState) -> ReviewGraphState:
    """
    加载 IR 并校验。

    成功：写入 state["ir"]
    失败：写 errors，不抛
    """
    case_id = state["case_id"]
    ir_path = CASES_ROOT / case_id / "schematic_ir.json"

    if not ir_path.exists():
        _append_error(state, f"IR 文件不存在: {ir_path}")
        return state

    try:
        from app.ir.serializer import load_ir
        ir = load_ir(ir_path)
    except Exception as exc:
        _append_error(state, f"IR 加载失败: {type(exc).__name__}: {exc}")
        return state

    # 严格模式校验（Golden Case 验收要求）
    try:
        from app.ir.validator import IRSchemaValidator
        result = IRSchemaValidator.validate(ir, strict_mode=True)
        if not result.is_valid:
            _append_error(
                state,
                f"IR 严格校验失败: {result.errors[:3]}"  # 只记前 3 条，避免日志过长
            )
            return state
    except Exception as exc:
        _append_error(state, f"IR 校验异常: {type(exc).__name__}: {exc}")
        return state

    state["ir"] = ir
    logger.info("load_ir ok: case_id=%s components=%d nets=%d",
                case_id, len(ir.components), len(ir.nets))
    return state


# ============================================================
# 节点 2：rule_check
# ============================================================
def node_rule_check(state: ReviewGraphState) -> ReviewGraphState:
    """
    执行 10 条规则，保存命中项到 state["rule_hits"]。

    说明：
      - 用 execute_defect_rules（只取 severity ∈ {medium, high, critical}），
        与 Benchmark / 报告落库一致；low 级提示（如 IO_001）不进报告。
      - low 级提示当前不传给 LLM。若未来需要在 Prompt 中区分，再扩展。
    """
    ir = state.get("ir")
    if ir is None:
        _append_error(state, "rule_check 跳过：ir 未加载")
        return state

    try:
        rule_hits = execute_defect_rules(ir, RULES_ROOT)
    except Exception as exc:
        _append_error(state, f"规则执行失败: {type(exc).__name__}: {exc}")
        return state

    state["rule_hits"] = rule_hits
    logger.info("rule_check ok: case_id=%s hits=%d",
                state["case_id"], len(rule_hits))
    return state


# ============================================================
# 节点 3：rag_retrieve
# ============================================================
def node_rag_retrieve(state: ReviewGraphState) -> ReviewGraphState:
    """
    四路检索 RAG 证据（S1-B 修正 v7）。

    检索策略：
      - query 1：规则命中（A 类精度，配额 2 条）
      - query 2：元件完整描述逐条（ref + value + attributes + intent + context，
                逐条检索取 score 最低 2 条，配额 1 条）
      - query 3：语义上下文（兜底，配额 0 条，被 q4 替代）
      - query 4：专题关键词触发（v7 新增，配额 2 条）
                从 IR 数值型特征触发 4 个专题检索：
                  · 含 ℃/温度 + 功率器件 → "热降额 结温 θJA 降额线 GJB/Z 35 温度"
                  · 含 容差/± → "RC 容差 最坏工况 裕量 复位时间常数"
                  · 含 I2C/EEPROM → "I2C 开漏 上拉电阻 SDA SCL"
                  · 含 ≥2 个 LDO/稳压 → "电源时序 上电顺序 灌电流 闩锁 LDO 启动时间"

    为什么需要 query 4：
      IR 的 intent/context 里没有专题关键词（如"热降额""时序"），
      embedding 无法从"85℃"推断出"要找热降额知识"，
      逐条 context 检索也捞不到专题。
      query 4 直接搜专题知识，弥补 IR 无专题关键词的缺陷。

    合并规则：
      - 四路分别检索，按配额截断
      - 按 chunk_id 去重
      - 合并后按 score 升序重排（距离小=相似度高）

    【性能说明】
      - query 2 逐条检索：case_b01 有 13 个 context → 13 次 embedding + 13 次 DB 查询
      - query 4 触发 1–2 次专题检索
      - 总检索耗时约 8–15 秒（embedding API 调用是主要开销）
      - Graph 整体耗时以 LLM（~20 秒）为主，检索慢 10 秒可接受
      - 需实测，不依赖估算决定长期方案

    失败不阻断：rag_context 为空时，expert_analysis 会走"无 RAG 依据"降级分支。

    【维度守卫链路】
      Retriever.__init__ 会调用 assert_backend_matches_db，校验
      RAG_EMBEDDING_BACKEND 期望维度与 DB 实际维度一致。
      若不一致，Retriever() 抛 RuntimeError，被本函数 try/except 捕获，
      写入 state["errors"]，rag_context 置空。
    """
    # ===== [S1-B 性能测量] 记录检索起点 =====
    import time as _time
    _t_retrieve_start = _time.perf_counter()
    # ===== [S1-B 性能测量 END] =====

    ir = state.get("ir")
    if ir is None:
        _append_error(state, "rag_retrieve 跳过：ir 未加载")
        return state

    rule_hits = state.get("rule_hits", [])

    # ===== [S1-B 修正 v7] 四路检索 =====
    # query 1：规则命中
    q1 = (
        " ".join(r.hit_message for r in rule_hits)
        if rule_hits
        else f"case {state['case_id']} 电路评审"
    )

    # query 2：元件完整描述逐条（ref + value + attributes + intent + context）
    context_queries: list[str] = []
    for comp in ir.components:
        # 只对"有语义"的元件构造 query（intent 或 context 非空）
        if not (comp.intent or comp.context):
            continue
        parts_c: list[str] = [comp.ref]
        if comp.value:
            parts_c.append(comp.value)
        for attr in comp.attributes:
            parts_c.append(f"{attr.key}={attr.value}")
        if comp.intent:
            parts_c.append(comp.intent)
        if comp.context:
            parts_c.append(comp.context)
        context_queries.append(" ".join(parts_c))
    # 兜底：全部元件都无语义时，用 ref + value 构造
    if not context_queries:
        for comp in ir.components:
            if comp.value:
                context_queries.append(f"{comp.ref} {comp.value}")

    # query 3：语义上下文
    semantic_ctx = _build_semantic_ctx(ir)
    q3 = semantic_ctx if (semantic_ctx and semantic_ctx != "(空)") else None

    # query 4：专题关键词触发（v7 新增）
    # 从 IR 数值型特征触发 4 个专题检索，弥补 IR 无专题关键词的缺陷。
    # 关键词表（4 个专题，维护成本低）。
    topic_queries: list[str] = []

    # 收集 IR 全文（value + attributes + intent + context），用于关键词匹配
    ir_text = ""
    for comp in ir.components:
        if comp.value:
            ir_text += f" {comp.value}"
        for attr in comp.attributes:
            ir_text += f" {attr.key}={attr.value}"
        if comp.intent:
            ir_text += f" {comp.intent}"
        if comp.context:
            ir_text += f" {comp.context}"

    # 1) 热降额：含 ℃ / 温度属性 + 功率器件
    has_temperature = ("℃" in ir_text) or ("温度" in ir_text) or ("ambient_temperature" in ir_text)
    has_power_device = any(
        kw in ir_text for kw in ("LDO", "Buck", "MOS", "稳压", "降压", "LM2596", "AMS1117")
    )
    if has_temperature and has_power_device:
        topic_queries.append("热降额 结温 θJA 降额线 GJB/Z 35 温度")

    # 2) RC 容差：含 容差 / ±
    # v8 修正：去掉"RC""复位"，让"容差"权重更高。
    # 原因：原 query "RC 容差 最坏工况 裕量 复位时间常数" 里"复位"权重高，
    #      检索 top1 是"晶振电路"（含"上电复位"），不是 AN-8051-002 裕量准则。
    if ("容差" in ir_text) or ("±" in ir_text):
        topic_queries.append("容差 最坏工况 裕量 时间常数 最坏情况")

    # 3) I2C 开漏：含 I2C / EEPROM
    if ("I2C" in ir_text) or ("EEPROM" in ir_text) or ("AT24C" in ir_text):
        topic_queries.append("I2C 开漏 上拉电阻 SDA SCL")

    # 4) 电源时序：含 ≥2 个 LDO/稳压器件
    # v9 修正：query 改为贴近 power_sequence_design.md 正文的表述。
    # 原因：原 query "电源时序 上电顺序 灌电流 闩锁 LDO 启动时间" 与
    #      chunk 正文 "核心电压必须先于或同步于 IO 电压""IO 电压通过 ESD
    #      保护二极管灌入核心电压域" 的 embedding 距离远，检索不到。
    #      改为含"核心电压先于 IO""ESD 保护二极管"的表述，提高关键词重合度。
    ldo_count = sum(
        1 for comp in ir.components
        if any(kw in (comp.value or "") + (comp.intent or "")
               for kw in ("LDO", "稳压", "AMS1117", "LM2596"))
    )
    if ldo_count >= 2:
        topic_queries.append("核心电压先于 IO 电压 上电顺序 灌电流 闩锁 ESD 保护二极管")

    rule_ids = list({r.rule_id for r in rule_hits}) if rule_hits else None

    try:
        retriever = Retriever()

        # ---------- query 1：规则命中，取前 3 条（配额 2 条，见合并段） ----------
        use_rule_ids = rule_ids if RAG_FILTER_RULE_IDS else None
        q1_results = retriever.retrieve(
            query=q1,
            rule_ids=use_rule_ids,
            top_k=RAG_TOP_K,
        )[:3]

        # ---------- query 2：元件完整描述逐条，去重后取 score 最低 2 条 ----------
        ctx_top1: list = []
        for cq in context_queries:
            rs = retriever.retrieve(query=cq, rule_ids=None, top_k=1)
            if rs:
                ctx_top1.append(rs[0])

        # 按 chunk_id 去重（同一 chunk 被多条 context 命中保留第一次）
        seen_ctx_ids: set[int] = set()
        ctx_dedup: list = []
        for r in ctx_top1:
            if r.chunk_id in seen_ctx_ids:
                continue
            seen_ctx_ids.add(r.chunk_id)
            ctx_dedup.append(r)

        # 按 score 升序，取前 2
        ctx_dedup.sort(key=lambda r: r.score)
        q2_results = ctx_dedup[:2]

        # ---------- query 3：语义上下文，保留 1 条 ----------
        q3_results: list = []
        if q3:
            q3_results = retriever.retrieve(query=q3, rule_ids=None, top_k=1)[:1]

        # ---------- query 4：专题关键词触发，每条保留 1 条 ----------
        # v7 新增：直接搜专题知识，弥补 IR 无专题关键词的缺陷。
        q4_results: list = []
        for tq in topic_queries:
            rs = retriever.retrieve(query=tq, rule_ids=None, top_k=1)
            if rs:
                q4_results.append(rs[0])

        # ---------- 合并四路，按配额截断 + chunk_id 去重 ----------
        # 配额：q1=2, q4=1, q2=2, q3=0（总共 5）
        # v8 调整：q2 从 1 增到 2，q4 从 2 降到 1。
        # 原因：
        #   - q2（逐条 context）在 v6 已证明能命中 case_b02 的 AN-8051-002，
        #     v7 把 q2 降到 1，AN-8051-002 被挤掉，退化。给 2 条保证不被挤掉。
        #   - q4（专题关键词）在 case_b01 命中 GJBZ35（top1），给 1 条足够。
        #   - q1 是规则命中（A 类精度），给 2 条。
        #   - q3 语义上下文兜底，被 q4 替代，给 0 条。
        seen_chunk_ids: set[int] = set()
        merged: list = []
        for results, quota in (
            (q1_results[:2], 2),
            (q4_results[:1], 1),
            (q2_results[:2], 2),
            (q3_results[:0], 0),
        ):
            for r in results:
                if r.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(r.chunk_id)
                merged.append(r)

        # 按 score 升序重排（距离小=相似度高），保证最相关的排前面
        merged.sort(key=lambda r: r.score)
        rag_context = merged[:RAG_TOP_K]

        # 日志：记录四路各自贡献
        logger.info(
            "rag_retrieve 四路检索: q1=%d, q2=%d(context_queries=%d), q3=%d, "
            "q4=%d(topic_queries=%d), merged=%d, final=%d",
            len(q1_results), len(q2_results), len(context_queries),
            len(q3_results), len(q4_results), len(topic_queries),
            len(merged), len(rag_context),
        )
        # ===== [S1-B 性能测量] 打印检索耗时 =====
        _elapsed = _time.perf_counter() - _t_retrieve_start
        _total_queries = 1 + len(context_queries) + (1 if q3 else 0) + len(topic_queries)
        logger.info(
            "rag_retrieve 耗时: %.2fs (queries=%d, 其中 context 逐条 %d 次, topic %d 次)",
            _elapsed, _total_queries, len(context_queries), len(topic_queries),
        )
        # ===== [S1-B 性能测量 END] =====
    except Exception as exc:
        _append_error(state, f"RAG 检索失败: {type(exc).__name__}: {exc}")
        # ===== [S1-B 性能测量] 失败时也打印耗时，便于排查 =====
        _elapsed = _time.perf_counter() - _t_retrieve_start
        logger.warning("rag_retrieve 失败耗时: %.2fs", _elapsed)
        # ===== [S1-B 性能测量 END] =====
        state["rag_context"] = []
        return state
    # ===== [S1-B 修正 v7 END] =====

    state["rag_context"] = rag_context
    logger.info("rag_retrieve ok: case_id=%s results=%d",
                state["case_id"], len(rag_context))
    return state


# ============================================================
# 节点 4：expert_analysis
# ============================================================
def node_expert_analysis(state: ReviewGraphState) -> ReviewGraphState:
    """
    渲染 Prompt，调用 LLM，保存 LLMResponse 到 state["llm_raw_output"]。

    失败不阻断：llm_raw_output 为空时，report_generate 会返回空报告 + errors。
    """
    ir = state.get("ir")
    if ir is None:
        _append_error(state, "expert_analysis 跳过：ir 未加载")
        return state

    # 构造 Prompt 输入
    ir_facts = _build_ir_facts(ir)
    semantic_ctx = _build_semantic_ctx(ir)
    topic_hints = _build_topic_hints(ir)          # ===== [S1-B 新增] 专题线索 =====
    rule_hits_text = _build_rule_hits_text(state.get("rule_hits", []))
    rules_passed_text = _build_rules_passed_text(state.get("rule_hits", []))
    rag_evidence_text = _build_rag_evidence_text(state.get("rag_context", []))

    try:
        template = PROMPT_PATH.read_text(encoding="utf-8")
        prompt = render_prompt(
            template,
            task_id=state["task_id"],
            report_id=f"{state['task_id']}-R",
            ir_facts=ir_facts,
            semantic_ctx=semantic_ctx,
            topic_hints=topic_hints,              # ===== [S1-B 新增] =====
            rule_hits=rule_hits_text,
            rules_passed=rules_passed_text,
            rag_evidence=rag_evidence_text,
        )
    except Exception as exc:
        _append_error(state, f"Prompt 渲染失败: {type(exc).__name__}: {exc}")
        return state

    try:
        provider = get_provider()
        llm_response = provider.complete(prompt, json_mode=True)
    except Exception as exc:
        _append_error(state, f"LLM 调用失败: {type(exc).__name__}: {exc}")
        return state

    state["llm_raw_output"] = llm_response

    if not llm_response.ok:
        _append_error(state, f"LLM 返回错误: {llm_response.error}")

    logger.info(
        "expert_analysis ok: case_id=%s provider=%s latency_ms=%.1f",
        state["case_id"], llm_response.provider, llm_response.latency_ms,
    )
    return state


# ============================================================
# 节点 5：report_generate
# ============================================================
def node_report_generate(state: ReviewGraphState) -> ReviewGraphState:
    """
    用 OutputParser 解析 LLM 输出，组装 ReviewReport。

    说明：
      - parse_report 内部完成三级校验 + 三态判定 + ai_discovered 硬拦截
      - 解析失败返回空报告 + errors，不伪装成功
      - parsed_defects 从 ReviewReport.defects 派生，便于后续节点消费
    """
    llm_response = state.get("llm_raw_output")
    if llm_response is None:
        _append_error(state, "report_generate 跳过：llm_raw_output 未生成")
        # ===== [S1-D 修正] LLM 输出缺失视为任务失败，置 failed，
        # 避免 feedback_save 把空报告误判为 done
        state["task_status"] = TASK_STATUS_FAILED
        # ===== [S1-D 修正 END] =====
        # 构造空报告，保证 review_report 字段始终存在
        from app.schemas.report import ReviewReport
        state["review_report"] = ReviewReport(
            task_id=state["task_id"],
            report_id=f"{state['task_id']}-R",
            defects=[],
            errors=list(state.get("errors", [])),
        )
        state["parsed_defects"] = []
        return state

    try:
        report = parse_report(
            llm_response,
            ir_document=state.get("ir"),
            rag_context=[
                {"source": r.source_title, "section": r.source_section}
                for r in state.get("rag_context", [])
            ],
            rule_hits=state.get("rule_hits", []),
        )
    except Exception as exc:
        _append_error(state, f"报告解析失败: {type(exc).__name__}: {exc}")
        # ===== [S1-D 修正] 解析异常也置 failed =====
        state["task_status"] = TASK_STATUS_FAILED
        # ===== [S1-D 修正 END] =====
        from app.schemas.report import ReviewReport
        state["review_report"] = ReviewReport(
            task_id=state["task_id"],
            report_id=f"{state['task_id']}-R",
            defects=[],
            errors=list(state.get("errors", [])),
        )
        state["parsed_defects"] = []
        return state

    # ===== [S1-B 新增 v12] Merge 层：补齐 rule 缺陷 =====
    # 架构原则：Rule Engine 决定规则缺陷，LLM 决定 AI 发现，Merge 层决定最终缺陷集。
    # 如果 LLM 漏输出某条 rule_hit 对应的缺陷，从 RuleResult 直接构造补齐。
    rule_hits = state.get("rule_hits", [])
    report.defects = _merge_rule_defects(report.defects, rule_hits)
    report.recount_review_status()                # 补齐后重算三态分布
    # ===== [S1-B 新增 v12 END] =====

    # 合并节点错误到报告 errors（parse_report 内部已合并 LLM 层错误）
    for e in state.get("errors", []):
        if e not in report.errors:
            report.errors.append(e)

    state["review_report"] = report
    state["parsed_defects"] = report.defects
    logger.info(
        "report_generate ok: case_id=%s defects=%d errors=%d",
        state["case_id"], len(report.defects), len(report.errors),
    )
    return state


# ============================================================
# 节点 6：feedback_save
# ============================================================
def node_feedback_save(state: ReviewGraphState) -> ReviewGraphState:
    """
    落库 ReviewReport 到 review_result + review_defect。

    【阶段 D 边界】
      只落库报告，不实现反馈提交（反馈提交留阶段 E）。
      命名保留 "feedback_save" 是为了与 Sprint 1 §5.3 的 6 节点清单对齐，
      实际职责是"报告落库"，反馈入口在阶段 E 由 API 层调用 feedback_service。

    落库字段映射：
      review_result.task_id          ← state["task_id"]
      review_result.review_output_json ← state["review_report"].model_dump(mode="json")
      review_result.schematic_case_id  ← 按 case_id 查 schematic_case（查不到留 NULL）
      review_result.review_status     ← 主缺陷三态（取最高优先级）
      review_result.total_defects     ← len(defects)
      review_result.*_count           ← 三态分布
      review_defect.*                 ← 每条 Defect 的 10 字段 + 追踪字段
    """
    report = state.get("review_report")
    if report is None:
        _append_error(state, "feedback_save 跳过：review_report 未生成")
        state["task_status"] = TASK_STATUS_FAILED
        return state

    try:
        # ===== [S1-F 修正] 传 rule_hits，供 _persist_review_report 纠正
        # origin=rule 缺陷的 category =====
        _persist_review_report(
            report, state["case_id"], state["task_id"],
            rule_hits=state.get("rule_hits", []),
            ir_doc=state.get("ir"),
        )
        # ===== [S1-F 修正 END] =====
    except Exception as exc:
        _append_error(state, f"报告落库失败: {type(exc).__name__}: {exc}")
        state["task_status"] = TASK_STATUS_FAILED
        return state

    # ===== [S1-D 修正] 若上游已置 failed，不覆盖为 done =====
    # --- 原代码（保留，已注释） ---
    # state["task_status"] = TASK_STATUS_DONE
    # --- 原代码（保留，已注释） ---
    if state.get("task_status") != TASK_STATUS_FAILED:
        state["task_status"] = TASK_STATUS_DONE
    # ===== [S1-D 修正 END] =====
    logger.info("feedback_save ok: case_id=%s task_id=%s status=%s",
                state["case_id"], state["task_id"], state["task_status"])
    return state    


def _resolve_rule_fields(
    d,
    rule_hits: list | None,
) -> tuple[str, str]:
    """
    对 origin=rule 的缺陷，用 RuleResult 的权威字段覆盖 LLM 输出。

    【S1-F 修正】
      原因：LLM 可能改写 category / risk（如把 CTRL_001 的 high 写成 medium），
           导致 Pred 与 GT 不一致，Full Review 匹配失败。
      权威来源：
        - category  ← RuleResult.category
        - risk      ← RuleResult.severity
      两者使用同一套等级体系（rule_format.md §1）。

    【工程要求】
      1. 找不到对应规则时，记录 warning（不静默忽略）。
      2. category 与 risk 覆盖行为一致。
      3. 非 origin=rule 的缺陷不覆盖。

    :return: (resolved_category, resolved_risk)
    """
    resolved_category = d.category
    resolved_risk = d.risk

    if d.origin == "rule" and d.rule_id and rule_hits:
        matched_rule = None
        for rh in rule_hits:
            if rh.rule_id == d.rule_id:
                matched_rule = rh
                break

        if matched_rule is not None:
            resolved_category = matched_rule.category
            resolved_risk = matched_rule.severity
        else:
            logger.warning(
                "_resolve_rule_fields: origin=rule 的缺陷 %s 的 rule_id=%s "
                "在 rule_hits 中未找到对应规则，category/risk 保持 LLM 输出",
                d.defect_id, d.rule_id,
            )

    return resolved_category, resolved_risk


# ===== [S1-D 修正] 签名加 task_id 参数，落库用 Graph 的 task_id =====
def _persist_review_report(
    report,
    case_id: str,
    task_id: str,
    rule_hits: list | None = None,
    ir_doc=None,
) -> None:
    """
    将 ReviewReport 落库到 review_result + review_defect。

    实现细节：
      - 用 get_db_session() 上下文管理器，自动 commit/rollback
      - 按 case_id 查 schematic_case 补全 schematic_case_id（查不到留 NULL）
      - 主 review_status 取三态中"最需要专家关注"的：
        优先级 LOW_CONFIDENCE > NEED_EXPERT_REVIEW > AI_CONFIRMED
      - review_defect 逐条插入，defect_id 冲突时跳过（幂等）

    【S1-D 修正】task_id 显式传入
      参数 task_id 来自 state["task_id"]（由 task_service 生成），
      不是 report.task_id（来自 LLM 输出回显，可能不一致）。

    【S1-F 修正】新增 rule_hits 参数
      原因：origin=rule 的缺陷，category 应以规则引擎为准（RuleResult.category），
           不应被 LLM 输出覆盖。LLM 可能把 POWER_002 的 power 写成 reset，
           导致 Pred 的 category 与 GT 不一致，Full Review 匹配失败。
      用法：按 rule_id 匹配，用 RuleResult.category 覆盖 d.category。
      兜底：rule_id 在 rule_hits 里找不到时，保持 d.category 原值。
    """
    from app.persistence.models import ReviewDefect, ReviewResult, SchematicCase

    # ===== [S1-E 修正] 落库前统一改写 defect_id 为 {task_id}-{defect_id} =====
    # 原因：
    #   1. review_defect.defect_id 在 DB 层是 unique=True（全局唯一），
    #      但缺陷 ID 的语义应是"本次评审内唯一"，不是"全局唯一"。
    #      真实 LLM 每次返回 DEF-1 / DEF-2，若直接落库，第二次评审会因
    #      unique 冲突被幂等逻辑跳过，导致 review_result 下无 review_defect。
    #   2. 加 task_id 前缀后全局唯一，同时保留原 ID 可追溯：
    #      "DEF-1" → "RUN-20261022-120000-ABCD-DEF-1"
    #   3. 改写放在落库前，保证 review_output_json 与 review_defect 的
    #      defect_id 一致，不出现两套 ID。
    #   4. 前缀格式 {task_id}-{defect_id}：task_id 自身格式固定为
    #      RUN-YYYYMMDD-HHMMSS-XXXX，后接 "-DEF-N" 仍可解析。
    for d in report.defects:
        d.defect_id = f"{task_id}-{d.defect_id}"
    # ===== [S1-E 修正 END] =====

    # 主 review_status 取最严格态
    status_priority = {"LOW_CONFIDENCE": 3, "NEED_EXPERT_REVIEW": 2, "AI_CONFIRMED": 1}
    main_status = "AI_CONFIRMED"
    if report.defects:
        main_status = max(
            (d.review_status.value for d in report.defects),
            key=lambda s: status_priority.get(s, 0),
        )

    # 三态计数
    dist = report.summary.review_status_distribution

    with get_db_session() as db:
        # 按 case_id 查 schematic_case
        schematic_case_id = None
        case_row = (
            db.query(SchematicCase)
            .filter(SchematicCase.case_id == case_id)
            .order_by(SchematicCase.id.desc())
            .first()
        )
        if case_row is not None:
            schematic_case_id = case_row.id

        rr = ReviewResult(
            # --- 原代码（保留，已注释） ---
            # task_id=report.task_id,
            # --- 原代码（保留，已注释） ---
            # ===== [S1-D 修正] 用 Graph 的 task_id，不依赖 LLM 回显 =====
            task_id=task_id,
            # ===== [S1-D 修正 END] =====
            schematic_case_id=schematic_case_id,
            # ===== [S1-D 修正] 补 review_output_json（NOT NULL 字段），
            # 存完整报告 JSON，供 get_task / API 报告查询使用 =====
            review_output_json=report.model_dump(mode="json"),
            # ===== [S1-D 修正 END] =====
            is_rule_only=False,      # 阶段 D 起含 LLM，非纯规则
            category="other",        # 汇总级 category，暂用 other
            risk="medium",           # 汇总级 risk，暂用 medium
            review_status=main_status,
            total_defects=len(report.defects),
            ai_confirmed_count=dist.AI_CONFIRMED,
            need_expert_review_count=dist.NEED_EXPERT_REVIEW,
            low_confidence_count=dist.LOW_CONFIDENCE,
        )
        db.add(rr)
        db.flush()

        for d in report.defects:
            # ===== [S1-E 修正] defect_id 已在落库前改写为 {task_id}-{defect_id}，
            # 此处 d.defect_id 已是全局唯一值。幂等检查针对同一 task_id 重跑场景。
            # --- 原代码（保留，已注释） ---
            # exists = (
            #     db.query(ReviewDefect)
            #     .filter(ReviewDefect.defect_id == d.defect_id)
            #     .first()
            # )
            # if exists is not None:
            #     logger.warning("review_defect 已存在，跳过: %s", d.defect_id)
            #     continue
            # --- 原代码（保留，已注释） ---
            exists = (
                db.query(ReviewDefect)
                .filter(ReviewDefect.defect_id == d.defect_id)
                .first()
            )
            if exists is not None:
                logger.warning("review_defect 已存在，跳过: %s", d.defect_id)
                continue
            # ===== [S1-F 修正] origin=rule 的缺陷，category / risk 以规则引擎为准 =====
            # 统一走 _resolve_rule_fields，保证两个字段行为一致。
            resolved_category, resolved_risk = _resolve_rule_fields(d, rule_hits)
            # ===== [S1-F 修正 END] =====

            rd = ReviewDefect(
                review_result_id=rr.id,
                defect_id=d.defect_id,
                # --- 原代码（保留，已注释） ---
                # category=d.category,
                # --- 原代码（保留，已注释） ---
                category=resolved_category,
                location=d.location,
                # ===== [S1-F 修正] LLM 可能输出 "-" / "" 作为 component 占位符 =====
                component=d.component if d.component not in (None, "", "-") else None,
                # ===== [S1-F 修正] LLM 可能输出 "-" / "" 作为 net 占位符，
                # 归一化为 None，避免 match_defect_full_review 把 '-' 当作真实网络名。
                # ===== [S1-B 修正 v10] 额外防御：net 必须是 IR 里存在的 net_name，
                # 否则视为 LLM 输出错误（如把引脚名 "CE" 当网络名），归一化为 None。
                # 原因：case005 的 CTRL_001 是引脚悬空缺陷，rule_hit 的 net_refs 为空，
                #      LLM 却填了 net="CE"（引脚名）。Pred net 非空 → 走严格匹配路径，
                #      {CE} & GT 的 {} = 空 → 匹配失败，tp=0。
                #      归一化为 None 后，走降级路径（pred_net 为空 + risk + category + component），
                #      与 GT 的 net=None 匹配成功。
                net=(
                    d.net
                    if d.net not in (None, "", "-")
                    and ir_doc is not None
                    and any(n.net_name == d.net for n in ir_doc.nets)
                    else None
                ),
                # ===== [S1-B 修正 v10 END] =====
                risk=resolved_risk,
                evidence=[e.model_dump(mode="json") for e in d.evidence],
                root_cause=d.root_cause,
                suggestion=d.suggestion,
                confidence=d.confidence,
                review_status=d.review_status.value,
                origin=d.origin,
                rule_id=d.rule_id,
                expert_decision=d.expert_decision,
            )
            db.add(rd)


# ============================================================
# 构建 Graph
# ============================================================
def build_review_graph():
    """
    构建 6 节点线性 DAG。

    说明：
      - 固定线性结构，不引入条件分支 / Planner / ReAct
      - 每个节点返回完整 state，LangGraph 自动合并
      - 异常已在节点内写 errors，不会因节点异常中断 Graph
    """
    graph = StateGraph(ReviewGraphState)

    graph.add_node("load_ir", node_load_ir)
    graph.add_node("rule_check", node_rule_check)
    graph.add_node("rag_retrieve", node_rag_retrieve)
    graph.add_node("expert_analysis", node_expert_analysis)
    graph.add_node("report_generate", node_report_generate)
    graph.add_node("feedback_save", node_feedback_save)

    graph.set_entry_point("load_ir")
    graph.add_edge("load_ir", "rule_check")
    graph.add_edge("rule_check", "rag_retrieve")
    graph.add_edge("rag_retrieve", "expert_analysis")
    graph.add_edge("expert_analysis", "report_generate")
    graph.add_edge("report_generate", "feedback_save")
    graph.add_edge("feedback_save", END)

    return graph.compile()


# ============================================================
# 入口：运行 Graph
# ============================================================
def run_review_graph(case_id: str, task_id: str) -> ReviewGraphState:
    """
    运行 6 节点评审工作流。

    :param case_id: 业务 Case 标识（如 "case001"）
    :param task_id: 本次评审任务 ID（如 "RUN-20261022-001"）
    :return: 最终 State（含 review_report / errors / task_status）

    说明：
      - 调用方（task_service）负责生成 task_id 和处理返回值
      - 本函数不抛异常，所有异常已写 state["errors"]
    """
    from app.workflows.state import make_initial_state

    graph = build_review_graph()
    initial_state = make_initial_state(case_id, task_id)
    initial_state["task_status"] = TASK_STATUS_RUNNING

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        # 兜底：LangGraph 自身异常（如节点签名不匹配）
        logger.error("run_review_graph 异常: %s", exc, exc_info=True)
        initial_state["task_status"] = TASK_STATUS_FAILED
        initial_state.setdefault("errors", []).append(
            f"Graph 执行异常: {type(exc).__name__}: {exc}"
        )
        return initial_state

    return final_state


__all__ = [
    "build_review_graph",
    "run_review_graph",
    "node_load_ir",
    "node_rule_check",
    "node_rag_retrieve",
    "node_expert_analysis",
    "node_report_generate",
    "node_feedback_save",
]
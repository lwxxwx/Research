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
  3. rag_retrieve     : 按 rule_hits + case_id 检索 RAG 证据
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
    """
    if not rag_context:
        return "(无检索结果)"
    blocks: list[str] = []
    for r in rag_context:
        blocks.append(
            f"source: {r.source_title}\n"
            f"section: {r.source_section}\n"
            f"reason: {r.snippet}\n"
            f"source_type: {r.source_type}"
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
    按 rule_hits + case_id 检索 RAG 证据，保存到 state["rag_context"]。

    检索策略：
      - query：从 rule_hits 的 hit_message 拼出查询文本；无命中时用 case_id
      - rule_ids 过滤：取 rule_hits 的 rule_id 去重
      - part_numbers 过滤：暂不用（Case IR 不强制带 part_number）
      - top_k：RAG_TOP_K

    失败不阻断：rag_context 为空时，expert_analysis 会走"无 RAG 依据"降级分支。

    【维度守卫链路】
      Retriever.__init__ 会调用 assert_backend_matches_db，校验
      RAG_EMBEDDING_BACKEND 期望维度与 DB 实际维度一致。
      若不一致（如 .env 切到 minilm 但 DB 仍是 1024 维），
      Retriever() 抛 RuntimeError，被本函数 try/except 捕获，
      写入 state["errors"]，rag_context 置空，
      expert_analysis 走"无 RAG 依据"降级分支。
      这是预期行为，不需要改代码；排查时看 errors 里的
      "RAG 检索失败: RuntimeError: Embedding 维度不匹配..." 即可。
    """
    ir = state.get("ir")
    if ir is None:
        _append_error(state, "rag_retrieve 跳过：ir 未加载")
        return state

    rule_hits = state.get("rule_hits", [])

    # 拼查询文本
    if rule_hits:
        query = " ".join(r.hit_message for r in rule_hits)
    else:
        query = f"case {state['case_id']} 电路评审"

    rule_ids = list({r.rule_id for r in rule_hits}) if rule_hits else None

    try:
        retriever = Retriever()
        rag_context = retriever.retrieve(
            query=query,
            rule_ids=rule_ids if RAG_FILTER_RULE_IDS else None,
            top_k=RAG_TOP_K,
        )
    except Exception as exc:
        _append_error(state, f"RAG 检索失败: {type(exc).__name__}: {exc}")
        state["rag_context"] = []
        return state

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
                # ===== [S1-F 修正 END] =====
                #component=d.component,
                # ===== [S1-F 修正] LLM 可能输出 "-" / "" 作为 net 占位符，
                # 归一化为 None，避免 match_defect_full_review 把 '-' 当作真实网络名。
                net=d.net if d.net not in (None, "", "-") else None,
                # ===== [S1-F 修正 END] =====
                # --- 原代码（保留，已注释） ---
                #net=d.net,
                # --- 原代码（保留，已注释） ---
                # risk=d.risk,
                # --- 原代码（保留，已注释） ---
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
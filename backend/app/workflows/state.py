# backend/app/workflows/state.py
"""
Graph State 定义（Sprint 1 阶段 D）

【职责】
  LangGraph 节点之间传递的唯一数据结构。所有节点读写本 State，
  不通过全局变量、不通过 DB 中转。

【设计原则】
  1. State 字段与 Sprint 1 §5.3 冻结清单一一对应：
     case_id / ir / rule_hits / rag_context / llm_raw_output /
     parsed_defects / review_report / errors / 任务状态
  2. 字段类型用"已冻结的 Pydantic 模型或原始类型"
  3. errors 用 list[str] 追加，不抛异常中断 Graph

【S1-D 修正】去掉 TYPE_CHECKING，改为运行时 import
  原因：LangGraph 的 StateGraph 构造时对 TypedDict 做 get_type_hints() 求值，
       TYPE_CHECKING 保护的字符串前向引用在运行时不生效，
       会抛 NameError: name 'SchematicIRDocument' is not defined。
       阶段 C 的五个类型模块均无重依赖（zhipuai 等在 provider 内延迟 import），
       运行时 import 安全。
"""
from __future__ import annotations

from typing import Any, TypedDict

# ===== [S1-D 修正] 运行时 import，不用 TYPE_CHECKING =====
# 原因见模块 docstring
from app.ir.schema import SchematicIRDocument
from app.llm.provider import LLMResponse
from app.rag.knowledge import RetrievalResult
from app.rules.engine import RuleResult
from app.schemas.report import ReviewReport
# ===== [S1-D 修正 END] =====


# ============================================================
# 任务状态枚举（字符串常量，与 review_result.review_status 区分）
# ============================================================
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"


class ReviewGraphState(TypedDict, total=False):
    """
    LangGraph 6 节点共享 State。

    total=False 表示所有字段可选：节点按需读写，未初始化的字段
    在访问前由节点自行判断（通常用 state.get("xxx")）。

    字段清单与 Sprint 1 §5.3 对齐：
      case_id         : 业务 Case 标识（如 "case001"）
      task_id         : 本次评审任务 ID（如 "RUN-20261022-001"）
      task_status     : pending / running / done / failed
      ir              : SchematicIRDocument
      rule_hits       : list[RuleResult]
      rag_context     : list[RetrievalResult]
      llm_raw_output  : LLMResponse
      parsed_defects  : list[Defect]（从 ReviewReport 派生，便于节点消费）
      review_report   : ReviewReport（最终产物）
      errors          : list[str]（节点异常累计，不中断 Graph）
    """

    # --- 任务标识 ---
    case_id: str
    task_id: str
    # ===== [S3-G3C] 审计字段：任务创建者 users.id =====
    created_by: int | None
    # ===== [S3-G3C END] =====
    task_status: str

    # --- 节点 1：load_ir 产出 ---
    ir: SchematicIRDocument

    # --- 节点 2：rule_check 产出 ---
    rule_hits: list[RuleResult]

    # --- 节点 3：rag_retrieve 产出 ---
    rag_context: list[RetrievalResult]

    # --- 节点 4：expert_analysis 产出 ---
    llm_raw_output: LLMResponse

    # --- 节点 5：report_generate 产出 ---
    parsed_defects: list[Any]          # list[Defect]，用 Any 避免 import 循环
    review_report: ReviewReport

    # --- 异常累计 ---
    errors: list[str]


# ============================================================
# 初始 State 工厂
# ============================================================
def make_initial_state(
    case_id: str,
    task_id: str,
    created_by: int | None = None,   # ← 加参数
) -> ReviewGraphState:
    """
    构造 Graph 初始 State。

    :param case_id: 业务 Case 标识（对应 data/cases/<case_id>/）
    :param task_id: 本次评审任务 ID（用于日志与 review_result.task_id）
    :param created_by: 任务创建者 users.id（S3-G3C 审计字段，可空）
    :return: 初始 State，仅含任务标识与空 errors
    """
    return ReviewGraphState(
        case_id=case_id,
        task_id=task_id,
        created_by=created_by,       # ← 加这行
        task_status=TASK_STATUS_PENDING,
        errors=[],
    )


__all__ = [
    "ReviewGraphState",
    "make_initial_state",
    "TASK_STATUS_PENDING",
    "TASK_STATUS_RUNNING",
    "TASK_STATUS_DONE",
    "TASK_STATUS_FAILED",
]
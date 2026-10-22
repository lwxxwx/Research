# backend/app/services/task_service.py
"""
评审任务服务（Sprint 1 阶段 D）

【职责】
  1. create_task(case_id)  : 生成 task_id，调用 run_review_graph，返回 task_id
  2. get_task(task_id)     : 查询 review_result，返回任务状态 + 报告
  3. list_tasks()          : 列出最近任务（阶段 G 的 API 会用到）

【与 Sprint 0 的约定】
  - 不建 tasks 表：Sprint 0 已冻结用 review_result 承担"一次评审"
  - task_id 对应 review_result.task_id（String 64，批次标识）
  - 报告 JSON 存 review_result.review_output_json，缺陷明细存 review_defect

【与阶段 C/D 的衔接】
  - create_task 调用 app.workflows.review_graph.run_review_graph
  - 落库由 review_graph 的 feedback_save 节点负责（阶段 D 边界）
  - 本服务不重复落库，只负责生成 task_id 和查询

【异常处理】
  - run_review_graph 内部已把所有异常写进 state["errors"]，不抛
  - 本服务只在 DB 查询层做 try/except，DB 异常向上抛（由 API 层统一处理）
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.persistence.db import get_db_session
from app.persistence.models import ReviewDefect, ReviewResult
from app.schemas.feedback import (
    FeedbackCreate,   # ← 新增
    FeedbackOut,
    FeedbackType,
    RuleCandidate,
    SuggestionDiff,
)
from app.services.feedback_service import FeedbackNotFoundError
from app.services.feedback_service import FeedbackService   # ← 新增
from app.workflows.review_graph import run_review_graph
from app.workflows.state import (
    TASK_STATUS_DONE,
    TASK_STATUS_FAILED,
    ReviewGraphState,
)

logger = logging.getLogger(__name__)


# ============================================================
# task_id 生成
# ------------------------------------------------------------
# 格式：RUN-YYYYMMDD-HHMMSS-XXXX
#   RUN      : 固定前缀，表示"一次评审运行"
#   YYYYMMDD : 日期
#   HHMMSS   : 时间
#   XXXX     : 4 位随机十六进制，避免同一秒并发冲突
#
# 与 review_result.task_id（String 64）兼容；不保证全局唯一（DB 未加唯一约束），
# 但实际并发概率极低，且 get_task 按 task_id 查最新一条即可。
# ============================================================
def _make_task_id() -> str:
    """
    生成任务 ID，格式 RUN-YYYYMMDD-HHMMSS-XXXX。

    说明：
      - 不依赖 DB 序列，避免 create_task 前多一次 DB 往返
      - 4 位随机后缀：同一秒内多次调用几乎不会撞
      - 与 review_result.task_id 字段（String 64）兼容
    """
    import uuid
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4].upper()
    return f"RUN-{ts}-{suffix}"


# ============================================================
# create_task：创建并运行评审任务
# ============================================================
def create_task(case_id: str) -> str:
    """
    创建并运行一次评审任务。

    流程：
      1. 生成 task_id
      2. 调用 run_review_graph(case_id, task_id)
         - 6 节点 DAG 执行
         - feedback_save 节点落库 review_result + review_defect
      3. 返回 task_id

    失败处理：
      - run_review_graph 不抛异常，失败信息在 state["errors"]
      - 若 Graph 整体失败（task_status == failed），仍返回 task_id
        （调用方可通过 get_task 查询错误详情）
      - 若 Graph 在 feedback_save 前失败，review_result 可能未落库，
        get_task 返回 None，调用方需处理

    :param case_id: 业务 Case 标识（如 "case001"）
    :return: task_id（即使失败也返回，供查询错误详情）
    """
    task_id = _make_task_id()
    logger.info("create_task start: case_id=%s task_id=%s", case_id, task_id)

    final_state: ReviewGraphState = run_review_graph(case_id, task_id)

    status = final_state.get("task_status", TASK_STATUS_FAILED)
    errors = final_state.get("errors", [])

    if status == TASK_STATUS_FAILED:
        logger.error(
            "create_task failed: case_id=%s task_id=%s errors=%s",
            case_id, task_id, errors,
        )
    else:
        logger.info(
            "create_task done: case_id=%s task_id=%s defects=%d errors=%d",
            case_id, task_id,
            len(final_state.get("parsed_defects", [])),
            len(errors),
        )

    return task_id


# ============================================================
# get_task：查询任务状态 + 报告
# ============================================================
def get_task(task_id: str) -> dict[str, Any] | None:
    """
    按 task_id 查询任务状态与报告。

    返回结构：
      {
        "task_id": str,
        "status": "done" | "failed" | "pending",   # 任务状态，非缺陷质量状态
        "case_id": str | None,
        "report": dict | None,                      # ReviewReport 的 JSON
        "errors": list[str],                        # 报告中的 errors
        "created_at": str | None,                   # ISO 格式
        "defects_count": int,
      }

    状态判断：
      - review_result 存在 → "done"（阶段 D 的 feedback_save 只在成功时落库）
      - review_result 不存在 → "pending"（Graph 未走到 feedback_save，或 task_id 不存在）
      - 报告中 errors 非空且 defects 为空 → "failed"

    :param task_id: 任务 ID
    :return: 任务信息 dict；不存在返回 None

    「有缺陷」和「无缺陷无错误」都判为 done，只有「无缺陷但有错误」判为 failed。这是合理的——评审正常完成但没发现问题，也是 done；只有 LLM 报错、IR 加载失败等才 failed。

    但有一种边界情况：LLM 输出了缺陷，但所有缺陷都被 Parser 降级为 LOW_CONFIDENCE——此时 defects 非空，判 done。这符合预期（评审完成了，只是质量低），但如果你希望这种情况单独标记，需要扩展状态枚举。当前不扩展，记入阶段 G API 层再做。
    """
    with get_db_session() as db:
        rr = (
            db.query(ReviewResult)
            .filter(ReviewResult.task_id == task_id)
            .order_by(ReviewResult.id.desc())
            .first()
        )
        if rr is None:
            logger.warning("get_task: task_id=%s 未找到 review_result", task_id)
            return None

        report_json = rr.review_output_json or {}
        errors = report_json.get("errors", []) if isinstance(report_json, dict) else []
        defects = report_json.get("defects", []) if isinstance(report_json, dict) else []

        # 状态判定
        if defects:
            status = TASK_STATUS_DONE
        elif errors:
            status = TASK_STATUS_FAILED
        else:
            status = TASK_STATUS_DONE  # 无缺陷无错误，视为正常完成（空报告）

        # 查 case_id（通过 schematic_case_id 反查）
        case_id = None
        if rr.schematic_case_id is not None:
            from app.persistence.models import SchematicCase
            case_row = db.get(SchematicCase, rr.schematic_case_id)
            if case_row is not None:
                case_id = case_row.case_id

        return {
            "task_id": rr.task_id,
            "status": status,
            "case_id": case_id,
            "report": report_json,
            "errors": errors,
            "created_at": rr.created_at.isoformat() if rr.created_at else None,
            "defects_count": len(defects),
        }


# ============================================================
# list_tasks：列出最近任务
# ============================================================
def list_tasks(limit: int = 20) -> list[dict[str, Any]]:
    """
    列出最近 N 条评审任务（按 created_at 降序）。

    返回结构：每个元素同 get_task 的简化版（不含完整 report），
    供阶段 G 的 API 列表页使用。

    :param limit: 返回条数上限，默认 20
    :return: 任务摘要列表
    """
    with get_db_session() as db:
        rows = (
            db.query(ReviewResult)
            .order_by(ReviewResult.created_at.desc(), ReviewResult.id.desc())
            .limit(limit)
            .all()
        )

        result: list[dict[str, Any]] = []
        for rr in rows:
            report_json = rr.review_output_json or {}
            defects = report_json.get("defects", []) if isinstance(report_json, dict) else []
            errors = report_json.get("errors", []) if isinstance(report_json, dict) else []

            if defects:
                status = TASK_STATUS_DONE
            elif errors:
                status = TASK_STATUS_FAILED
            else:
                status = TASK_STATUS_DONE

            result.append({
                "task_id": rr.task_id,
                "status": status,
                "review_status": rr.review_status,
                "total_defects": rr.total_defects,
                "created_at": rr.created_at.isoformat() if rr.created_at else None,
            })

        return result


# ============================================================
# get_task_defects：查询任务下的缺陷明细
# ============================================================
def get_task_defects(task_id: str) -> list[dict[str, Any]]:
    """
    查询任务下的缺陷明细（review_defect 表）。

    说明：
      - review_result 与 review_defect 是 1:N 关系
      - 返回每条 defect 的 10 字段 + 追踪字段 + 专家判定
      - 供阶段 G 的 API 报告详情页使用

    :param task_id: 任务 ID
    :return: 缺陷明细列表；任务不存在返回空列表
    """
    with get_db_session() as db:
        rr = (
            db.query(ReviewResult)
            .filter(ReviewResult.task_id == task_id)
            .order_by(ReviewResult.id.desc())
            .first()
        )
        if rr is None:
            return []

        defects = (
            db.query(ReviewDefect)
            .filter(ReviewDefect.review_result_id == rr.id)
            .order_by(ReviewDefect.id.asc())
            .all()
        )

        return [
            {
                "defect_id": d.defect_id,
                "category": d.category,
                "location": d.location,
                "component": d.component,
                "net": d.net,
                "risk": d.risk,
                "evidence": d.evidence,
                "root_cause": d.root_cause,
                "suggestion": d.suggestion,
                "confidence": d.confidence,
                "review_status": d.review_status,
                "origin": d.origin,
                "rule_id": d.rule_id,
                "expert_decision": d.expert_decision,
            }
            for d in defects
        ]

# ============================================================
# submit_task_feedback：反馈提交入口（Sprint 1 阶段 E1）
# ------------------------------------------------------------
# 设计理由：
#   1. 调用方（API / Demo / CLI）只需知道 task_id + defect_id，
#      不需要自己查 DB 拿 review_result_id / review_defect_id 外键。
#      这是"对外入口"的语义，符合 Sprint 1 §10.1 的 FeedbackService 边界。
#   2. 内部把 task_id + defect_id 翻译成 review_result_id / review_defect_id，
#      再委托 feedback_service.submit_feedback 落库 + 触发规则演进。
#   3. 六类反馈全部走同一入口，不按类型分支（分支在 feedback_service 内部）。
# ============================================================
def submit_task_feedback(
    task_id: str,
    defect_id: str,
    feedback_type: FeedbackType,
    *,
    expert_suggestion: str | None = None,
    suggestion_adopted: str | None = None,
    suggestion_diff: list[SuggestionDiff] | None = None,
    rule_candidate: RuleCandidate | None = None,
    created_by: int | None = None,
) -> FeedbackOut:
    """
    按 task_id + defect_id 提交专家反馈。

    参数：
      task_id           : 评审任务 ID（review_result.task_id）
      defect_id         : 缺陷业务 ID（review_defect.defect_id）
      feedback_type     : 六类反馈类型
      expert_suggestion : 专家建议（correct_defect / false_positive / false_negative /
                          suggestion_update / knowledge_gap 时用）
      suggestion_adopted: 是否采纳建议（suggestion_update 时用，yes/partial/no）
      suggestion_diff   : 新旧建议对比（suggestion_update 时用）
      rule_candidate    : 规则候选草稿（false_negative / new_rule_candidate 时用）
      created_by        : 反馈人 users.id（可选）

    返回：
      FeedbackOut（含 JOIN 回填的 task_id / defect_id）

    异常：
      - task_id 不存在 → FeedbackNotFoundError（继承 ValueError）
      - defect_id 不存在或不属于该 task_id → FeedbackNotFoundError
    """
    #原代码注释#
    #from app.persistence.models import ReviewDefect, ReviewResult
    #from app.schemas.feedback import FeedbackCreate
    #from app.services.feedback_service import FeedbackService

    with get_db_session() as db:
        # 1. 按 task_id 查 review_result
        rr = (
            db.query(ReviewResult)
            .filter(ReviewResult.task_id == task_id)
            .order_by(ReviewResult.id.desc())
            .first()
        )
        if rr is None:
            raise FeedbackNotFoundError(
                f"review_result task_id={task_id} 不存在"
            )

        # 2. 按 defect_id + review_result_id 查 review_defect
        #    注意：defect_id 是全局唯一（DB unique 约束），但为了防御
        #    "同一 defect_id 挂在多个 review_result 下"的异常数据，
        #    额外用 review_result_id 限定，确保反馈挂在正确的任务下。
        rd = (
            db.query(ReviewDefect)
            .filter(
                ReviewDefect.defect_id == defect_id,
                ReviewDefect.review_result_id == rr.id,
            )
            .first()
        )
        if rd is None:
            raise FeedbackNotFoundError(
                f"review_defect defect_id={defect_id} "
                f"(review_result_id={rr.id}) 不存在"
            )

        # 3. 构造 FeedbackCreate，补全外键
        fb_in = FeedbackCreate(
            review_result_id=rr.id,
            review_defect_id=rd.id,
            feedback_type=feedback_type,
            expert_suggestion=expert_suggestion,
            suggestion_adopted=suggestion_adopted,
            suggestion_diff=suggestion_diff,
            rule_candidate=rule_candidate,
            created_by=created_by,
        )

        # 4. 委托 feedback_service
        svc = FeedbackService()
        return svc.submit_feedback(db, fb_in)

__all__ = [
    "create_task",
    "get_task",
    "list_tasks",
    "get_task_defects",
    "submit_task_feedback",
]
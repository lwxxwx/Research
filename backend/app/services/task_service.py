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

import re
import logging
from datetime import datetime, timezone
from typing import Any

from app.persistence.db import get_db_session
from app.persistence.models import ReviewDefect, ReviewResult
# ===== [S3-G3] 业务参数校验异常 → 422 =====
from app.core.exceptions import BusinessValueError
# ===== [S3-G3 END] =====
from app.schemas.feedback import (
    FeedbackCreate,   # ← 新增
    FeedbackOut,
    FeedbackType,
    RuleCandidate,
    SuggestionDiff,
)
from app.services.feedback_service import FeedbackNotFoundError
from app.services.feedback_service import FeedbackService   # ← 新增
# ===== [S2-C] 真实指标函数在 get_task_summary_metrics 内 import，避免循环 =====
# ===== [S2-C] 真实指标聚合 =====
from app.services.benchmark_service import (
    compute_real_ai_ner,
    compute_real_ear,
    compute_sample_info,
)
# ===== [S2-C END] =====
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
def create_task(case_id: str, created_by: int | None = None) -> str:
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
    :param created_by: 创建人 ID（可选）
    :return: {
        "task_id": str,
        "status": str,              # done / failed / pending
        "defects_count": int,       # 优先取 Graph parsed_defects；为空时单次查 ReviewResult 兜底
    }
    失败时仍返回 dict（status=failed，defects_count 取已解析缺陷数，不查库）。
    """
    task_id = _make_task_id()
    logger.info("create_task start: case_id=%s task_id=%s", case_id, task_id)

    final_state: ReviewGraphState = run_review_graph(
        case_id, task_id, created_by=created_by,
    )

    status = final_state.get("task_status", TASK_STATUS_FAILED)
    errors = final_state.get("errors", [])
    parsed_defects = final_state.get("parsed_defects", []) or []
    defects_count = len(parsed_defects)

    if status == TASK_STATUS_FAILED:
        logger.error(...)
    else:
        logger.info(
            "create_task done: case_id=%s task_id=%s defects=%s errors=%s",
            case_id, task_id, defects_count, len(errors),
        )

    # ===== [S3-LEGACY-B1] D24：parsed_defects 为空时单次查 ReviewResult 兜底 =====
    if status != TASK_STATUS_FAILED and defects_count == 0:
        with get_db_session() as db:
            rr = (
                db.query(ReviewResult)
                .filter(ReviewResult.task_id == task_id)
                .order_by(ReviewResult.id.desc())
                .first()
            )
            if rr is not None:
                report_json = rr.review_output_json or {}
                defects_count = len(report_json.get("defects", []) or [])
    # ===== [S3-LEGACY-B1 END] =====

    return {
        "task_id": task_id,
        "status": status,
        "defects_count": defects_count,
    }


# ============================================================
# get_task：查询任务状态 + 报告
# ============================================================
def get_task(task_id: str, current_user=None) -> dict[str, Any] | None:
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
            # 分支 1：真不存在
            logger.info(
                "task.not_found: task_id=%s current_user_id=%s current_user_role=%s",
                task_id,
                getattr(current_user, "id", None),
                getattr(current_user, "role", None),
            )
            return None

        # ===== [S3-G3B] engineer 对象级数据范围（D6：对外 404，日志区分）=====
        if current_user is not None and getattr(current_user, "role", None) == "engineer":
            if rr.created_by is None:
                # 分支 2：历史数据（created_by NULL），engineer 无权访问
                logger.warning(
                    "audit.historical_unauthorized: task_id=%s current_user_id=%s reason=created_by_is_null",
                    task_id, current_user.id,
                )
                return None
            if rr.created_by != current_user.id:
                # 分支 3：真实越权（engineer 访问他人任务）
                logger.warning(
                    "audit.idor_hidden: task_id=%s current_user_id=%s owner_id=%s",
                    task_id, current_user.id, rr.created_by,
                )
                return None
        # ===== [S3-G3B END] =====

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

def _infer_case_class(case_id: str | None) -> str | None:
    """
    从 case_id 推断 case_class（A / B）。

    【G2 决策 2：不默认归 A】
      - A 类：`^case\\d{3}$`（case001–case006）
      - B 类：`^case_b\\d{2}_[a-z0-9_]+_B$`（case_b01–case_b04）
      - 未知或缺失 → 返回 None（不默认归 A）

    【为什么返回 None 而不是 "A"】
      - 默认归 A 会让未知 case 被误归类为 A 类
      - 返回 None 让上层明确知道"这个 case 分类未知"
      - `list_tasks_filtered(case_class="A")` 只返回明确匹配 A 类的，未知分类不命中
      - `list_tasks_filtered(case_class=None)` 时，未知分类仍返回（不丢失）

    :param case_id: 业务 Case ID，可能为 None（如 schematic_case_id 为 NULL 查不到 case_id）
    :return: "A" / "B" / None
    """
    if not case_id:
        return None

    # A 类：case + 3 位数字
    if re.match(r"^case\d{3}$", case_id):
        return "A"

    # B 类：case_b + 2 位数字 + _ + 描述 + _B
    if re.match(r"^case_b\d{2}_[a-z0-9_]+_B$", case_id):
        return "B"

    return None

def list_tasks_filtered(
    *,
    current_user=None,   # ← 加这行（CurrentUser | None，避免循环 import）
    case_class: str | None = None,
    review_status: str | None = None,
    status: str | None = None,   # ← [S2-C] 任务状态过滤（pending/done/failed）
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """
    按条件筛选任务列表（G2 新增，供 API 层用）。

    【与 list_tasks 的区别】
      - list_tasks(limit=20) -> list[dict]（原函数，不动，保持兼容）
      - list_tasks_filtered(...) -> dict{total, items, errors}（新函数，支持筛选 + 分页）

    【G2 决策与约束】
      - 决策 1（status 过滤）：G2 不实现。review_result 表无 task_status 列，
        从 review_output_json.errors 推断不可靠（errors 含非阻断 warnings，
        非纯任务级错误标志）。S2 加真实 task_status 落库后再做。
      - 决策 2（case_class 推断）：Python 层过滤。DB 无 case_class 列，
        用 `_infer_case_class(case_id)` 从 case_id 前缀推断。
        未知/缺失 case_id 返回 None，不默认归 A。
      - 决策 3（JOIN）：用 outerjoin 避免 N+1。保留没有关联 Case 的 ReviewResult。
      - 决策 4（参数校验）：Service 层对非法 case_class / limit / offset 抛 ValueError。
        API 层用 Query/Literal 校验，明确参数错误转 422。

    【筛选逻辑（先过滤后 total 再分页）】
      1. DB 层：按 review_status 过滤（如果传了）
      2. JOIN 拿 case_id（outerjoin，无关联的 case_id 为 None）
      3. Python 层：按 case_class 过滤（如果传了）——未知分类不命中
      4. Python 层：收集 errors（未知/缺失 case_id）
      5. total = 过滤后的总数
      6. 分页：items = 过滤后的 [offset:offset+limit]

    【返回结构】
      {
        "total": int,                  # 过滤后的总数（不是分页后）
        "items": list[dict],           # 当前页的任务
        "errors": list[str],           # 未知/缺失 case_id 的说明
      }
      item 字段：task_id / case_id / case_class / defects_count / review_status / created_at
      **不含 status 字段**（G2 决策 1：status 过滤不实现）

    :param case_class: "A" / "B" / None（None 表示不过滤）
    :param review_status: "AI_CONFIRMED" / "NEED_EXPERT_REVIEW" / "LOW_CONFIDENCE" / None
    :param limit: 每页数量，必须 >= 0
    :param offset: 偏移量，必须 >= 0
    :return: {"total": int, "items": list[dict], "errors": list[str]}
    :raises ValueError: case_class / review_status 非法值，或 limit / offset 负数
    """
    from app.persistence.models import ReviewResult, SchematicCase

    # ===== 决策 4：Service 层参数校验（ValueError） =====
    # 不无差别捕获所有 ValueError，只在本函数入口校验并抛；
    # 调用方（API 层）负责将其转换为 HTTP 422。
    _VALID_CASE_CLASS = {"A", "B"}
    _VALID_REVIEW_STATUS = {"AI_CONFIRMED", "NEED_EXPERT_REVIEW", "LOW_CONFIDENCE"}
    # ===== [S2-C] status 校验 =====
    _VALID_STATUS = {"pending", "done", "failed"}

    if case_class is not None and case_class not in _VALID_CASE_CLASS:
        raise BusinessValueError(
            f"case_class 必须为 'A' / 'B' / None，实际: {case_class!r}"
        )
    # ============ 新增下面这一段 ============
    if review_status is not None and review_status not in _VALID_REVIEW_STATUS:
        raise BusinessValueError(f"review_status 必须为 {_VALID_REVIEW_STATUS} 之一，实际: {review_status!r}")
    #if review_status is not None and review_status not in _VALID_REVIEW_STATUS:
    #    raise BusinessValueError(
    #        f"review_status 必须为 AI_CONFIRMED / NEED_EXPERT_REVIEW / "
    #        f"LOW_CONFIDENCE / None，实际: {review_status!r}"
    #    )

    # ===== [S2-C] status 校验 =====
    if status is not None and status not in _VALID_STATUS:
        raise BusinessValueError(
            f"status 必须为 'pending' / 'done' / 'failed' / None，实际: {status!r}"
        )
    # ===== [S2-C END] =====
    if limit < 0:
        raise BusinessValueError(f"limit 必须 >= 0，实际: {limit}")
    if offset < 0:
        raise BusinessValueError(f"offset 必须 >= 0，实际: {offset}")

    # ===== 决策 3：JOIN 拿 case_id，避免 N+1 =====
    errors: list[str] = []

    with get_db_session() as db:
        # outerjoin：保留没有关联 Case 的 ReviewResult（schematic_case_id 为 NULL 的）
        query = (
            db.query(ReviewResult, SchematicCase.case_id)
            .outerjoin(
                SchematicCase,
                ReviewResult.schematic_case_id == SchematicCase.id,
            )
            .order_by(ReviewResult.id.desc())
        )

        # ===== [S3-G3B] created_by 过滤：仅 engineer =====
        if current_user is not None and getattr(current_user, "role", None) == "engineer":
            query = query.filter(ReviewResult.created_by == current_user.id)
        # ===== [S3-G3B END] =====

        # DB 层过滤：review_status
        if review_status:
            query = query.filter(ReviewResult.review_status == review_status)

        # DB 层过滤：review_status
        if review_status:
            query = query.filter(ReviewResult.review_status == review_status)

        # ===== [S2-C] DB 层过滤：task_status（S2-A 已加列） =====
        if status:
            query = query.filter(ReviewResult.task_status == status)
        # ===== [S2-C END] =====

        rows = query.all()

        # Python 层：按 case_class 过滤 + 收集 errors
        filtered_items: list[dict[str, Any]] = []
        for rr, case_id in rows:
            item_case_class = _infer_case_class(case_id)

            # 收集 errors（未知/缺失 case_id）
            if case_id is None:
                errors.append(
                    f"task_id={rr.task_id}: schematic_case_id 为 NULL 或关联的 case 不存在"
                )
            elif item_case_class is None:
                errors.append(
                    f"task_id={rr.task_id}: case_id={case_id!r} 不符合 A/B 命名规范，"
                    f"case_class=Unknown"
                )

            # case_class 过滤（如果传了）
            # 决策 2：传 A/B 时，未知分类不命中；不传时，未知分类仍返回
            if case_class is not None and item_case_class != case_class:
                continue

            filtered_items.append({
                "task_id": rr.task_id,
                "case_id": case_id,
                "case_class": item_case_class,
                "defects_count": rr.total_defects,
                "review_status": rr.review_status,
                "created_at": rr.created_at.isoformat() if rr.created_at else None,
                # ===== [S2-C] task_status 透出（S2-A 已加列，S2-C 补字段） =====
                "status": rr.task_status,
                # ===== [S2-C END] =====
            })

        # 决策 3（后半句）：先过滤后 total 再分页
        total = len(filtered_items)
        paginated = filtered_items[offset: offset + limit]

        return {
            "total": total,
            "items": paginated,
            "errors": errors,
        }

# ============================================================
# get_task_defects：查询任务下的缺陷明细
# ============================================================
def get_task_defects(task_id: str, current_user=None) -> list[dict[str, Any]]:
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
            logger.info(
                "task.not_found: task_id=%s endpoint=get_task_defects current_user_id=%s",
                task_id, getattr(current_user, "id", None),
            )
            return []

        # ===== [S3-G3B] engineer 对象级数据范围（D6：对外 404，日志区分）=====
        if current_user is not None and getattr(current_user, "role", None) == "engineer":
            if rr.created_by is None:
                logger.warning(
                    "audit.historical_unauthorized: task_id=%s endpoint=get_task_defects current_user_id=%s",
                    task_id, current_user.id,
                )
                return []
            if rr.created_by != current_user.id:
                logger.warning(
                    "audit.idor_hidden: task_id=%s endpoint=get_task_defects current_user_id=%s owner_id=%s",
                    task_id, current_user.id, rr.created_by,
                )
                return []
        # ===== [S3-G3B END] =====

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
                # ===== [S1-G G1 修正] 补齐 §23.1 要求的 expert_decision 3 字段 =====
                # 原因：models.py::ReviewDefect 已有 3 字段（阶段 A 迁移落地），
                #      但 get_task_defects 序列化只带了 expert_decision，
                #      漏了 expert_decision_updated_at / expert_decision_by，
                #      导致 API 层无法展示"谁在什么时候做了什么判定"。
                # 修法：只补 2 个字段，expert_decision 已有。
                #      不改查询、不改签名，保持向后兼容。
                #[S3-LEGACY-B1] 薄兼容层：task/defect → 外键的翻译逻辑暂留此处，
                #实际反馈落库与规则演进由 FeedbackService 负责。逻辑迁入 feedback_service
                #安排在后续批次（D25），本层届时改为直接转发或删除。
                # --- 原代码（保留，已注释） ---
                # "expert_decision": d.expert_decision,
                # --- 原代码（保留，已注释） ---
                "expert_decision": d.expert_decision,
                "expert_decision_updated_at": (
                    d.expert_decision_updated_at.isoformat()
                    if d.expert_decision_updated_at is not None
                    else None
                ),
                "expert_decision_by": d.expert_decision_by,
                # ===== [S1-G G1 修正 END] =====
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

# ============================================================
# list_cases：列出所有 Case 元信息（G3 新增，供 API 层用）
# ------------------------------------------------------------
# 数据来源：data/cases/*/info.yaml（由 review_graph.CASES_ROOT 统一决定根路径）
#
# 【G3 决策（已定稿）】
#   - 决策 1（CASES_ROOT）：复用 review_graph.CASES_ROOT；函数内 import，便于
#                          测试 monkeypatch 原模块属性（`rg.CASES_ROOT`）
#   - 决策 2（必填字段）：case_id + case_class 必填；case_class 先检查类型再检查
#                        合法值（非 str / 空 str / 非 A/B 都按非法资产处理）
#   - 决策 3（排序）：A 类在前、B 类在后，组内 case_id 字典序
#   - 决策 4（错误处理）：只校验含 info.yaml 的目录；坏 Case 跳过 + 记 errors；
#                        纯空目录（无 info.yaml）静默跳过
#   - 决策 5（tags）：统一 list[str]，兼容 list / str(逗号分隔) / None
#   - total == len(items)（恒等，不做额外过滤）
# ============================================================
def list_cases() -> dict[str, Any]:
    """
    扫描 data/cases/*/info.yaml，返回 Case 元信息列表（G3 新增）。

    返回结构：
      {
        "total": int,                 # == len(items)
        "items": list[dict],          # A 类在前、B 类在后，组内 case_id 字典序
        "errors": list[str],          # 坏 Case 说明（含目录名 + 原因）
      }

    item 字段（6 个）：
      case_id / case_class / title / description / calibration_status / tags

    【case_class 校验顺序（决策 2 补充）】
      1. 类型：必须是 str（非 str → 非法）
      2. 非空：str.strip() 非空（空串 / 纯空白 → 非法）
      3. 合法值：必须 in ("A", "B")（其他 → 非法）
      任一不满足 → 跳过 + errors 记 "目录名: 原因"

    【tags 归一化（决策 5）】
      - list/tuple → 元素转 str + strip，过滤空串
      - str        → split(",") + strip，过滤空串
      - None/缺失  → []

    :return: {"total": int, "items": list[dict], "errors": list[str]}
    """
    import yaml

    # 决策 1：函数内 import，便于测试 monkeypatch rg.CASES_ROOT
    from app.workflows.review_graph import CASES_ROOT

    errors: list[str] = []
    items: list[dict[str, Any]] = []

    if not CASES_ROOT.exists():
        errors.append(f"CASES_ROOT 不存在: {CASES_ROOT}")
        return {"total": 0, "items": [], "errors": errors}

    def _normalize_tags(raw: Any) -> list[str]:
        """决策 5：tags 统一为 list[str]。"""
        if raw is None:
            return []
        if isinstance(raw, str):
            return [t.strip() for t in raw.split(",") if t.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(t).strip() for t in raw if str(t).strip()]
        # 其他类型保守转 str 包一层
        return [str(raw)]

    # 遍历 CASES_ROOT 下的直接子目录
    for case_dir in sorted(CASES_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue

        info_file = case_dir / "info.yaml"
        if not info_file.exists():
            # 决策 4：纯空目录（无 info.yaml）静默跳过
            continue

        # 解析 YAML
        try:
            raw = yaml.safe_load(info_file.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(
                f"{case_dir.name}: info.yaml 解析失败: {type(exc).__name__}: {exc}"
            )
            continue

        if not isinstance(raw, dict):
            errors.append(
                f"{case_dir.name}: info.yaml 顶层不是 dict，实际 {type(raw).__name__}"
            )
            continue

        # ===== 决策 2：case_id 必填 =====
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(
                f"{case_dir.name}: case_id 缺失或非法（需非空字符串），"
                f"实际 {case_id!r}"
            )
            continue
        case_id = case_id.strip()

        # ===== 决策 2 补充：case_class 先类型、再非空、再合法值 =====
        case_class = raw.get("case_class")
        if not isinstance(case_class, str):
            errors.append(
                f"{case_dir.name}: case_class 类型非法（需 str），"
                f"实际 {type(case_class).__name__}={case_class!r}"
            )
            continue
        case_class = case_class.strip()
        if not case_class:
            errors.append(f"{case_dir.name}: case_class 为空字符串")
            continue
        if case_class not in ("A", "B"):
            errors.append(
                f"{case_dir.name}: case_class={case_class!r} 非 A/B，跳过"
            )
            continue

        items.append({
            "case_id": case_id,
            "case_class": case_class,
            "title": raw.get("title"),
            "description": raw.get("description"),
            "calibration_status": raw.get("calibration_status"),
            "tags": _normalize_tags(raw.get("tags")),
        })

    # ===== 决策 3：A 类在前、B 类在后，组内 case_id 字典序 =====
    items.sort(key=lambda it: (0 if it["case_class"] == "A" else 1, it["case_id"]))

    # ===== total == len(items)（恒等） =====
    return {
        "total": len(items),
        "items": items,
        "errors": errors,
    }

# ============================================================
# update_expert_decision：专家判定 PATCH（S2-B）
# ------------------------------------------------------------
# 设计理由：
#   1. 与 get_task_defects / submit_task_feedback 一致，用 task_id + defect_id
#      业务标识，不暴露 DB 外键。
#   2. 双校验：task 存在 + defect 属于该 task，防越权。
#   3. 只更新 expert_decision 3 字段，不改 review_status（AI 质量状态）。
# ============================================================
def update_expert_decision(
    task_id: str,
    defect_id: str,
    expert_decision: str,
    updated_by: int | None = None,
) -> dict[str, Any]:
    """
    更新单条缺陷的专家最终判定（S2-B）。

    校验：
      - task 存在（按 task_id 查 review_result）
      - defect 存在且属于该 task（review_defect.review_result_id == review_result.id）
      - expert_decision 枚举合法

    返回：{defect_id, expert_decision, expert_decision_updated_at, expert_decision_by}

    异常：
      - task 不存在 → FeedbackNotFoundError
      - defect 不存在或不属于该 task → FeedbackNotFoundError
      - expert_decision 非法 → ValueError
    """
    #from datetime import datetime, timezone

    _ALLOWED = {"pending", "confirmed", "rejected", "needs_verification"}
    if expert_decision not in _ALLOWED:
        raise BusinessValueError(f"非法 expert_decision: {expert_decision!r}")

    with get_db_session() as db:
        # 1. 查 review_result
        rr = (
            db.query(ReviewResult)
            .filter(ReviewResult.task_id == task_id)
            .order_by(ReviewResult.id.desc())
            .first()
        )
        if rr is None:
            raise FeedbackNotFoundError(f"review_result task_id={task_id} 不存在")

        # 2. 查 review_defect，校验归属
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

        # 3. 更新 3 字段
        rd.expert_decision = expert_decision
        rd.expert_decision_updated_at = datetime.now(timezone.utc)
        rd.expert_decision_by = updated_by
        db.commit()
        db.refresh(rd)

        return {
            "defect_id": rd.defect_id,
            "expert_decision": rd.expert_decision,
            "expert_decision_updated_at": rd.expert_decision_updated_at,
            "expert_decision_by": rd.expert_decision_by,
        }

# ============================================================
# get_task_summary_metrics：任务详情指标聚合（S2-C）
# ------------------------------------------------------------
# 设计理由：
#   1. 实时聚合，不写 review_result 静态列（S2-C 冻结：反馈异步产生，落库会过期）
#   2. 指标口径严格对齐 metrics_definition.md v3.2
#   3. sample_info 供前端显示「N/A · 样本不足」时拿具体样本数 / 门槛
# ============================================================
def get_task_summary_metrics(task_id: str) -> dict[str, Any] | None:
    """
    聚合任务详情指标（S2-C）。

    返回：
      {
        "evidence_coverage": float | None,
        "ai_new_effective_rate": float | None,
        "expert_adoption_rate": float | None,
        "review_status_distribution": {
          "AI_CONFIRMED": int,
          "NEED_EXPERT_REVIEW": int,
          "LOW_CONFIDENCE": int,
        },
        "sample_info": {
          "ear_sample_size": int,
          "ear_sample_threshold": 20,
          "ai_ner_sample_size": int,
          "ai_ner_sample_threshold": 10,
        },
      }

    task 不存在 → None。

    【数据来源】
      - evidence_coverage      : review_result.evidence_coverage
      - review_status_distribution : review_defect.review_status 计数
      - expert_adoption_rate   : feedback_item 聚合（compute_real_ear）
      - ai_new_effective_rate  : review_defect 聚合（compute_real_ai_ner）
      - sample_info            : compute_sample_info
    """
    # 函数内 import 避免循环依赖（benchmark_service 已 import task_service）
    from app.services.benchmark_service import (
        compute_real_ai_ner,
        compute_real_ear,
        compute_sample_info,
    )
    from app.persistence.models import FeedbackItem

    with get_db_session() as db:
        rr = (
            db.query(ReviewResult)
            .filter(ReviewResult.task_id == task_id)
            .order_by(ReviewResult.id.desc())
            .first()
        )
        if rr is None:
            return None

        # 1. 缺陷列表（S2-C 的 review_defect 已含 expert_decision 3 字段）
        defects = (
            db.query(ReviewDefect)
            .filter(ReviewDefect.review_result_id == rr.id)
            .all()
        )
        defect_dicts = [
            {
                "defect_id": d.defect_id,
                "review_status": d.review_status,
                "origin": d.origin,
                "expert_decision": d.expert_decision,
                "feedback_type": d.feedback_type,
            }
            for d in defects
        ]

        # 2. 三态分布
        distribution = {
            "AI_CONFIRMED": 0,
            "NEED_EXPERT_REVIEW": 0,
            "LOW_CONFIDENCE": 0,
        }
        for d in defects:
            if d.review_status in distribution:
                distribution[d.review_status] += 1

        # 3. 反馈列表（用于 EAR）
        feedbacks = (
            db.query(FeedbackItem)
            .filter(FeedbackItem.review_result_id == rr.id)
            .all()
        )
        # ===== [S2-C 修正] 把 review_defect_id 翻译成业务 defect_id =====
        # compute_real_ear / compute_sample_info 期望业务 ID（str），
        # 不是 DB 外键 id（int）。
        defect_id_map = {d.id: d.defect_id for d in defects}
        feedback_dicts = [
            {
                "id": fb.id,
                "defect_id": (
                    defect_id_map.get(fb.review_defect_id)
                    if fb.review_defect_id is not None
                    else None
                ),
                "feedback_type": fb.feedback_type,
                "suggestion_adopted": fb.suggestion_adopted,
                "comment": fb.comment,
                "created_at": fb.created_at.isoformat() if fb.created_at else None,
            }
            for fb in feedbacks
        ]
        # ===== [S2-C 修正 END] =====

        # 4. 真实指标
        expert_adoption_rate = compute_real_ear(feedback_dicts, defect_dicts)
        ai_new_effective_rate = compute_real_ai_ner(defect_dicts)
        sample_info = compute_sample_info(feedback_dicts, defect_dicts)

        return {
            "evidence_coverage": rr.evidence_coverage,
            "ai_new_effective_rate": ai_new_effective_rate,
            "expert_adoption_rate": expert_adoption_rate,
            "review_status_distribution": distribution,
            "sample_info": sample_info,
        }

__all__ = [
    "create_task",
    "get_task",
    "list_tasks",
    "list_tasks_filtered",   # ← G2 新增（补进 __all__）
    "list_cases",             # ← G3 新增
    "get_task_defects",
    "submit_task_feedback",
    "update_expert_decision",  # ← S2-B 新增
    "get_task_summary_metrics",  # ← S2-C 新增
]

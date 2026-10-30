# backend/app/services/report_service.py
"""报告导出服务（S2-H 新增）。

职责：
  1. 查询 task_id 对应的 review_result 和 review_defect
  2. 渲染成结构化 Markdown 字符串
  3. 供 API 层返回 PlainText 下载
"""
from __future__ import annotations

from typing import Dict, Any

from app.persistence.db import get_db_session
from app.persistence.models import ReviewResult, ReviewDefect


def generate_markdown_report(task_id: str) -> str:
    """
    生成任务的 Markdown 评审报告。

    :param task_id: 任务标识
    :return: Markdown 字符串
    :raises ValueError: 任务不存在 / 任务执行失败
    """
    with get_db_session() as db:
        rr = (
            db.query(ReviewResult)
            .filter(ReviewResult.task_id == task_id)
            .order_by(ReviewResult.id.desc())
            .first()
        )
        if rr is None:
            raise ValueError(f"任务不存在或未落库: {task_id}")

        # 边界处理：不导出 failed 状态的任务
        if rr.task_status == "failed":
            raise ValueError(f"任务 {task_id} 执行失败，无法导出报告")

        defects = (
            db.query(ReviewDefect)
            .filter(ReviewDefect.review_result_id == rr.id)
            .order_by(ReviewDefect.id.asc())
            .all()
        )

        # 基础信息
        lines = []
        lines.append(f"# 原理图评审报告\n")
        lines.append(f"- **Task ID**: `{rr.task_id}`")
        lines.append(f"- **评审状态**: {rr.task_status}")
        lines.append(f"- **缺陷总数**: {len(defects)}")
        lines.append(f"- **AI 质量分布**: AI 已确认 {rr.ai_confirmed_count} 条 | 需专家复核 {rr.need_expert_review_count} 条 | 低置信 {rr.low_confidence_count} 条")
        lines.append(f"- **生成时间**: {rr.created_at.isoformat() if rr.created_at else '未知'}\n")
        lines.append("---\n")
        
        lines.append("## 缺陷明细\n")

        if not defects:
            lines.append("> 本次评审未发现任何缺陷。\n")
            return "\n".join(lines)

        # 遍历缺陷
        for idx, d in enumerate(defects, 1):
            origin_text = {
                "rule": "规则发现",
                "ai_discovered": "AI 发现",
                "hybrid": "混合发现",
            }.get(d.origin, d.origin)

            decision_text = {
                "pending": "待判定",
                "confirmed": "专家确认",
                "rejected": "专家驳回",
                "needs_verification": "需进一步核验",
            }.get(d.expert_decision, d.expert_decision)

            lines.append(f"### {idx}. {d.defect_id} (来源: {origin_text})\n")
            lines.append(f"- **分类**: {d.category}")
            lines.append(f"- **元件**: {d.component or '-'}")
            lines.append(f"- **网络**: {d.net or '-'}")
            lines.append(f"- **风险等级**: {d.risk.upper()}")
            lines.append(f"- **置信度**: {int(d.confidence * 100)}%")
            if d.rule_id:
                lines.append(f"- **规则 ID**: `{d.rule_id}`")
            lines.append(f"- **专家判定**: **{decision_text}**\n")

            lines.append("**根因分析**")
            lines.append(f"> {d.root_cause.strip() if d.root_cause else '无'}\n")

            lines.append("**修改建议**")
            lines.append(f"> {d.suggestion.strip() if d.suggestion else '无'}\n")

            lines.append("**证据链**")
            evidence_list = d.evidence or []
            if not evidence_list:
                lines.append("- 无证据\n")
            else:
                for ev in evidence_list:
                    ev_type = ev.get("type", "unknown")
                    if ev_type == "rag_ref":
                        src = ev.get("source", "未知来源")
                        sec = ev.get("section", "")
                        reason = ev.get("reason", "")
                        lines.append(f"- [{ev_type.upper()}] {src} {sec} - {reason}")
                    elif ev_type == "ir_ref":
                        detail = ev.get("detail", "")
                        lines.append(f"- [{ev_type.upper()}] {detail}")
                    else:
                        lines.append(f"- [{ev_type.upper()}] {str(ev)}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)
# backend/app/services/rule_evolution_service.py
"""
Phase‑I RuleEvolutionService
Sprint‑0边界：
1. 仅生成proposed状态rule_candidate YAML草稿；草稿**禁止直接合并正式rules库**；
2. 草稿必须人工编辑，候选规则benchmark校验逻辑放到Sprint‑1；
3. 提供CLI：list / export‑yaml / export‑knowledge‑gap

⚠️重要：
1. FeedbackItem没有task_id/defect_id字段！
   task_id(业务字符串) → ReviewResult.task_id，通过feedback_item.review_result_id外键join获取
   defect_id(业务字符串) → ReviewDefect.defect_id，通过feedback_item.review_defect_id外键join获取
2. rule_candidates DB表 **不存在task_id字段！禁止传入task_id参数给RuleCandidate ORM构造**
   task信息查询走 from_feedback_id → feedback_item → review_result JOIN，不冗余存储

v1.3 变更（None 值兜底）：
- ★ 修复：hint_payload.get(k, default) 在 key 存在但值为 None 时返回 None，
  导致 enumerate(None) 报 TypeError / ORM evidence_refs=None 违反 NOT NULL。
- ★ 统一用 hint.get(k) or default 兜底：key 不存在或值为 None 都走 default。
- ★ 影响函数：_build_proposed_rule_yaml / generate_candidate_from_feedback
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.persistence.models import RuleCandidate, FeedbackItem, ReviewResult, ReviewDefect


def _make_candidate_id() -> str:
    """生成候选规则ID RC‑YYYYMMDD‑001"""
    ts = datetime.now().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"RC‑{ts}‑{suffix}"


def _build_proposed_rule_yaml(hint_payload: Optional[Dict[str, Any]]) -> str:
    """
    根据feedback生成候选规则YAML草稿模板
    Sprint‑0仅生成草稿，字段填充基础占位，专家后续人工修正
    ✅修复：把 >‑(全角破折号) 替换为标准YAML >‑ → >- (ASCII半角减号)

    ★ v1.3：所有 hint.get(k, default) 改为 hint.get(k) or default，
           兼容「key 存在但值为 None」的情况（RuleCandidate 可选字段默认 None）
    """
    hint = hint_payload or {}
    # --- 原代码（保留，已弃用） ---
    # rule_id = hint.get("rule_id", "POWER_RC_AUTO_001")
    # rule_name = hint.get("rule_name", "自动生成候选规则(草稿，需人工编辑)")
    # category = hint.get("category", "power")
    # severity = hint.get("severity", "high")
    # applicable_condition = hint.get("applicable_condition", {"scope": "schematic"})
    # check_logic = hint.get("check_logic", {"function": "builtin.power_checks.check_decoupling_cap"})
    # rule_basis = hint.get("rule_basis", "来自专家反馈自动生成草稿，请补充datasheet/reference依据")
    # suggestion = hint.get("suggestion", "请人工完善修改建议")
    # ------------------------------------------------------------
    # ★ v1.3：用 or 兜底 None
    rule_id = hint.get("rule_id") or "POWER_RC_AUTO_001"
    rule_name = hint.get("rule_name") or "自动生成候选规则(草稿，需人工编辑)"
    category = hint.get("category") or "power"
    severity = hint.get("severity") or "high"
    applicable_condition = hint.get("applicable_condition") or {"scope": "schematic"}
    check_logic = hint.get("check_logic") or {"function": "builtin.power_checks.check_decoupling_cap"}
    rule_basis = hint.get("rule_basis") or "来自专家反馈自动生成草稿，请补充datasheet/reference依据"
    suggestion = hint.get("suggestion") or "请人工完善修改建议"

    yaml_text = f"""# WARNING: AUTO-GENERATED DRAFT RULE CANDIDATE
# Sprint-0仅草稿,**禁止直接合并正式rules库**,必须人工编辑后Sprint-1跑benchmark校验
rule_id: {rule_id}
rule_name: {rule_name}
version: "0.1-proposed"
category: {category}
severity: {severity}
applicable_condition: {json.dumps(applicable_condition, ensure_ascii=False)}
check_logic: {json.dumps(check_logic, ensure_ascii=False)}
rule_basis: >-
  {rule_basis}
suggestion: >-
  {suggestion}
enabled: true
"""
    return yaml_text


class RuleEvolutionService:
    def generate_candidate_from_feedback(
        self,
        db: Session,
        feedback_id: int,
        case_id: Optional[str] = None,
        hint_payload: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        根据false_negative / new_rule_candidate反馈生成rule_candidate草稿记录
        return candidate_id

        ★ v1.3：所有 hint_payload.get(k, default) 改为 (hint_payload or {}).get(k) or default，
               兼容「key 存在但值为 None」的情况，避免 enumerate(None) / ORM evidence_refs=None
        """
        fb: Optional[FeedbackItem] = db.query(FeedbackItem).filter(FeedbackItem.id == feedback_id).first()
        if fb is None:
            raise ValueError(f"feedback_item id={feedback_id} not found")

        candidate_id = _make_candidate_id()
        proposed_yaml = _build_proposed_rule_yaml(hint_payload)

        # ★ v1.3：统一兜底 None
        _hint = hint_payload or {}

        # ✅ NEW: DeepSeek‑3 evidence_refs V1.2 §3.2 三要素结构化校验
        # V1.2 §3.2：每条evidence必须具备 source / section / reason 三要素；最小结构化校验；只校验key存在，不校验内容语义
        # --- 原代码（保留，已弃用） ---
        # evidence_refs = hint_payload.get("evidence_refs", []) if hint_payload else []
        evidence_refs = _hint.get("evidence_refs") or []
        for idx, e in enumerate(evidence_refs):
            assert "source" in e, f"evidence_refs[{idx}] 缺失必填字段 source"
            assert "section" in e, f"evidence_refs[{idx}] 缺失必填字段 section"
            assert "reason" in e, f"evidence_refs[{idx}] 缺失必填字段 reason"

        # 关键修复：彻底移除 task_id 参数！rule_candidates表没有task_id列
        # --- 原代码（保留，已弃用） ---
        # rc = RuleCandidate(
        #     candidate_id=candidate_id,
        #     from_feedback_id=feedback_id,
        #     case_id=case_id,
        #     title=hint_payload.get("title", "专家反馈生成候选规则草稿") if hint_payload else "专家反馈生成候选规则草稿",
        #     description=hint_payload.get("description", "自动草稿，需要专家人工完善字段、依据、check_logic")
        #     if hint_payload else "自动草稿，需要专家人工完善字段、依据、check_logic",
        #     severity=hint_payload.get("severity", "high") if hint_payload else "high",
        #     evidence_refs=hint_payload.get("evidence_refs", []) if hint_payload else [],
        #     proposed_yaml=proposed_yaml,
        #     status="proposed"
        # )
        # ------------------------------------------------------------
        # ★ v1.3：所有 .get 加 or 兜底 None
        rc = RuleCandidate(
            candidate_id=candidate_id,
            from_feedback_id=feedback_id,
            case_id=case_id,
            title=_hint.get("title") or "专家反馈生成候选规则草稿",
            description=_hint.get("description") or "自动草稿，需要专家人工完善字段、依据、check_logic",
            severity=_hint.get("severity") or "high",
            evidence_refs=_hint.get("evidence_refs") or [],
            proposed_yaml=proposed_yaml,
            status="proposed"
        )
        db.add(rc)
        db.flush()
        return candidate_id

    def list_candidates(self, db: Session, status: str = "proposed") -> List[Dict[str, Any]]:
        rows = db.query(RuleCandidate).filter(RuleCandidate.status == status).order_by(RuleCandidate.created_at.desc()).all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "candidate_id": r.candidate_id,
                "from_feedback_id": r.from_feedback_id,
                "case_id": r.case_id,
                "title": r.title,
                "severity": r.severity,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        return out

    def export_candidate_yaml(self, db: Session, candidate_id: str, out_path: str | Path) -> None:
        rc: Optional[RuleCandidate] = db.query(RuleCandidate).filter(RuleCandidate.candidate_id == candidate_id).first()
        if rc is None:
            raise ValueError(f"rule_candidate candidate_id={candidate_id} not found")
        Path(out_path).write_text(rc.proposed_yaml, encoding="utf‑8")

    def export_knowledge_gap_backlog(self, db: Session, out_path: str | Path) -> None:
        """导出feedback_item中feedback_type=knowledge_gap全部记录作为知识缺口backlog"""
        # JOIN查询拿到ReviewResult、ReviewDefect获取业务task_id/defect_id
        stmt = (
            select(FeedbackItem, ReviewResult, ReviewDefect)
            .outerjoin(ReviewResult, FeedbackItem.review_result_id == ReviewResult.id)
            .outerjoin(ReviewDefect, FeedbackItem.review_defect_id == ReviewDefect.id)
            .where(FeedbackItem.feedback_type == "knowledge_gap")
        )
        rows = db.execute(stmt).all()
        payload = []
        for fb, rr, rd in rows:
            payload.append({
                "feedback_id": fb.id,
                "task_id": rr.task_id if rr else None,
                "defect_id": rd.defect_id if rd else None,
                "expert_suggestion": fb.expert_suggestion,
                "created_at": fb.created_at.isoformat() if fb.created_at else None
            })
        Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf‑8")


if __name__ == "__main__":
    """CLI入口，供demo脚本调用"""
    import argparse
    from app.persistence.db import get_db_session
    svc = RuleEvolutionService()
    parser = argparse.ArgumentParser(description="RuleEvolutionService CLI Sprint‑0")
    parser.add_argument("--list", action="store_true", help="list rule_candidate，默认status=proposed")
    parser.add_argument("--status", default="proposed")
    parser.add_argument("--output-json", help="list结果写入指定json文件（容器内路径，避免docker stdout管道乱码）")
    parser.add_argument("--export-yaml", help="export candidate yaml, require candidate-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--export-knowledge-gap", help="export knowledge_gap backlog json output path")
    args = parser.parse_args()

    with get_db_session() as db:
        if args.list:
            data = svc.list_candidates(db, status=args.status)
            if args.output_json:
                # ✅容器内部直接写文件，不打印stdout！规避Windows docker exec管道编码损坏
                import json
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"✅ list candidate写入文件: {args.output_json}")
            else:
                print(json.dumps(data, indent=2, ensure_ascii=False))
        elif args.export_yaml:
            if not args.candidate_id:
                raise RuntimeError("--export-yaml必须传--candidate-id")
            svc.export_candidate_yaml(db, args.candidate_id, args.export_yaml)
            print(f"✅ export candidate yaml → {args.export_yaml}")
        elif args.export_knowledge_gap:
            svc.export_knowledge_gap_backlog(db, args.export_knowledge_gap)
            print(f"✅ export knowledge-gap backlog → {args.export_knowledge_gap}")
        else:
            parser.print_help()
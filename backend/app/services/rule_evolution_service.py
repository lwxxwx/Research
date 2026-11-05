# backend/app/services/rule_evolution_service.py
"""
Phase-I RuleEvolutionService
Sprint-0边界：
1. 仅生成proposed状态rule_candidate YAML草稿；草稿**禁止直接合并正式rules库**；
2. 草稿必须人工编辑，候选规则benchmark校验逻辑放到Sprint‑1；
3. 提供CLI：list / export-yaml / export-knowledge-gap

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
import os
import shutil
import tempfile
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.models import (
    FeedbackItem,
    ReviewDefect,
    ReviewResult,
    RuleCandidate,
    RuleDefinition,      # F05 新增
)
from app.core.exceptions import BusinessValueError
from app.rules.engine import load_rule_definitions  # F05 新增

# ============================================================
# [F05] 正式规则发布：异常类型 + 辅助函数
# ============================================================
class RulePublishError(Exception):
    """发布失败（技术异常，方案 A 下不直接 raise 给 API，由方法内部返回）。"""
    pass


class RulePublishConflictError(RulePublishError):
    """候选状态不允许发布（未批准 / 已发布）→ API 409。"""
    pass


class RulePublishValidationError(RulePublishError):
    """YAML 校验失败（语法 / 字段 / 规则 ID 冲突）→ API 422。"""
    pass


def _rollback_publish(
    db: Session,
    rc: RuleCandidate,
    target_path: Path,
) -> str:
    """发布失败补偿：删 target_path + 标记 publish_failed。

    返回错误描述（写入 publish_error）。
    注意：本函数内部 commit，确保 publish_failed 状态落库。
    """
    error_msg = "发布失败"

    # 文件补偿：删除已替换的 target_path（F05 不产生备份，直接删）
    try:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
    except Exception as exc:
        error_msg = f"发布失败（文件清理异常: {exc}）"

    # DB 状态补偿
    try:
        rc.publish_status = "publish_failed"
        rc.publish_error = error_msg
        db.commit()
    except Exception:
        db.rollback()

    return error_msg

def _make_candidate_id() -> str:
    """生成候选规则ID RC‑YYYYMMDD‑XXXXXX（6 位大写 hex）"""
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
    check_logic = hint.get("check_logic") or {"function": "builtin.circuit_checks.check_decoupling"}
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
    """
    ⚠️ 待解决问题（Sprint 1，不是 Sprint 0 阻塞项）：

    delete_candidate 方法缺失
      - 当前无「物理删除 RuleCandidate」的对外方法
      - 若未来直接 db.delete(rc)，会残留 FeedbackItem.rule_candidate_ref
        悬空字符串（该列无 DB 外键约束）
      - 建议（Sprint 1）新增 delete_candidate(db, candidate_id, commit=True)：
          1) 查候选，不存在抛 RuleCandidateNotFoundError
          2) db.query(FeedbackItem).filter(
                 FeedbackItem.rule_candidate_ref == candidate_id
             ).update({FeedbackItem.rule_candidate_ref: None},
                      synchronize_session="fetch")
          3) db.delete(rc)
          4) commit 由参数控制（与 submit_feedback 保持一致）

    v1.4 新增（Sprint 1 阶段 E4）：
      - run_benchmark_for_candidate：候选规则临时 Benchmark 骨架
        （内存加载，不写磁盘，不落库；完整实现留阶段 F）
    """
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
            # ★ 用 raise 替代 assert，避免 python -O 跳过校验
            # --- 原代码（保留，已弃用） ---
            # assert "source" in e, f"evidence_refs[{idx}] 缺失必填字段 source"
            # assert "section" in e, f"evidence_refs[{idx}] 缺失必填字段 section"
            # assert "reason" in e, f"evidence_refs[{idx}] 缺失必填字段 reason"
            if "source" not in e:
                raise ValueError(f"evidence_refs[{idx}] 缺失必填字段 source")
            if "section" not in e:
                raise ValueError(f"evidence_refs[{idx}] 缺失必填字段 section")
            if "reason" not in e:
                raise ValueError(f"evidence_refs[{idx}] 缺失必填字段 reason")
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

    # ============================================================
    # S2-G：Web API 所需的 3 个方法
    # ============================================================
    def get_all_candidates(
        self, db: Session, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查询候选规则列表（S2-G）。"""
        query = db.query(RuleCandidate)
        if status:
            query = query.filter(RuleCandidate.status == status)
        rows = query.order_by(RuleCandidate.created_at.desc()).all()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "candidate_id": r.candidate_id,
                "from_feedback_id": r.from_feedback_id,
                "case_id": r.case_id,
                "title": r.title,
                "description": r.description,
                "severity": r.severity,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        return out

    def get_candidate(
        self, db: Session, candidate_id: str
    ) -> Optional[Dict[str, Any]]:
        """查询单条候选规则详情（S2-G + F05）。"""
        rc: Optional[RuleCandidate] = (
            db.query(RuleCandidate)
            .filter(RuleCandidate.candidate_id == candidate_id)
            .first()
        )
        if rc is None:
            return None

        return {
            "candidate_id": rc.candidate_id,
            "from_feedback_id": rc.from_feedback_id,
            "case_id": rc.case_id,
            "title": rc.title,
            "description": rc.description,
            "severity": rc.severity,
            "status": rc.status,
            "proposed_yaml": rc.proposed_yaml,
            "evidence_refs": rc.evidence_refs,
            "benchmark_result": rc.benchmark_result,
            "benchmarked_at": (
                rc.benchmarked_at.isoformat() if rc.benchmarked_at else None
            ),
            "rejected_reason": rc.rejected_reason,
            "created_at": rc.created_at.isoformat() if rc.created_at else None,
            "updated_at": rc.updated_at.isoformat() if rc.updated_at else None,
            # ===== [F05] 发布状态字段 =====
            "publish_status": rc.publish_status,
            "published_at": (
                rc.published_at.isoformat() if rc.published_at else None
            ),
            "published_version": rc.published_version,
            "publish_error": rc.publish_error,
            # ===== [F05 END] =====
        }

    def update_candidate_status(
        self,
        db: Session,
        candidate_id: str,
        status: str,
        rejected_reason: Optional[str] = None,
        reviewed_by: int | None = None,   # ← 加这行
    ) -> Dict[str, Any]:
        """更新候选规则状态（S2-G，approve / reject）。"""
        _ALLOWED = {"accepted", "rejected"}
        if status not in _ALLOWED:
            raise BusinessValueError(
                f"status 必须为 'accepted' / 'rejected'，实际: {status!r}"
            )

        rc: Optional[RuleCandidate] = (
            db.query(RuleCandidate)
            .filter(RuleCandidate.candidate_id == candidate_id)
            .first()
        )
        if rc is None:
            raise ValueError(f"rule_candidate candidate_id={candidate_id} not found")

        rc.status = status
        if status == "rejected":
            rc.rejected_reason = rejected_reason
        # ===== [S3-G3C] 审计字段：任务创建者 users.id =====
        rc.reviewed_by = reviewed_by
        # ===== [S3-G3C END] =====
        db.commit()
        db.refresh(rc)

        return {
            "candidate_id": rc.candidate_id,
            "status": rc.status,
            "rejected_reason": rc.rejected_reason,
            "reviewed_by": rc.reviewed_by,
            "updated_at": rc.updated_at.isoformat() if rc.updated_at else None,
        }    

    def export_candidate_yaml(self, db: Session, candidate_id: str, out_path: str | Path) -> None:
        rc: Optional[RuleCandidate] = db.query(RuleCandidate).filter(RuleCandidate.candidate_id == candidate_id).first()
        if rc is None:
            raise ValueError(f"rule_candidate candidate_id={candidate_id} not found")
        p = Path(out_path)
        # 自动创建父文件夹
        p.parent.mkdir(parents=True, exist_ok=True)
        # 可选：禁止覆盖已有文件
        if p.exists():
            raise FileExistsError(f"File {out_path} already exists, refuse overwrite.")
        # 写入文件
        p.write_text(rc.proposed_yaml, encoding="utf-8")

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
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"File {out_path} already exists, refuse overwrite.")
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ============================================================
    # run_benchmark_for_candidate：候选规则临时 Benchmark（Sprint 1 阶段 E4）
    # ------------------------------------------------------------
    # 设计理由：
    #   1. 只做骨架：内存加载候选 YAML → 对比基线 → 返回指标变化。
    #      不落库、不写文件、不生成 CSV（完整实现留阶段 F）。
    #   2. 候选规则只在内存中追加，不写入 data/rules/，
    #      不污染正式规则目录（Sprint 1 §10.3）。
    #   3. per_case 字段保留，便于定位"候选规则在哪个 Case 上引入 FP"。
    #   4. precision / recall 分母为 0 时返回 None，与 benchmark_service 一致，
    #      避免 0.0 被误读为"精度为零"。
    #   5. 风险字段映射：RuleResult.severity 对应 expected_review.json 的 risk，
    #      沿用 benchmark_service.match_defect 的约定（Sprint 0 冻结）。
    # ============================================================
    def run_benchmark_for_candidate(
        self,
        candidate_yaml: str,
        case_ids: List[str],
        cases_root: str | Path = "/data/cases",
        rules_root: str | Path = "/data/rules",
    ) -> Dict[str, Any]:
        """
        对候选规则做临时 Benchmark：对比"磁盘基线"与"磁盘 + 候选"的指标变化。

        参数：
          candidate_yaml : 候选规则 YAML 字符串（来自 proposed_yaml）
          case_ids       : 要跑的 Case 列表（如 ["case001", "case_b01_..."]）
          cases_root     : Case 根目录（默认 /data/cases）
          rules_root     : 规则根目录（默认 /data/rules）

        返回：
          {
            "candidate_rule_id": str,
            "case_ids": list[str],
            "baseline":  {"tp": int, "fp": int, "fn": int,
                          "precision": float | None, "recall": float | None},
            "candidate": {"tp": int, "fp": int, "fn": int,
                          "precision": float | None, "recall": float | None},
            "delta":     {"tp": int, "fp": int, "fn": int,
                          "precision": float | None, "recall": float | None},
            "per_case": [
                {"case_id": str,
                 "baseline":  {"tp": int, "fp": int, "fn": int},
                 "candidate": {"tp": int, "fp": int, "fn": int}},
                ...
            ],
          }

        异常：
          - candidate_yaml 解析失败 → ValueError
          - case_id 目录不存在 → ValueError
        """
        from app.ir.serializer import load_ir
        from app.rules.engine import (
            RuleDefinition,
            execute_defect_rules,
            execute_defect_rules_with_extra,
        )
        from app.services.benchmark_service import match_defect

        # 1. 解析候选规则
        try:
            raw = yaml.safe_load(candidate_yaml)
            candidate_rule = RuleDefinition(**raw)
        except Exception as exc:
            raise ValueError(
                f"候选规则 YAML 解析失败: {type(exc).__name__}: {exc}"
            ) from exc

        cases_root_p = Path(cases_root)
        rules_root_p = Path(rules_root)

        # 2. 逐 case 跑基线与候选
        per_case: List[Dict[str, Any]] = []
        total_base = {"tp": 0, "fp": 0, "fn": 0}
        total_cand = {"tp": 0, "fp": 0, "fn": 0}

        for case_id in case_ids:
            case_dir = cases_root_p / case_id
            ir_path = case_dir / "schematic_ir.json"
            gt_path = case_dir / "expected_review.json"
            if not ir_path.exists() or not gt_path.exists():
                raise ValueError(
                    f"Case {case_id} 缺少 schematic_ir.json 或 expected_review.json"
                )

            ir = load_ir(ir_path)
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_json = json.load(f)
            gt_defects = gt_json.get("defects", [])

            # 基线：磁盘规则
            base_hits = execute_defect_rules(ir, rules_root_p)
            base_pred = [r.model_dump() for r in base_hits]
            base_tp, base_fp, base_fn = _match_tp_fp_fn(
                base_pred, gt_defects, match_defect
            )

            # 候选：磁盘 + 内存候选
            cand_hits = execute_defect_rules_with_extra(
                ir, rules_root_p, [candidate_rule]
            )
            cand_pred = [r.model_dump() for r in cand_hits]
            cand_tp, cand_fp, cand_fn = _match_tp_fp_fn(
                cand_pred, gt_defects, match_defect
            )

            per_case.append({
                "case_id": case_id,
                "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn},
                "candidate": {"tp": cand_tp, "fp": cand_fp, "fn": cand_fn},
            })
            total_base["tp"] += base_tp
            total_base["fp"] += base_fp
            total_base["fn"] += base_fn
            total_cand["tp"] += cand_tp
            total_cand["fp"] += cand_fp
            total_cand["fn"] += cand_fn

        # 3. 组装结果
        baseline = _with_pr(total_base)
        candidate = _with_pr(total_cand)
        delta = {
            "tp": candidate["tp"] - baseline["tp"],
            "fp": candidate["fp"] - baseline["fp"],
            "fn": candidate["fn"] - baseline["fn"],
            "precision": _safe_delta(baseline["precision"], candidate["precision"]),
            "recall": _safe_delta(baseline["recall"], candidate["recall"]),
        }

        return {
            "candidate_rule_id": candidate_rule.rule_id,
            "case_ids": list(case_ids),
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
            "per_case": per_case,
        }

    # ============================================================
    # [F05] 正式规则发布
    # ============================================================
    def publish_rule_candidate(
        self,
        db: Session,
        candidate_id: str,
        idempotency_key: str,
        rules_dir: str | Path = "/data/rules",
    ) -> Dict[str, Any]:
        """发布候选到正式规则库。

        方案 A：失败时返回 200 + publish_status="publish_failed"，状态落库。
        只有业务冲突（未批准 / 已发布）raise RulePublishConflictError（→ 409），
        校验失败（YAML / 规则 ID）raise RulePublishValidationError（→ 422）。

        状态机：accepted → publishing → published | publish_failed
        顺序：临时文件校验 → 原子替换 → DB 同步 → 引擎验证 → 状态落 published
        关键：文件替换后、DB 提交前不做任何规则执行（不跑 Benchmark）
        """
        rules_dir = Path(rules_dir)
        rules_dir.mkdir(parents=True, exist_ok=True)

        # 1. 幂等检查
        existing = (
            db.query(RuleCandidate)
            .filter(RuleCandidate.publish_idempotency_key == idempotency_key)
            .first()
        )
        if existing is not None:
            return {
                "candidate_id": existing.candidate_id,
                "publish_status": existing.publish_status,
                "published_version": existing.published_version,
                "publish_error": existing.publish_error,
                "idempotent_replay": True,
                "benchmark_regression": None,
            }

        # 2. 候选校验（业务冲突 → 409）
        rc: Optional[RuleCandidate] = (
            db.query(RuleCandidate)
            .filter(RuleCandidate.candidate_id == candidate_id)
            .first()
        )
        if rc is None:
            raise RulePublishConflictError(f"候选不存在: {candidate_id}")
        if rc.status != "accepted":
            raise RulePublishConflictError(
                f"候选未批准，status={rc.status}，不允许发布"
            )
        if rc.publish_status == "published":
            raise RulePublishConflictError(
                f"候选已发布，published_version={rc.published_version}"
            )
        if not rc.proposed_yaml:
            raise RulePublishValidationError("候选缺少 proposed_yaml")

        # 3. YAML 解析 + 字段校验（校验失败 → 422）
        try:
            parsed = yaml.safe_load(rc.proposed_yaml)
        except yaml.YAMLError as exc:
            raise RulePublishValidationError(f"YAML 语法错误: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RulePublishValidationError("YAML 顶层必须是 dict")
        rule_id = parsed.get("rule_id")
        if not rule_id:
            raise RulePublishValidationError("YAML 缺少 rule_id")

        # 4. 规则 ID 唯一性（校验失败 → 422）
        target_path = rules_dir / f"{rule_id}.yaml"
        if target_path.exists():
            raise RulePublishValidationError(
                f"规则 ID 已存在: {rule_id}（{target_path}）"
            )

        # 5. 标记 publishing + 记录幂等 Key（先落状态，防重复）
        rc.publish_status = "publishing"
        rc.publish_error = None
        rc.publish_idempotency_key = idempotency_key
        db.commit()

        # 6. 临时文件 + 原子替换 + DB 同步 + 引擎验证
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".tmp_{candidate_id}_", suffix=".yaml", dir=str(rules_dir)
        )
        tmp_path = Path(tmp_path_str)
        #backup_path: Path | None = None
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(rc.proposed_yaml)

            # 验证临时文件可解析
            try:
                yaml.safe_load(tmp_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise RulePublishValidationError(
                    f"临时文件 YAML 校验失败: {exc}"
                ) from exc

            # 原子替换
            os.replace(tmp_path, target_path)

            # DB 同步：更新 / 插入 rule_definition
            version = parsed.get("version") or "v1.0"
            existing_rd = (
                db.query(RuleDefinition)
                .filter(RuleDefinition.rule_id == rule_id)
                .first()
            )
            if existing_rd is None:
                db.add(RuleDefinition(
                    rule_id=rule_id,
                    rule_name=parsed.get("rule_name") or rule_id,
                    rule_category=parsed.get("category") or "other",
                    rule_yaml=rc.proposed_yaml,
                    severity=parsed.get("severity") or "medium",
                    enabled=True,
                    version=version,
                ))
            else:
                existing_rd.rule_name = parsed.get("rule_name") or rule_id
                existing_rd.rule_category = parsed.get("category") or "other"
                existing_rd.rule_yaml = rc.proposed_yaml
                existing_rd.severity = parsed.get("severity") or "medium"
                existing_rd.enabled = True
                existing_rd.version = version

            # 引擎验证：调 load_rule_definitions 确认新文件可解析（不执行规则）
            loaded = load_rule_definitions(rules_dir)
            loaded_ids = {r.rule_id for r in loaded}
            if rule_id not in loaded_ids:
                raise RulePublishError(
                    f"引擎加载后未找到 {rule_id}，文件可能损坏"
                )

            # 状态落 published
            rc.publish_status = "published"
            rc.published_at = datetime.now(timezone.utc)
            rc.published_version = version
            rc.publish_error = None
            db.commit()

            return {
                "candidate_id": candidate_id,
                "publish_status": "published",
                "published_version": version,
                "publish_error": None,
                "idempotent_replay": False,
                "benchmark_regression": None,
            }

        except RulePublishValidationError as exc:
            # 发布中校验失败（临时文件 YAML 错）→ 补偿 + 返回 200 + publish_failed
            _rollback_publish(db, rc, target_path)
            return {
                "candidate_id": candidate_id,
                "publish_status": "publish_failed",
                "published_version": None,
                "publish_error": str(exc),
                "idempotent_replay": False,
                "benchmark_regression": None,
            } 
        except RulePublishError as exc:
            # 技术失败（引擎验证失败）→ 补偿 + 返回 200 + publish_failed
            _rollback_publish(db, rc, target_path)
            return {
                "candidate_id": candidate_id,
                "publish_status": "publish_failed",
                "published_version": None,
                "publish_error": str(exc),
                "idempotent_replay": False,
                "benchmark_regression": None,
            }
        except Exception as exc:
            # 未预期异常 → 补偿 + 返回 200 + publish_failed
            _rollback_publish(db, rc, target_path)
            return {
                "candidate_id": candidate_id,
                "publish_status": "publish_failed",
                "published_version": None,
                "publish_error": f"{type(exc).__name__}: {exc}",
                "idempotent_replay": False,
                "benchmark_regression": None,
            }
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

# ============================================================
# 辅助函数：TP/FP/FN 匹配与 Precision/Recall 计算
# ------------------------------------------------------------
# 独立成函数，便于单测；逻辑与 benchmark_service.run_single_case 一致，
# 但只算 TP/FP/FN，不算 EVC / 三态（候选 Benchmark 不需要）。
# ============================================================
def _match_tp_fp_fn(
    pred_defects: List[Dict[str, Any]],
    gt_defects: List[Dict[str, Any]],
    match_fn,
) -> tuple[int, int, int]:
    """
    贪心匹配 pred 与 gt，返回 (tp, fp, fn)。

    贪心策略与 benchmark_service.run_single_case 一致：
      - 外层遍历 GT，内层遍历未占用的 pred
      - 一个 GT 命中一个 pred 后立即 break
      - 未命中的 pred 计 FP，未命中的 GT 计 FN
    """
    tp_set_gt: set[int] = set()
    tp_set_pred: set[int] = set()
    for gt_idx, gt_d in enumerate(gt_defects):
        for pred_idx, pred_d in enumerate(pred_defects):
            if pred_idx in tp_set_pred:
                continue
            if match_fn(pred_d, gt_d):
                tp_set_gt.add(gt_idx)
                tp_set_pred.add(pred_idx)
                break
    tp = len(tp_set_gt)
    fp = len(pred_defects) - len(tp_set_pred)
    fn = len(gt_defects) - len(tp_set_gt)
    return tp, fp, fn


def _with_pr(counts: Dict[str, int]) -> Dict[str, Any]:
    """
    在 tp/fp/fn 基础上补 precision / recall。
    分母为 0 时返回 None（与 benchmark_service 一致，避免 0.0 误读）。
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def _safe_delta(base: float | None, cand: float | None) -> float | None:
    """两值都非 None 时返回差值，否则返回 None。"""
    if base is None or cand is None:
        return None
    return cand - base

if __name__ == "__main__":
    # CLI入口，供demo脚本调用
    import argparse
    import sys

    from app.persistence.db import get_db_session

    svc = RuleEvolutionService()
    parser = argparse.ArgumentParser(description="RuleEvolutionService CLI Sprint-0")

    # ★ 主操作互斥：--list / --export-yaml / --export-knowledge-gap 三选一
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--list", action="store_true", help="list rule_candidate，默认status=proposed")
    group.add_argument("--export-yaml", help="export candidate yaml, require --candidate-id", metavar="OUT_YAML_PATH")
    group.add_argument("--export-knowledge-gap", help="export knowledge_gap backlog json output path")

    # ★ 修饰参数（不属于互斥组）
    parser.add_argument("--status", default="proposed")
    parser.add_argument("--output-json", help="list结果写入指定json文件（容器内路径，避免docker stdout管道乱码）")
    parser.add_argument("--candidate-id", help="rule candidate id，配合--export-yaml使用")

    args = parser.parse_args()
    try:
        with get_db_session() as db:
            if args.list:
                data = svc.list_candidates(db, status=args.status)
                if args.output_json:
                    # ⚠️ V1.3 修改：删除重复 import（第 24 行已 import json）
                    # [原逻辑 - 保留注释，便于对照回滚]
                    # import json
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
    except Exception as e:
        print(f"❌ Error: {e}")
        # ★ sys.exit 而非 exit
        sys.exit(1)
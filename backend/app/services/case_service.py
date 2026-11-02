# backend/app/services/case_service.py
"""F07 Case 管理服务。

职责：
  - get_case_detail：Case 元数据 + 资产预览
  - run_case_benchmark：跑单 Case Benchmark，结果写 case_benchmark_result
  - list_case_benchmarks：历史结果查询
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.persistence.models import CaseBenchmarkResult
from app.services.benchmark_service import (
    STATUS_DONE,
    STATUS_FAILED,
    run_single_case,
    run_single_case_full_review,
)

CASES_ROOT = Path("/data/cases")

_ALLOWED_MODES = {"rule_only", "full_review"}


def get_case_detail(case_id: str) -> Optional[dict[str, Any]]:
    """Case 元数据 + 资产预览。

    读 /data/cases/{case_id}/ 下的：
      - info.yaml（元数据）
      - evaluation.yaml（评估配置）
      - expert_reasoning.md（专家推理）
      - expected_review.json（GT）
      - schematic_ir.json（IR）
    """
    case_dir = CASES_ROOT / case_id
    if not case_dir.is_dir():
        return None

    def _read_text(name: str) -> Optional[str]:
        p = case_dir / name
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def _read_json(name: str) -> Optional[dict]:
        p = case_dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "info_yaml": _read_text("info.yaml"),
        "evaluation_yaml": _read_text("evaluation.yaml"),
        "expert_reasoning_md": _read_text("expert_reasoning.md"),
        "expected_review_json": _read_json("expected_review.json"),
        "schematic_ir_json": _read_json("schematic_ir.json"),
    }


def run_case_benchmark(
    db: Session,
    case_id: str,
    mode: str,
    cases_root: Path | str = CASES_ROOT,
) -> dict[str, Any]:
    """跑单 Case Benchmark，结果写 case_benchmark_result。

    mode:
      - rule_only：调 run_single_case(f"{cases_root}/{case_id}")
      - full_review：调 run_single_case_full_review(case_id, cases_root)
        （内部走 6 节点 Graph，真实 LLM 评审，耗时 20–30 秒）

    结果落库，返回 dict（含 id / case_id / mode / status / 指标）。
    """
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode 必须为 {_ALLOWED_MODES}，实际: {mode!r}")

    cases_root_p = Path(cases_root)
    case_dir = cases_root_p / case_id
    if not case_dir.is_dir():
        raise ValueError(f"Case 目录不存在: {case_dir}")

    if mode == "rule_only":
        result = run_single_case(str(case_dir))
    else:  # full_review
        result = run_single_case_full_review(case_id, str(cases_root_p))

    # 落库
    row = CaseBenchmarkResult(
        case_id=result.case_id,
        mode=result.mode,
        status=result.status,
        total_gt_defects=result.total_gt_defects,
        total_pred_defects=result.total_pred_defects,
        tp=result.tp,
        fp=result.fp,
        fn=result.fn,
        rule_precision=result.rule_precision,
        rule_recall=result.rule_recall,
        evc=result.evc,
        ear=result.ear,
        ai_ner=result.ai_ner,
        metric_source=result.metric_source,
        task_id=result.task_id,
        error=result.error,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "id": row.id,
        "case_id": row.case_id,
        "mode": row.mode,
        "status": row.status,
        "total_gt_defects": row.total_gt_defects,
        "total_pred_defects": row.total_pred_defects,
        "tp": row.tp,
        "fp": row.fp,
        "fn": row.fn,
        "rule_precision": row.rule_precision,
        "rule_recall": row.rule_recall,
        "evc": row.evc,
        "ear": row.ear,
        "ai_ner": row.ai_ner,
        "metric_source": row.metric_source,
        "task_id": row.task_id,
        "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def list_case_benchmarks(
    db: Session,
    case_id: str,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """历史结果查询（按 created_at 倒序）。"""
    q = db.query(CaseBenchmarkResult).filter(CaseBenchmarkResult.case_id == case_id)
    total = q.count()
    rows = q.order_by(CaseBenchmarkResult.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id": r.id,
                "case_id": r.case_id,
                "mode": r.mode,
                "status": r.status,
                "total_gt_defects": r.total_gt_defects,
                "total_pred_defects": r.total_pred_defects,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "rule_precision": r.rule_precision,
                "rule_recall": r.rule_recall,
                "evc": r.evc,
                "ear": r.ear,
                "ai_ner": r.ai_ner,
                "metric_source": r.metric_source,
                "task_id": r.task_id,
                "error": r.error,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }
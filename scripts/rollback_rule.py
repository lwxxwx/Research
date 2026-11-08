"""规则回滚 CLI（S3 legacy 批次④ #8）。

退出码（对齐 bootstrap_admin.py）：0 成功 / 1 业务拒绝 / 2 系统故障。
"""
from __future__ import annotations

import argparse
import sys

from app.persistence.db import get_db_session
from app.services.rule_evolution_service import RuleEvolutionService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rollback a published rule to a historical version"
    )
    parser.add_argument("--rule-id", required=True)
    parser.add_argument("--to-version", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    svc = RuleEvolutionService()
    try:
        with get_db_session() as db:
            result = svc.rollback_rule(
                db, args.rule_id, args.to_version, dry_run=args.dry_run,
            )
        if result is None:
            print(f"REFUSE: rule_id={args.rule_id} 目标版本 {args.to_version} 不存在")
            return 1
        if result.get("dry_run"):
            print(
                f"DRY-RUN OK: rule_id={args.rule_id} "
                f"from={result['from_version']} to={result['to_version']}"
            )
        else:
            print(
                f"OK: rule_id={args.rule_id} "
                f"{result['from_version']} -> {result['to_version']}"
            )
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
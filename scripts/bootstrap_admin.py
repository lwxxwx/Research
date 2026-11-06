#!/usr/bin/env python
"""Sprint 3 G2 初始管理员 CLI。

用法：
    uv run python /scripts/bootstrap_admin.py --username admin [--email admin@x]
    uv run python /scripts/bootstrap_admin.py --username admin --i-understand

5 种判断场景（Sprint3_详细执行计划_V1.0.md §2.2）：
  1. 用户名存在，角色不是 admin       → 退出 1
  2. 用户名存在，是已禁用 admin       → 退出 1
  3. 用户名存在，是启用 admin         → 幂等成功，退出 0
  4. 用户名不存在，系统已有启用 admin  → 退出 0
  5. 用户名不存在，系统无启用 admin    → 创建初始 admin，退出 0

并发保护：pg_advisory_xact_lock（与 API 侧共用 ADMIN_LOCK_KEY）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从容器内 /scripts 直接执行
sys.path.insert(0, "/app")

from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError

from app.persistence.db import get_db_session
from app.persistence.models import User
from app.services.admin_lock import (
    ADMIN_LOCK_KEY,
    AdminLockTimeoutError,
    acquire_admin_lock,
    audit_admin_change,
)


def _print(msg: str) -> None:
    """纯英文输出，避免 PowerShell 重定向的 UTF-16 坑。"""
    print(msg, flush=True)


_MAX_INTEGRITY_RETRIES = 3


def _bootstrap_once(args) -> int:
    """单次 bootstrap 尝试（在独立事务内执行 5 种场景判断）。

    返回：
      0 = 成功（场景 3 / 4 / 5）
      1 = 拒绝（场景 1 / 2）
      抛 IntegrityError = 唯一约束冲突（由外层重试）
      抛 AdminLockTimeoutError = 锁超时（不重试）
    """
    with get_db_session() as db:
        acquire_admin_lock(db, timeout_ms=5000)
        # 场景 1-3：用户名已存在
        existing = db.query(User).filter(User.username == args.username).first()
        if existing is not None:
            if existing.role != "admin":
                _print(f"REFUSE: user '{args.username}' exists with role={existing.role}, not admin")
                return 1
            if not existing.is_active:
                _print(f"REFUSE: admin '{args.username}' is disabled")
                return 1
            _print(f"OK: admin '{args.username}' already exists (id={existing.id})")
            return 0

        # 场景 4-5：用户名不存在，检查当前 admin 数量
        active_admin_count = (
            db.query(func.count(User.id))
            .filter(User.role == "admin", User.is_active.is_(True))
            .scalar()
        ) or 0
        if active_admin_count > 0:
            _print(f"OK: system already has {active_admin_count} active admin(s), no need to create")
            return 0

        # 场景 5：创建初始 admin
        user = User(
            username=args.username,
            email=args.email,
            role="admin",
            is_active=True,
        )
        db.add(user)
        db.flush()
        user_id = user.id

        audit_admin_change(
            source="cli",
            actor_id=None,
            target_user_id=user_id,
            before_role=None,
            after_role="admin",
            before_is_active=None,
            after_is_active=True,
        )
        _print(f"OK: created admin '{args.username}' (id={user_id})")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap initial admin user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", default=None)
    parser.add_argument(
        "--i-understand",
        action="store_true",
        help="Confirm this is a production operation",
    )
    args = parser.parse_args()

    # ===== [B 类] 唯一约束冲突恢复：回滚 → 重试 → 重读 → 按场景判断 =====
    for attempt in range(_MAX_INTEGRITY_RETRIES):
        try:
            return _bootstrap_once(args)
        except IntegrityError as exc:
            # 唯一约束冲突：本次 with 块已 rollback（get_db_session 退出时）
            # 下一次 attempt 会开新 session + 新事务 + 重新 acquire_admin_lock
            if attempt < _MAX_INTEGRITY_RETRIES - 1:
                _print(
                    f"RETRY: integrity error on attempt {attempt + 1}/{_MAX_INTEGRITY_RETRIES}, "
                    f"retrying after rollback..."
                )
                import time
                time.sleep(0.1 * (attempt + 1))  # 指数退避
                continue
            _print(
                f"ERROR: integrity error after {_MAX_INTEGRITY_RETRIES} attempts: {exc}"
            )
            return 2
        except AdminLockTimeoutError:
            _print("ERROR: admin lock timeout, please retry")
            return 2
        except Exception as exc:
            _print(f"ERROR: {type(exc).__name__}: {exc}")
            return 2

    return 2  # 理论不可达


if __name__ == "__main__":
    sys.exit(main())
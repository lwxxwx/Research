# backend/app/services/admin_lock.py
"""Sprint 3 管理员变更互斥锁（PostgreSQL Advisory Lock）。

依据 Sprint3_详细执行计划_V1.0.md §2.3 / §3.3：
  - 使用 pg_advisory_xact_lock（事务级锁），事务提交/回滚自动释放。
  - 锁 key 固定为 ADMIN_LOCK_KEY，CLI 与 API 共用同一 key。
  - 锁超时视为基础设施异常，需输出结构化审计日志 + 告警，不伪装业务成功。

决策已冻结：
  - ADMIN_LOCK_KEY = 0x5342445F41444D31（ASCII "SBD_ADM1"）
  - 锁等待超时默认 5000ms
"""
from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


# ============================================================
# 固定锁 key（CLI 与 API 共用）
# ============================================================
# 0x5342445F41444D31 = ASCII "SBD_ADM1"
ADMIN_LOCK_KEY = 0x5342445F41444D31


# ============================================================
# 异常类型
# ============================================================
class LastAdminProtectionError(Exception):
    """业务异常：不能降级或停用最后一个启用管理员。"""
    pass


class AdminLockTimeoutError(Exception):
    """基础设施异常：管理员变更锁等待超时。"""
    pass


# ============================================================
# 锁获取
# ============================================================
def acquire_admin_lock(db: Session, timeout_ms: int = 5000) -> None:
    """在传入的 Session 内获取事务级管理员变更互斥锁。

    必须在同一个事务内连续执行"锁 → 查询 → 更新"，锁随事务提交/回滚自动释放。

    参数：
        db: SQLAlchemy Session（必须已在事务中，或即将在事务中使用）
        timeout_ms: 锁等待超时（毫秒），默认 5000ms

    异常：
        AdminLockTimeoutError: 锁等待超时
    """
    try:
        # SET LOCAL 只在当前事务内生效
        db.execute(text(f"SET LOCAL lock_timeout = '{timeout_ms}ms'"))
        db.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": ADMIN_LOCK_KEY},
        )
    except OperationalError as exc:
        # 锁等待超时：PostgreSQL 报错信息一般包含 "canceling statement due to lock timeout"
        if "lock timeout" in str(exc).lower() or "canceling statement" in str(exc).lower():
            logger.error(
                "audit.admin_lock_timeout: key=%s timeout_ms=%d",
                ADMIN_LOCK_KEY,
                timeout_ms,
            )
            raise AdminLockTimeoutError(
                f"管理员变更锁等待超时（{timeout_ms}ms），请稍后重试"
            ) from exc
        raise


# ============================================================
# 结构化审计日志（Sprint3_详细执行计划_V1.0.md §2.3 P1 要求）
# ============================================================
def audit_admin_change(
    *,
    source: str,           # "cli" / "api"
    actor_id: int | None,  # system=0 or None, 用户 id
    target_user_id: int | None,
    before_role: str | None,
    after_role: str | None,
    before_is_active: bool | None,
    after_is_active: bool | None,
) -> None:
    """输出结构化审计日志（KV 形式，便于检索）。

    依据：Sprint3_详细执行计划_V1.0.md §2.3 结构化审计日志字段约定。
    """
    logger.info(
        "audit.admin_change: source=%s actor_id=%s target_user_id=%s "
        "before_role=%s after_role=%s before_is_active=%s after_is_active=%s",
        source,
        actor_id,
        target_user_id,
        before_role,
        after_role,
        before_is_active,
        after_is_active,
    )
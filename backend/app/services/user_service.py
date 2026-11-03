# backend/app/services/user_service.py
"""Sprint 3 用户管理服务。

职责：
  - 用户 CRUD（create / get / list / update / soft_delete）
  - 最后管理员保护（update / soft_delete 时触发）
  - 结构化审计日志输出

【关键设计约束（决策已冻结，写入代码注释）】

D1. 事务块内禁止嵌入慢外部调用（Sprint3_详细执行计划_V1.0.md §3.3 补充）：
    - 本模块所有方法不调用 LLM / HTTP / 大文件 I/O。
    - 若未来业务需要审计消息回写外部系统，必须放到事务提交之后（异步或 after-commit hook），
      绝不能在 `with get_db_session() as db:` 块内发起，否则会持锁期间做慢操作，
      导致 admin_lock 长时间阻塞其他管理员变更请求。

D2. 锁顺序必须严格如下（不得改变）：
    1. acquire_admin_lock(db)   ← 统一管理员变更互斥锁（pg_advisory_xact_lock）
    2. db.query(User).with_for_update()  ← 目标用户行锁
    3. db.query(func.count(...))  ← 查询当前启用 admin 数量（在锁 + 行锁内）
    4. 更新字段 + db.flush()
    5. 事务提交（由 get_db_session 退出时统一 commit）
    反例：先 with_for_update() 再 acquire_admin_lock → 与 CLI 的锁顺序冲突 → 死锁。

D3. 禁止 db.commit()（Sprint3_详细执行计划_V1.0.md §3.3）：
    - 本模块所有方法只 db.flush()，不 db.commit()。
    - 事务生命周期由 get_db_session 上下文管理器统一管理。
    - 若在 Service 层 db.commit()，会导致"锁提前释放 + 事务提前结束"，
      后续步骤变成非事务性操作，最后管理员保护失效。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.persistence.models import User
from app.services.admin_lock import (
    LastAdminProtectionError,
    acquire_admin_lock,
    audit_admin_change,
)


logger = logging.getLogger(__name__)


# ============================================================
# 异常类型（Service 层业务异常）
# ============================================================
class UserNotFoundError(Exception):
    """用户不存在。"""
    pass


class UsernameConflictError(Exception):
    """用户名冲突。"""
    pass


# ============================================================
# 内部辅助：用户对象转 dict（避免 ORM detached）
# ============================================================
def _user_to_dict(user: User) -> dict[str, Any]:
    """在 with 块内调用，把 ORM 字段提取成纯 dict（避免 detached）。"""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


# ============================================================
# 查询
# ============================================================
def get_user_by_id(db: Session, user_id: int) -> dict[str, Any] | None:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return None
    return _user_to_dict(user)


def get_user_by_username(db: Session, username: str) -> dict[str, Any] | None:
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return None
    return _user_to_dict(user)


def list_users(
    db: Session, limit: int = 100, offset: int = 0
) -> dict[str, Any]:
    q = db.query(User)
    total = q.count()
    rows = q.order_by(User.id.asc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [_user_to_dict(u) for u in rows],
        "limit": limit,
        "offset": offset,
    }


# ============================================================
# 创建
# ============================================================
def create_user(
    db: Session,
    *,
    username: str,
    email: str | None,
    role: str,
    actor_id: int | None,
) -> dict[str, Any]:
    """创建用户。D1 约束：不调用外部服务。"""
    # 唯一性校验
    existing = db.query(User).filter(User.username == username).first()
    if existing is not None:
        raise UsernameConflictError(f"用户名已存在: {username}")

    user = User(
        username=username,
        email=email,
        role=role,
        is_active=True,
    )
    db.add(user)
    db.flush()  # 拿 id，不 commit（D3）
    db.refresh(user)

    audit_admin_change(
        source="api",
        actor_id=actor_id,
        target_user_id=user.id,
        before_role=None,
        after_role=user.role,
        before_is_active=None,
        after_is_active=user.is_active,
    )
    return _user_to_dict(user)


# ============================================================
# 更新（含最后管理员保护）
# ============================================================
def update_user(
    db: Session,
    *,
    user_id: int,
    patch: dict[str, Any],
    actor_id: int | None,
) -> dict[str, Any]:
    """更新用户 role / is_active。

    patch 只包含本次需要更新的字段，例如 {"role": "reviewer"} 或
    {"is_active": False}；空 dict 表示不更新任何字段。

    D2 锁顺序：acquire_admin_lock → with_for_update → count → update → flush。
    D3：不 commit，由 get_db_session 统一 commit。
    D1：不调用外部服务。
    """
    # 步骤 1：取统一管理员变更锁（D2）
    acquire_admin_lock(db)

    # 步骤 2：锁定目标用户行（D2）
    target: User | None = (
        db.query(User).filter(User.id == user_id).with_for_update().first()
    )
    if target is None:
        raise UserNotFoundError(f"用户 id={user_id} 不存在")

    # 记录更新前状态
    current_role = target.role
    current_is_active = target.is_active

    # 计算最终状态
    effective_role = patch.get("role", current_role)
    effective_is_active = patch.get("is_active", current_is_active)

    was_active_admin = current_role == "admin" and current_is_active is True
    will_be_active_admin = (
        effective_role == "admin" and effective_is_active is True
    )
    should_protect = was_active_admin and not will_be_active_admin

    # 步骤 3：最后管理员保护（D2）
    if should_protect:
        active_admin_count = (
            db.query(func.count(User.id))
            .filter(User.role == "admin", User.is_active.is_(True))
            .scalar()
        )
        if (active_admin_count or 0) <= 1:
            raise LastAdminProtectionError(
                "不能降级或停用最后一个启用管理员"
            )

    # 步骤 4：只更新实际提供的字段（D2）
    for key, value in patch.items():
        setattr(target, key, value)
    db.flush()  # 不 commit（D3）

    # 步骤 5：结构化审计日志
    audit_admin_change(
        source="api",
        actor_id=actor_id,
        target_user_id=target.id,
        before_role=current_role,
        after_role=effective_role,
        before_is_active=current_is_active,
        after_is_active=effective_is_active,
    )
    db.refresh(target)
    return _user_to_dict(target)


# ============================================================
# 软禁用
# ============================================================
def soft_delete_user(
    db: Session,
    *,
    user_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    """软禁用用户（语义 = is_active=false）。

    D2：使用统一的 admin_lock + 最后管理员保护。
    D3：不 commit。
    D1：不调用外部服务。
    """
    acquire_admin_lock(db)

    target: User | None = (
        db.query(User).filter(User.id == user_id).with_for_update().first()
    )
    if target is None:
        raise UserNotFoundError(f"用户 id={user_id} 不存在")

    current_role = target.role
    current_is_active = target.is_active

    # 只有启用中的 admin 被禁用时，才触发保护
    if current_role == "admin" and current_is_active is True:
        active_admin_count = (
            db.query(func.count(User.id))
            .filter(User.role == "admin", User.is_active.is_(True))
            .scalar()
        )
        if (active_admin_count or 0) <= 1:
            raise LastAdminProtectionError(
                "不能降级或停用最后一个启用管理员"
            )

    # 已禁用的用户直接返回（幂等）
    if current_is_active is False:
        return _user_to_dict(target)

    target.is_active = False
    db.flush()

    audit_admin_change(
        source="api",
        actor_id=actor_id,
        target_user_id=target.id,
        before_role=current_role,
        after_role=current_role,  # 角色不变
        before_is_active=True,
        after_is_active=False,
    )
    db.refresh(target)
    return _user_to_dict(target)
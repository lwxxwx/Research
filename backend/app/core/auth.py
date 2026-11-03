# backend/app/core/auth.py
"""Sprint 3 认证底座：当前用户依赖与角色校验。

设计约束（决策已冻结）：
  1. 第一阶段不交付 JWT / 密码登录，身份来自 X-User-Id 请求头（username-only 身份模拟）。
  2. get_current_user 返回 CurrentUser(NamedTuple)，避免 with get_db_session() 退出后
     ORM 对象 detached 报错。
  3. is_active=false 用户，任何接口一律 401。
  4. 角色：admin / engineer / reviewer / rule_admin。
"""
from __future__ import annotations

from typing import NamedTuple

from fastapi import Header, HTTPException
from fastapi import Depends

from app.persistence.db import get_db_session
from app.persistence.models import User


# ============================================================
# 角色常量
# ============================================================
ROLE_ADMIN = "admin"
ROLE_ENGINEER = "engineer"
ROLE_REVIEWER = "reviewer"
ROLE_RULE_ADMIN = "rule_admin"
ALL_ROLES = {ROLE_ADMIN, ROLE_ENGINEER, ROLE_REVIEWER, ROLE_RULE_ADMIN}


# ============================================================
# 当前用户数据结构（避免 ORM detached）
# ============================================================
class CurrentUser(NamedTuple):
    id: int
    username: str
    role: str
    is_active: bool


# ============================================================
# 内部辅助：解析 X-User-Id 头并查库
# ============================================================
def _resolve_user(x_user_id: str | None) -> CurrentUser:
    """解析 X-User-Id 并校验，返回 CurrentUser。

    校验顺序：
      1. 请求头缺失 → 401
      2. 格式非法（非整数）→ 401
      3. 用户不存在 → 401
      4. is_active=false → 401
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="缺少 X-User-Id 请求头")
    try:
        user_id = int(x_user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="X-User-Id 格式非法，需整数")

    with get_db_session() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail=f"用户 id={user_id} 不存在")
        if not user.is_active:
            raise HTTPException(status_code=401, detail=f"用户 {user.username} 已禁用")
        # 关键：在 with 块内提取字段，避免退出后 detached
        return CurrentUser(
            id=user.id,
            username=user.username,
            role=user.role,
            is_active=user.is_active,
        )


# ============================================================
# 依赖：强制登录
# ============================================================
def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> CurrentUser:
    """强制登录：必须携带合法的 X-User-Id，否则 401。"""
    return _resolve_user(x_user_id)


# ============================================================
# 依赖：可选登录（携带非法身份仍 401）
# ============================================================
def get_optional_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> CurrentUser | None:
    """可选登录：
      - 未携带 X-User-Id → 返回 None（匿名访问）
      - 携带了 X-User-Id 但非法 / 不存在 / 已禁用 → 401（不降级为匿名）
    """
    if x_user_id is None:
        return None
    return _resolve_user(x_user_id)


# ============================================================
# 依赖：角色校验
# ============================================================
def require_roles(*roles: str) -> callable:
    """生成一个依赖：要求当前用户角色在 roles 里，否则 403。

    用法：
        @router.get("/admin-only", dependencies=[Depends(require_roles(ROLE_ADMIN))])
        def admin_only(): ...

        # 或注入当前用户：
        @router.get("/me")
        def me(user: CurrentUser = Depends(require_roles(ROLE_ADMIN))): ...
    """
    for r in roles:
        if r not in ALL_ROLES:
            raise ValueError(f"非法角色: {r!r}，允许: {ALL_ROLES}")
    allowed = frozenset(roles)

    def _dependency(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"权限不足：需要角色 {sorted(allowed)}，当前 {current_user.role}",
            )
        return current_user

    return _dependency
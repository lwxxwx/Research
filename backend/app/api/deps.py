# backend/app/api/deps.py
"""API 层依赖注入 re-export。

业务模块从此处 import get_current_user / require_roles，不直接 import app.core.auth，
以便未来替换底层实现（如 JWT）时只需改这里。
"""
from __future__ import annotations

from app.core.auth import (
    ROLE_ADMIN,
    ROLE_ENGINEER,
    ROLE_REVIEWER,
    ROLE_RULE_ADMIN,
    CurrentUser,
    get_current_user,
    get_optional_user,
    require_roles,
)

__all__ = [
    "ROLE_ADMIN",
    "ROLE_ENGINEER",
    "ROLE_REVIEWER",
    "ROLE_RULE_ADMIN",
    "CurrentUser",
    "get_current_user",
    "get_optional_user",
    "require_roles",
]
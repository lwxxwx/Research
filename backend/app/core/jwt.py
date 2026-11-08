# backend/app/core/jwt.py
"""Sprint 3 legacy 批次③：用户名模式 JWT（D30–D32）。

范围边界：
  - 只做签发 / 校验，不含密码登录（password_hash 后置迭代）。
  - payload = {sub, username, role, iat, exp}；role 仅供日志，业务鉴权以 DB 为准（D32）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.core.config import settings


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8   # D31


def create_access_token(*, user_id: int, username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """校验并解码。过期 / 签名错 / 格式错一律抛 jwt.PyJWTError，由调用方转 401。"""
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
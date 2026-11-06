# backend/app/core/ids.py
"""统一 ID 生成器（Sprint 3 G5 新增）。

设计约束（D12 冻结）：
  1. 格式固定 `PREFIX-YYYYMMDD-XXXXXX`，随机后缀位数可配（RC=4 位向后兼容，GAP=6 位）。
  2. 分隔符必须是 ASCII `-`（U+002D），**禁止 U+2011 非断连字符**——这是
     Sprint 2 遗留的候选 ID 技术债（附录 D 第 1 项），G3-C 手测时复现过 404。
  3. 不依赖 DB 序列，避免 `create_*` 前多一次 DB 往返。
"""
from __future__ import annotations

import uuid
from datetime import datetime


# ============================================================
# 统一 ID 生成
# ============================================================
def make_prefixed_id(prefix: str, random_len: int = 6) -> str:
    """生成 `PREFIX-YYYYMMDD-XXXXXX` 格式的 ID。

    Args:
        prefix: 业务前缀（如 "RC" / "GAP"），自动大写。
        random_len: 随机后缀长度（默认 6 位，RC 用 4 位保持向后兼容）。

    Returns:
        形如 `GAP-20261106-A1B2C3` / `RC-20261106-A1B2`。

    约束：
        - 分隔符用 ASCII `-`（U+002D），禁止 U+2011（会导致 URL path 404）。
        - 不保证全局唯一（调用方需做碰撞重试，或依赖 DB UNIQUE 约束）。
    """
    ts = datetime.now().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:random_len].upper()
    return f"{prefix.upper()}-{ts}-{suffix}"


__all__ = ["make_prefixed_id"]
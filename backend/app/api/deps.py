# backend/app/api/deps.py
"""
API 层依赖注入（G4 新增）。

【为什么只有这个文件】
  - service 层的函数（create_task / get_task 等）都是模块级函数，不是类方法
  - 不需要 FastAPI 的 Depends 注入 DB session（service 内部自己 get_db_session）
  - 本文件预留给未来：认证、限流、trace_id 等横切关注点
  - G4 暂不注入任何依赖，保留空壳
"""
from __future__ import annotations

# G4 暂无依赖注入项。
# 预留位置：认证（get_current_user）、限流（rate_limit）、trace（get_trace_id）。
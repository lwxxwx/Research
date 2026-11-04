# app/core/exceptions.py
"""Sprint 3 G3-A 业务异常类型（不依赖任何 service，避免循环 import）。"""


class BusinessValueError(ValueError):
    """业务层参数校验失败 → HTTP 422。

    与标准 ValueError 区分开：
      - BusinessValueError：明确的业务校验失败（枚举非法、范围越界、格式不符）→ 422
      - ValueError：其他代码层面的值错误（解析失败、类型转换失败）→ 500
    """
    pass
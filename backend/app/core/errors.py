# app/core/errors.py
"""全局异常处理器。

Sprint 3 G3-A 修正（debug_03.md §3.5 问题 10）：
  - 原实现注册了全局 ValueError → 422 handler，过度捕获，会把代码 bug
    （如 json.loads 解析 DB 脏数据失败、int() 转换失败）伪装成"用户输入错误"。
  - 改为：业务校验失败用 BusinessValueError 显式抛出 → 422；
    其他 ValueError 不注册 handler，走 unhandled_exception_handler → 500。

使用约定：
  - Service 层校验业务参数（枚举/范围/格式）失败时，raise BusinessValueError。
  - API 层/框架层的其他 ValueError 一律不转 422，作为代码 bug 暴露为 500。
"""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.feedback_service import FeedbackNotFoundError
from app.core.exceptions import BusinessValueError   # ← 从新文件 import


logger = logging.getLogger(__name__)


# ============================================================
# 业务异常类型
# ============================================================
class BusinessException(Exception):
    """业务自定义异常（携带 HTTP 状态码）。"""
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

# ============================================================
# 注册所有异常处理器
# ============================================================
def register_exception_handlers(app: FastAPI) -> None:

    # 1. 业务异常（携带 status code）
    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException):
        logger.warning(f"BusinessException code={exc.code} msg={exc.message}")
        return JSONResponse(
            status_code=exc.code,
            content={"error": exc.code, "message": exc.message},
        )

    # 2. 业务参数校验失败 → 422（仅 BusinessValueError，不捕获标准 ValueError）
    @app.exception_handler(BusinessValueError)
    async def business_value_error_handler(request: Request, exc: BusinessValueError):
        logger.warning(f"BusinessValueError: {str(exc)}")
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    # 3. 反馈资源未找到 → 404
    @app.exception_handler(FeedbackNotFoundError)
    async def feedback_not_found_handler(request: Request, exc: FeedbackNotFoundError):
        logger.warning(f"FeedbackNotFoundError: {str(exc)}")
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc)},
        )

    # 4. 兜底：未捕获异常 → 500（不再把标准 ValueError 当 422）
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception")
        return JSONResponse(
            status_code=500,
            content={"error": 500, "message": "internal server error"},
        )
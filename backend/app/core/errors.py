# app/core/errors.py
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)

class BusinessException(Exception):
    """业务自定义异常"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

def register_exception_handlers(app: FastAPI):

    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException):
        logger.warning(f"BusinessException code={exc.code} msg={exc.message}")
        return JSONResponse(
            status_code=exc.code,
            content={"error": exc.code, "message": exc.message}
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception")
        return JSONResponse(
            status_code=500,
            content={"error": 500, "message": "internal server error"}
        )

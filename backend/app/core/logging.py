# app/core/logging.py
import logging
import sys
from pythonjsonlogger import jsonlogger
from app.core.config import settings

# 保存本模块添加的 handler 引用，便于只清自己加的
_configured_handler: logging.Handler | None = None

def setup_logging() -> None:
    global _configured_handler

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    #root_logger.handlers.clear()
    # 只移除本模块之前添加的 handler，避免清掉别人的
    if _configured_handler is not None:
        root_logger.removeHandler(_configured_handler)
        _configured_handler = None

    handler = logging.StreamHandler(stream=sys.stdout)
    
    #formatter = jsonlogger.JsonFormatter(
    #    "%(asctime)s %(levelname)s %(name)s %(message)s"
    #)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        #    "message": "message",
        },
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _configured_handler = handler

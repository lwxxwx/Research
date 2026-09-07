# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.errors import register_exception_handlers

# 初始化日志
setup_logging()

app = FastAPI(
    title="AI原理图评审 MVP Backend",
    version="sprint0-v1.0",
    docs_url="/docs" if settings.is_dev else None,
    redoc_url="/redoc" if settings.is_dev else None
)

# ========== 新增：CORS配置 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册全局异常处理器
register_exception_handlers(app)

# 健康检查接口
@app.get("/api/v1/healthz")
async def healthz():
    # 适度增强：返回环境信息
    return {
        "status": "ok",
        "env": settings.app_env,
        "version": "sprint0-v1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000
    )
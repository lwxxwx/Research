# app/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# 添加 /api/v1/healthz 以匹配 Dockerfile 中的健康检查
@app.get("/api/v1/healthz")
async def healthz():
    return {"status": "healthy"}
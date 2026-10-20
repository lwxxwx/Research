# backend/app/rag/embedding_factory.py
"""
Embedding 工厂与维度守卫（Sprint 1 阶段 C 新增）

【为什么新建本文件】
  app/rag/ingest.py 与 app/rag/retriever.py 各有一份逐字相同的
  get_embedding_client()，仅支持 fake/openai 两分支。
  本文件收敛为唯一实现，新增 zhipu 分支，并加维度守卫。

【维度守卫】
  切后端时若忘记迁移 DB 列维度，pgvector 会抛难懂的
  "different vector dimensions" 错误。本守卫在写入/检索前对照
  rag_embedding_backend 的期望维度与 DB 实际维度，不匹配则 raise
  可操作的错误提示，指向切换脚本。

【切换契约】
  - 日常切换：只改 RAG_EMBEDDING_BACKEND 环境变量
  - 换后端：改参数 + 跑 scripts/switch_embedding_backend.py（迁移 + 清库 + 重灌）

【S1-C 修正】zhipu 2048 -> 1024
  原因：pgvector HNSW 对 vector 类型上限 2000 维。
"""
from __future__ import annotations

from sqlalchemy import text

from app.core.config import settings
from app.llm.embeddings.base import EmbeddingProvider
from app.llm.embeddings.fake import FakeEmbedding
from app.llm.embeddings.openai import OpenAIEmbedding
from app.llm.embeddings.zhipu import ZhipuEmbedding

# 后端 → 期望维度
# --- 原代码（保留，已注释） ---
# BACKEND_DIM: dict[str, int] = {
#     "fake": 1536,
#     "openai": 1536,
#     "zhipu": 2048,
# }
# --- 原代码（保留，已注释） ---
# ===== [S1-C 修正] zhipu 2048 -> 1024：pgvector HNSW 对 vector 上限 2000 维 =====
BACKEND_DIM: dict[str, int] = {
    "fake": 1536,
    "openai": 1536,
    "zhipu": 1024,
}
# ===== [S1-C 修正 END] =====

_PROVIDERS = {
    "fake": FakeEmbedding,
    "openai": OpenAIEmbedding,
    "zhipu": ZhipuEmbedding,
}


def get_embedding_client(backend: str | None = None) -> EmbeddingProvider:
    """
    按 settings.rag_embedding_backend（或显式传入 backend）返回 EmbeddingProvider。
    """
    key = (backend or settings.rag_embedding_backend or "fake").lower().strip()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(
            f"不支持的 RAG_EMBEDDING_BACKEND={key}，"
            f"可选值: {sorted(_PROVIDERS.keys())}"
        )
    return cls()


def assert_backend_matches_db(db_session) -> None:
    """
    守卫：校验当前后端期望维度与 knowledge_chunk.embedding 列实际维度一致。
    不匹配时 raise RuntimeError，提示先跑切换脚本。
    """
    backend = (settings.rag_embedding_backend or "fake").lower().strip()
    expected = BACKEND_DIM.get(backend)
    if expected is None:
        raise ValueError(
            f"不支持的 RAG_EMBEDDING_BACKEND={backend}，"
            f"可选值: {sorted(BACKEND_DIM.keys())}"
        )

    # --- 原代码（保留，已注释） ---
    # actual = db_session.execute(
    #     text(
    #         "SELECT atttypmod - 4 FROM pg_attribute "
    #         "WHERE attrelid = 'knowledge_chunk'::regclass AND attname = 'embedding'"
    #     )
    # ).scalar()
    # --- 原代码（保留，已注释） ---
    # ===== [S1-C 修正] atttypmod - 4 在某些环境偏差 4（1024 算成 1020），
    # 改用 pgvector 的 format_type 解析，直接拿 "vector(1024)" 里的 1024。
    actual_type = db_session.execute(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'knowledge_chunk'::regclass AND attname = 'embedding'"
        )
    ).scalar()
    # actual_type 形如 "vector(1024)"，解析出维度
    actual = None
    if actual_type and "(" in actual_type and ")" in actual_type:
        try:
            actual = int(actual_type.split("(")[1].rstrip(")"))
        except (ValueError, IndexError):
            actual = None
    # ===== [S1-C 修正 END] =====

    if actual != expected:
        raise RuntimeError(
            f"Embedding 维度不匹配：后端 {backend} 期望 {expected} 维，"
            f"DB 当前 {actual} 维。请先执行切换脚本："
            f"python /scripts/switch_embedding_backend.py --to {backend}"
        )


__all__ = [
    "BACKEND_DIM",
    "get_embedding_client",
    "assert_backend_matches_db",
]
# backend/app/llm/embeddings/fake.py
"""
Fake Embedding（Sprint 1 阶段 C 新增）

包装 langchain_core 的 DeterministicFakeEmbedding，维度 1536。
CI 默认后端，不触网。

保持与 app/rag/ingest.py 现有行为一致：
  DeterministicFakeEmbedding(size=1536).embed_documents / embed_query
"""
from __future__ import annotations

from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from app.llm.embeddings.base import EmbeddingProvider


class FakeEmbedding(EmbeddingProvider):
    provider_name = "fake"
    dimension = 1536

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self._impl = DeterministicFakeEmbedding(size=dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)


__all__ = ["FakeEmbedding"]
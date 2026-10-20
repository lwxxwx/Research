# backend/app/llm/embeddings/openai.py
"""
OpenAI Embedding（Sprint 1 阶段 C 保留）

- 保留为后续可选后端
- 模型: settings.openai_embedding_model（默认 text-embedding-3-small）
- 维度: 1536

Key 取值：settings.openai_embedding_api_key or settings.llm_api_key
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.llm.embeddings.base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    provider_name = "openai"
    dimension = 1536

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = (
            api_key
            or settings.openai_embedding_api_key
            or settings.llm_api_key
        )
        self.model = model or settings.openai_embedding_model
        if not self.api_key:
            raise RuntimeError(
                "OpenAIEmbedding 需要配置 OPENAI_EMBEDDING_API_KEY 或 LLM_API_KEY"
            )
        self._impl = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)


__all__ = ["OpenAIEmbedding"]
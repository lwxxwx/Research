# backend/app/llm/embeddings/base.py
"""
Embedding Provider 抽象（Sprint 1 阶段 C 新增）

【职责】
  - 统一 embed_documents / embed_query 接口
  - 供 app/rag/embedding_factory.py 分发

【边界】
  - 本模块只负责向量化，不涉及检索/入库
  - 维度由实现类通过 dimension 属性声明，工厂用它做守卫
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Embedding 统一接口"""

    provider_name: str = "base"
    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（文档侧）。返回与 texts 等长的向量列表。"""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """单条向量化（查询侧）。返回一条向量。"""
        raise NotImplementedError


__all__ = ["EmbeddingProvider"]
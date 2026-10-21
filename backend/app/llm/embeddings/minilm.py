# backend/app/llm/embeddings/minilm.py
"""
本地化 MiniLM Embedding 骨架（Sprint 1 阶段 C 占位）

【状态】
  骨架占位：未实现、未注册到工厂、未加入 BACKEND_DIM、未新增迁移。
  不影响当前 zhipu 1024 维的闭环。

【为什么现在建骨架】
  - 固定接口签名，未来实现只填函数体，不改签名
  - 声明 dimension=384，未来 BACKEND_DIM 加一行即可
  - 不注册到工厂，不会被任何路径调到，不会半可用

【接入步骤】
  见 docs/sprint1/onprem_minilm_embedding_integration.md

【接入前需确定的 4 件事】
  1. 加载方式：sentence-transformers / onnxruntime / llama.cpp
  2. 模型文件来源：本地挂载 / 打进镜像 / 首次运行下载
  3. 归一化：是否 L2 normalize（sentence-transformers 默认是）
  4. 批大小：默认 32 / 64
"""
from __future__ import annotations

from app.llm.embeddings.base import EmbeddingProvider


class MiniLMEmbedding(EmbeddingProvider):
    provider_name = "minilm"
    dimension = 384

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or "/data/models/all-MiniLM-L6-v2"
        raise NotImplementedError(
            "MiniLM 尚未启用；本地化部署就绪后按 "
            "docs/sprint1/onprem_minilm_embedding_integration.md 实现"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


__all__ = ["MiniLMEmbedding"]
# backend/app/llm/embeddings/zhipu.py
"""
智谱 Embedding（Sprint 1 阶段 C 新增）

- SDK: zhipuai.ZhipuAI(api_key=...)
- 模型: settings.zhipuai_embedding_model（默认 embedding-3）
- 维度: 1024（见下方 [S1-C 修正]）

Key 取值：settings.zhipuai_api_key or settings.llm_api_key

【S1-C 修正】2048 -> 1024
  原因：pgvector HNSW 对 vector 类型上限 2000 维，2048 维无法建 HNSW 索引。
       智谱 embedding-3 是 Matryoshka 模型，支持 dimensions 参数降维，
       选 1024 维（HNSW 友好，检索质量损失可忽略）。
  配套：app/rag/embedding_factory.py 的 BACKEND_DIM["zhipu"] 同步为 1024；
       Alembic 迁移 c1_embedding_dim_1024 把列改为 Vector(1024)。
"""
from __future__ import annotations

from app.core.config import settings
from app.llm.embeddings.base import EmbeddingProvider


class ZhipuEmbedding(EmbeddingProvider):
    provider_name = "zhipu"
    # --- 原代码（保留，已注释） ---
    # dimension = 2048
    # --- 原代码（保留，已注释） ---
    # ===== [S1-C 修正] 2048 -> 1024：绕开 pgvector HNSW 2000 维上限 =====
    dimension = 1024
    # ===== [S1-C 修正 END] =====

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.zhipuai_api_key or settings.llm_api_key
        self.model = model or settings.zhipuai_embedding_model
        if not self.api_key:
            raise RuntimeError(
                "ZhipuEmbedding 需要配置 ZHIPUAI_API_KEY 或 LLM_API_KEY"
            )

    def _client(self):
        # 延迟导入，无 Key 的单测不会触发 SDK import
        from zhipuai import ZhipuAI
        return ZhipuAI(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        # --- 原代码（保留，已注释） ---
        # resp = client.embeddings.create(model=self.model, input=texts)
        # --- 原代码（保留，已注释） ---
        # ===== [S1-C 修正] 加 dimensions，让智谱按 1024 维返回 =====
        resp = client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimension
        )
        # ===== [S1-C 修正 END] =====
        return [list(item.embedding) for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        client = self._client()
        # --- 原代码（保留，已注释） ---
        # resp = client.embeddings.create(model=self.model, input=text)
        # --- 原代码（保留，已注释） ---
        # ===== [S1-C 修正] 加 dimensions，让智谱按 1024 维返回 =====
        resp = client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimension
        )
        # ===== [S1-C 修正 END] =====
        return list(resp.data[0].embedding)


__all__ = ["ZhipuEmbedding"]
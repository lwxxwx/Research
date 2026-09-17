"""
app/rag/retriever.py
Sprint‑0 RAG检索器
功能：
1. query生成embedding做pgvector余弦相似度召回
2. 支持 part_numbers、related_rule_ids元数据过滤
3. 返回标准化RetrievalResult对象
Sprint‑0限制：仅内部Python API，无重排，不对外暴露HTTP接口；Sprint‑1接入LangGraph knowledge_retrieve节点
"""
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from app.core.config import settings
from app.persistence.db import get_db_session
from app.persistence.models import KnowledgeChunk
from app.rag.knowledge import RetrievalResult

EMBEDDING_MODEL_NAME = settings.openai_embedding_model


def get_embedding_client():
    backend = settings.rag_embedding_backend.lower()
    if backend == "fake":
        return DeterministicFakeEmbedding(size=1536)
    elif backend == "openai":
        _embedding_api_key = settings.openai_embedding_api_key or settings.llm_api_key
        if not _embedding_api_key:
            raise RuntimeError(
                "RAG_EMBEDDING_BACKEND=openai 模式需要配置 OPENAI_EMBEDDING_API_KEY 或者 LLM_API_KEY"
            )
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_key=_embedding_api_key
        )
    else:
        raise ValueError(f"不支持的 RAG_EMBEDDING_BACKEND={backend}，可选值：fake / openai")


class Retriever:
    def __init__(self):
        self._embed_client = get_embedding_client()

    def retrieve(
        self,
        query: str,
        part_numbers: Optional[List[str]] = None,
        rule_ids: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        :param query: 查询文本
        :param part_numbers: 器件型号过滤列表，为空不做过滤
        :param rule_ids: rule_id过滤列表，为空不做过滤
        :param top_k: 期望返回结果条数
        :return: List[RetrievalResult]
        """
        query_embedding = self._embed_client.embed_query(query)
        filter_parts = part_numbers or []
        filter_rules = rule_ids or []
        output: List[RetrievalResult] = []

        # ===🔴全部读取实体、访问属性逻辑全部放在with会话内部！不要把对象带出session
        with get_db_session() as db:
            stmt = (
                select(
                    KnowledgeChunk,
                    KnowledgeChunk.embedding.cosine_distance(query_embedding).label("distance")
                )
                .order_by("distance")
                .limit(top_k * 4)
            )
            rows = db.execute(stmt).all()

            for row in rows:
                chunk: KnowledgeChunk = row[0]
                dist = row[1]

                meta_raw = chunk.meta_json
                if meta_raw is not None:
                    meta = dict(meta_raw.items())
                else:
                    meta = {}

                chunk_parts: List[str] = meta.get("part_numbers", [])
                chunk_rules: List[str] = meta.get("related_rule_ids", [])

                if filter_parts and not set(filter_parts) & set(chunk_parts):
                    continue
                if filter_rules and not set(filter_rules) & set(chunk_rules):
                    continue

                doc = chunk.doc
                doc_meta_raw = doc.meta_json
                if doc_meta_raw is not None:
                    doc_meta = dict(doc_meta_raw.items())
                else:
                    doc_meta = {}

                item = RetrievalResult(
                    source_type=doc.source_type or "",
                    source_title=doc.title or "",
                    source_section=doc_meta.get("source_section", ""),
                    snippet=chunk.chunk_text,
                    score=dist,
                    part_numbers=chunk_parts,
                    related_rule_ids=chunk_rules,
                    doc_id=doc.id,
                    chunk_id=chunk.id
                )
                output.append(item)
                if len(output) >= top_k:
                    break
        # with块结束，session关闭；此时output里面只有Pydantic数据对象RetrievalResult，没有ORM实体对象，安全返回
        return output

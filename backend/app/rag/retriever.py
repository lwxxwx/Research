"""
app/rag/retriever.py
Sprint-0 v1.5 RAG 检索器

 用 embedding 工厂 + SQLAlchemy 2.0 方法链 + pgvector 余弦距离 从 DB 捞候选，
 用 joinedload 消除 N+1、过采样 + 集合交集过滤 精选结果，
 用 _as_dict 四层 fallback + 限流告警 兜住 JSONB 边界，用 assert 哨兵
 守护数据完整性，最终组装成 RetrievalResult 返回——是一个防御式编程 + 性能优化并重的检索实现。


v1.1 变更：
- M4：joinedload(KnowledgeChunk.doc) 消除 N+1 查询
- M5：dict(meta_raw) 替代 dict(meta_raw.items())
- 兼容 v1.1 chunk 元数据：优先使用 chunk 级 source_section（M3 落库）

v1.2 变更（豆包优化 5）：
- 优化 5：检索结果透传 is_continuation 字段

v1.3 变更（豆包阻断 BUG3）：
- BUG3：_as_dict 优先 .items()，兼容 psycopg Json 包装类型；
         转换失败输出 WARNING（限流一次），不静默返回 {}

v1.4 变更（豆包新优化 4）：
- 优化 4：doc = chunk.doc 后加 assert doc is not None 防御断言；
         删除 if doc else 分支（DB 层 FK + NOT NULL 保证 doc 必存在）

v1.5 变更（豆包新优化 1）：
- 优化 1：删除未使用的 import：Union、Session（PEP 8 / ruff F401）

Sprint-0 限制：仅内部 Python API，无重排，不对外暴露 HTTP 接口
"""
import logging
import threading
from typing import List, Optional

from langchain_core.embeddings.fake import DeterministicFakeEmbedding

# ★ 优化 1：删除未使用 import Session
# --- 原 import（保留，已删） ---
# from sqlalchemy.orm import joinedload, Session
from langchain_openai import OpenAIEmbeddings

# ★ 优化 1：删除未使用 import
# --- 原 import（保留，已删） ---
# from typing import List, Optional, Union
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.persistence.db import get_db_session
from app.persistence.models import KnowledgeChunk
from app.rag.knowledge import RetrievalResult

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = settings.openai_embedding_model


def get_embedding_client():
    backend = settings.rag_embedding_backend.lower()
    if backend == "fake":
        return DeterministicFakeEmbedding(size=1536)
    elif backend == "openai":
        _embedding_api_key = settings.openai_embedding_api_key or settings.llm_api_key
        if not _embedding_api_key:
            raise RuntimeError(
                "RAG_EMBEDDING_BACKEND=openai 模式需要配置 "
                "OPENAI_EMBEDDING_API_KEY 或者 LLM_API_KEY"
            )
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_key=_embedding_api_key,
        )
    else:
        raise ValueError(
            f"不支持的 RAG_EMBEDDING_BACKEND={backend}，可选值：fake / openai"
        )


# ============================================================
# ★ BUG3 修复：_as_dict 增强
# ============================================================
_warned_types: set = set()
_warned_types_lock = threading.Lock()


def _as_dict(meta_raw) -> dict:
    """
    M5 / BUG3：统一 JSONB 列可能返回的类型（dict / None / Json wrapper）

    BUG3 修复要点：
    1. 优先 .items()：兼容 psycopg 的 Json 包装类型
    2. isinstance(dict) 亦命中（兼容 SQLAlchemy MutableDict）
    3. 转换失败输出 WARNING（每种类型只 warn 一次）
    4. 仍返回 {}，避免单个 chunk 解析失败导致整个检索崩溃
    """
    if meta_raw is None:
        return {}

    # dict 及其子类（含 SQLAlchemy MutableDict）
    if isinstance(meta_raw, dict):
        return meta_raw

    # 非 dict 但有 .items() 的对象
    if hasattr(meta_raw, "items"):
        try:
            return dict(meta_raw.items())
        except (TypeError, ValueError):
            pass

    # Fallback：尝试 dict() 转换
    try:
        return dict(meta_raw)
    except (TypeError, ValueError):
        # ★ BUG3：不静默，输出 WARNING（限流）
        type_name = type(meta_raw).__name__
        with _warned_types_lock:
            if type_name not in _warned_types:
                _warned_types.add(type_name)
                logger.warning(
                    f"_as_dict: 无法将类型 {type_name} 转换为 dict，"
                    f"返回空字典（此类型只告警一次）"
                )
        return {}


class Retriever:
    def __init__(self):
        self._embed_client = get_embedding_client()

    def retrieve(
        self,
        query: str,
        part_numbers: Optional[List[str]] = None,
        rule_ids: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """
        :param query: 查询文本
        :param part_numbers: 器件型号过滤列表
        :param rule_ids: rule_id 过滤列表
        :param top_k: 期望返回结果条数
        :return: List[RetrievalResult]
        """
        query_embedding = self._embed_client.embed_query(query)
        filter_parts = part_numbers or []
        filter_rules = rule_ids or []
        output: List[RetrievalResult] = []

        # M4：joinedload 一次性拉出 doc，消除 N+1
        with get_db_session() as db:
            stmt = (
                select(
                    KnowledgeChunk,
                    KnowledgeChunk.embedding
                    .cosine_distance(query_embedding)
                    .label("distance"),
                )
                .options(joinedload(KnowledgeChunk.doc))
                .order_by("distance")
                .limit(top_k * 4)
            )
            rows = db.execute(stmt).all()

            for row in rows:
                chunk: KnowledgeChunk = row[0]
                dist = row[1]

                # M5 / BUG3：统一 JSONB 解析
                meta = _as_dict(chunk.meta_json)
                chunk_parts: List[str] = meta.get("part_numbers", []) or []
                chunk_rules: List[str] = meta.get("related_rule_ids", []) or []

                if filter_parts and not set(filter_parts) & set(chunk_parts):
                    continue
                if filter_rules and not set(filter_rules) & set(chunk_rules):
                    continue

                # ★ 优化 4：加 assert 防御断言，删除 if doc else 分支
                doc = chunk.doc
                # --- 原代码（保留，已弃用） ---
                # doc_meta = _as_dict(doc.meta_json) if doc else {}
                assert doc is not None, (
                    f"KnowledgeChunk(id={chunk.id}).doc 为 None，数据完整性被破坏"
                )
                doc_meta = _as_dict(doc.meta_json)

                # M3：优先 chunk 级 source_section，缺失时回落到 doc 级
                chunk_section = (
                    meta.get("source_section")
                    or doc_meta.get("source_section")
                    or ""
                )

                # ★ 优化 4：删除 if doc else 三目分支，直接访问 doc 属性
                item = RetrievalResult(
                    # --- 原代码（保留，已弃用） ---
                    # source_type=doc.source_type or "" if doc else "",
                    # source_title=doc.title or "" if doc else "",
                    # doc_id=doc.id if doc else None,
                    source_type=doc.source_type or "",
                    source_title=doc.title or "",
                    source_section=chunk_section,
                    snippet=chunk.chunk_text,
                    score=dist,
                    part_numbers=chunk_parts,
                    related_rule_ids=chunk_rules,
                    doc_id=doc.id,
                    chunk_id=chunk.id,

                    # === 优化 5：透传 is_continuation ===
                    is_continuation=bool(meta.get("is_continuation", False)),
                )
                output.append(item)
                if len(output) >= top_k:
                    break

        return output

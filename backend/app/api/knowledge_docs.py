# backend/app/api/knowledge_docs.py
"""RAG 知识库管理路由（S2-H 新增）。

2 端点：
  GET  /api/v1/knowledge-docs          列表
  POST /api/v1/knowledge-docs/upload   上传 Markdown 并切片入库

【约束】
  - 仅支持 .md 后缀
  - 不支持启停与物理删除（S2 冻结）
"""
from __future__ import annotations

import uuid
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import func

from app.api.schemas import (
    KnowledgeDocItem,
    KnowledgeDocListResponse,
    KnowledgeUploadResponse,
)
from app.persistence.db import get_db_session
from app.persistence.models import KnowledgeDoc, KnowledgeChunk
from app.rag.ingest import ingest_single_file
from app.rag.embedding_factory import get_embedding_client

router = APIRouter(prefix="/knowledge-docs", tags=["knowledge_docs"])


# ============================================================
# GET /knowledge-docs
# ============================================================
@router.get("", response_model=KnowledgeDocListResponse)
def list_knowledge_docs() -> KnowledgeDocListResponse:
    """查询已入库的知识文档列表。"""
    with get_db_session() as db:
        docs = db.query(KnowledgeDoc).order_by(KnowledgeDoc.id.desc()).all()

        items = [
            KnowledgeDocItem(
                id=d.id,
                title=d.title,
                source=d.source,
                source_type=d.source_type,
                created_at=d.created_at.isoformat() if d.created_at else None,
            )
            for d in docs
        ]
        return KnowledgeDocListResponse(total=len(items), items=items)


# ============================================================
# POST /knowledge-docs/upload
# ============================================================
@router.post("/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge_doc(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    """上传 Markdown 文件，落盘临时文件 -> 切片向量化 -> 删除临时文件。"""
    if not file.filename or not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持上传 .md 后缀的 Markdown 文件")

    temp_dir = Path("/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload_{uuid.uuid4().hex}.md"

    try:
        # 1. 写入临时文件
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        temp_path.write_bytes(content)

        # 2. 调用 ingest_single_file
        embeddings = get_embedding_client()
        with get_db_session() as db:
            # 记录导入前的 doc 数量，用于异常捕获后的防御
            before_count = db.query(func.count(KnowledgeDoc.id)).scalar() or 0
            
            # ingest 内部会做 db.add + db.commit
            ingest_single_file(db, temp_path, embeddings)
            
            # 查最新的一条 doc 来获取 id 和 title
            latest_doc = db.query(KnowledgeDoc).order_by(KnowledgeDoc.id.desc()).first()
            if not latest_doc or latest_doc.id <= before_count:
                # 理论上不该发生，除非 ingest 逻辑有变
                raise HTTPException(status_code=500, detail="向量化入库异常，未生成新文档")
            
            chunk_count = (
                db.query(func.count(KnowledgeChunk.id))
                .filter(KnowledgeChunk.knowledge_doc_id == latest_doc.id)
                .scalar() or 0
            )

            return KnowledgeUploadResponse(
                doc_id=latest_doc.id,
                title=latest_doc.title,
                chunk_count=chunk_count,
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"知识库切片/向量化失败: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        # 3. 无论成功失败，清理临时文件
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass  # 清理失败不影响主流程，仅日志记录
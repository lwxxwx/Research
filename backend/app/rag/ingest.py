"""
app/rag/ingest.py
Sprint‑0 RAG种子知识库导入CLI
职责：
1. 读取seed_ingest_list.yaml配置
2. 调用chunking.py完成md解析、校验、切片
3. 根据RAG_EMBEDDING_BACKEND生成向量(fake模拟 / openai真实)
4. 写入数据库 knowledge_doc / knowledge_chunk
CLI入口：python -m app.rag.ingest --config ./data/knowledge/seed_ingest_list.yaml
Sprint‑0约束：不接入LLM、不接入LangGraph workflow
"""
import argparse
import logging
import pathlib
import yaml
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from sqlalchemy.orm import Session
from app.core.config import settings
from app.persistence.db import get_db_session
from app.persistence.models import KnowledgeDoc, KnowledgeChunk
from app.rag.knowledge import IngestConfig
from app.rag.chunking import process_markdown_file

logger = logging.getLogger(__name__)

# Sprint0 切片参数
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL_NAME = settings.openai_embedding_model


def get_embedding_client():
    """根据 settings.rag_embedding_backend 返回embedding实例"""
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


def scan_case_evidence(root_dir: str, glob_pattern: str) -> List[pathlib.Path]:
    """扫描B‑Case目录下evidence/*.md"""
    root = pathlib.Path(root_dir)
    return list(root.glob(glob_pattern))


def ingest_single_file(db: Session, file_path: pathlib.Path, embeddings):
    logger.info(f"开始处理文件：{file_path.as_posix()}")
    meta, raw_content, chunk_texts = process_markdown_file(
        file_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    # 使用 ORM属性 meta_json，数据库列依旧 metadata
    doc_entity = KnowledgeDoc(
        title=meta["source_title"],
        source=file_path.as_posix(),
        source_type=meta["source_type"],
        content_md=raw_content,
        meta_json={
            "source_section": meta["source_section"],
            "part_numbers": meta["part_numbers"],
            "related_rule_ids": meta["related_rule_ids"],
            "embedding_provider": settings.rag_embedding_backend,
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
        version="v1.0"
    )
    db.add(doc_entity)
    db.flush()
    doc_id = doc_entity.id

    for seg in chunk_texts:
        emb_vec = embeddings.embed_query(seg)
        chunk_entity = KnowledgeChunk(
            knowledge_doc_id=doc_id,
            chunk_text=seg,
            embedding=emb_vec,
            meta_json={
                "source_section": meta["source_section"],
                "part_numbers": meta["part_numbers"],
                "related_rule_ids": meta["related_rule_ids"],
            }
        )
        db.add(chunk_entity)
    db.commit()
    logger.info(f"✅ {file_path.name} 导入完成 doc_id={doc_id}, chunk数量={len(chunk_texts)}")


def run_ingest(config_path: str):
    cfg_file = pathlib.Path(config_path)
    cfg_raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    ingest_cfg = IngestConfig.model_validate(cfg_raw)

    embeddings = get_embedding_client()
    file_list: List[pathlib.Path] = []

    for entry in ingest_cfg.ingest_entries:
        fp = pathlib.Path(entry.file)
        if not fp.exists():
            logger.warning(f"文件不存在跳过：{fp.as_posix()}")
            continue
        file_list.append(fp)

    if ingest_cfg.case_evidence_scan:
        scan_root = ingest_cfg.case_evidence_scan.get("root_dir")
        glob_pat = ingest_cfg.case_evidence_scan.get("glob_pattern")
        if scan_root and glob_pat:
            found_files = scan_case_evidence(scan_root, glob_pat)
            file_list.extend(found_files)
            logger.info(f"case_evidence_scan扫描到 {len(found_files)} 个md文件")

    logger.info(f"待导入总文件数：{len(file_list)}")
    with get_db_session() as db:
        for fpath in file_list:
            try:
                ingest_single_file(db, fpath, embeddings)
            except Exception as exc:
                logger.error(f"导入失败 {fpath.as_posix()} : {exc}", exc_info=True)
                raise
    logger.info("🎉 RAG种子知识库全部导入任务完成")


def main():
    parser = argparse.ArgumentParser(description="Sprint0 RAG种子知识库导入CLI工具")
    parser.add_argument("--config", required=True, help="seed_ingest_list.yaml配置文件路径")
    args = parser.parse_args()
    run_ingest(args.config)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    main()

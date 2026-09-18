"""
app/rag/ingest.py
Sprint-0 v1.4 RAG 种子知识库导入 CLI

v1.1 变更（M3）：
- 改用 process_markdown_file_v2
- KnowledgeChunk.meta_json 的 source_section 使用 chunk 级 section
- 新增 is_continuation 标记

v1.2 变更（豆包优化 2）：
- 优化 2：日志打印 section_count dict

v1.3 变更（豆包阻断 BUG1）：
- BUG1：CHUNK_OVERLAP 常量加注释，说明 v1.3 起语义切片不支持 overlap

v1.4：无代码变更（保留 v1.3 状态）

CLI 入口：
    python -m app.rag.ingest --config ./data/knowledge/seed_ingest_list.yaml
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
from app.rag.chunking import process_markdown_file_v2

logger = logging.getLogger(__name__)

# Sprint0 v1.1 切片参数（M2：调大以匹配段落粒度）
CHUNK_SIZE = 800

# ★ BUG1 修复：v1.3 起语义切片不支持 overlap
# 此常量保留仅用于文档追溯；实际调用 process_markdown_file_v2 时已忽略
CHUNK_OVERLAP = 0

EMBEDDING_MODEL_NAME = settings.openai_embedding_model


def get_embedding_client():
    """根据 settings.rag_embedding_backend 返回 embedding 实例"""
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


def scan_case_evidence(root_dir: str, glob_pattern: str) -> List[pathlib.Path]:
    """扫描 B-Case 目录下 evidence/*.md"""
    root = pathlib.Path(root_dir)
    return list(root.glob(glob_pattern))


def ingest_single_file(db: Session, file_path: pathlib.Path, embeddings):
    logger.info(f"开始处理文件：{file_path.as_posix()}")
    meta, raw_content, chunks = process_markdown_file_v2(
        file_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,   # v1.3 起函数内部忽略此参数
    )

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
        version="v1.0",
    )
    db.add(doc_entity)
    db.flush()
    doc_id = doc_entity.id

    section_count: dict[str, int] = {}
    for chunk in chunks:
        emb_vec = embeddings.embed_query(chunk.text)
        section_count[chunk.section] = section_count.get(chunk.section, 0) + 1

        chunk_entity = KnowledgeChunk(
            knowledge_doc_id=doc_id,
            chunk_text=chunk.text,
            embedding=emb_vec,
            meta_json={
                "source_section": chunk.section,
                "is_continuation": chunk.is_continuation,
                "part_numbers": meta["part_numbers"],
                "related_rule_ids": meta["related_rule_ids"],
                "doc_source_section": meta["source_section"],
            },
        )
        db.add(chunk_entity)

    db.commit()

    # === 优化 2：日志打印 section_count dict ===
    logger.info(
        f"✅ {file_path.name} 导入完成 doc_id={doc_id}, "
        f"chunk数量={len(chunks)}, sections={section_count}"
    )


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
            logger.info(f"case_evidence_scan 扫描到 {len(found_files)} 个 md 文件")

    logger.info(f"待导入总文件数：{len(file_list)}")
    with get_db_session() as db:
        for fpath in file_list:
            try:
                ingest_single_file(db, fpath, embeddings)
            except Exception as exc:
                logger.error(f"导入失败 {fpath.as_posix()} : {exc}", exc_info=True)
                raise
    logger.info("🎉 RAG 种子知识库全部导入任务完成")


def main():
    parser = argparse.ArgumentParser(description="Sprint0 RAG 种子知识库导入 CLI 工具")
    parser.add_argument("--config", required=True, help="seed_ingest_list.yaml 配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只解析切片，不入库")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(args.config)
    else:
        run_ingest(args.config)


def _dry_run(config_path: str):
    """M2 切片验证：只打印切片结果，不触网、不写库"""
    cfg_file = pathlib.Path(config_path)
    cfg_raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    ingest_cfg = IngestConfig.model_validate(cfg_raw)

    for entry in ingest_cfg.ingest_entries:
        fp = pathlib.Path(entry.file)
        if not fp.exists():
            print(f"[SKIP] {fp} 不存在")
            continue
        meta, raw, chunks = process_markdown_file_v2(fp, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"\n=== {fp.name} ({meta['source_type']}) ===")
        print(f"文档级 source_section: {meta['source_section']}")
        for i, c in enumerate(chunks):
            cont = " [continuation]" if c.is_continuation else ""
            preview = c.text[:60].replace("\n", " ")
            print(f"  chunk[{i}] section='{c.section}'{cont} len={len(c.text)} | {preview}...")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    main()
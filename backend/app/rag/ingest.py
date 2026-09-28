"""
app/rag/ingest.py
Sprint-0 v1.5 RAG 种子知识库导入 CLI

v1.1 变更（M3）：
- 改用 process_markdown_file_v2
- KnowledgeChunk.meta_json 的 source_section 使用 chunk 级 section
- 新增 is_continuation 标记

v1.2 变更（豆包优化 2）：
- 优化 2：日志打印 section_count dict

v1.3 变更（豆包阻断 BUG1）：
- BUG1：CHUNK_OVERLAP 常量加注释，说明 v1.3 起语义切片不支持 overlap

v1.4：
- 无代码变更（保留 v1.3 状态）

v1.5 变更：
- BUG 修复：ingest_single_file 用 embed_documents 替代 embed_query，并批量调用
- ingest_single_file：embedding 返回数量校验
- run_ingest：case_evidence_scan 收集阶段逐个 exists() 检查
- _dry_run：对齐 run_ingest，支持 case_evidence_scan，收集阶段 exists() 检查，单文件异常隔离
- _dry_run：[ERROR] 打印附带异常类型名

CLI 入口（容器内，data/ 挂载到 /data）：
    python -m app.rag.ingest --config /data/knowledge/seed_ingest_list.yaml
    python -m app.rag.ingest --config /data/knowledge/seed_ingest_list.yaml --dry-run
"""
import argparse
import logging
import pathlib
import sys
from typing import List

import yaml

from langchain_core.embeddings import Embeddings
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


def get_embedding_client() -> Embeddings:
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


"""
【输入】db, file_path, embeddings
    ↓
日志打印：开始处理文件路径
    ↓
调用 process_markdown_file_v2(file_path, chunk_size, chunk_overlap)
    ├─meta：markdown头部yaml元信息 dict
    ├─raw_content：markdown全文文本 str
    └─chunks：List[Chunk]，经过切片+碎片抢救后的Chunk对象列表
    ↓
组装 KnowledgeDoc（文档主记录ORM对象）
    字段来源：meta、file_path、settings全局配置
    ↓
db.add(doc_entity)
    → 加入session待写入缓存，尚未写入数据库
    ↓
db.flush()
    → 执行SQL插入KnowledgeDoc；数据库生成id并回填到doc_entity
    ↓
doc_id = doc_entity.id  // 获取文档主键，作为分片外键
    ↓
统计 section_count（遍历 chunks，累计每个 section 的 chunk 数）
    ↓
批量 embedding：emb_vecs = embeddings.embed_documents([c.text for c in chunks])
    → 一次网络请求，返回与 chunks 等长的向量列表
    ↓
校验 len(emb_vecs) == len(chunks)，不等则抛 RuntimeError
    ↓
for chunk, emb_vec in zip(chunks, emb_vecs):  // 配对遍历
    ├─组装 KnowledgeChunk（分片ORM对象），外键 knowledge_doc_id 绑定 doc_id
    └─db.add(chunk_entity)  // 加入 session 缓存，暂不提交
循环结束，所有 chunk 全部 add 完成
    ↓
db.commit()
    → 一次性提交事务，将doc_entity + 全部chunk_entity持久写入数据库
    ↓
日志打印：导入完成，doc_id、chunk总数、各section统计
【结束】
"""


def ingest_single_file(db: Session, file_path: pathlib.Path, embeddings: Embeddings):
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

    # 统计 section 分布（用于日志）
    section_count: dict[str, int] = {}
    for chunk in chunks:
        section_count[chunk.section] = section_count.get(chunk.section, 0) + 1

    # ★ BUG 修复：用 embed_documents（文档侧），而非 embed_query（查询侧）
    # 并且批量调用，减少网络往返
    chunk_texts = [c.text for c in chunks]
    emb_vecs = embeddings.embed_documents(chunk_texts)

    if len(emb_vecs) != len(chunks):
        raise RuntimeError(
            f"embedding 返回数量不匹配：期望 {len(chunks)}，实际 {len(emb_vecs)}"
        )

    for chunk, emb_vec in zip(chunks, emb_vecs):
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


"""
入参：/data/knowledge/seed_ingest_list.yaml
    ↓
读取yaml → cfg_raw
    ↓
Pydantic校验得到ingest_cfg
    ↓
创建embeddings客户端
    ↓
file_list = []
    ↓
循环 ingest_entries 条目
    文件存在 → append进file_list
    文件缺失 → warning日志，continue跳过
    ↓
若 case_evidence_scan 非空：
    glob 扫描 → 逐个 exists() 检查 → 有效的追加到 file_list
    ↓
输出：file_list，等待后续循环调用ingest_single_file入库
"""


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
            # 逐个检查存在性，与 ingest_entries 一致
            valid_found: List[pathlib.Path] = []
            for fp in found_files:
                if not fp.exists():
                    logger.warning(f"文件不存在跳过：{fp.as_posix()}")
                    continue
                valid_found.append(fp)
            file_list.extend(valid_found)
            logger.info(
                f"case_evidence_scan 扫描到 {len(found_files)} 个 md 文件，"
                f"其中 {len(valid_found)} 个有效"
            )

    logger.info(f"待导入总文件数：{len(file_list)}")
    with get_db_session() as db:
        for fpath in file_list:
            try:
                ingest_single_file(db, fpath, embeddings)
            except Exception as exc:
                logger.error(f"导入失败 {fpath.as_posix()} : {exc}", exc_info=True)
                raise
    logger.info("🎉 RAG 种子知识库全部导入任务完成")


"""
命令行输入参数
    ↓
argparse解析参数得到 args（config路径 + dry_run布尔标记）
    ↓
if args.dry_run == True:
    调用 _dry_run(config_path) → 预览模式，解析+切片，无数据库操作
else:
    调用 run_ingest(config_path) → 正式模式，解析yaml、收集file_list
        ↓（后续业务代码循环 file_list，调用ingest_single_file 真正入库）
"""


def main():
    parser = argparse.ArgumentParser(description="Sprint0 RAG 种子知识库导入 CLI 工具")
    parser.add_argument("--config", required=True, help="seed_ingest_list.yaml 配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只解析切片，不入库")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(args.config)
    else:
        run_ingest(args.config)


"""
输入 config_path
    ↓
读取yaml配置文件 → cfg_raw
    ↓
IngestConfig.model_validate 配置校验
    ↓
先收集全部待处理文件到 file_list：
    ingest_entries 条目逐个 exists() 检查，存在的加入
    case_evidence_scan 非空时 glob 扫描，逐个 exists() 检查，有效的追加
    ↓
打印待校验总文件数
    ↓
遍历 file_list：
    ├ 文件不存在（收集后、处理前被删）→ [ERROR] 打印异常类型名，continue
    ├ 文件存在
    │   ↓
    │   调用 process_markdown_file_v2()：md解析、切片、碎片抢救
    │   ↓
    │   打印文件名、source_section
    │   ↓
    │   遍历 chunks，打印 chunk 序号、section、continuation 标记、长度、文本预览
    └ 异常被 try/except 捕获，continue 下一个文件
    ↓
结束；无数据库、无向量网络请求
"""


def _dry_run(config_path: str):
    """M2 切片验证：只打印切片结果，不触网、不写库

    与 run_ingest 对齐：
        1. 同时处理 ingest_entries 与 case_evidence_scan
        2. 收集阶段统一做 exists() 检查
        3. 单文件异常隔离，不中断整体

    注意：
        - 本函数只打印，不返回数据
        - 所有输出走 stdout，不依赖 logging 配置
    """
    cfg_file = pathlib.Path(config_path)
    cfg_raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    ingest_cfg = IngestConfig.model_validate(cfg_raw)

    # ========== 收集全部待处理文件（与 run_ingest 对齐） ==========
    file_list: List[pathlib.Path] = []

    # 收集 ingest_entries 中的文件（不存在则跳过）
    for entry in ingest_cfg.ingest_entries:
        fp = pathlib.Path(entry.file)
        if not fp.exists():
            print(f"[SKIP] {fp.as_posix()} 文件不存在")
            continue
        file_list.append(fp)

    # 如果配置开启 case_evidence_scan，则 glob 扫描追加文件
    if ingest_cfg.case_evidence_scan:
        scan_root = ingest_cfg.case_evidence_scan.get("root_dir")
        glob_pat = ingest_cfg.case_evidence_scan.get("glob_pattern")
        if scan_root and glob_pat:
            found_files = scan_case_evidence(scan_root, glob_pat)
            # 逐个检查存在性，与 ingest_entries 一致
            valid_found: List[pathlib.Path] = []
            for fp in found_files:
                if not fp.exists():
                    print(f"[SKIP] {fp.as_posix()} 文件不存在")
                    continue
                valid_found.append(fp)
            file_list.extend(valid_found)
            print(f"[INFO] case_evidence_scan 扫描到 {len(found_files)} 个 md 文件，"
                  f"其中 {len(valid_found)} 个有效")
    # ==============================================================

    print(f"[INFO] dry-run 待校验总文件数：{len(file_list)}")

    for fp in file_list:
        # ========== 单文件异常隔离，不中断整体 dry-run ==========
        try:
            meta, raw, chunks = process_markdown_file_v2(fp, CHUNK_SIZE, CHUNK_OVERLAP)
        except Exception as e:
            print(f"[ERROR] {fp.as_posix()} 处理失败：{type(e).__name__}: {e}")
            continue
        # =======================================================

        print(f"\n=== {fp.name} ({meta['source_type']}) ===")
        print(f"文档级 source_section: {meta['source_section']}")
        for i, c in enumerate(chunks):
            cont = " [continuation]" if c.is_continuation else ""
            preview = c.text[:60].replace("\n", " ")
            print(f"  chunk[{i}] section='{c.section}'{cont} len={len(c.text)} | {preview}...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    main()
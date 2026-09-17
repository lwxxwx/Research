"""
tests/test_rag_seed.py
Sprint‑0 Phase‑H RAG Seed Knowledge单元&集成测试
约束：
1. CI流水线默认使用RAG_EMBEDDING_BACKEND=fake，不调用任何外网OpenAI接口，无token消耗
2. 纯解析逻辑为普通pytest用例；涉及数据库读写标记 @pytest.mark.integration
3. 标记integration的用例需要pgvector数据库，CI可通过 -m "not integration" 跳过
4. Sprint‑0：Retriever仅内部Python API，不接入LangGraph，不暴露HTTP接口
"""
import json
import pathlib
import tempfile
import pytest
from sqlalchemy.orm import Session
from app.services.benchmark_service import _evidence_is_complete, EVIDENCE_MIN_LEN
from app.rag.knowledge import RetrievalResult
from app.rag.chunking import (
    parse_markdown_frontmatter,
    validate_frontmatter,
    split_text_chunk,
    process_markdown_file
)
from app.rag.sources import DocumentType
from app.rag.ingest import get_embedding_client
from app.rag.retriever import Retriever
from app.persistence.db import get_db_session
from app.persistence.models import KnowledgeDoc, KnowledgeChunk


def test_parse_markdown_frontmatter_normal():
    md_content = """---
source_type: datasheet
source_title: STC89C55RC datasheet
source_section: Oscillator Circuit 晶振电路设计
part_numbers: ["STC89C55RC"]
related_rule_ids: ["MCU_001"]
---
STC89C55RC使用片内振荡器，XTAL1、XTAL2引脚外接石英晶振Y1。
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as fp:
        fp.write(md_content)
        tmp_path = pathlib.Path(fp.name)
    try:
        meta, content = parse_markdown_frontmatter(tmp_path)
        assert meta["source_type"] == "datasheet"
        assert meta["source_title"] == "STC89C55RC datasheet"
        assert meta["part_numbers"] == ["STC89C55RC"]
        assert meta["related_rule_ids"] == ["MCU_001"]
        assert "STC89C55RC使用片内振荡器" in content
    finally:
        tmp_path.unlink(missing_ok=True)


def test_parse_markdown_frontmatter_no_frontmatter():
    md_content = "just plain text without front matter"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as fp:
        fp.write(md_content)
        tmp_path = pathlib.Path(fp.name)
    try:
        with pytest.raises(ValueError, match="缺失YAML frontmatter"):
            parse_markdown_frontmatter(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_validate_frontmatter_ok():
    meta = {
        "source_type": DocumentType.DATASHEET,
        "source_title": "test datasheet",
        "source_section": "chapter 2",
        "part_numbers": ["STC89C55RC"],
        "related_rule_ids": ["MCU_001"]
    }
    validate_frontmatter(meta)


def test_validate_frontmatter_bad_source_type():
    meta = {
        "source_type": "wrong_type",
        "source_title": "test datasheet",
        "source_section": "chapter 2",
        "part_numbers": ["STC89C55RC"],
        "related_rule_ids": ["MCU_001"]
    }
    with pytest.raises(ValueError, match="非法source_type"):
        validate_frontmatter(meta)


def test_split_text_chunk_basic():
    raw_text = "AAA BBB CCC DDD EEE FFF GGG HHH III JJJ"
    chunks = split_text_chunk(raw_text, chunk_size=10, chunk_overlap=3)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_retrieval_result_convert_to_evidence_pass():
    """RetrievalResult构造rag_ref证据对象，可以通过benchmark证据完整性校验"""
    res = RetrievalResult(
        source_type="datasheet",
        source_title="STC89C55RC datasheet",
        source_section="Oscillator Circuit 晶振电路设计",
        snippet="STC89C55RC使用片内振荡器，XTAL1、XTAL2引脚外接石英晶振Y1。典型晶振规格11.0592MHz，两侧匹配电容C1、C2取值22pF。",
        score=0.11,
        part_numbers=["STC89C55RC"],
        related_rule_ids=["MCU_001"]
    )
    ev_dict = {
        "type": "rag_ref",
        "source": res.source_title,
        "section": res.source_section,
        "reason": res.snippet
    }
    assert _evidence_is_complete(ev_dict) is True


def test_retrieval_result_snippet_too_short_fail():
    res = RetrievalResult(
        source_type="datasheet",
        source_title="STC89C55RC datasheet",
        source_section="Oscillator Circuit",
        snippet="hi",
        score=0.3,
        part_numbers=["STC89C55RC"],
        related_rule_ids=["MCU_001"]
    )
    ev_dict = {
        "type": "rag_ref",
        "source": res.source_title,
        "section": res.source_section,
        "reason": res.snippet
    }
    assert _evidence_is_complete(ev_dict) is False
    assert len(res.snippet) < EVIDENCE_MIN_LEN


def test_process_markdown_file_content_too_short():
    md_content = """---
source_type: datasheet
source_title: short test
source_section: sec1
part_numbers: ["U1"]
related_rule_ids: ["R001"]
---
abc
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as fp:
        fp.write(md_content)
        tmp_path = pathlib.Path(fp.name)
    try:
        with pytest.raises(ValueError, match="小于最小阈值"):
            process_markdown_file(tmp_path, chunk_size=100, chunk_overlap=10)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_get_embedding_client_fake_mode():
    """测试embedding工厂函数，fake模式输出1536维向量，无需外网"""
    emb = get_embedding_client()
    vec = emb.embed_query("测试输入文本")
    assert len(vec) == 1536, "fake模式向量维度必须等于1536，适配pgvector(1536)"


def test_retriever_instance_create():
    """测试Retriever实例化，fake模式，不访问网络"""
    ret = Retriever()
    assert hasattr(ret, "_embed_client")


@pytest.mark.integration
def test_retriever_basic_query():
    """
    集成测试：调用Retriever.retrieve，依赖已经ingest导入的5条种子知识库
    注意：需要先执行ingest把5份md写入pg；CI可使用 -m "not integration" 跳过本用例
    """
    ret = Retriever()
    results = ret.retrieve(
        query="P0端口上拉电阻",
        part_numbers=["STC89C55RC"],
        top_k=2
    )
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, RetrievalResult)
        assert "STC89C55RC" in item.part_numbers


@pytest.mark.integration
def test_knowledge_table_has_seed_data():
    """集成测试：验证数据库存在Phase‑H导入的种子知识库记录"""
    with get_db_session() as db:
        doc_count = db.query(KnowledgeDoc).count()
        chunk_count = db.query(KnowledgeChunk).count()
        assert doc_count >= 5, f"期望至少5条knowledge_doc，实际{doc_count}"
        assert chunk_count >= 5, f"期望至少5条knowledge_chunk，实际{chunk_count}"

        sample_doc = db.query(KnowledgeDoc).first()
        assert sample_doc.meta_json is not None
        doc_meta = dict(sample_doc.meta_json.items())
        assert doc_meta.get("embedding_provider") == "fake"



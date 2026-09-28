"""
tests/test_rag_seed.py
Sprint-0 Phase-H v1.8 RAG Seed Knowledge 测试

v1.1 新增：
- 语义切片边界测试（M2）
- chunk 级 source_section 测试（M3）
- Retriever dict 兼容测试（M5）

v1.2 新增（豆包优化 3 / 5）：
- 优化 3：FakeMapping 补 __len__
- 优化 5：RetrievalResult.is_continuation 单测与集成测试

v1.3 新增（豆包阻断 BUG1 / 2 / 3）：
- BUG1：验证 split_text_chunk_v2 不再支持 overlap
- BUG2：验证单段超长时字符硬切 + is_continuation 判定
- BUG3：验证 _as_dict 对非 dict Mapping 的兼容

v1.4 新增（豆包新优化 2 / 3 / 6）：
- 优化 2：集成测试统一用 _as_dict()；补 is_continuation 布尔断言
- 优化 3：pytest.raises match 恢复精确字符串匹配（含空格）
- 优化 6：test_split_v2_paragraph_aggregation 断言加 strip()

v1.5 新增（豆包新优化 3 / 4 / 5）：
- 优化 3：新增 test_hard_split_choose_closest_punct_near_limit 等 3 条硬切边界测试
- 优化 4：删除未使用的 import json
- 优化 5：新增 test_split_v2_long_code_block_not_cut（超长代码块不切割）

v1.6 新增（豆包轻微缺陷 1 / 2）：
- 轻微缺陷 1：test_hard_split_choose_closest_punct_with_multiple 补强切点断言
- 轻微缺陷 2：test_split_v2_single_long_paragraph 的 +50 余量加注释

v1.7 变更（豆包方案 A）：
- 方案 A.1：test_hard_split_choose_closest_punct_with_multiple 改用字符串全等断言
- 方案 A.2：新增 HARD_SPLIT_TOLERANCE 常量（=50），替代 +50 魔法数字
- 方案 A.3：同步更新文件头部 v1.7 变更注释

v1.8 变更（并入逻辑集成验证 + ingest 回归）：
- 新增 test_process_v2_merges_short_tail_chunk 等 3 个并入逻辑集成测试
- 新增 test_ingest_uses_embed_documents_not_embed_query 回归测试
- test_split_v2_single_long_paragraph / test_split_v2_hard_split_section_preserved
  加 monkeypatch 禁用并入，与全局 EVIDENCE_MIN_LEN 解耦

约束：
1. CI 默认 RAG_EMBEDDING_BACKEND=fake，不触网
2. 纯解析逻辑为 pytest 用例；DB 读写标记 @pytest.mark.integration
"""
import pathlib
import tempfile

import pytest

from app.services.benchmark_service import _evidence_is_complete, EVIDENCE_MIN_LEN
from app.rag.knowledge import RetrievalResult
from app.rag.chunking import (
    parse_markdown_frontmatter,
    validate_frontmatter,
    split_text_chunk,
    split_text_chunk_v2,
    process_markdown_file,
    process_markdown_file_v2,
    Chunk,
    _hard_split,
)
from app.rag.sources import DocumentType
from app.rag.ingest import get_embedding_client
from app.rag.retriever import Retriever, _as_dict
from app.persistence.db import get_db_session
from app.persistence.models import KnowledgeDoc, KnowledgeChunk


# ============================================================
# ★ 方案 A.2（v1.7）：测试用容差常量
# ============================================================
# _hard_split 切点可能因句子边界回退而略超 limit；
# 当前实现理论 <= limit（cut_pos ∈ [limit-99, limit]），保留此容差以适配未来调整
HARD_SPLIT_TOLERANCE = 50


# ============================================================
# v1.0 兼容测试（保持原有行为）
# ============================================================
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
        # ★ 优化 3（v1.4）：恢复精确字符串匹配（含空格，与业务文案一致）
        with pytest.raises(ValueError, match="文件缺失 YAML frontmatter"):
            parse_markdown_frontmatter(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_validate_frontmatter_ok():
    meta = {
        "source_type": DocumentType.DATASHEET,
        "source_title": "test datasheet",
        "source_section": "chapter 2",
        "part_numbers": ["STC89C55RC"],
        "related_rule_ids": ["MCU_001"],
    }
    validate_frontmatter(meta)


def test_validate_frontmatter_bad_source_type():
    meta = {
        "source_type": "wrong_type",
        "source_title": "test datasheet",
        "source_section": "chapter 2",
        "part_numbers": ["STC89C55RC"],
        "related_rule_ids": ["MCU_001"],
    }
    # ★ 优化 3（v1.4）：恢复精确字符串匹配（含空格）
    with pytest.raises(ValueError, match="非法 source_type"):
        validate_frontmatter(meta)


def test_split_text_chunk_basic():
    raw_text = "AAA BBB CCC DDD EEE FFF GGG HHH III JJJ"
    with pytest.warns(DeprecationWarning, match="split_text_chunk is deprecated"):
        chunks = split_text_chunk(raw_text, chunk_size=10, chunk_overlap=3)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_retrieval_result_convert_to_evidence_pass():
    res = RetrievalResult(
        source_type="datasheet",
        source_title="STC89C55RC datasheet",
        source_section="Oscillator Circuit 晶振电路设计",
        snippet="STC89C55RC使用片内振荡器，XTAL1、XTAL2引脚外接石英晶振Y1。"
                "典型晶振规格11.0592MHz，两侧匹配电容C1、C2取值22pF。",
        score=0.11,
        part_numbers=["STC89C55RC"],
        related_rule_ids=["MCU_001"],
    )
    ev_dict = {
        "type": "rag_ref",
        "source": res.source_title,
        "section": res.source_section,
        "reason": res.snippet,
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
        related_rule_ids=["MCU_001"],
    )
    ev_dict = {
        "type": "rag_ref",
        "source": res.source_title,
        "section": res.source_section,
        "reason": res.snippet,
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
        with pytest.warns(DeprecationWarning, match="process_markdown_file is deprecated"):
            with pytest.raises(ValueError, match="小于最小阈值"):
                process_markdown_file(tmp_path, chunk_size=100, chunk_overlap=10)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_get_embedding_client_fake_mode():
    emb = get_embedding_client()
    vec = emb.embed_query("测试输入文本")
    assert len(vec) == 1536


def test_retriever_instance_create():
    ret = Retriever()
    assert hasattr(ret, "_embed_client")


# ============================================================
# v1.1 新增测试（M2 语义切片）
# ============================================================
def _make_md(text: str) -> pathlib.Path:
    """辅助：写临时 md 文件（含 frontmatter）"""
    content = f"""---
source_type: datasheet
source_title: test doc
source_section: doc level section
part_numbers: ["U1"]
related_rule_ids: ["R001"]
---
{text}
"""
    fp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    fp.write(content)
    fp.close()
    return pathlib.Path(fp.name)


def test_split_v2_h2_sections():
    """M2/M3：按 ## 切分，每个 chunk 携带独立 section"""
    md = """## Section A
段落 A1。

段落 A2。

## Section B
段落 B1。
"""
    chunks = split_text_chunk_v2(md, chunk_size=200)
    sections = [c.section for c in chunks]
    assert "Section A" in sections
    assert "Section B" in sections
    section_a_chunks = [c for c in chunks if c.section == "Section A"]
    section_b_chunks = [c for c in chunks if c.section == "Section B"]
    assert len(section_a_chunks) == 1
    assert len(section_b_chunks) == 1


def test_split_v2_paragraph_aggregation():
    """
    M2：超长章节按段落聚合，不切断句子

    ★ 优化 6（v1.4）：断言加 .strip()，容忍首尾空白
    """
    long_para = "这是一句完整的话。" * 30
    md = f"## Long Section\n{long_para}\n\n{long_para}\n\n{long_para}\n"
    chunks = split_text_chunk_v2(md, chunk_size=400)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.section == "Long Section"
        # ★ 优化 6：strip 后断言
        assert c.text.strip().endswith("。"), f"chunk 未以句号结尾: ...{c.text[-20:]}"


def test_split_v2_code_block_protected():
    """M2：含代码块时整段不切（宁可超长）"""
    code = "```\nline1\nline2\nline3\n```"
    md = f"## Code Section\n{code}\n" + ("填充文本。" * 200)
    chunks = split_text_chunk_v2(md, chunk_size=200)
    assert len(chunks) == 1


# ★ 优化 5（v1.5）：新增超长完整代码块不切割单测
def test_split_v2_long_code_block_not_cut():
    """
    M2 / 优化 5：超长完整代码块不切割

    场景：代码块总长 > chunk_size，但仍应作为 1 个 chunk 完整保留
    """
    code_lines = "\n".join([f"line_{i} = {i};" for i in range(50)])
    code_block = f"```c\n{code_lines}\n```"
    md = f"## Code Section\n{code_block}\n"

    chunks = split_text_chunk_v2(md, chunk_size=200)

    # 代码块完整保留：只有 1 个 chunk
    assert len(chunks) == 1
    # 代码块起止符完整
    assert chunks[0].text.startswith("```")
    assert chunks[0].text.endswith("```")
    # 应包含全部 50 行
    assert "line_0" in chunks[0].text
    assert "line_49" in chunks[0].text


def test_split_v2_chunk_section_independent():
    """M3：不同 chunk 的 section 字段独立"""
    md = """## Alpha
A 段。

## Beta
B 段。

## Gamma
C 段。
"""
    chunks = split_text_chunk_v2(md, chunk_size=200)
    assert {c.section for c in chunks} >= {"Alpha", "Beta", "Gamma"}


# ============================================================
# v1.2 新增测试（豆包优化 1：(intro) 继承 doc_section）
# ============================================================
def test_split_v2_intro_uses_doc_section():
    """优化 1：引言块 section 应继承文档级 doc_section"""
    md = """本文是引言段落，介绍文档整体背景。

## Section A
段落 A1 内容。
"""
    chunks = split_text_chunk_v2(
        md, chunk_size=200,
        doc_section="Oscillator Circuit 晶振电路设计",
    )
    intro_chunks = [c for c in chunks if "引言" in c.text]
    assert len(intro_chunks) == 1
    assert intro_chunks[0].section == "Oscillator Circuit 晶振电路设计"
    assert intro_chunks[0].section != "(intro)"


def test_split_v2_intro_fallback_when_no_doc_section():
    """优化 1 回退：doc_section 为空时，引言块 section 落回 '(intro)'"""
    md = """引言段落。

## Section A
内容。
"""
    chunks = split_text_chunk_v2(md, chunk_size=200, doc_section="")
    intro_chunks = [c for c in chunks if "引言" in c.text]
    assert len(intro_chunks) == 1
    assert intro_chunks[0].section == "(intro)"


def test_process_v2_returns_chunks_with_section():
    """v1.1 入口：返回 Chunk 对象，携带 section"""
    md = """## Oscillator
STC89C55RC 使用片内振荡器，XTAL1、XTAL2 引脚外接石英晶振 Y1。
典型晶振规格 11.0592MHz，两侧匹配电容 C1、C2 取值 22pF。
"""
    tmp = _make_md(md)
    try:
        meta, content, chunks = process_markdown_file_v2(tmp, chunk_size=500)
        assert meta["source_title"] == "test doc"
        assert len(chunks) == 1
        assert chunks[0].section == "Oscillator"
        assert isinstance(chunks[0], Chunk)
    finally:
        tmp.unlink(missing_ok=True)


# ============================================================
# v1.1 新增测试（M5 dict 兼容）
# ============================================================
def test_as_dict_none():
    assert _as_dict(None) == {}


def test_as_dict_dict():
    assert _as_dict({"a": 1}) == {"a": 1}


def test_as_dict_mapping_like():
    """优化 3：FakeMapping 补 __len__"""
    class FakeMapping:
        _data = {"k1": "v1", "k2": "v2"}

        def __iter__(self):
            return iter(self._data.items())

        def __getitem__(self, k):
            return self._data[k]

        def __len__(self):
            return len(self._data)

        def keys(self):
            return list(self._data.keys())

    m = FakeMapping()
    assert len(m) == 2
    assert _as_dict(m) == {"k1": "v1", "k2": "v2"}


# ============================================================
# v1.2 新增测试（豆包优化 5：is_continuation）
# ============================================================
def test_retrieval_result_is_continuation_default_false():
    """优化 5：RetrievalResult 默认 is_continuation=False"""
    res = RetrievalResult(
        source_type="datasheet",
        source_title="test",
        source_section="sec",
        snippet="abc" * 10,
        score=0.1,
    )
    assert res.is_continuation is False


def test_retrieval_result_is_continuation_explicit_true():
    """优化 5：显式设置 is_continuation=True"""
    res = RetrievalResult(
        source_type="datasheet",
        source_title="test",
        source_section="sec",
        snippet="abc" * 10,
        score=0.1,
        is_continuation=True,
    )
    assert res.is_continuation is True


# ============================================================
# v1.3 新增测试（豆包阻断 BUG2：字符硬切兜底）
# ============================================================
def test_hard_split_basic():
    """BUG2：_hard_split 基本功能"""
    text = "A" * 1000
    pieces = _hard_split(text, limit=300)
    assert len(pieces) >= 4
    for p in pieces:
        assert len(p) <= 300


def test_hard_split_prefers_sentence_boundary():
    """BUG2：硬切优先在句子边界（。！？；\\n）切"""
    text = "第一句话。" + "第二句话。" * 100
    pieces = _hard_split(text, limit=100)
    for p in pieces[:-1]:
        assert p.endswith("。"), f"片段未以句号结尾: ...{p[-20:]}"


def test_hard_split_short_text_no_split():
    """BUG2：短文本直接返回单元素"""
    text = "短文本"
    pieces = _hard_split(text, limit=300)
    assert len(pieces) == 1
    assert pieces[0] == text


# ★ 优化 3（v1.5）：新增 _hard_split 就近标点边界测试
def test_hard_split_choose_closest_punct_near_limit():
    """
    BUG2 / 优化 3：验证 _hard_split 在 limit 附近选「离 limit 最近」的标点

    构造：limit=20，位置 19 有 "。"
    期望：第一块切在位置 19 之后，即 "A"*19 + "。"
    """
    text = "A" * 19 + "。" + "B" * 50
    pieces = _hard_split(text, limit=20)
    assert pieces[0] == "A" * 19 + "。"


def test_hard_split_choose_closest_punct_with_multiple():
    """
    BUG2 / 优化 3：多标点场景，优先选最近 limit 的标点

    构造：limit=20，位置 15 有 "！"，位置 19 有 "。"
    期望：选位置 19 的 "。"（离 limit 更近），而非位置 15 的 "！"

    ★ 方案 A.1（v1.7）：改用字符串全等断言，同时锁定内容 + 长度 + 切点
    """
    text = "A" * 15 + "！" + "A" * 3 + "。" + "B" * 50
    pieces = _hard_split(text, limit=20)

    # ★ 方案 A.1：字符串全等断言（蕴含 len == 20 与末位标点类型）
    assert pieces[0] == "A" * 15 + "！" + "A" * 3 + "。"


def test_hard_split_falls_back_when_no_punct_within_window():
    """
    BUG2 / 优化 3：窗口内无标点时按 limit 硬切
    """
    text = "A" * 500
    pieces = _hard_split(text, limit=20)
    assert pieces[0] == "A" * 20


def test_split_v2_single_long_paragraph(monkeypatch):
    """
    BUG2：单段超长时字符硬切兜底

    ★ 方案 A.2（v1.7）：+50 抽为 HARD_SPLIT_TOLERANCE 常量
    ★ v1.8：加 monkeypatch 禁用并入逻辑，稳定断言，与全局阈值解耦
    """
    import app.rag.chunking as chunking_module
    # 临时禁用并入：EVIDENCE_MIN_LEN=0，尾块长度 > 0 永不触发并入
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 0)

    long_para = "这是一句完整的话。" * 200
    md = f"## Long Section\n{long_para}\n"
    chunks = split_text_chunk_v2(md, chunk_size=400)

    assert len(chunks) >= 4
    for c in chunks:
        # ★ 方案 A.2：用常量替代魔法数字
        #   - 当前 _hard_split 保证 cut_pos ∈ [limit-99, limit]，每块 <= limit
        #   - HARD_SPLIT_TOLERANCE 为防御性余量
        assert len(c.text) <= 400 + HARD_SPLIT_TOLERANCE, f"chunk 超长: {len(c.text)}"
    assert chunks[0].is_continuation is False
    for c in chunks[1:]:
        assert c.is_continuation is True, f"续块 is_continuation 应为 True"


def test_split_v2_hard_split_section_preserved(monkeypatch):
    """
    BUG2：硬切产生的所有块沿用同一 section

    ★ v1.8：加 monkeypatch 禁用并入，稳定断言
    """
    import app.rag.chunking as chunking_module
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 0)

    long_para = "这是一句完整的话。" * 200
    md = f"## Long Section\n{long_para}\n"
    chunks = split_text_chunk_v2(md, chunk_size=400)
    for c in chunks:
        assert c.section == "Long Section"


# ============================================================
# v1.3 新增测试（豆包阻断 BUG3：_as_dict 增强）
# ============================================================
def test_as_dict_object_with_items():
    """BUG3：非 dict 但有 .items() 的对象应走 .items() 路径"""
    class ObjectWithItems:
        def items(self):
            return [("k1", "v1"), ("k2", "v2")]

    m = ObjectWithItems()
    assert _as_dict(m) == {"k1": "v1", "k2": "v2"}


def test_as_dict_unsupported_type_warns(caplog):
    """BUG3：不可转换的类型应输出 WARNING"""
    from app.rag import retriever as retriever_module

    retriever_module._warned_types.clear()

    class WeirdType:
        pass

    with caplog.at_level("WARNING"):
        result = _as_dict(WeirdType())

    assert result == {}
    assert any("无法将类型" in rec.message for rec in caplog.records)


def test_as_dict_warning_deduplicated(caplog):
    """BUG3：同类型只 warn 一次（限流）"""
    from app.rag import retriever as retriever_module

    retriever_module._warned_types.clear()

    class AnotherWeirdType:
        pass

    with caplog.at_level("WARNING"):
        _as_dict(AnotherWeirdType())
        _as_dict(AnotherWeirdType())
        _as_dict(AnotherWeirdType())

    warnings = [r for r in caplog.records if "无法将类型" in r.message]
    assert len(warnings) == 1


# ============================================================
# v1.3 新增测试（豆包阻断 BUG1：overlap 不支持）
# ============================================================
def test_split_v2_no_overlap_cross_section():
    """BUG1：语义切片不再支持 overlap，验证跨 section 无污染"""
    md = """## Section A
AAA 内容。

## Section B
BBB 内容。
"""
    chunks = split_text_chunk_v2(md, chunk_size=100)

    section_a_chunks = [c for c in chunks if c.section == "Section A"]
    section_b_chunks = [c for c in chunks if c.section == "Section B"]

    for c in section_a_chunks:
        assert "AAA" in c.text
        assert "BBB" not in c.text
    for c in section_b_chunks:
        assert "BBB" in c.text
        assert "AAA" not in c.text


# ============================================================
# v1.8 新增测试（并入前一块逻辑的集成验证）
# ============================================================
def test_process_v2_merges_short_tail_chunk(monkeypatch):
    """
    v1.5 并入逻辑集成验证：
      通过 process_markdown_file_v2 入口，验证尾块 < EVIDENCE_MIN_LEN 时被并入前一块。

    构造：
      - 段落1：290 字符（接近 chunk_size=300）
      - 段落2：5 字符（远 < EVIDENCE_MIN_LEN=10）
    临时把 EVIDENCE_MIN_LEN 调到 20，触发并入。

    预期：
      - 尾块 5 < 20 被并入前一块
      - 最终 1 个 chunk，长度 290 + 2 + 5 = 297
    """
    import app.rag.chunking as chunking_module
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 20)

    para1 = "甲" * 290
    para2 = "乙" * 5
    md = f"## Merge Test\n{para1}\n\n{para2}\n"

    tmp = _make_md(md)
    try:
        _meta, _content, chunks = process_markdown_file_v2(tmp, chunk_size=300)
        # 并入后只有 1 块
        assert len(chunks) == 1, (
            f"尾块 5 < 20 应并入前一块，实际 {len(chunks)} 块"
        )
        # 长度 = 290 + 2 + 5 = 297
        assert len(chunks[0].text) == 297, (
            f"并入后长度应为 297，实际 {len(chunks[0].text)}"
        )
        assert chunks[0].section == "Merge Test"
        assert chunks[0].is_continuation is False
    finally:
        tmp.unlink(missing_ok=True)


def test_process_v2_no_merge_when_tail_above_threshold(monkeypatch):
    """
    v1.5 并入逻辑边界：
      尾块 >= EVIDENCE_MIN_LEN 时不并入。

    构造：
      - 段落1：290 字符
      - 段落2：15 字符
    临时把 EVIDENCE_MIN_LEN 调到 10，尾块 15 >= 10，不并入。

    预期：保持 2 块。
    """
    import app.rag.chunking as chunking_module
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 10)

    para1 = "甲" * 290
    para2 = "乙" * 15
    md = f"## No Merge Test\n{para1}\n\n{para2}\n"

    tmp = _make_md(md)
    try:
        _meta, _content, chunks = process_markdown_file_v2(tmp, chunk_size=300)
        assert len(chunks) == 2, (
            f"尾块 15 >= 10 不应并入，实际 {len(chunks)} 块"
        )
        assert chunks[0].section == "No Merge Test"
        assert chunks[1].section == "No Merge Test"
        # 尾块标记续块
        assert chunks[0].is_continuation is False
        assert chunks[1].is_continuation is True
    finally:
        tmp.unlink(missing_ok=True)


def test_process_v2_merge_only_within_same_section(monkeypatch):
    """
    v1.5 并入逻辑边界：不同 section 不合并。

    构造：
      - Section A：280 字符
      - Section B：50 字符（独立 section，长度 >= 阈值）
    临时把 EVIDENCE_MIN_LEN 调到 20。

    预期：Section A 1 块（280 >= 20）；Section B 1 块（50 >= 20）；
         两块 section 不同，不合并。
    """
    import app.rag.chunking as chunking_module
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 20)

    md = (
        f"## Section A\n{'A' * 280}\n\n"
        f"## Section B\n{'B' * 50}\n"
    )

    tmp = _make_md(md)
    try:
        _meta, _content, chunks = process_markdown_file_v2(tmp, chunk_size=300)
        a_chunks = [c for c in chunks if c.section == "Section A"]
        b_chunks = [c for c in chunks if c.section == "Section B"]
        assert len(a_chunks) == 1
        assert len(b_chunks) == 1
        assert a_chunks[0].text == "A" * 280
        assert b_chunks[0].text == "B" * 50
    finally:
        tmp.unlink(missing_ok=True)


# ============================================================
# 集成测试（依赖 DB + 已 ingest）
# ============================================================
@pytest.mark.integration
def test_retriever_basic_query():
    ret = Retriever()
    results = ret.retrieve(query="P0端口上拉电阻", part_numbers=["STC89C55RC"], top_k=2)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, RetrievalResult)
        assert "STC89C55RC" in item.part_numbers


@pytest.mark.integration
def test_knowledge_table_has_seed_data():
    """★ 优化 2（v1.4）：统一用 _as_dict() 读取 meta_json"""
    with get_db_session() as db:
        doc_count = db.query(KnowledgeDoc).count()
        chunk_count = db.query(KnowledgeChunk).count()
        assert doc_count >= 5
        assert chunk_count >= 5

        sample_doc = db.query(KnowledgeDoc).first()
        meta = _as_dict(sample_doc.meta_json)
        assert meta.get("embedding_provider") == "fake"


@pytest.mark.integration
def test_chunk_meta_has_chunk_level_section():
    """
    M3：chunk 级 source_section 已落库
    ★ 优化 2（v1.4）：统一用 _as_dict()；补 is_continuation 布尔断言
    """
    with get_db_session() as db:
        chunks = db.query(KnowledgeChunk).limit(5).all()
        for c in chunks:
            meta = _as_dict(c.meta_json)
            assert "source_section" in meta
            assert len(meta["source_section"]) > 0
            assert "doc_source_section" in meta
            # ★ 优化 2：补 is_continuation 布尔断言
            assert "is_continuation" in meta
            assert isinstance(meta["is_continuation"], bool)


@pytest.mark.integration
def test_retriever_returns_chunk_level_section():
    """M3 + M4：检索结果的 source_section 来自 chunk 级"""
    ret = Retriever()
    results = ret.retrieve(query="复位电路", top_k=1)
    if results:
        assert results[0].source_section != ""


@pytest.mark.integration
def test_retriever_passes_is_continuation():
    """优化 5（v1.2）：检索结果携带 is_continuation，且类型为 bool"""
    ret = Retriever()
    results = ret.retrieve(query="晶振", top_k=5)
    for r in results:
        assert hasattr(r, "is_continuation")
        assert isinstance(r.is_continuation, bool)


@pytest.mark.integration
def test_ingest_uses_embed_documents_not_embed_query():
    """
    v1.5 BUG 修复回归：
      验证 ingest_single_file 用 embed_documents，而不是 embed_query。

    方法：
      - 用 spy 包装 embedding client，记录调用次数
      - 调用 ingest_single_file，写入一条测试数据
      - 断言 embed_documents 被调用、embed_query 未被调用

    注意：
      - 标记 integration，会写库（每次运行新增一条 doc/chunk）
      - 仅适用于 dev/test 环境
    """
    from app.rag import ingest as ingest_module

    md_content = """---
source_type: datasheet
source_title: embed method regression test
source_section: test section
part_numbers: ["T1"]
related_rule_ids: ["R1"]
---
这是一段足够长的测试正文，用于触发切片逻辑并验证 embedding 方法。
补充文字使其超过最小长度阈值，确保切片器产出至少一个有效 chunk。
"""
    tmp = _make_md(md_content)
    try:
        real_client = get_embedding_client()
        call_log = {"embed_documents": 0, "embed_query": 0}

        class SpyEmbedding:
            def embed_documents(self, texts):
                call_log["embed_documents"] += 1
                return real_client.embed_documents(texts)

            def embed_query(self, text):
                call_log["embed_query"] += 1
                return real_client.embed_query(text)

        with get_db_session() as db:
            ingest_module.ingest_single_file(db, tmp, SpyEmbedding())

        assert call_log["embed_documents"] >= 1, (
            "ingest_single_file 应调用 embed_documents"
        )
        assert call_log["embed_query"] == 0, (
            "ingest_single_file 不应调用 embed_query"
        )
    finally:
        tmp.unlink(missing_ok=True)
"""
tests/test_chunking.py
验证 process_markdown_file_v2 的切片行为 + _split_long_section 的并入逻辑。

运行：
    uv run pytest tests/test_chunking.py -v -s
"""
from __future__ import annotations

import pathlib

import pytest

# ★ 关键：导入 chunking 模块本身，便于 monkeypatch 其 EVIDENCE_MIN_LEN
import app.rag.chunking as chunking_module
from app.rag.chunking import (
    Chunk,
    _split_by_h2,
    _split_long_section,
    process_markdown_file_v2,
)
from app.services.benchmark_service import EVIDENCE_MIN_LEN


FIXTURE_MD = pathlib.Path(__file__).parent / "fixtures" / "test_sample.md"
CHUNK_SIZE = 300


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    """读取样例 md，调用 process_markdown_file_v2，返回切片结果。"""
    assert FIXTURE_MD.exists(), f"样例文件不存在: {FIXTURE_MD}"
    _meta, _content, chunk_list = process_markdown_file_v2(
        FIXTURE_MD,
        chunk_size=CHUNK_SIZE,
    )

    # ========== 打印切片结果，供人工核对 ==========
    print("\n" + "=" * 78)
    print(f"【切片结果】 EVIDENCE_MIN_LEN = {EVIDENCE_MIN_LEN}  "
          f"chunk_size = {CHUNK_SIZE}  总 chunk 数 = {len(chunk_list)}")
    print("=" * 78)
    for i, c in enumerate(chunk_list):
        preview = c.text[:70].replace("\n", "\\n")
        if len(c.text) > 70:
            preview += "..."
        print(
            f"[{i:02d}] section={c.section!r:<30} "
            f"len={len(c.text):>4} "
            f"is_continuation={c.is_continuation} "
            f"text={preview!r}"
        )
    print("=" * 78 + "\n")
    # =============================================

    return chunk_list


@pytest.fixture(scope="module")
def sections() -> list[tuple[str, str]]:
    """读取样例 md，按 ## 切分为 [(section_title, section_body), ...]。"""
    assert FIXTURE_MD.exists(), f"样例文件不存在: {FIXTURE_MD}"
    _meta, content = chunking_module.parse_markdown_frontmatter(FIXTURE_MD)
    return _split_by_h2(content.strip(), doc_section=_meta["source_section"])


# ============================================================
# 辅助函数
# ============================================================
def _split_all_sections(
    sections: list[tuple[str, str]],
    chunk_size: int,
) -> dict[str, list[int]]:
    """对每个 section 调用 _split_long_section，返回 {section: [len, ...]}。"""
    result: dict[str, list[int]] = {}
    for title, body in sections:
        if not body.strip():
            continue
        chunks = _split_long_section(title, body, chunk_size)
        result[title] = [len(c.text) for c in chunks]
    return result


# ============================================================
# 基础断言
# ============================================================
def test_all_chunks_meet_min_len(chunks: list[Chunk]) -> None:
    """所有 chunk 均达到最小证据长度。"""
    for i, c in enumerate(chunks):
        assert len(c.text.strip()) >= EVIDENCE_MIN_LEN, (
            f"chunk[{i}] 长度 {len(c.text.strip())} < {EVIDENCE_MIN_LEN}"
        )


def test_all_chunks_have_section(chunks: list[Chunk]) -> None:
    """所有 chunk 均携带非空 section。"""
    for i, c in enumerate(chunks):
        assert c.section.strip(), f"chunk[{i}] section 为空"


# ============================================================
# 各场景断言
# ============================================================
def test_scene_a_paragraph_aggregation(chunks: list[Chunk]) -> None:
    """场景A：两段正常长度段落，应被段落聚合或各自成块。"""
    a_blocks = [c for c in chunks if "场景A" in c.section]
    assert len(a_blocks) >= 1, (
        f"场景A 应有至少 1 个 chunk；现有块: {[c.text for c in a_blocks]}"
    )


def test_scene_b_paragraph_aggregation(chunks: list[Chunk]) -> None:
    """场景B：碎一、碎二、碎三被段落聚合为一个 chunk。"""
    b_blocks = [c for c in chunks if "场景B" in c.section]
    assert len(b_blocks) >= 1, "场景B 应有至少一个 chunk"
    merged = [c for c in b_blocks if "碎一" in c.text and "碎二" in c.text]
    assert merged, (
        f"场景B 未聚合为一块；现有块: {[c.text for c in b_blocks]}"
    )


def test_scene_c_paragraph_aggregation(chunks: list[Chunk]) -> None:
    """场景C：碎四、合格段、碎五 被段落聚合为一个 chunk。"""
    c_blocks = [c for c in chunks if "场景C" in c.section]
    assert len(c_blocks) >= 1, "场景C 应有至少一个 chunk"
    merged = [c for c in c_blocks if "碎四" in c.text and "碎五" in c.text]
    assert merged, (
        f"场景C 未聚合为一块；现有块: {[c.text for c in c_blocks]}"
    )


def test_scene_d_code_block_preserved(chunks: list[Chunk]) -> None:
    """场景D：代码块整体保留。"""
    d_blocks = [c for c in chunks if "场景D" in c.section]
    assert any("def reset_mcu" in c.text for c in d_blocks), (
        f"场景D 代码块丢失；现有块: {[c.text for c in d_blocks]}"
    )


def test_scene_e_table_preserved(chunks: list[Chunk]) -> None:
    """场景E：表格整体保留。"""
    e_blocks = [c for c in chunks if "场景E" in c.section]
    assert any("| 引脚 |" in c.text for c in e_blocks), (
        f"场景E 表格丢失；现有块: {[c.text for c in e_blocks]}"
    )


def test_scene_f_hard_split_continuation(chunks: list[Chunk]) -> None:
    """
    场景F：超长段落被硬切为多块，且存在 is_continuation=True 的续块。

    兼容两种情况：
      - 尾块 >= EVIDENCE_MIN_LEN：保留 2 块，尾块 is_continuation=True
      - 尾块 <  EVIDENCE_MIN_LEN：并入前一块，只剩 1 块
    """
    f_blocks = [c for c in chunks if "场景F" in c.section]
    assert len(f_blocks) >= 1, f"场景F 应有至少 1 块；实际 {len(f_blocks)} 块"
    if len(f_blocks) >= 2:
        assert any(c.is_continuation for c in f_blocks), (
            f"场景F 多块时应有续块；is_continuation: "
            f"{[c.is_continuation for c in f_blocks]}"
        )


def test_no_cross_section_merge(chunks: list[Chunk]) -> None:
    """不同 section 不合并。"""
    sections_set = {c.section for c in chunks}
    assert len(sections_set) >= 2, f"section 集合: {sections_set}"


# ============================================================
# 并入逻辑（直接调用 _split_long_section + monkeypatch 阈值）
# ============================================================
def test_merge_disabled_at_min_len_10(
    sections: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阈值 10 时，场景 F 尾块 18 >= 10，并入不触发，保持 2 块。"""
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 10)

    result = _split_all_sections(sections, CHUNK_SIZE)

    print("\n【并入测试】min_len=10  merge 期望不触发")
    for title, lens in result.items():
        print(f"  {title!r:<30} → {lens}")

    f_title = next(t for t in result if "场景F" in t)
    assert result[f_title] == [299, 18], (
        f"min_len=10 时场景F 应保持 [299, 18]，实际 {result[f_title]}"
    )


def test_merge_triggered_at_min_len_20(
    sections: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阈值 20 时，场景 F 尾块 18 < 20，并入触发，变 1 块 319 字符。"""
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 20)

    result = _split_all_sections(sections, CHUNK_SIZE)

    print("\n【并入测试】min_len=20  merge 期望触发")
    for title, lens in result.items():
        print(f"  {title!r:<30} → {lens}")

    f_title = next(t for t in result if "场景F" in t)
    assert result[f_title] == [319], (
        f"min_len=20 时场景F 应并入为 [319]，实际 {result[f_title]}"
    )


def test_merge_triggered_at_min_len_50(
    sections: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阈值 50 时，场景 F 尾块 18 < 50，并入触发，变 1 块 319 字符。"""
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 50)

    result = _split_all_sections(sections, CHUNK_SIZE)

    print("\n【并入测试】min_len=50  merge 期望触发")
    for title, lens in result.items():
        print(f"  {title!r:<30} → {lens}")

    f_title = next(t for t in result if "场景F" in t)
    assert result[f_title] == [319], (
        f"min_len=50 时场景F 应并入为 [319]，实际 {result[f_title]}"
    )


def test_other_sections_unaffected_by_merge(
    sections: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无论阈值多少，B/C/D/E/G 等单块 section 不受并入影响。"""
    for min_len in [10, 20, 50]:
        monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", min_len)
        result = _split_all_sections(sections, CHUNK_SIZE)

        for keyword in ["场景B", "场景C", "场景D", "场景E", "场景G"]:
            title = next((t for t in result if keyword in t), None)
            if title is None:
                continue
            assert len(result[title]) == 1, (
                f"min_len={min_len} 时 {keyword} 应只有 1 块，"
                f"实际 {result[title]}"
            )


def test_merge_length_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """构造精确场景，验证并入后长度 = len(主块) + 2 + len(尾块)。"""
    monkeypatch.setattr(chunking_module, "EVIDENCE_MIN_LEN", 20)

    para1 = "甲" * 290
    para2 = "乙" * 15
    section_body = f"{para1}\n\n{para2}"

    chunks = _split_long_section("精确测试", section_body, CHUNK_SIZE)

    print(f"\n【精确长度测试】para1={len(para1)} para2={len(para2)} "
          f"EVIDENCE_MIN_LEN=20")
    for i, c in enumerate(chunks):
        print(f"  [{i}] len={len(c.text)} is_continuation={c.is_continuation}")

    assert len(chunks) == 1, f"应并入为 1 块，实际 {len(chunks)} 块"
    assert len(chunks[0].text) == 290 + 2 + 15, (
        f"并入后长度应为 307，实际 {len(chunks[0].text)}"
    )
    assert chunks[0].is_continuation is False, (
        "并入后应继承前一块的 is_continuation=False"
    )


def test_no_merge_when_single_chunk() -> None:
    """单块 section 不并入（无前一块）。"""
    section_body = "丙" * 5

    chunks = _split_long_section("单块测试", section_body, CHUNK_SIZE)

    print(f"\n【并入边界验证】section_body 长度={len(section_body)}")
    print(f"切片结果: {len(chunks)} 块")

    assert len(chunks) == 1, f"应返回 1 块，实际 {len(chunks)} 块"
    assert chunks[0].text == section_body, "内容应保持原样"
    assert chunks[0].is_continuation is False
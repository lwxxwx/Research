"""
app/rag/chunking.py
Sprint-0 v1.5：Markdown frontmatter 解析 + 语义优先切片

职责：
1. 解析 md 文件 YAML frontmatter 元数据
2. frontmatter 字段合法性校验
3. 文本切片：
   - v1.0: 固定字符切片（保留为 process_markdown_file，标注 deprecated）
   - v1.1: ## 二级标题优先 + 段落聚合 + 硬切兜底（process_markdown_file_v2）
4. 片段长度校验，复用 benchmark_service.EVIDENCE_MIN_LEN

v1.1 变更（M2 + M3）：
- M2：切片以 Markdown 语义单位（## / 段落 / 代码块 / 表格）为边界
- M3：每个 chunk 携带独立的 source_section（chunk 级）
- M3.1（补丁）：无 ## 时 chunk 级 section 回落到文档级 section

v1.2 变更（豆包优化 1 / 4 / 6）：
- 优化 1：(intro) 引言块继承 doc_section
- 优化 4：deprecation 注释精确化
- 优化 6：process_markdown_file_v2 异常信息附带 chunk 长度列表

v1.3 变更（豆包阻断 BUG 1 / 2）：
- BUG1：split_text_chunk_v2 移除 chunk_overlap 拼接逻辑
- BUG2：_split_long_section 单段超长时字符硬切兜底

v1.4 变更（豆包新优化 1 / 5 / 7）：
- 优化 1：import yaml 移到文件头部
- 优化 5：_hard_split 标点查找逻辑简化（单层 offset 遍历，遇标点即切）
- 优化 7：Chunk dataclass 删除未使用 meta 字段；移除 field import

v1.5 变更（豆包新优化 2）：
- 优化 2：_hard_split 注释与循环语义显式对齐（说明 offset=0 检查 limit-1 位置）

对齐规范：
- Sprint0 V1.2 §13 Phase H
- 第一阶段 V1.2 §6.2 知识来源标签
"""
import re
import pathlib
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# ★ 优化 1：import yaml 移到文件头部
import yaml

from app.rag.sources import DOCUMENT_TYPE_ALLOWED
from app.services.benchmark_service import EVIDENCE_MIN_LEN


# ============================================================
# frontmatter 解析（v1.0 保持不变）
# ============================================================
def parse_markdown_frontmatter(file_path: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    """
    解析 markdown 头部 YAML frontmatter
    return (metadata_dict, content_without_frontmatter)
    """
    raw_text = file_path.read_text(encoding="utf-8")
    frontmatter_pat = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match_obj = frontmatter_pat.match(raw_text)
    if not match_obj:
        raise ValueError(f"[{file_path.name}] 文件缺失 YAML frontmatter，必须以 --- 开头结尾")

    meta_raw = match_obj.group(1)
    meta: Dict[str, Any] = yaml.safe_load(meta_raw) or {}
    pure_content = raw_text[match_obj.end():]
    return meta, pure_content


def validate_frontmatter(meta: Dict[str, Any]) -> None:
    """
    校验 frontmatter 必填字段与数据类型，不合法直接抛 ValueError
    必填：source_type, source_title, source_section, part_numbers, related_rule_ids
    """
    required_keys = ["source_type", "source_title", "source_section",
                     "part_numbers", "related_rule_ids"]
    for k in required_keys:
        if k not in meta:
            raise ValueError(f"frontmatter 缺少必填字段：{k}")

    source_type = meta["source_type"]
    if source_type not in DOCUMENT_TYPE_ALLOWED:
        raise ValueError(
            f"非法 source_type={source_type}，允许值：{sorted(DOCUMENT_TYPE_ALLOWED)}"
        )

    if not isinstance(meta["part_numbers"], list):
        raise ValueError("part_numbers 必须为数组，空数组写 []")
    if not isinstance(meta["related_rule_ids"], list):
        raise ValueError("related_rule_ids 必须为数组，空数组写 []")


# ============================================================
# v1.0 固定字符切片（保留，标注 deprecated）
# ============================================================
def split_text_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Sprint-0 v1.0 简单固定字符分片
    ⚠️ deprecated：请使用 split_text_chunk_v2

    === 优化 4：deprecation 注释精确化 ===
    计划在 Sprint1 或后续大版本中删除；删除前将通过 changelog 提前公告
    """
    warnings.warn(
        "split_text_chunk is deprecated, use split_text_chunk_v2 instead",
        DeprecationWarning,
        stacklevel=2,
    )
    chunks: List[str] = []
    start = 0
    total_len = len(text)
    while start < total_len:
        end = start + chunk_size
        seg = text[start:end]
        chunks.append(seg)
        start += (chunk_size - chunk_overlap)
    return chunks


# ============================================================
# v1.1 语义优先切片（M2 + M3 核心）
# ============================================================
@dataclass
class Chunk:
    """
    v1.1 chunk 数据模型：携带 chunk 级 source_section

    ★ 优化 7：删除未使用的 meta 字段
    原定义（保留）：
        section: str
        text: str
        is_continuation: bool = False
        meta: Dict[str, Any] = field(default_factory=dict)
    新定义：仅保留实际使用的 3 个字段
    """
    section: str           # chunk 级章节标题（M3）
    text: str              # chunk 正文
    is_continuation: bool = False  # 是否因超长被拆分的续块
    # --- 原字段（保留，已删除） ---
    # meta: Dict[str, Any] = field(default_factory=dict)


# Markdown 语义单位识别
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def _split_by_h2(body: str, doc_section: str = "(untitled)") -> List[Tuple[str, str]]:
    """
    按 ## 二级标题切分，返回 [(section_title, section_body), ...]

    - 首个 ## 之前的内容（引言/文档标题）标记为文档级 section（优化 1）
    - 无 ## 时回落到文档级 section（M3.1 补丁）
    """
    matches = list(_H2_RE.finditer(body))
    if not matches:
        return [(doc_section or "(untitled)", body.strip())]

    parts: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        intro = body[: matches[0].start()].strip()
        if intro:
            # ★ 优化 1：引言块继承文档级 section
            parts.append((doc_section or "(intro)", intro))
            # --- 原内容（保留） ---
            # parts.append(("(intro)", intro))

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        parts.append((title, body[start:end].strip()))
    return parts


# ============================================================
# ★ BUG2 修复：字符硬切兜底辅助函数
# ============================================================
def _hard_split(text: str, limit: int) -> List[str]:
    """
    ★ BUG2 修复：字符硬切兜底

    仅用于「单个段落超过 limit 且无 \\n\\n 边界」的极端场景。
    优先在句号 / 感叹号 / 问号 / 分号 / 换行处切；若找不到边界，再按 limit 硬切。

    === 优化 5：简化标点查找逻辑 ===
    用单层 offset 遍历替代原「外层 punct + 内层 offset」的双层遍历。

    === 优化 2（v1.5）：注释与循环语义显式对齐 ===
    循环语义说明：
      - `offset = 0` 检查 `remaining[limit - 1]`（不含 limit 位置自身）
      - `offset = 1` 检查 `remaining[limit - 2]`
      - `offset = n` 检查 `remaining[limit - n - 1]`
      - offset 递增 = 从 limit 位置往前扫描；一旦遇到标点即切
      - 因此切点选中的是「[limit-100, limit-1] 窗口内离 limit 最近的标点」
    """
    if len(text) <= limit:
        return [text]

    result: List[str] = []
    remaining = text
    puncts_set = {"。", "！", "？", "；", "\n"}

    while len(remaining) > limit:
        cut_pos = limit
        # ★ 优化 5：从 limit-1 位置往前找最近标点（不含 limit 位置自身），找到即切
        # offset=0 检查 remaining[limit-1]，offset=1 检查 remaining[limit-2]，依此类推
        for offset in range(0, min(100, limit)):
            if remaining[limit - offset - 1] in puncts_set:
                cut_pos = limit - offset
                break

        result.append(remaining[:cut_pos])
        remaining = remaining[cut_pos:]

    if remaining:
        result.append(remaining)
    return result


def _split_long_section(section: str, section_body: str, limit: int) -> List[Chunk]:
    """
    单章节超过 limit 时，按段落聚合拆分（M2 核心）

    - 段落边界：\\n\\n
    - 保护代码块与表格：如整体包含 ``` 或表格行，则整段不切（宁可超长）
    - ★ BUG2 修复：单个段落超 limit 时字符硬切兜底
    - 拆分出的续块按「段内序号 > 0」标记 is_continuation=True
    """
    if len(section_body) <= limit:
        return [Chunk(section=section, text=section_body, is_continuation=False)]

    # 保护代码块与表格：整体不切
    has_code = bool(_CODE_BLOCK_RE.search(section_body))
    has_table = bool(_TABLE_LINE_RE.search(section_body))
    if has_code or has_table:
        return [Chunk(section=section, text=section_body, is_continuation=False)]

    # 段落聚合
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_body) if p.strip()]
    chunks: List[Chunk] = []
    buf = ""

    for p in paragraphs:
        # ★ BUG2 修复：单段超长时，先落盘 buf，再用 _hard_split 按句子边界硬切
        if len(p) > limit:
            if buf:
                chunks.append(Chunk(
                    section=section,
                    text=buf,
                    is_continuation=bool(chunks),
                ))
                buf = ""
            pieces = _hard_split(p, limit)
            for piece_idx, piece in enumerate(pieces):
                # ★ BUG2：is_continuation 用「段内序号 > 0」精确判定
                chunks.append(Chunk(
                    section=section,
                    text=piece,
                    is_continuation=(piece_idx > 0),
                ))
            continue

        # 正常段落：聚合到 limit
        if len(buf) + len(p) + 2 <= limit:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(Chunk(
                    section=section,
                    text=buf,
                    is_continuation=bool(chunks),
                ))
            buf = p

    if buf:
        chunks.append(Chunk(
            section=section,
            text=buf,
            is_continuation=bool(chunks),
        ))
    return chunks


def split_text_chunk_v2(
    text: str,
    chunk_size: int,
    # ★ BUG1 修复：移除 chunk_overlap 参数
    doc_section: str = "(untitled)",
) -> List[Chunk]:
    """
    v1.1/v1.3 语义优先切片（BUG1 修复版）

    ★ BUG1 修复说明：
      本函数**不支持 overlap**。段落聚合的 chunk 边界已是语义完整单元，
      overlap 会引入跨 section 污染。如需字符级 overlap，请使用 v1.0 的 split_text_chunk。
    """
    chunks: List[Chunk] = []
    for section_title, section_body in _split_by_h2(text, doc_section):
        if not section_body.strip():
            continue
        for c in _split_long_section(section_title, section_body, chunk_size):
            chunks.append(c)

    # ★ BUG1 修复：删除 overlap 拼接逻辑
    # --- 原代码（保留，已弃用） ---
    # if chunk_overlap > 0 and len(chunks) > 1:
    #     for i in range(1, len(chunks)):
    #         prev_tail = chunks[i - 1].text[-chunk_overlap:]
    #         chunks[i].text = prev_tail + chunks[i].text

    return chunks


# ============================================================
# 统一入口
# ============================================================
def process_markdown_file(
    file_path: pathlib.Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    v1.0 入口（保留兼容）
    ⚠️ deprecated：返回 List[str]，不携带 chunk 级 section

    === 优化 4：deprecation 注释精确化 ===
    计划在 Sprint1 或后续大版本中删除；删除前将通过 changelog 提前公告
    """
    warnings.warn(
        "process_markdown_file is deprecated, "
        "use process_markdown_file_v2 to get chunk-level source_section",
        DeprecationWarning,
        stacklevel=2,
    )
    meta, content = parse_markdown_frontmatter(file_path)
    validate_frontmatter(meta)

    content_stripped = content.strip()
    if len(content_stripped) < EVIDENCE_MIN_LEN:
        raise ValueError(
            f"[{file_path.name}] 正文长度 {len(content_stripped)} "
            f"小于最小阈值 {EVIDENCE_MIN_LEN}，拒绝入库"
        )

    chunk_list = split_text_chunk(content_stripped, chunk_size, chunk_overlap)
    return meta, content, chunk_list


def process_markdown_file_v2(
    file_path: pathlib.Path,
    chunk_size: int,
    # ★ BUG1 修复：保留 chunk_overlap 参数以兼容调用方，但内部忽略
    chunk_overlap: int = 0,
) -> Tuple[Dict[str, Any], str, List[Chunk]]:
    """
    v1.1/v1.3 统一入口（推荐）
    :returns: (meta_dict, original_content_text, [Chunk, ...])
              Chunk.section 为 chunk 级 source_section（M3）

    ★ BUG1 修复：chunk_overlap 参数保留但忽略（语义切片不支持 overlap）
    """
    meta, content = parse_markdown_frontmatter(file_path)
    validate_frontmatter(meta)

    content_stripped = content.strip()
    if len(content_stripped) < EVIDENCE_MIN_LEN:
        raise ValueError(
            f"[{file_path.name}] 正文长度 {len(content_stripped)} "
            f"小于最小阈值 {EVIDENCE_MIN_LEN}，拒绝入库"
        )

    # ★ BUG1 修复：不再传递 chunk_overlap
    chunks = split_text_chunk_v2(
        content_stripped,
        chunk_size,
        # chunk_overlap,   # --- 原代码（保留，已弃用） ---
        doc_section=meta["source_section"],
    )

    # 过滤过短碎片
    valid_chunks = [c for c in chunks if len(c.text.strip()) >= EVIDENCE_MIN_LEN]

    # === 优化 6：异常信息附带每个 chunk 的长度列表 ===
    if not valid_chunks:
        lengths = [len(c.text.strip()) for c in chunks]
        raise ValueError(
            f"[{file_path.name}] 切片后无有效 chunk "
            f"(全部 < {EVIDENCE_MIN_LEN} 字符)；chunk 长度列表: {lengths}"
        )
        # --- 原内容（保留） ---
        # raise ValueError(
        #     f"[{file_path.name}] 切片后无有效 chunk（全部 < {EVIDENCE_MIN_LEN} 字符）"
        # )

    return meta, content, valid_chunks
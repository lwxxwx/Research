"""
app/rag/chunking.py
Sprint‑0：Markdown frontmatter解析 + 文本分片工具
职责：
1. 解析md文件YAML frontmatter元数据
2. frontmatter字段合法性校验
3. 文本切片（Sprint0简单字符切片；Sprint‑1可替换为langchain语义切分）
4. 片段长度校验，复用benchmark_service.EVIDENCE_MIN_LEN

输出：(meta:dict[str,Any], pure_content:str, chunk_list:list[str])
"""
import re
import pathlib
from typing import Dict, Any, List

from app.rag.sources import DOCUMENT_TYPE_ALLOWED
from app.services.benchmark_service import EVIDENCE_MIN_LEN


def parse_markdown_frontmatter(file_path: pathlib.Path) -> tuple[Dict[str, Any], str]:
    """
    解析markdown头部YAML frontmatter
    return (metadata_dict, content_without_frontmatter)
    """
    raw_text = file_path.read_text(encoding="utf-8")
    frontmatter_pat = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match_obj = frontmatter_pat.match(raw_text)
    if not match_obj:
        raise ValueError(f"[{file_path.name}] 文件缺失YAML frontmatter，必须以 --- 开头结尾")

    import yaml
    meta_raw = match_obj.group(1)
    meta: Dict[str, Any] = yaml.safe_load(meta_raw) or {}
    pure_content = raw_text[match_obj.end():]
    return meta, pure_content


def validate_frontmatter(meta: Dict[str, Any]) -> None:
    """
    校验frontmatter必填字段与数据类型，不合法直接抛ValueError
    必填：source_type, source_title, source_section, part_numbers, related_rule_ids
    """
    required_keys = ["source_type", "source_title", "source_section", "part_numbers", "related_rule_ids"]
    for k in required_keys:
        if k not in meta:
            raise ValueError(f"frontmatter缺少必填字段：{k}")

    source_type = meta["source_type"]
    if source_type not in DOCUMENT_TYPE_ALLOWED:
        raise ValueError(
            f"非法source_type={source_type}，允许值：{sorted(DOCUMENT_TYPE_ALLOWED)}"
        )

    if not isinstance(meta["part_numbers"], list):
        raise ValueError("part_numbers 必须为数组，空数组写 []")
    if not isinstance(meta["related_rule_ids"], list):
        raise ValueError("related_rule_ids 必须为数组，空数组写 []")


def split_text_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Sprint‑0 简单固定字符分片
    Sprint‑1 替换为RecursiveCharacterTextSplitter语义切分
    """
    chunks: List[str] = []
    start = 0
    total_len = len(text)
    while start < total_len:
        end = start + chunk_size
        seg = text[start:end]
        chunks.append(seg)
        start += (chunk_size - chunk_overlap)
    return chunks


def process_markdown_file(
    file_path: pathlib.Path,
    chunk_size: int,
    chunk_overlap: int
) -> tuple[Dict[str, Any], str, List[str]]:
    """
    对外统一入口：解析+校验+切片
    :returns: (meta_dict, original_content_text, chunk_list)
    """
    meta, content = parse_markdown_frontmatter(file_path)
    validate_frontmatter(meta)

    content_stripped = content.strip()
    if len(content_stripped) < EVIDENCE_MIN_LEN:
        raise ValueError(
            f"[{file_path.name}] 正文长度 {len(content_stripped)} "
            f"小于最小阈值{EVIDENCE_MIN_LEN}，拒绝入库"
        )

    chunk_list = split_text_chunk(content_stripped, chunk_size, chunk_overlap)
    return meta, content, chunk_list

"""
app/rag/knowledge.py
RAG模块内部Pydantic模型
- RetrievalResult：检索返回结果，可直接映射为 type=rag_ref evidence
- IngestConfig / IngestConfigEntry：ingest导入yaml配置模型
注意：数据库ORM模型仍然在 app/persistence/models.py，此处仅为内存schema，不操作DB
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class RetrievalResult(BaseModel):
    """RAG单条检索结果，可直接映射为evidence字典"""
    source_type: str = Field(description="对应evidence.source_type")
    source_title: str = Field(description="对应evidence.source，原始文档全名")
    source_section: str = Field(description="对应evidence.section，章节/页码信息")
    snippet: str = Field(description="对应evidence.reason，原文片段正文")
    score: float = Field(description="向量余弦距离，数值越小相似度越高")
    part_numbers: List[str] = Field(default_factory=list, description="关联器件型号列表")
    related_rule_ids: List[str] = Field(default_factory=list, description="关联规则ID列表")
    doc_id: Optional[int] = Field(default=None, description="knowledge_doc表主键")
    chunk_id: Optional[int] = Field(default=None, description="knowledge_chunk表主键")


class IngestConfigEntry(BaseModel):
    file: str = Field(description="待导入md文件相对路径")


class IngestConfig(BaseModel):
    ingest_entries: List[IngestConfigEntry] = Field(default_factory=list, description="显式指定导入文件列表")
    case_evidence_scan: Optional[dict] = Field(default=None, description="B‑Case evidence目录扫描配置")

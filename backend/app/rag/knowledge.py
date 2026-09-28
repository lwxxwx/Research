"""
app/rag/knowledge.py
RAG 模块内部 Pydantic 模型
- RetrievalResult：检索返回结果，可直接映射为 type=rag_ref evidence
- IngestConfig / IngestConfigEntry：ingest 导入 yaml 配置模型
注意：数据库 ORM 模型仍然在 app/persistence/models.py，此处仅为内存 schema，不操作 DB

用三个 Pydantic 模型把 RAG 模块的"配置输入"（IngestConfig / IngestConfigEntry）
和"检索输出"（RetrievalResult）固化成类型安全的 schema，核心语法点是 default_factory 
规避可变默认值陷阱、Optional 表达可空语义、嵌套模型 vs dict 的强弱类型权衡，
以及 is_continuation 这个为 Sprint1 重排预留的血缘字段。

v1.2 变更（豆包优化 5）：
- 优化 5：RetrievalResult 增加 is_continuation 字段，供 Sprint1 重排时降权续块
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class RetrievalResult(BaseModel):
    """RAG 单条检索结果，可直接映射为 evidence 字典"""
    source_type: str = Field(description="对应evidence.source_type")
    source_title: str = Field(description="对应evidence.source，原始文档全名")
    source_section: str = Field(description="对应evidence.section，章节/页码信息")
    snippet: str = Field(description="对应evidence.reason，原文片段正文")
    score: float = Field(description="向量余弦距离，数值越小相似度越高")
    part_numbers: List[str] = Field(default_factory=list, description="关联器件型号列表")
    related_rule_ids: List[str] = Field(default_factory=list, description="关联规则ID列表")
    doc_id: Optional[int] = Field(default=None, description="knowledge_doc表主键")
    chunk_id: Optional[int] = Field(default=None, description="knowledge_chunk表主键")

    # === 优化 5：新增 is_continuation 字段 ===
    # 说明：chunk 落库时 meta_json 已有此字段，但 RetrievalResult 此前未透传；
    #       Sprint1 做重排（Rerank）时用于降权「续块」（信息不完整，非首选）
    is_continuation: bool = Field(
        default=False,
        description="是否为超长章节的续块；Sprint1 重排时用于降权",
    )


class IngestConfigEntry(BaseModel):
    file: str = Field(description="待导入md文件相对路径")


class IngestConfig(BaseModel):
    ingest_entries: List[IngestConfigEntry] = Field(default_factory=list, description="显式指定导入文件列表")
    case_evidence_scan: Optional[dict] = Field(default=None, description="B-Case evidence目录扫描配置")
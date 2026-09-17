"""
app/rag/sources.py
DocumentType 枚举，RAG知识库来源类型
对齐V1.2 evidence.source_type枚举定义
"""
from enum import StrEnum


class DocumentType(StrEnum):
    DATASHEET = "datasheet"
    REFERENCE_DESIGN = "reference_design"
    APPLICATION_NOTE = "application_note"
    REVIEW_CASE = "review_case"
    DESIGN_STANDARD = "design_standard"


DOCUMENT_TYPE_ALLOWED = {
    DocumentType.DATASHEET,
    DocumentType.REFERENCE_DESIGN,
    DocumentType.APPLICATION_NOTE,
    DocumentType.REVIEW_CASE,
    DocumentType.DESIGN_STANDARD,
}

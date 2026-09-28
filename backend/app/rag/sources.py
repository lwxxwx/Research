"""
app/rag/sources.py
DocumentType 枚举，RAG知识库来源类型
对齐V1.2 evidence.source_type枚举定义(expected_review.json中定义)

定义 5 种文档类型，关键红利是"枚举成员本身即字符串"，
使得 DOCUMENT_TYPE_ALLOWED 这个成员集合能直接对
YAML 里的字符串做值域校验；它虽小，却是 RAG 模块与
 V1.2 evidence 规范之间的唯一契约锚点。

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

# app/persistence/models.py
from sqlalchemy import (
    Column, BigInteger, String, Text, Boolean, Float, Integer,
    ForeignKey, TIMESTAMP, Index, UniqueConstraint, CheckConstraint, text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional, List, Dict, Any
class Base(DeclarativeBase):
    """注意：Sprint0禁止 Base.metadata.create_all()，schema唯一来源：infra/docker/initdb/002_schema.sql"""
    pass
# ============================================================
# 1. User
# ============================================================
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    projects: Mapped[List["Project"]] = relationship(back_populates="owner")
    feedbacks: Mapped[List["FeedbackItem"]] = relationship(back_populates="creator")
# ============================================================
# 2. Project
# ============================================================
class Project(Base):
    __tablename__ = "project"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    owner_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    owner: Mapped[Optional["User"]] = relationship(back_populates="projects")
    schematic_cases: Mapped[List["SchematicCase"]] = relationship(back_populates="project", cascade="all, delete-orphan")
# ============================================================
# 3. SchematicCase
# ============================================================
class SchematicCase(Base):
    __tablename__ = "schematic_case"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    project_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("project.id", ondelete="CASCADE"))
    case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    case_type: Mapped[str] = mapped_column(String(32), nullable=False)
    case_path: Mapped[Optional[str]] = mapped_column(String(512))
    expert_calibration_md: Mapped[Optional[str]] = mapped_column(Text)
    expected_review_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    evaluation_yaml: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    project: Mapped[Optional["Project"]] = relationship(back_populates="schematic_cases")
    ir_documents: Mapped[List["IRDocument"]] = relationship(back_populates="schematic_case", cascade="all, delete-orphan")

    # ✅ 对齐 SQL：UNIQUE(project_id, case_id)
    __table_args__ = (
        UniqueConstraint("project_id", "case_id", name="uq_schematic_case_project_case"),
    )    
# ============================================================
# 4. IRDocument
# ============================================================
class IRDocument(Base):
    __tablename__ = "ir_document"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    schematic_case_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("schematic_case.id", ondelete="CASCADE"))
    ir_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    ir_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    schematic_case: Mapped[Optional["SchematicCase"]] = relationship(back_populates="ir_documents")
    rule_executions: Mapped[List["RuleExecution"]] = relationship(back_populates="ir_document", cascade="all, delete-orphan")
    review_results: Mapped[List["ReviewResult"]] = relationship(back_populates="ir_document", cascade="all, delete-orphan")
# ============================================================
# 5. RuleDefinition
# ============================================================
class RuleDefinition(Base):
    __tablename__ = "rule_definition"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    rule_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    rule_name: Mapped[str] = mapped_column(String(256), nullable=False)
    rule_category: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_yaml: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(20), server_default="medium")
    enabled: Mapped[bool] = mapped_column(Boolean, server_default="true")
    version: Mapped[str] = mapped_column(String(20), server_default="v1.0")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    executions: Mapped[List["RuleExecution"]] = relationship(back_populates="rule_def")
# ============================================================
# 6. RuleExecution
# ============================================================
class RuleExecution(Base):
    __tablename__ = "rule_execution"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ir_document_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("ir_document.id", ondelete="CASCADE"))
    rule_def_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("rule_definition.id"))
    hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    evidence_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    execution_log: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
#    ir_document: Mapped[Optional["IRDocument"]] = relationship(back_populates="ir_document")
    ir_document: Mapped[Optional["IRDocument"]] = relationship(back_populates="rule_executions")
    rule_def: Mapped[Optional["RuleDefinition"]] = relationship(back_populates="executions")
# ============================================================
# 7. ReviewResult
# ============================================================
class ReviewResult(Base):
    __tablename__ = "review_result"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ir_document_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("ir_document.id", ondelete="CASCADE"))
    task_id: Mapped[Optional[str]] = mapped_column(String(64))
    review_output_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_rule_only: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    # V1.2 摘要字段
    category: Mapped[str] = mapped_column(String(64), nullable=False, server_default="other")
    risk: Mapped[str] = mapped_column(String(16), nullable=False, server_default="medium")
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="AI_CONFIRMED")
    evidence_coverage: Mapped[Optional[float]] = mapped_column(Float)
    ai_new_effective_rate: Mapped[Optional[float]] = mapped_column(Float)
    expert_adoption_rate: Mapped[Optional[float]] = mapped_column(Float)
    total_defects: Mapped[int] = mapped_column(Integer, server_default="0")
    ai_confirmed_count: Mapped[int] = mapped_column(Integer, server_default="0")
    need_expert_review_count: Mapped[int] = mapped_column(Integer, server_default="0")
    low_confidence_count: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    # 关系
    ir_document: Mapped[Optional["IRDocument"]] = relationship(back_populates="review_results")
    defects: Mapped[List["ReviewDefect"]] = relationship(back_populates="review_result", cascade="all, delete-orphan")
    feedbacks: Mapped[List["FeedbackItem"]] = relationship(back_populates="review_result", cascade="all, delete-orphan")
    __table_args__ = (
        Index("idx_review_result_review_status", "review_status"),
        Index("idx_review_result_category", "category"),
        Index("idx_review_result_task_id", "task_id"),
    )
# ============================================================
# 8. ReviewDefect（新增：缺陷明细表）
# ============================================================
class ReviewDefect(Base):
    __tablename__ = "review_defect"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    review_result_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("review_result.id", ondelete="CASCADE"))
    defect_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # V1.2 10字段规范
    category: Mapped[str] = mapped_column(String(64), nullable=False, server_default="other")
    location: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    component: Mapped[Optional[str]] = mapped_column(String(100))
    net: Mapped[Optional[str]] = mapped_column(String(100))
    risk: Mapped[str] = mapped_column(String(16), nullable=False, server_default="medium")
    #evidence: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default="[]")
    evidence: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    suggestion: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, server_default="0.5")
    # 质量状态（三态）
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="AI_CONFIRMED")
    # 来源追踪
    origin: Mapped[str] = mapped_column(String(32), nullable=False, server_default="rule")
    rule_id: Mapped[Optional[str]] = mapped_column(String(64))
    # 反馈聚合
    feedback_type: Mapped[Optional[str]] = mapped_column(String(64))
    feedback_count: Mapped[int] = mapped_column(Integer, server_default="0")
    latest_feedback_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    # 关系
    review_result: Mapped[Optional["ReviewResult"]] = relationship(back_populates="defects")
    feedbacks: Mapped[List["FeedbackItem"]] = relationship(back_populates="defect", cascade="all, delete-orphan")
    __table_args__ = (
        Index("idx_review_defect_review_result", "review_result_id"),
        Index("idx_review_defect_defect_id", "defect_id"),
        Index("idx_review_defect_review_status", "review_status"),
        Index("idx_review_defect_category", "category"),
        Index("idx_review_defect_origin", "origin"),
        Index("idx_review_defect_rule_id", "rule_id"),
    )
# ============================================================
# 9. FeedbackItem（增强：关联 review_defect）
# ===== ✅ MODIFIED Sprint0 Phase‑I：对齐002_schema.sql全部字段 =====
#
# ⚠️ 待解决问题（Sprint 1，不是 Sprint 0 阻塞项）：
#
# 问题 1：FeedbackItem.rule_candidate_ref 无 DB 外键约束
#   - 该列是普通 String(64)，仅存 RuleCandidate.candidate_id 字符串
#   - 若 RuleCandidate 记录被删除，FeedbackItem.rule_candidate_ref 会残留
#     悬空 candidate_id 字符串，DB 不校验
#   - 建议（Sprint 1）：
#       RuleEvolutionService.delete_candidate() 中先反向更新
#       FeedbackItem.rule_candidate_ref = NULL，再 db.delete(rc)
#
# 问题 2：review_result_id / review_defect_id 均使用 ondelete="CASCADE"
#   - 删除 ReviewResult → 关联 FeedbackItem 级联删除（业务上可能合理）
#   - 删除 ReviewDefect → 关联 FeedbackItem 级联删除（业务上需评估）
#   - 若 FeedbackItem 只应依附 ReviewResult、不应依附 ReviewDefect，
#     则 review_defect_id 的 ondelete 应改为 "SET NULL"
#   - 待 Sprint 1 明确业务语义后决定
# ============================================================
class FeedbackItem(Base):
class FeedbackItem(Base):
    __tablename__ = "feedback_item"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    review_result_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("review_result.id", ondelete="CASCADE"))
    review_defect_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("review_defect.id", ondelete="CASCADE"))
    feedback_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # ----- 【注意】SQL字段名 comment，Pydantic使用expert_suggestion做业务字段 -----
    comment: Mapped[Optional[str]] = mapped_column(Text)
    expert_suggestion: Mapped[Optional[str]] = mapped_column(Text)
    suggestion_adopted: Mapped[Optional[str]] = mapped_column(String(16))
    suggestion_diff_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default="[]")
    rule_candidate_ref: Mapped[Optional[str]] = mapped_column(String(64))
    attached_refs: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    # 关系
    review_result: Mapped[Optional["ReviewResult"]] = relationship(back_populates="feedbacks")
    defect: Mapped[Optional["ReviewDefect"]] = relationship(back_populates="feedbacks")
    creator: Mapped[Optional["User"]] = relationship(back_populates="feedbacks")
    __table_args__ = (
        # ✅ 对齐 SQL：DB 层强制 6 种反馈类型
        CheckConstraint(
            "feedback_type IN ("
            "'correct_defect','false_positive','false_negative',"
            "'suggestion_update','new_rule_candidate','knowledge_gap'"
            ")",
            name="check_feedback_item_type",
        ),
        Index("idx_feedback_item_type", "feedback_type"),
        Index("idx_feedback_item_result", "review_result_id"),
        Index("idx_feedback_item_defect", "review_defect_id"),
    )
# ============================================================
# 10. KnowledgeDoc
# ============================================================
class KnowledgeDoc(Base):
    __tablename__ = "knowledge_doc"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(256))
    source_type: Mapped[Optional[str]] = mapped_column(String(32))
    content_md: Mapped[str] = mapped_column(Text, nullable=False)
    # 注意：ORM属性名 meta_json；数据库真实列名仍然 metadata（保留原SQL schema不动）
    meta_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, name="metadata", server_default="{}")
    version: Mapped[str] = mapped_column(String(20), server_default="v1.0")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    chunks: Mapped[List["KnowledgeChunk"]] = relationship(back_populates="doc", cascade="all, delete-orphan")
# ============================================================
# 11. KnowledgeChunk
# ============================================================
class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunk"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    knowledge_doc_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("knowledge_doc.id", ondelete="CASCADE"))
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1536))
    # ORM属性 meta_json，DB列名 metadata
    meta_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, name="metadata", server_default="{}")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    doc: Mapped[Optional["KnowledgeDoc"]] = relationship(back_populates="chunks")
    __table_args__ = (
        Index("idx_knowledge_chunk_embedding", "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
    )
# ============================================================
# 12. RuleCandidate（P1，Sprint‑1业务使用）
# ===== ✅ MODIFIED Sprint0 Phase‑I：对齐002_schema.sql触发器、时区字段 =====
# ===== ✅注意：数据库表名是复数 rule_candidates，不是 rule_candidate =====
class RuleCandidate(Base):
    __tablename__ = "rule_candidates"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    from_feedback_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("feedback_item.id"))
    case_id: Mapped[Optional[str]] = mapped_column(String(64))
    # ✅ Sprint0：普通字段，无外键；Sprint-1 tasks 表创建后再补 ForeignKey
    task_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[Optional[str]] = mapped_column(String(16))
    evidence_refs: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default="[]")
    proposed_yaml: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="proposed")
    # ----- ✅ 对齐SQL：TIMESTAMPTZ + 触发器自动更新updated_at -----
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"))
    __table_args__ = (
        Index("idx_rule_candidates_status", "status"),
        Index("idx_rule_candidates_case_id", "case_id"),
         # ✅ 对齐 SQL
        Index("idx_rule_candidates_from_feedback_id", "from_feedback_id"), 
    )

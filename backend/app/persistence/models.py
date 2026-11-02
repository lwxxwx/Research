# app/persistence/models.py
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    # ⚠️ V1.3 修改：删除下方错误注释行（back_populates="ir_document" 是笔误，
    # IRDocument 侧是 rule_executions，恢复该注释会导致 relationship 配置错误）。
    # [原逻辑 - 保留注释，便于对照回滚]
    # #    ir_document: Mapped[Optional["IRDocument"]] = relationship(back_populates="ir_document")
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
    # ===== [S1-CHANGE-3] ReviewResult 新增 schematic_case_id（Sprint1 §8：评审-案例关联，ON DELETE SET NULL）=====
    schematic_case_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("schematic_case.id", ondelete="SET NULL")
    )
    # ===== [S1-CHANGE-3 END] =====
    # ===== [S2-A 新增] task_status：任务执行结果，与 review_status 分离 =====
    task_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending"
    )
    # ===== [S2-A 新增 END] =====
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
        # ===== [S2-A 新增] =====
        Index("idx_review_result_task_status", "task_status"),
        # ===== [S2-A 新增 END] =====
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
    # ===== [S1-CHANGE-1] ReviewDefect 新增专家最终判定（Sprint1 §23.1：质量状态与专家决定分离）=====
    expert_decision: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending")
    expert_decision_updated_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    expert_decision_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    # ===== [S1-CHANGE-1 END] =====
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
        # ===== [S1-CHANGE-2] ReviewDefect 新增 expert_decision 约束与索引（Sprint1 §23.1）=====
        CheckConstraint(
            "expert_decision IN ('pending','confirmed','rejected','needs_verification')",
            name="check_review_defect_expert_decision",
        ),
        Index("idx_review_defect_expert_decision", "expert_decision"),
        # ===== [S1-CHANGE-2 END] =====
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
    # ⚠️ V1.3 修改：补 nullable=False，对齐 002_schema.sql（attached_refs JSONB NOT NULL DEFAULT '[]'::jsonb）。
    # 原代码允许 NULL，与 DB NOT NULL 冲突，显式传 None 会报错。
    # [原逻辑 - 保留注释，便于对照回滚]
    # attached_refs: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    attached_refs: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default="[]")
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
    # ===== [S1-C 修改]  embedding 维度改为 1024 =====
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1024))
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
    # ===== [S1-CHANGE-4/5] RuleCandidate：新增 schematic_case_id，删除 task_id（Sprint1 §8：候选独立于 case，ON DELETE SET NULL）=====
    schematic_case_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("schematic_case.id", ondelete="SET NULL")
    )
    # ===== [S1-CHANGE-4/5 END] =====
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[Optional[str]] = mapped_column(String(16))
    evidence_refs: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default="[]")
    proposed_yaml: Mapped[Optional[str]] = mapped_column(Text)
    # ===== [S2-A 新增] benchmark_result / benchmarked_at / rejected_reason =====
    benchmark_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    benchmarked_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    rejected_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # ===== [S2-A 新增 END] =====
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="proposed")
    # ===== [F05] 发布状态机字段 =====
    publish_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="not_published"
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    published_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    publish_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publish_idempotency_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # ===== [F05 END] =====
    # ----- ✅ 对齐SQL：TIMESTAMPTZ + 触发器自动更新updated_at -----
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"))
    __table_args__ = (
        Index("idx_rule_candidates_status", "status"),
        Index("idx_rule_candidates_case_id", "case_id"),
        # ✅ 对齐 SQL
        Index("idx_rule_candidates_from_feedback_id", "from_feedback_id"),
        # ===== [S1-CHANGE-6] RuleCandidate 新增 schematic_case_id 索引（Sprint1 §8）=====
        Index("idx_rule_candidates_schematic_case", "schematic_case_id"),
        CheckConstraint(
            "publish_status IN ('not_published','publishing','published','publish_failed')",
            name="check_rule_candidates_publish_status",
        ),
        Index("idx_rule_candidates_publish_status", "publish_status"),
        Index(
            "uniq_rule_candidates_publish_idempotency_key",
            "publish_idempotency_key",
            unique=True,
            postgresql_where=text("publish_idempotency_key IS NOT NULL"),
        ),
        # ===== [S1-CHANGE-6 END] =====
    )
# ============================================================
# 13. KnowledgeGap（S2-A 新增，一对多：一条反馈可派生多条缺口）
# ============================================================
class KnowledgeGap(Base):
    __tablename__ = "knowledge_gap"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    gap_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # 主来源冗余字段，不唯一（实际一对多关系由 gap_feedback_link 承载）
    source_feedback_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("feedback_item.id", ondelete="SET NULL"), nullable=True
    )
    case_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    defect_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="open")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("NOW()"))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    resolved_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=True)

    # 关系
    source_feedback: Mapped[Optional["FeedbackItem"]] = relationship(foreign_keys=[source_feedback_id])
    feedback_links: Mapped[List["GapFeedbackLink"]] = relationship(
        back_populates="gap", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('open','in_progress','resolved','wontfix')",
            name="check_knowledge_gap_status",
        ),
        Index("idx_knowledge_gap_status", "status"),
        Index("idx_knowledge_gap_source_feedback", "source_feedback_id"),
        Index("idx_knowledge_gap_case", "case_id"),
    )
# ============================================================
# 14. GapFeedbackLink（S2-A 新增，中间表：支撑一对多）
# ============================================================
class GapFeedbackLink(Base):
    __tablename__ = "gap_feedback_link"
    gap_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_gap.id", ondelete="CASCADE"), primary_key=True
    )
    feedback_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("feedback_item.id", ondelete="CASCADE"), primary_key=True
    )

    # 关系
    gap: Mapped["KnowledgeGap"] = relationship(back_populates="feedback_links")
    feedback: Mapped["FeedbackItem"] = relationship()

    __table_args__ = (
        Index("idx_gap_feedback_link_feedback", "feedback_id"),
    )

# ============================================================
# 15. CaseBenchmarkResult（F07 新增，单 Case Benchmark 历史结果）
# ============================================================
class CaseBenchmarkResult(Base):
    __tablename__ = "case_benchmark_result"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    total_gt_defects: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_pred_defects: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    tp: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    fp: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    fn: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    rule_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rule_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ear: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_ner: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metric_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("NOW()")
    )
    __table_args__ = (
        CheckConstraint(
            "mode IN ('rule_only','full_review')",
            name="check_case_benchmark_result_mode",
        ),
        CheckConstraint(
            "status IN ('done','failed')",
            name="check_case_benchmark_result_status",
        ),
        Index("idx_case_benchmark_result_case_id", "case_id"),
        Index("idx_case_benchmark_result_created_at", "created_at"),
    )
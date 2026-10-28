"""S2-A: task_status + rule_candidates benchmark fields + knowledge_gap + gap_feedback_link

Revision ID: s2a_schema_updates
Revises: c1_embedding_dim_1024
Create Date: 2026-10-28

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "s2a_schema_updates"
down_revision = "c1_embedding_dim_1024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ===== 1. review_result: task_status + 索引 =====
    op.add_column(
        "review_result",
        sa.Column("task_status", sa.String(32), nullable=False, server_default="pending"),
    )
    op.create_index(
        "idx_review_result_task_status", "review_result", ["task_status"]
    )

    # ===== 2. rule_candidates: benchmark_result / benchmarked_at / rejected_reason =====
    op.add_column(
        "rule_candidates",
        sa.Column("benchmark_result", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "rule_candidates",
        sa.Column("benchmarked_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "rule_candidates",
        sa.Column("rejected_reason", sa.Text(), nullable=True),
    )

    # ===== 3. knowledge_gap 建表（一对多，不加唯一约束） =====
    op.create_table(
        "knowledge_gap",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("gap_id", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "source_feedback_id",
            sa.BigInteger(),
            sa.ForeignKey("feedback_item.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("case_id", sa.String(64), nullable=True),
        sa.Column("task_id", sa.String(64), nullable=True),
        sa.Column("defect_id", sa.String(64), nullable=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="open"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "resolved_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "status IN ('open','in_progress','resolved','wontfix')",
            name="check_knowledge_gap_status",
        ),
    )
    op.create_index("idx_knowledge_gap_status", "knowledge_gap", ["status"])
    op.create_index(
        "idx_knowledge_gap_source_feedback", "knowledge_gap", ["source_feedback_id"]
    )
    op.create_index("idx_knowledge_gap_case", "knowledge_gap", ["case_id"])

    # ===== 4. gap_feedback_link 中间表（一对多） =====
    op.create_table(
        "gap_feedback_link",
        sa.Column(
            "gap_id",
            sa.BigInteger(),
            sa.ForeignKey("knowledge_gap.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "feedback_id",
            sa.BigInteger(),
            sa.ForeignKey("feedback_item.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    op.create_index(
        "idx_gap_feedback_link_feedback", "gap_feedback_link", ["feedback_id"]
    )


def downgrade() -> None:
    # 逆序回滚
    op.drop_index("idx_gap_feedback_link_feedback", table_name="gap_feedback_link")
    op.drop_table("gap_feedback_link")

    op.drop_index("idx_knowledge_gap_case", table_name="knowledge_gap")
    op.drop_index("idx_knowledge_gap_source_feedback", table_name="knowledge_gap")
    op.drop_index("idx_knowledge_gap_status", table_name="knowledge_gap")
    op.drop_table("knowledge_gap")

    op.drop_column("rule_candidates", "rejected_reason")
    op.drop_column("rule_candidates", "benchmarked_at")
    op.drop_column("rule_candidates", "benchmark_result")

    op.drop_index("idx_review_result_task_status", table_name="review_result")
    op.drop_column("review_result", "task_status")
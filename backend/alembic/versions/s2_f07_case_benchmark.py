"""F07: case_benchmark_result 表（单 Case Benchmark 历史结果）

Revision ID: s2_f07_case_benchmark
Revises: s2_f05_rule_publish
Create Date: 2026-11-02

变更：
  - 新增 case_benchmark_result 表（14 列 + 2 索引）
  - 索引 idx_case_benchmark_result_case_id
  - 索引 idx_case_benchmark_result_created_at
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "s2_f07_case_benchmark"
down_revision = "s2_f05_rule_publish"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "case_benchmark_result",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("total_gt_defects", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_pred_defects", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tp", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fp", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fn", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rule_precision", sa.Float(), nullable=True),
        sa.Column("rule_recall", sa.Float(), nullable=True),
        sa.Column("evc", sa.Float(), nullable=True),
        sa.Column("ear", sa.Float(), nullable=True),
        sa.Column("ai_ner", sa.Float(), nullable=True),
        sa.Column("metric_source", sa.String(32), nullable=True),
        sa.Column("task_id", sa.String(64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        sa.CheckConstraint(
            "mode IN ('rule_only','full_review')",
            name="check_case_benchmark_result_mode",
        ),
        sa.CheckConstraint(
            "status IN ('done','failed')",
            name="check_case_benchmark_result_status",
        ),
    )
    op.create_index(
        "idx_case_benchmark_result_case_id",
        "case_benchmark_result",
        ["case_id"],
    )
    op.create_index(
        "idx_case_benchmark_result_created_at",
        "case_benchmark_result",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_case_benchmark_result_created_at",
        table_name="case_benchmark_result",
    )
    op.drop_index(
        "idx_case_benchmark_result_case_id",
        table_name="case_benchmark_result",
    )
    op.drop_table("case_benchmark_result")
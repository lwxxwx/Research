"""S3 legacy batch4: rule_definition_history

Revision ID: s3_b4_rule_history
Revises: s3_b2_feedback_soft_delete
Create Date: 2026-11-08

D33：规则版本历史表。仅 DDL，不做数据回填。
"""
from alembic import op
import sqlalchemy as sa


revision = "s3_b4_rule_history"
down_revision = "s3_b2_feedback_soft_delete"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rule_definition_history",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("rule_id", sa.String(64), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("rule_name", sa.String(256), nullable=False),
        sa.Column("rule_category", sa.String(64), nullable=False),
        sa.Column("rule_yaml", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(20), nullable=False, server_default="medium"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("change_source", sa.String(16), nullable=False, server_default="publish"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint(
            "rule_id", "version", name="uq_rule_definition_history_rule_version"
        ),
    )
    op.create_index(
        "idx_rule_definition_history_rule",
        "rule_definition_history",
        ["rule_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_rule_definition_history_rule", table_name="rule_definition_history"
    )
    op.drop_table("rule_definition_history")
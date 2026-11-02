"""F05: rule_candidates 发布状态机字段

Revision ID: s2_f05_rule_publish
Revises: s2a_schema_updates
Create Date: 2026-11-01
"""
from alembic import op
import sqlalchemy as sa


revision = "s2_f05_rule_publish"
down_revision = "s2a_schema_updates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "rule_candidates",
        sa.Column(
            "publish_status",
            sa.String(32),
            nullable=False,
            server_default="not_published",
        ),
    )
    op.add_column(
        "rule_candidates",
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "rule_candidates",
        sa.Column("published_version", sa.String(32), nullable=True),
    )
    op.add_column(
        "rule_candidates",
        sa.Column("publish_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "rule_candidates",
        sa.Column("publish_idempotency_key", sa.String(128), nullable=True),
    )

    op.create_check_constraint(
        "check_rule_candidates_publish_status",
        "rule_candidates",
        "publish_status IN ('not_published','publishing','published','publish_failed')",
    )

    op.create_index(
        "idx_rule_candidates_publish_status",
        "rule_candidates",
        ["publish_status"],
    )
    op.create_index(
        "uniq_rule_candidates_publish_idempotency_key",
        "rule_candidates",
        ["publish_idempotency_key"],
        unique=True,
        postgresql_where=sa.text("publish_idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uniq_rule_candidates_publish_idempotency_key",
        table_name="rule_candidates",
    )
    op.drop_index(
        "idx_rule_candidates_publish_status",
        table_name="rule_candidates",
    )
    op.drop_constraint(
        "check_rule_candidates_publish_status",
        "rule_candidates",
        type_="check",
    )
    op.drop_column("rule_candidates", "publish_idempotency_key")
    op.drop_column("rule_candidates", "publish_error")
    op.drop_column("rule_candidates", "published_version")
    op.drop_column("rule_candidates", "published_at")
    op.drop_column("rule_candidates", "publish_status")
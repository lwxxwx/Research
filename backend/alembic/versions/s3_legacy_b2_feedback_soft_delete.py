"""S3 legacy batch2: feedback_item soft delete

Revision ID: s3_legacy_b2_feedback_soft_delete
Revises: s2j_g2_roles
Create Date: 2026-11-08

D26 / D28：仅 DDL，不做数据回填；server_default 保证存量行自动 false。
"""
from alembic import op
import sqlalchemy as sa


revision = "s3_b2_feedback_soft_delete"
down_revision = "s2j_g2_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "feedback_item",
        sa.Column(
            "is_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "idx_feedback_item_is_deleted",
        "feedback_item",
        ["is_deleted"],
    )


def downgrade() -> None:
    op.drop_index("idx_feedback_item_is_deleted", table_name="feedback_item")
    op.drop_column("feedback_item", "is_deleted")
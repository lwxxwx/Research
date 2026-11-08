"""Sprint 3 G2: users 加 role/is_active, review_result 加 created_by, rule_candidates 加 reviewed_by

Revision ID: s2j_g2_roles
Revises: s2a_schema_updates
Create Date: 2026-11-03

变更：
  - users.role VARCHAR(32) NOT NULL DEFAULT 'engineer' + CHECK
  - users.is_active BOOLEAN NOT NULL DEFAULT true
  - review_result.created_by BIGINT FK users.id ON DELETE SET NULL（允许 NULL，517 条历史数据不回填）
  - rule_candidates.reviewed_by BIGINT FK users.id ON DELETE SET NULL（允许 NULL）
  - 索引：idx_users_role、idx_review_result_created_by
  - 索引：idx_rule_candidates_reviewed_by（便于按审核人查询）
"""
from alembic import op
import sqlalchemy as sa


revision = "s2j_g2_roles"
down_revision = "s2_f07_case_benchmark"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. users 加 role + is_active
    op.add_column(
        "users",
        sa.Column(
            "role",
            sa.String(32),
            nullable=False,
            server_default="engineer",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    op.create_check_constraint(
        "check_users_role",
        "users",
        "role IN ('admin','engineer','reviewer','rule_admin')",
    )
    op.create_index("idx_users_role", "users", ["role"])

    # 2. review_result 加 created_by（FK SET NULL）
    op.add_column(
        "review_result",
        sa.Column("created_by", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        "fk_review_result_created_by_users",
        "review_result",
        "users",
        ["created_by"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_review_result_created_by",
        "review_result",
        ["created_by"],
    )

    # 3. rule_candidates 加 reviewed_by（FK SET NULL）
    op.add_column(
        "rule_candidates",
        sa.Column("reviewed_by", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        "fk_rule_candidates_reviewed_by_users",
        "rule_candidates",
        "users",
        ["reviewed_by"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_rule_candidates_reviewed_by",
        "rule_candidates",
        ["reviewed_by"],
    )


def downgrade() -> None:
    # 按相反顺序回滚
    # 3. rule_candidates
    op.drop_index("idx_rule_candidates_reviewed_by", table_name="rule_candidates")
    op.drop_constraint(
        "fk_rule_candidates_reviewed_by_users",
        "rule_candidates",
        type_="foreignkey",
    )
    op.drop_column("rule_candidates", "reviewed_by")

    # 2. review_result
    op.drop_index("idx_review_result_created_by", table_name="review_result")
    op.drop_constraint(
        "fk_review_result_created_by_users",
        "review_result",
        type_="foreignkey",
    )
    op.drop_column("review_result", "created_by")

    # 1. users
    op.drop_index("idx_users_role", table_name="users")
    op.drop_constraint("check_users_role", "users", type_="check")
    op.drop_column("users", "is_active")
    op.drop_column("users", "role")
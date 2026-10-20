"""embedding dim: -> 1024 (zhipu)

Revision ID: c1_embedding_dim_1024
Revises: 42104f6cb89d
Create Date: 2026-10-20

【S1-C 修正】
  原计划 2048 维，但 pgvector HNSW 对 vector 类型上限 2000 维，
  2048 维无法建 HNSW 索引。智谱 embedding-3 支持 Matryoshka 降维，
  选 1024 维（HNSW 友好，检索质量损失可忽略）。

【为何用原生 SQL 而非 op.alter_column】
  诊断发现当前列维度为异常值 1532（非 1536），
  op.alter_column(existing_type=Vector(1536)) 会因类型不匹配失败。
  改用 ALTER TABLE ... TYPE vector(1024)，不校验旧维度，直接改。

  本迁移只改 knowledge_chunk.embedding 列 + 重建 HNSW 索引，不建表。
  清空旧向量与重灌由 scripts/switch_embedding_backend.py 负责。
"""
from alembic import op


revision = "c1_embedding_dim_1024"
down_revision = "42104f6cb89d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_knowledge_chunk_embedding")
    op.execute("UPDATE knowledge_chunk SET embedding = NULL")
    # 不指定 existing_type，直接改列类型，兼容异常维度（如 1532）
    op.execute(
        "ALTER TABLE knowledge_chunk "
        "ALTER COLUMN embedding TYPE vector(1024) "
        "USING embedding::vector(1024)"
    )
    op.execute(
        "CREATE INDEX idx_knowledge_chunk_embedding "
        "ON knowledge_chunk USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_knowledge_chunk_embedding")
    op.execute("UPDATE knowledge_chunk SET embedding = NULL")
    op.execute(
        "ALTER TABLE knowledge_chunk "
        "ALTER COLUMN embedding TYPE vector(1536) "
        "USING embedding::vector(1536)"
    )
    op.execute(
        "CREATE INDEX idx_knowledge_chunk_embedding "
        "ON knowledge_chunk USING hnsw (embedding vector_cosine_ops)"
    )
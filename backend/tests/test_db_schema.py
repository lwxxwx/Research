# tests/test_db_schema.py
"""
Phase‑C Database Schema 冒烟测试
执行命令：
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest tests/test_db_schema.py -v
⚠️ 重要：本测试只做读校验，建表必须通过 scripts.init_db 执行，禁止 Base.metadata.create_all()
"""
from sqlalchemy import create_engine, inspect, text
from app.core.config import settings
from app.persistence.models import Base
import pytest
@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine(settings.database_url, echo=False)
    yield engine
    engine.dispose()
def test_db_connect(db_engine):
    """测试数据库连通性"""
    with db_engine.connect() as conn:
        res = conn.execute(text("SELECT 1 as ok"))
        assert res.scalar() == 1
def test_pgvector_extension_enabled(db_engine):
    """校验pgvector数据库扩展已启用"""
    with db_engine.connect() as conn:
        row = conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).fetchone()
        assert row is not None, "pgvector extension NOT enabled, run init_db.py first"
def test_p0_mandatory_tables_exist(db_engine):
    """校验P0必选业务表全部存在"""
    p0_tables = {
        'users', 'project', 'schematic_case', 'ir_document',
        'rule_definition', 'rule_execution', 'review_result', 'feedback_item',
        'knowledge_doc', 'knowledge_chunk', 'review_defect'
    }
    insp = inspect(db_engine)
    actual_tables = set(insp.get_table_names(schema="public"))
    missing = p0_tables - actual_tables
    assert len(missing) == 0, f"Missing P0 tables: {missing}"
def test_review_defect_required_columns(db_engine):
    """校验review_defect关键字段"""
    insp = inspect(db_engine)
    cols = {c["name"] for c in insp.get_columns("review_defect", schema="public")}
    required = {"defect_id", "category", "risk", "review_status", "origin"}
    missing = required - cols
    assert len(missing) == 0, f"review_defect missing columns: {missing}"
def test_feedback_item_has_review_defect_id(db_engine):
    """校验feedback_item存在review_defect_id字段"""
    insp = inspect(db_engine)
    cols = {c["name"] for c in insp.get_columns("feedback_item", schema="public")}
    assert "review_defect_id" in cols, "feedback_item missing review_defect_id column"
def test_review_result_v12_columns(db_engine):
    """校验review_result V1.2关键字段 category/risk/review_status"""
    insp = inspect(db_engine)
    cols = {c["name"] for c in insp.get_columns("review_result", schema="public")}
    required = {"category", "risk", "review_status"}
    missing = required - cols
    assert len(missing) == 0, f"review_result missing V1.2 columns: {missing}"
def test_feedback_check_constraint_exists(db_engine):
    """校验feedback_item check_feedback_item_type约束存在"""
    with db_engine.connect() as conn:
        row = conn.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_schema='public' AND table_name='feedback_item'
            AND constraint_type='CHECK' AND constraint_name='check_feedback_item_type'
        """)).fetchone()
        assert row is not None, "feedback_item check_feedback_item_type constraint missing"
def test_orm_metadata_table_names_match_db(db_engine):
    """ORM模型元数据表集合 和数据库实际表名比对（只比对表名，不做列级映射）"""
    insp = inspect(db_engine)
    db_tables = set(insp.get_table_names(schema="public"))
    orm_tables = set(Base.metadata.tables.keys())
    # ORM内全部表必须真实存在DB
    orm_only = orm_tables - db_tables
    assert len(orm_only) == 0, f"ORM has tables which not exist in DB: {orm_only}"

# ===== ✅ NEW Sprint0 Phase‑I 新增冒烟测试 =====
def test_rule_candidates_table_exists(db_engine):
    """Phase‑I：校验新增rule_candidates表存在"""
    insp = inspect(db_engine)
    actual_tables = set(insp.get_table_names(schema="public"))
    assert "rule_candidates" in actual_tables, "Missing table rule_candidates (Phase‑I)"

def test_feedback_item_phasei_new_columns(db_engine):
    """Phase‑I：校验feedback_item新增suggestion_diff_json / rule_candidate_ref字段"""
    insp = inspect(db_engine)
    cols = {c["name"] for c in insp.get_columns("feedback_item", schema="public")}
    require_cols = {"suggestion_diff_json","rule_candidate_ref","attached_refs"}
    missing = require_cols - cols
    assert len(missing)==0, f"feedback_item missing Phase‑I columns: {missing}"

def test_rule_candidates_indexes(db_engine):
    """校验rule_candidates关键索引存在"""
    with db_engine.connect() as conn:
        res = conn.execute(text("""
        SELECT indexname FROM pg_indexes WHERE tablename='rule_candidates'
        """)).fetchall()
        idx_names = {r[0] for r in res}
        expect = {"idx_rule_candidates_status","idx_rule_candidates_case_id"}
        missing = expect - idx_names
        assert len(missing)==0, f"rule_candidates missing indexes: {missing}"

def test_rule_candidates_trigger_exists(db_engine):
    """校验rule_candidates的updated_at自动更新触发器"""
    with db_engine.connect() as conn:
        res = conn.execute(text("""
        SELECT tgname FROM pg_trigger WHERE tgrelid='rule_candidates'::regclass
        """)).fetchall()
        trigger_names = {r[0] for r in res}
        assert "update_rule_candidates_updated_at" in trigger_names, "rule_candidates missing update_rule_candidates_updated_at trigger"

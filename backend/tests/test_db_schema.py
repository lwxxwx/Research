# tests/test_db_schema.py
"""
Phase-C Database Schema 冒烟测试（含 Sprint0 Phase-I 扩展）

执行命令（PowerShell 单行，推荐）：
    docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec -T backend uv run pytest tests/test_db_schema.py -v

⚠️ 重要：本测试只做读校验，建表必须通过 scripts.init_db 执行，禁止 Base.metadata.create_all()
"""
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
from app.core.config import settings
from app.persistence.models import Base
import pytest


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine(settings.database_url, echo=False)
    yield engine
    engine.dispose()


# ============================================================
# 0. 连通性 & 扩展
# ============================================================
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


# ============================================================
# 1. 表存在性
# ============================================================
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


def test_rule_candidates_table_exists(db_engine):
    """Phase-I：校验新增rule_candidates表存在"""
    insp = inspect(db_engine)
    actual_tables = set(insp.get_table_names(schema="public"))
    assert "rule_candidates" in actual_tables, "Missing table rule_candidates (Phase-I)"


# ============================================================
# 2. 关键列存在性
# ============================================================
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


def test_feedback_item_phasei_new_columns(db_engine):
    """Phase-I：校验feedback_item新增suggestion_diff_json / rule_candidate_ref字段"""
    insp = inspect(db_engine)
    cols = {c["name"] for c in insp.get_columns("feedback_item", schema="public")}
    require_cols = {"suggestion_diff_json", "rule_candidate_ref", "attached_refs"}
    missing = require_cols - cols
    assert len(missing) == 0, f"feedback_item missing Phase-I columns: {missing}"


# ============================================================
# 3. ORM ↔ DB 双向一致性
# ============================================================
def test_orm_metadata_table_names_match_db(db_engine):
    """ORM↔DB 表名双向比对：DB 有 ORM 无、ORM 有 DB 无，都要报出来"""
    insp = inspect(db_engine)
    db_tables = set(insp.get_table_names(schema="public"))
    orm_tables = set(Base.metadata.tables.keys())

    orm_only = orm_tables - db_tables
    db_only = db_tables - orm_tables
    assert not orm_only, f"ORM 有但 DB 缺失的表: {sorted(orm_only)}"
    assert not db_only, f"DB 有但 ORM 未声明的表: {sorted(db_only)}"


def test_orm_columns_match_db(db_engine):
    """ORM↔DB 列级双向比对：抓 task_id 这类 DB 有 ORM 缺的字段"""
    insp = inspect(db_engine)
    mismatches = {}
    for tname in sorted(Base.metadata.tables.keys()):
        db_cols = {c["name"] for c in insp.get_columns(tname, schema="public")}
        orm_cols = {c.name for c in Base.metadata.tables[tname].columns}
        miss = db_cols - orm_cols   # DB 有、ORM 缺
        extra = orm_cols - db_cols  # ORM 有、DB 缺
        if miss or extra:
            mismatches[tname] = {
                "db_only": sorted(miss),
                "orm_only": sorted(extra),
            }
    assert not mismatches, f"列级不一致: {mismatches}"


# ============================================================
# 4. 索引
# ============================================================
def test_rule_candidates_indexes(db_engine):
    """校验 rule_candidates 三个关键索引全部存在"""
    with db_engine.connect() as conn:
        res = conn.execute(text("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname='public' AND tablename='rule_candidates'
        """)).fetchall()
        idx_names = {r[0] for r in res}
        expect = {
            "idx_rule_candidates_status",
            "idx_rule_candidates_case_id",
            "idx_rule_candidates_from_feedback_id",
        }
        missing = expect - idx_names
        assert not missing, f"rule_candidates missing indexes: {missing}"


def test_all_declared_indexes_exist(db_engine):
    """校验 ORM 声明的所有 Index 在 DB 中都真实存在"""
    insp = inspect(db_engine)
    missing = {}
    for tname, table in Base.metadata.tables.items():
        db_idx = {i["name"] for i in insp.get_indexes(tname, schema="public")}
        orm_idx = {i.name for i in table.indexes}
        miss = orm_idx - db_idx
        if miss:
            missing[tname] = sorted(miss)
    assert not missing, f"ORM 声明但 DB 缺失的索引: {missing}"


def test_pgvector_hnsw_index_exists(db_engine):
    """校验 knowledge_chunk 的 HNSW 向量索引存在"""
    with db_engine.connect() as conn:
        row = conn.execute(text("""
            SELECT indexdef FROM pg_indexes
            WHERE schemaname='public'
              AND tablename='knowledge_chunk'
              AND indexdef ILIKE '%hnsw%'
        """)).fetchone()
        assert row is not None, "knowledge_chunk 缺 HNSW 向量索引"


# ============================================================
# 5. CHECK 约束
# ============================================================
def test_feedback_check_constraint_exists(db_engine):
    """校验feedback_item check_feedback_item_type约束存在"""
    with db_engine.connect() as conn:
        row = conn.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_schema='public' AND table_name='feedback_item'
            AND constraint_type='CHECK' AND constraint_name='check_feedback_item_type'
        """)).fetchone()
        assert row is not None, "feedback_item check_feedback_item_type constraint missing"


def _fetch_check_names(conn, table):
    rows = conn.execute(text("""
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_schema='public'
          AND table_name = :t
          AND constraint_type = 'CHECK'
    """), {"t": table}).fetchall()
    return {r[0] for r in rows}


def test_check_constraints_all_present(db_engine):
    """核对各表关键 CHECK 约束"""
    expected = {
        "feedback_item": {"check_feedback_item_type"},
    }
    with db_engine.connect() as conn:
        for tbl, want in expected.items():
            got = _fetch_check_names(conn, tbl)
            missing = want - got
            assert not missing, f"{tbl} 缺 CHECK: {missing}，实际: {got}"


# ============================================================
# 6. 触发器
# ============================================================
def test_rule_candidates_trigger_exists(db_engine):
    """校验 rule_candidates 的 updated_at 自动更新触发器（不依赖 ::regclass）"""
    with db_engine.connect() as conn:
        res = conn.execute(text("""
            SELECT t.tgname
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'rule_candidates' AND n.nspname = 'public'
        """)).fetchall()
        trigger_names = {r[0] for r in res}
        assert "update_rule_candidates_updated_at" in trigger_names, \
            f"rule_candidates missing trigger, got: {trigger_names}"


def test_all_updated_at_triggers_exist(db_engine):
    """所有应带 updated_at 的表都要有对应触发器"""
    expect_triggers = {
        "update_users_updated_at",
        "update_project_updated_at",
        "update_schematic_case_updated_at",
        "update_rule_definition_updated_at",
        "update_knowledge_doc_updated_at",
        "update_rule_candidates_updated_at",
    }
    with db_engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.tgname
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname='public' AND NOT t.tgisinternal
        """)).fetchall()
        got = {r[0] for r in rows}
    missing = expect_triggers - got
    assert not missing, f"缺失 updated_at 触发器: {missing}，实际: {sorted(got)}"


def test_trigger_function_exists(db_engine):
    """通用触发器函数 update_updated_at_column 存在"""
    with db_engine.connect() as conn:
        row = conn.execute(text("""
            SELECT proname FROM pg_proc
            WHERE proname='update_updated_at_column'
        """)).fetchone()
        assert row is not None, "缺函数 update_updated_at_column"


# ============================================================
# 7. UNIQUE 约束
# ============================================================
def _fetch_unique_constraints(conn, table):
    rows = conn.execute(text("""
        SELECT tc.constraint_name, kcu.column_name, kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_schema='public'
          AND tc.table_name = :t
          AND tc.constraint_type = 'UNIQUE'
        ORDER BY tc.constraint_name, kcu.ordinal_position
    """), {"t": table}).fetchall()
    grouped = {}
    for cname, col, _ in rows:
        grouped.setdefault(cname, []).append(col)
    return {tuple(v) for v in grouped.values()}


def test_users_username_unique(db_engine):
    with db_engine.connect() as conn:
        got = _fetch_unique_constraints(conn, "users")
    assert ("username",) in got, f"users 缺 UNIQUE(username)，实际: {got}"


def test_schematic_case_composite_unique(db_engine):
    """schematic_case UNIQUE(project_id, case_id)"""
    with db_engine.connect() as conn:
        got = _fetch_unique_constraints(conn, "schematic_case")
    assert ("project_id", "case_id") in got, \
        f"schematic_case 缺 UNIQUE(project_id, case_id)，实际: {got}"


def test_rule_definition_rule_id_unique(db_engine):
    with db_engine.connect() as conn:
        got = _fetch_unique_constraints(conn, "rule_definition")
    assert ("rule_id",) in got, f"rule_definition 缺 UNIQUE(rule_id)，实际: {got}"


def test_review_defect_defect_id_unique(db_engine):
    with db_engine.connect() as conn:
        got = _fetch_unique_constraints(conn, "review_defect")
    assert ("defect_id",) in got, f"review_defect 缺 UNIQUE(defect_id)，实际: {got}"


def test_rule_candidates_candidate_id_unique(db_engine):
    with db_engine.connect() as conn:
        got = _fetch_unique_constraints(conn, "rule_candidates")
    assert ("candidate_id",) in got, \
        f"rule_candidates 缺 UNIQUE(candidate_id)，实际: {got}"


# ============================================================
# 8. 外键 & 级联
# ============================================================
def _fetch_fks(conn, table):
    rows = conn.execute(text("""
        SELECT
            kcu.column_name,
            ccu.table_name   AS ref_table,
            ccu.column_name  AS ref_column,
            rc.delete_rule
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        JOIN information_schema.referential_constraints rc
          ON tc.constraint_name = rc.constraint_name
        WHERE tc.table_schema='public'
          AND tc.table_name = :t
          AND tc.constraint_type = 'FOREIGN KEY'
    """), {"t": table}).fetchall()
    return {(r[0], r[1], r[2], r[3]) for r in rows}


def test_schematic_case_cascade_delete(db_engine):
    """schematic_case.project_id → project.id ON DELETE CASCADE"""
    with db_engine.connect() as conn:
        fks = _fetch_fks(conn, "schematic_case")
    match = [fk for fk in fks if fk[0] == "project_id" and fk[1] == "project"]
    assert match, f"schematic_case 缺 project_id 外键，实际: {fks}"
    assert match[0][3] == "CASCADE", \
        f"schematic_case.project_id 应为 CASCADE，实际: {match[0][3]}"


def test_ir_document_cascade_delete(db_engine):
    """ir_document.schematic_case_id → schematic_case.id ON DELETE CASCADE"""
    with db_engine.connect() as conn:
        fks = _fetch_fks(conn, "ir_document")
    match = [fk for fk in fks if fk[0] == "schematic_case_id"]
    assert match, f"ir_document 缺 schematic_case_id 外键，实际: {fks}"
    assert match[0][3] == "CASCADE", \
        f"ir_document.schematic_case_id 应为 CASCADE，实际: {match[0][3]}"


def test_feedback_item_dual_fk(db_engine):
    """feedback_item 同时外键 review_result 与 review_defect"""
    with db_engine.connect() as conn:
        fks = _fetch_fks(conn, "feedback_item")
    cols = {fk[0] for fk in fks}
    assert "review_result_id" in cols, f"feedback_item 缺 review_result_id 外键，实际: {fks}"
    assert "review_defect_id" in cols, f"feedback_item 缺 review_defect_id 外键，实际: {fks}"


# ============================================================
# 9. 列类型抽查
# ============================================================
def _fetch_column_types(conn, table):
    rows = conn.execute(text("""
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name = :t
    """), {"t": table}).fetchall()
    return {r[0]: {"data_type": r[1], "udt_name": r[2]} for r in rows}


def test_timestamp_with_timezone(db_engine):
    """关键表的时间列必须带时区"""
    check = {
        "users": ["created_at", "updated_at"],
        "review_result": ["created_at"],
        "rule_candidates": ["created_at", "updated_at"],
    }
    with db_engine.connect() as conn:
        for tbl, cols in check.items():
            types = _fetch_column_types(conn, tbl)
            for col in cols:
                assert col in types, f"{tbl}.{col} 不存在"
                assert types[col]["data_type"] == "timestamp with time zone", \
                    f"{tbl}.{col} 应为 TIMESTAMPTZ，实际: {types[col]['data_type']}"


def test_jsonb_columns_are_jsonb(db_engine):
    """关键 JSONB 列类型正确"""
    check = {
        "ir_document": ["ir_json"],
        "review_result": ["review_output_json"],
        "review_defect": ["location", "evidence"],
        "feedback_item": ["suggestion_diff_json", "attached_refs"],
        "knowledge_doc": ["metadata"],
        "knowledge_chunk": ["metadata"],
    }
    with db_engine.connect() as conn:
        for tbl, cols in check.items():
            types = _fetch_column_types(conn, tbl)
            for col in cols:
                assert col in types, f"{tbl}.{col} 不存在"
                assert types[col]["data_type"] == "jsonb", \
                    f"{tbl}.{col} 应为 JSONB，实际: {types[col]['data_type']}"


def test_knowledge_chunk_embedding_is_vector(db_engine):
    """knowledge_chunk.embedding 是 vector 类型"""
    with db_engine.connect() as conn:
        types = _fetch_column_types(conn, "knowledge_chunk")
    assert "embedding" in types, "knowledge_chunk 缺 embedding 列"
    assert types["embedding"]["udt_name"] == "vector", \
        f"embedding 应为 vector，实际: {types['embedding']['udt_name']}"


# ============================================================
# 10. ORM 写入冒烟（不破坏数据）
# ============================================================
def test_orm_can_query_every_table(db_engine):
    """ORM 能对每张表执行一次 SELECT，验证映射没坏"""
    errors = {}
    with Session(db_engine) as session:
        for tname, table in Base.metadata.tables.items():
            try:
                session.execute(table.select().limit(1)).fetchall()
            except Exception as e:
                errors[tname] = str(e)
    assert not errors, f"ORM 查询失败: {errors}"



# ============================================================
# 11. 级联删除运行时行为（真正跑 DELETE 验证 CASCADE）
# ============================================================
#
# 说明：
# - 使用独立 Session，测试结束显式回滚，不留脏数据。
# - 只依赖已在 002_schema.sql 里声明的 ON DELETE CASCADE。
# - 若某张表 NOT NULL 字段没给，测试会在 setup 阶段报错，说明 models.py
#   与 SQL 不一致，属于有效失败。

from sqlalchemy.orm import Session
from app.persistence.models import (
    User, Project, SchematicCase, IRDocument,
    ReviewResult, ReviewDefect, FeedbackItem,
    KnowledgeDoc, KnowledgeChunk,
    RuleDefinition, RuleCandidate,   # ✅ 新增：约束测试用
)


@pytest.fixture()
def db_session(db_engine):
    """
    每个测试独享一个 Session，绑定到外层事务；
    测试结束无条件 rollback 外层事务，DB 回到测试前状态。
    序列（BIGSERIAL）不回滚，但不影响任何断言。
    """
    connection = db_engine.connect()
    trans = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()


# ---------- 11.1 Project → SchematicCase → IRDocument ----------

def test_cascade_project_deletes_cases_and_ir(db_session):
    """删 Project 应级联删 SchematicCase 及其下的 IRDocument"""
    # --- setup ---
    user = User(username="cascade_u1", email="c1@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="cascade_proj_1", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(
        project_id=proj.id,
        case_id="case_cascade_1",
        case_type="A",
    )
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(
        schematic_case_id=case.id,
        ir_json={"nodes": [], "edges": []},
        ir_schema_version="v1.0",
    )
    db_session.add(ir)
    db_session.flush()

    proj_id, case_id, ir_id = proj.id, case.id, ir.id

    # --- action: 删除 Project ---
    db_session.delete(proj)
    db_session.flush()

    # --- assert ---
    assert db_session.get(Project, proj_id) is None, "Project 未删除"
    assert db_session.get(SchematicCase, case_id) is None, \
        "SchematicCase 未随 Project 级联删除"
    assert db_session.get(IRDocument, ir_id) is None, \
        "IRDocument 未随 SchematicCase 级联删除"

    # 清理 User（避免残留；rollback 也能兜底）
    db_session.delete(user)
    db_session.flush()


# ---------- 11.2 SchematicCase 独立删除 → IRDocument ----------

def test_cascade_schematic_case_deletes_ir(db_session):
    """单独删 SchematicCase 应级联删其下 IRDocument（不动 Project）"""
    user = User(username="cascade_u2", email="c2@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="cascade_proj_2", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(
        project_id=proj.id,
        case_id="case_cascade_2",
        case_type="B",
    )
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(
        schematic_case_id=case.id,
        ir_json={"nodes": [{"id": "n1"}]},
        ir_schema_version="v1.0",
    )
    db_session.add(ir)
    db_session.flush()

    proj_id, case_id, ir_id = proj.id, case.id, ir.id

    db_session.delete(case)
    db_session.flush()

    assert db_session.get(SchematicCase, case_id) is None
    assert db_session.get(IRDocument, ir_id) is None, \
        "IRDocument 未随 SchematicCase 级联删除"
    # Project 应保留
    assert db_session.get(Project, proj_id) is not None, \
        "删 Case 不应影响 Project"

    db_session.delete(proj)
    db_session.delete(user)
    db_session.flush()


# ---------- 11.3 ReviewResult → ReviewDefect → FeedbackItem ----------

def test_cascade_review_result_deletes_defects_and_feedbacks(db_session):
    """删 ReviewResult 应级联删 ReviewDefect 与其下 FeedbackItem"""
    user = User(username="cascade_u3", email="c3@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="cascade_proj_3", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(
        project_id=proj.id,
        case_id="case_cascade_3",
        case_type="A",
    )
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(
        schematic_case_id=case.id,
        ir_json={"nodes": []},
        ir_schema_version="v1.0",
    )
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(
        ir_document_id=ir.id,
        review_output_json={"summary": "test"},
        is_rule_only=True,
    )
    db_session.add(rr)
    db_session.flush()

    defect = ReviewDefect(
        review_result_id=rr.id,
        defect_id="DEF-CASCADE-001",
        location={"sheet": "s1", "path": "/", "coords": [0, 0]},
        root_cause="test",
        suggestion="fix it",
    )
    db_session.add(defect)
    db_session.flush()

    fb = FeedbackItem(
        review_result_id=rr.id,
        review_defect_id=defect.id,
        feedback_type="correct_defect",
        comment="test comment",
        created_by=user.id,
    )
    db_session.add(fb)
    db_session.flush()

    rr_id, defect_id, fb_id = rr.id, defect.id, fb.id

    # 删 ReviewResult
    db_session.delete(rr)
    db_session.flush()

    assert db_session.get(ReviewResult, rr_id) is None
    assert db_session.get(ReviewDefect, defect_id) is None, \
        "ReviewDefect 未随 ReviewResult 级联删除"
    assert db_session.get(FeedbackItem, fb_id) is None, \
        "FeedbackItem 未随 ReviewResult 级联删除"

    # 清理上游
    db_session.delete(ir)
    db_session.delete(case)
    db_session.delete(proj)
    db_session.delete(user)
    db_session.flush()


# ---------- 11.4 ReviewDefect 独立删除 → FeedbackItem ----------

def test_cascade_review_defect_deletes_feedbacks(db_session):
    """单独删 ReviewDefect 应级联删其下 FeedbackItem（不动 ReviewResult）"""
    user = User(username="cascade_u4", email="c4@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="cascade_proj_4", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(
        project_id=proj.id,
        case_id="case_cascade_4",
        case_type="A",
    )
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(
        schematic_case_id=case.id,
        ir_json={"nodes": []},
        ir_schema_version="v1.0",
    )
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(
        ir_document_id=ir.id,
        review_output_json={"summary": "test4"},
        is_rule_only=True,
    )
    db_session.add(rr)
    db_session.flush()

    defect = ReviewDefect(
        review_result_id=rr.id,
        defect_id="DEF-CASCADE-002",
        location={"sheet": "s2", "path": "/x", "coords": [1, 1]},
        root_cause="rc",
        suggestion="sug",
    )
    db_session.add(defect)
    db_session.flush()

    fb = FeedbackItem(
        review_result_id=rr.id,
        review_defect_id=defect.id,
        feedback_type="false_positive",
        comment="fp",
        created_by=user.id,
    )
    db_session.add(fb)
    db_session.flush()

    rr_id, defect_id, fb_id = rr.id, defect.id, fb.id

    db_session.delete(defect)
    db_session.flush()

    assert db_session.get(ReviewDefect, defect_id) is None
    assert db_session.get(FeedbackItem, fb_id) is None, \
        "FeedbackItem 未随 ReviewDefect 级联删除"
    # ReviewResult 应保留
    assert db_session.get(ReviewResult, rr_id) is not None, \
        "删 Defect 不应影响 ReviewResult"

    db_session.delete(rr)
    db_session.delete(ir)
    db_session.delete(case)
    db_session.delete(proj)
    db_session.delete(user)
    db_session.flush()


# ---------- 11.5 KnowledgeDoc → KnowledgeChunk ----------

def test_cascade_knowledge_doc_deletes_chunks(db_session):
    """删 KnowledgeDoc 应级联删其下 KnowledgeChunk"""
    doc = KnowledgeDoc(
        title="cascade_doc_1",
        source="test",
        source_type="datasheet",
        content_md="# test",
    )
    db_session.add(doc)
    db_session.flush()

    chunk = KnowledgeChunk(
        knowledge_doc_id=doc.id,
        chunk_text="chunk body",
    )
    db_session.add(chunk)
    db_session.flush()

    doc_id, chunk_id = doc.id, chunk.id

    db_session.delete(doc)
    db_session.flush()

    assert db_session.get(KnowledgeDoc, doc_id) is None
    assert db_session.get(KnowledgeChunk, chunk_id) is None, \
        "KnowledgeChunk 未随 KnowledgeDoc 级联删除"


# ---------- 11.6 Project.owner_id 无 CASCADE：删 User 不应删 Project ----------

def test_project_owner_is_not_cascade(db_session):
    """
    验证 DB 层 project.owner_id 的外键是 NO ACTION：
    用原生 SQL 直接 DELETE users，若还有 Project 引用，应抛 IntegrityError。
    绕过 ORM 的自动置空逻辑。

    技巧：用嵌套 SAVEPOINT，让失败的 DELETE 只回滚它自己，不动 setup 数据。
    """
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    # --- setup：用 ORM 建数据 ---
    user = User(username="cascade_u5", email="c5@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="cascade_proj_5", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    user_id, proj_id = user.id, proj.id

    # --- action：用嵌套 SAVEPOINT 包住那条 DELETE ---
    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})
        db_session.flush()
    except IntegrityError:
        raised = True
        nested.rollback()   # 只回滚嵌套层，setup 数据保留
    else:
        nested.commit()

    assert raised, (
        "DB 层 project.owner_id 应为 NO ACTION，"
        "原生 SQL 删 users 时若仍有 Project 引用必须抛 IntegrityError。"
        "请检查 002_schema.sql 里 project.owner_id 的 REFERENCES 是否被误加了 ON DELETE CASCADE/SET NULL"
    )

    # --- assert：Project 应仍在（嵌套 savepoint 回滚后 DB 未变） ---
    proj_after = db_session.get(Project, proj_id)
    assert proj_after is not None, "User 删除被 DB 拒绝后，Project 应仍存在"



# ============================================================
# 12. 隔离性验证（确保测试后回到初始状态）
# ============================================================

def test_isolation_no_dirty_data_left(db_engine):
    """
    验证：造一批脏数据，事务回滚后，各表行数与之前一致。
    """
    from sqlalchemy import text
    tables = [
        "users", "project", "schematic_case", "ir_document",
        "rule_definition", "rule_execution", "review_result",
        "review_defect", "feedback_item", "knowledge_doc",
        "knowledge_chunk", "rule_candidates",
    ]

    def counts(engine):
        with engine.connect() as conn:
            return {
                t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                for t in tables
            }

    before = counts(db_engine)

    # 造脏数据
    connection = db_engine.connect()
    trans = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        u = User(username="isolation_test_u", email="iso@test.local")
        session.add(u)
        session.flush()
        p = Project(name="isolation_test_p", owner_id=u.id)
        session.add(p)
        session.flush()
        session.commit()  # 被 SAVEPOINT 拦截，不会真提交
    finally:
        session.close()
        trans.rollback()
        connection.close()

    after = counts(db_engine)
    assert before == after, f"存在脏数据: before={before}, after={after}"


def test_orm_nullifies_project_owner_when_deleting_user(db_session):
    """
    固化 SQLAlchemy ORM 的默认行为：
    - owner_id 可空
    - relationship 未配 cascade="delete"
    => session.delete(user) 会把引用该 user 的 project.owner_id 置 NULL，而不是删 project。
    这是 ORM 预期行为，非 bug。
    """
    user = User(username="orm_null_u1", email="nu1@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="orm_null_proj_1", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    user_id, proj_id = user.id, proj.id

    db_session.delete(user)
    db_session.flush()

    assert db_session.get(User, user_id) is None, "User 未被 ORM 删除"
    proj_after = db_session.get(Project, proj_id)
    assert proj_after is not None, "Project 不应被 ORM 删除"
    assert proj_after.owner_id is None, \
        "ORM 删 User 后应把 project.owner_id 置 NULL（默认行为）" 



# ============================================================
# 13. 约束运行时行为（故意插脏数据，断言 DB 抛 IntegrityError）
# ============================================================
#
# 设计说明：
# - 每条测试用 db_session（外层事务 + SAVEPOINT），结束无条件回滚，不留脏数据。
# - 关键技巧：用 db_session.begin_nested() 开嵌套 SAVEPOINT，
#   让"故意失败"的 INSERT 只回滚它自己，不影响后续断言与清理。
# - 只依赖 002_schema.sql 里声明的约束，任何一条不生效都应 FAILED。
#
# 覆盖清单：
#   13.1  CHECK  feedback_item.feedback_type 六选一
#   13.2  UNIQUE users.username
#   13.3  UNIQUE schematic_case(project_id, case_id) 复合
#   13.4  UNIQUE rule_definition.rule_id
#   13.5  UNIQUE review_defect.defect_id
#   13.6  UNIQUE rule_candidates.candidate_id
#   13.7  NOT NULL users.username
#   13.8  NOT NULL review_result.review_output_json
#   13.9  NOT NULL review_defect.location / root_cause / suggestion
#   13.10 FK   feedback_item.created_by → users.id（不存在用户）
#   13.11 FK   ir_document.schematic_case_id → schematic_case.id（不存在 case）
#   13.12 正向对照：合法插入应成功（防止约束误伤）

from sqlalchemy import text as _sql_text
from sqlalchemy.exc import IntegrityError as _IntegrityError


# ---------- 13.1 CHECK：feedback_type 六选一 ----------

def test_constraint_feedback_type_check_rejects_invalid(db_session):
    """
    故意插入非法 feedback_type，DB 的 CHECK 约束应抛 IntegrityError。
    覆盖 002_schema.sql 里的 check_feedback_item_type。
    """
    nested = db_session.begin_nested()
    raised = False
    try:
        # 直接走原生 SQL，避免 ORM 侧可能的 Enum 拦截（这里模型是 String，无拦截）
        db_session.execute(_sql_text("""
            INSERT INTO feedback_item (feedback_type, comment)
            VALUES ('hacked_type', 'should fail')
        """))
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "feedback_type='hacked_type' 应被 CHECK 约束拒绝，实际写入成功"


def test_constraint_feedback_type_check_accepts_all_six(db_session):
    """正向对照：6 种合法 feedback_type 全部能写入（防止约束误伤）"""
    legal = [
        "correct_defect",
        "false_positive",
        "false_negative",
        "suggestion_update",
        "new_rule_candidate",
        "knowledge_gap",
    ]
    for ft in legal:
        nested = db_session.begin_nested()
        try:
            db_session.execute(_sql_text("""
                INSERT INTO feedback_item (feedback_type, comment)
                VALUES (:ft, 'ok')
            """), {"ft": ft})
            db_session.flush()
        finally:
            nested.rollback()   # 立即清掉，避免互相影响


# ---------- 13.2 UNIQUE：users.username ----------

def test_constraint_users_username_unique(db_session):
    """重复 username 应抛 IntegrityError"""
    u1 = User(username="uniq_user_a", email="a@test.local")
    db_session.add(u1)
    db_session.flush()

    nested = db_session.begin_nested()
    raised = False
    try:
        u2 = User(username="uniq_user_a", email="b@test.local")
        db_session.add(u2)
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "重复 username 应被 UNIQUE 约束拒绝"


# ---------- 13.3 UNIQUE 复合：schematic_case(project_id, case_id) ----------

def test_constraint_schematic_case_composite_unique(db_session):
    """
    同一个 project 下不能有重复 case_id；
    不同 project 下可以有相同 case_id（复合唯一语义）。
    """
    user = User(username="uniq_u3", email="u3@test.local")
    db_session.add(user)
    db_session.flush()

    proj_a = Project(name="uniq_proj_a", owner_id=user.id)
    proj_b = Project(name="uniq_proj_b", owner_id=user.id)
    db_session.add_all([proj_a, proj_b])
    db_session.flush()

    # 第一次插入：OK
    c1 = SchematicCase(project_id=proj_a.id, case_id="case_x", case_type="A")
    db_session.add(c1)
    db_session.flush()

    # 同 project 下重复 case_id：应失败
    nested = db_session.begin_nested()
    raised_same = False
    try:
        c2 = SchematicCase(project_id=proj_a.id, case_id="case_x", case_type="A")
        db_session.add(c2)
        db_session.flush()
    except _IntegrityError:
        raised_same = True
        nested.rollback()
    else:
        nested.commit()
    assert raised_same, "同一 project 下重复 case_id 应被复合 UNIQUE 拒绝"

    # 不同 project 下同 case_id：应成功（证明复合语义）
    nested2 = db_session.begin_nested()
    try:
        c3 = SchematicCase(project_id=proj_b.id, case_id="case_x", case_type="A")
        db_session.add(c3)
        db_session.flush()
    finally:
        nested2.rollback()


# ---------- 13.4 UNIQUE：rule_definition.rule_id ----------

def test_constraint_rule_definition_rule_id_unique(db_session):
    """重复 rule_id 应抛 IntegrityError"""
    r1 = RuleDefinition(
        rule_id="RULE_UNIQ_001",
        rule_name="r1",
        rule_category="POWER",
        rule_yaml="x: 1",
    )
    db_session.add(r1)
    db_session.flush()

    nested = db_session.begin_nested()
    raised = False
    try:
        r2 = RuleDefinition(
            rule_id="RULE_UNIQ_001",
            rule_name="r2",
            rule_category="CLOCK",
            rule_yaml="y: 2",
        )
        db_session.add(r2)
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "重复 rule_id 应被 UNIQUE 拒绝"


# ---------- 13.5 UNIQUE：review_defect.defect_id ----------

def test_constraint_review_defect_defect_id_unique(db_session):
    """重复 defect_id 应抛 IntegrityError"""
    # setup 上游
    user = User(username="uniq_u5", email="u5@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="uniq_proj_5", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="uniq_case_5", case_type="A")
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(schematic_case_id=case.id, ir_json={"a": 1}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(ir_document_id=ir.id, review_output_json={"s": 1})
    db_session.add(rr)
    db_session.flush()

    d1 = ReviewDefect(
        review_result_id=rr.id,
        defect_id="DEF-UNIQ-001",
        location={"sheet": "s"},
        root_cause="rc",
        suggestion="sg",
    )
    db_session.add(d1)
    db_session.flush()

    nested = db_session.begin_nested()
    raised = False
    try:
        d2 = ReviewDefect(
            review_result_id=rr.id,
            defect_id="DEF-UNIQ-001",   # 重复
            location={"sheet": "s2"},
            root_cause="rc2",
            suggestion="sg2",
        )
        db_session.add(d2)
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "重复 defect_id 应被 UNIQUE 拒绝"


# ---------- 13.6 UNIQUE：rule_candidates.candidate_id ----------

def test_constraint_rule_candidates_candidate_id_unique(db_session):
    """重复 candidate_id 应抛 IntegrityError"""
    r1 = RuleCandidate(
        candidate_id="RC-UNIQ-001",
        title="t1",
        description="d1",
    )
    db_session.add(r1)
    db_session.flush()

    nested = db_session.begin_nested()
    raised = False
    try:
        r2 = RuleCandidate(
            candidate_id="RC-UNIQ-001",   # 重复
            title="t2",
            description="d2",
        )
        db_session.add(r2)
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "重复 candidate_id 应被 UNIQUE 拒绝"


# ---------- 13.7 NOT NULL：users.username ----------

def test_constraint_users_username_not_null(db_session):
    """username 为 NULL 应抛 IntegrityError"""
    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(_sql_text("""
            INSERT INTO users (username, email) VALUES (NULL, 'x@test.local')
        """))
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "users.username 为 NULL 应被 NOT NULL 拒绝"


# ---------- 13.8 NOT NULL：review_result.review_output_json ----------

def test_constraint_review_result_output_json_not_null(db_session):
    """review_output_json 为 NULL 应抛 IntegrityError"""
    # setup 上游
    user = User(username="uniq_u8", email="u8@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="uniq_proj_8", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="uniq_case_8", case_type="A")
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(schematic_case_id=case.id, ir_json={"a": 8}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()
    ir_id = ir.id

    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(_sql_text("""
            INSERT INTO review_result (ir_document_id, review_output_json)
            VALUES (:ir_id, NULL)
        """), {"ir_id": ir_id})
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "review_result.review_output_json 为 NULL 应被 NOT NULL 拒绝"


# ---------- 13.9 NOT NULL：review_defect 关键列 ----------

def test_constraint_review_defect_location_not_null(db_session):
    """review_defect.location 为 NULL 应抛 IntegrityError"""
    # setup 上游
    user = User(username="uniq_u9", email="u9@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="uniq_proj_9", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="uniq_case_9", case_type="A")
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(schematic_case_id=case.id, ir_json={"a": 9}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(ir_document_id=ir.id, review_output_json={"s": 9})
    db_session.add(rr)
    db_session.flush()
    rr_id = rr.id

    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(_sql_text("""
            INSERT INTO review_defect
                (review_result_id, defect_id, location, root_cause, suggestion)
            VALUES (:rr_id, 'DEF-NOTNULL-001', NULL, 'rc', 'sg')
        """), {"rr_id": rr_id})
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "review_defect.location 为 NULL 应被 NOT NULL 拒绝"


def test_constraint_review_defect_root_cause_not_null(db_session):
    """review_defect.root_cause 为 NULL 应抛 IntegrityError"""
    user = User(username="uniq_u9b", email="u9b@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="uniq_proj_9b", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="uniq_case_9b", case_type="A")
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(schematic_case_id=case.id, ir_json={"a": 1}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(ir_document_id=ir.id, review_output_json={"s": 1})
    db_session.add(rr)
    db_session.flush()
    rr_id = rr.id

    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(_sql_text("""
            INSERT INTO review_defect
                (review_result_id, defect_id, location, root_cause, suggestion)
            VALUES (:rr_id, 'DEF-NOTNULL-002', '{}'::jsonb, NULL, 'sg')
        """), {"rr_id": rr_id})
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "review_defect.root_cause 为 NULL 应被 NOT NULL 拒绝"


# ---------- 13.10 FK：feedback_item.created_by ----------

def test_constraint_feedback_item_created_by_fk(db_session):
    """created_by 指向不存在的 users.id 应抛 IntegrityError"""
    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(_sql_text("""
            INSERT INTO feedback_item (feedback_type, comment, created_by)
            VALUES ('correct_defect', 'x', 999999999)
        """))
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "feedback_item.created_by 指向不存在用户应被 FK 拒绝"


# ---------- 13.11 FK：ir_document.schematic_case_id ----------

def test_constraint_ir_document_schematic_case_fk(db_session):
    """schematic_case_id 指向不存在的 case 应抛 IntegrityError"""
    nested = db_session.begin_nested()
    raised = False
    try:
        db_session.execute(_sql_text("""
            INSERT INTO ir_document (schematic_case_id, ir_json, ir_schema_version)
            VALUES (999999999, '{}'::jsonb, 'v1.0')
        """))
        db_session.flush()
    except _IntegrityError:
        raised = True
        nested.rollback()
    else:
        nested.commit()

    assert raised, "ir_document.schematic_case_id 指向不存在 case 应被 FK 拒绝"


# ---------- 13.12 正向对照：合法插入应成功 ----------

def test_constraint_legal_insert_succeeds(db_session):
    """
    正向对照：全合法插入应成功。
    防止"约束过严"把正常数据也拒了。
    """
    user = User(username="ok_u12", email="ok12@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="ok_proj_12", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="ok_case_12", case_type="A")
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(schematic_case_id=case.id, ir_json={"a": 12}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(ir_document_id=ir.id, review_output_json={"s": 12})
    db_session.add(rr)
    db_session.flush()

    defect = ReviewDefect(
        review_result_id=rr.id,
        defect_id="DEF-OK-012",
        location={"sheet": "s12"},
        root_cause="rc",
        suggestion="sg",
    )
    db_session.add(defect)
    db_session.flush()

    fb = FeedbackItem(
        review_result_id=rr.id,
        review_defect_id=defect.id,
        feedback_type="correct_defect",   # 合法类型
        comment="ok",
        created_by=user.id,                # 合法 FK
    )
    db_session.add(fb)
    db_session.flush()

    # 断言都能查到
    assert db_session.get(FeedbackItem, fb.id) is not None
    assert db_session.get(ReviewDefect, defect.id) is not None



# ============================================================
# 14. 触发器运行时行为（验证 updated_at 真的被 BEFORE UPDATE 刷新）
# ============================================================
#
# 设计说明：
# - 用 db_session（外层事务 + SAVEPOINT），结束无条件回滚，不留脏数据。
# - 核心手法：
#     1) insert 一行，flush，记下 updated_at = t1
#     2) sleep 一小段（PG CURRENT_TIMESTAMP 精度到微秒，sleep 0.05s 足够）
#     3) update 一个字段，flush
#     4) 新查同一行，断言 updated_at > t1
# - 用 db_session.expire_all() 强制从 DB 重新加载，避免 ORM identity map 缓存旧值。
# - 只依赖 002_schema.sql 里 BEFORE UPDATE 触发器 + update_updated_at_column()。

import time as _time


def _reload(obj):
    """把 ORM 对象标记为过期，下次访问属性会从 DB 重新拉取。"""
    from sqlalchemy import inspect as _sa_inspect
    _sa_inspect(obj).session.expire(obj)


def test_trigger_users_updated_at_refresh(db_session):
    """users.updated_at 在 UPDATE 后应被触发器自动刷新"""
    u = User(username="trig_u1", email="t1@test.local")
    db_session.add(u)
    db_session.flush()

    uid = u.id
    t1 = u.updated_at

    _time.sleep(0.05)

    # 触发 UPDATE
    u.username = "trig_u1_modified"
    db_session.flush()

    # 强制从 DB 重新加载
    db_session.expire(u)
    u2 = db_session.get(User, uid)
    t2 = u2.updated_at

    assert t2 > t1, f"users.updated_at 未刷新: t1={t1}, t2={t2}"


def test_trigger_project_updated_at_refresh(db_session):
    """project.updated_at 在 UPDATE 后应被触发器自动刷新"""
    user = User(username="trig_u2", email="t2@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="trig_proj_2", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    pid = proj.id
    t1 = proj.updated_at

    _time.sleep(0.05)

    proj.name = "trig_proj_2_modified"
    db_session.flush()

    db_session.expire(proj)
    proj2 = db_session.get(Project, pid)
    t2 = proj2.updated_at

    assert t2 > t1, f"project.updated_at 未刷新: t1={t1}, t2={t2}"


def test_trigger_rule_definition_updated_at_refresh(db_session):
    """rule_definition.updated_at 在 UPDATE 后应被触发器自动刷新"""
    rd = RuleDefinition(
        rule_id="TRIG_RULE_003",
        rule_name="trig rule",
        rule_category="POWER",
        rule_yaml="x: 1",
    )
    db_session.add(rd)
    db_session.flush()

    rid = rd.id
    t1 = rd.updated_at

    _time.sleep(0.05)

    rd.rule_name = "trig rule modified"
    db_session.flush()

    db_session.expire(rd)
    rd2 = db_session.get(RuleDefinition, rid)
    t2 = rd2.updated_at

    assert t2 > t1, f"rule_definition.updated_at 未刷新: t1={t1}, t2={t2}"


def test_trigger_rule_candidates_updated_at_refresh(db_session):
    """rule_candidates.updated_at 在 UPDATE 后应被触发器自动刷新"""
    rc = RuleCandidate(
        candidate_id="RC-TRIG-004",
        title="trig cand",
        description="d",
    )
    db_session.add(rc)
    db_session.flush()

    rcid = rc.id
    t1 = rc.updated_at

    _time.sleep(0.05)

    rc.status = "designing"
    db_session.flush()

    db_session.expire(rc)
    rc2 = db_session.get(RuleCandidate, rcid)
    t2 = rc2.updated_at

    assert t2 > t1, f"rule_candidates.updated_at 未刷新: t1={t1}, t2={t2}"


def test_trigger_knowledge_doc_updated_at_refresh(db_session):
    """knowledge_doc.updated_at 在 UPDATE 后应被触发器自动刷新"""
    doc = KnowledgeDoc(
        title="trig doc",
        content_md="# x",
    )
    db_session.add(doc)
    db_session.flush()

    did = doc.id
    t1 = doc.updated_at

    _time.sleep(0.05)

    doc.title = "trig doc modified"
    db_session.flush()

    db_session.expire(doc)
    doc2 = db_session.get(KnowledgeDoc, did)
    t2 = doc2.updated_at

    assert t2 > t1, f"knowledge_doc.updated_at 未刷新: t1={t1}, t2={t2}"


def test_trigger_does_not_refresh_created_at(db_session):
    """
    负向验证：UPDATE 时 created_at 不应被触发器改动（触发器只动 updated_at）。
    """
    u = User(username="trig_u6", email="t6@test.local")
    db_session.add(u)
    db_session.flush()

    uid = u.id
    c1 = u.created_at
    u1 = u.updated_at

    _time.sleep(0.05)

    u.username = "trig_u6_modified"
    db_session.flush()

    db_session.expire(u)
    u2 = db_session.get(User, uid)

    assert u2.created_at == c1, \
        f"created_at 不应被 UPDATE 触发器改变: before={c1}, after={u2.created_at}"
    assert u2.updated_at > u1, "updated_at 应被刷新"


# ============================================================
# 15. ORM 映射真读写（CRUD + 高级映射）
# ============================================================
#
# 目标：验证 ORM 层在真实读写中行为正确，特别是：
#   - KnowledgeDoc/KnowledgeChunk 的 meta_json → DB 列名 metadata 映射
#   - RuleCandidate.task_id（Sprint0 预留字段）能读能写
#   - JSONB 字段 dict/list 往返不丢
#   - server_default 让 ORM 未显式赋值的列自动填充
#   - ORM 侧 update 能被 DB 接收并回读


# ---------- 15.1 通用 insert/select 验证 ----------

def test_crud_user_full_cycle(db_session):
    """User: insert → select → update → 回读"""
    u = User(username="crud_u1", email="c1@test.local")
    db_session.add(u)
    db_session.flush()

    uid = u.id
    db_session.expire(u)
    u2 = db_session.get(User, uid)
    assert u2 is not None
    assert u2.username == "crud_u1"
    assert u2.email == "c1@test.local"
    assert u2.created_at is not None, "server_default 未生效：created_at 为空"
    assert u2.updated_at is not None, "server_default 未生效：updated_at 为空"

    u2.email = "c1_updated@test.local"
    db_session.flush()
    db_session.expire(u2)
    u3 = db_session.get(User, uid)
    assert u3.email == "c1_updated@test.local"


def test_crud_project_full_cycle(db_session):
    """Project: insert → select → update"""
    user = User(username="crud_u2", email="c2@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="crud_proj_2", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    pid = proj.id
    db_session.expire(proj)
    p2 = db_session.get(Project, pid)
    assert p2.name == "crud_proj_2"
    assert p2.owner_id == user.id

    p2.description = "updated desc"
    db_session.flush()
    db_session.expire(p2)
    p3 = db_session.get(Project, pid)
    assert p3.description == "updated desc"


# ---------- 15.2 JSONB 往返：IRDocument.ir_json ----------

def test_crud_ir_document_jsonb_roundtrip(db_session):
    """IRDocument.ir_json 的 dict 应能原样存回，结构不丢"""
    user = User(username="crud_u3", email="c3@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="crud_proj_3", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="crud_case_3", case_type="A")
    db_session.add(case)
    db_session.flush()

    payload = {
        "nodes": [
            {"id": "n1", "type": "resistor", "value": "10k"},
            {"id": "n2", "type": "capacitor", "value": "100nF"},
        ],
        "edges": [{"from": "n1", "to": "n2"}],
        "meta": {"sheet": "power", "unicode": "中文测试"},
    }

    ir = IRDocument(
        schematic_case_id=case.id,
        ir_json=payload,
        ir_schema_version="v1.0",
    )
    db_session.add(ir)
    db_session.flush()

    ir_id = ir.id
    db_session.expire(ir)
    ir2 = db_session.get(IRDocument, ir_id)
    assert ir2.ir_json == payload, "IRDocument.ir_json 往返失真"
    # Unicode 不丢
    assert ir2.ir_json["meta"]["unicode"] == "中文测试"


# ---------- 15.3 JSONB 往返：ReviewDefect.evidence (list) ----------

def test_crud_review_defect_jsonb_evidence(db_session):
    """ReviewDefect.evidence 的 list[dict] 应能原样存回"""
    # setup 上游
    user = User(username="crud_u4", email="c4@test.local")
    db_session.add(user)
    db_session.flush()
    proj = Project(name="crud_proj_4", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()
    case = SchematicCase(project_id=proj.id, case_id="crud_case_4", case_type="A")
    db_session.add(case)
    db_session.flush()
    ir = IRDocument(schematic_case_id=case.id, ir_json={}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()
    rr = ReviewResult(ir_document_id=ir.id, review_output_json={"s": 1})
    db_session.add(rr)
    db_session.flush()

    evidence = [
        {"source": "datasheet", "section": "p12", "reason": "overvoltage"},
        {"source": "rule_engine", "section": "POWER_001", "reason": "no decoupling cap"},
    ]

    d = ReviewDefect(
        review_result_id=rr.id,
        defect_id="DEF-CRUD-004",
        location={"sheet": "power", "path": "/U1", "coords": [10, 20]},
        evidence=evidence,
        root_cause="missing cap",
        suggestion="add 100nF",
    )
    db_session.add(d)
    db_session.flush()

    did = d.id
    db_session.expire(d)
    d2 = db_session.get(ReviewDefect, did)
    assert d2.evidence == evidence, "ReviewDefect.evidence 往返失真"
    assert d2.location == {"sheet": "power", "path": "/U1", "coords": [10, 20]}


# ---------- 15.4 JSONB 往返：FeedbackItem.suggestion_diff_json (list) ----------

def test_crud_feedback_item_jsonb_diff(db_session):
    """FeedbackItem.suggestion_diff_json 的 list[dict] 应能原样存回"""
    fb = FeedbackItem(
        feedback_type="suggestion_update",
        comment="fix",
        suggestion_diff_json=[
            {"op": "add", "path": "/root/cap", "value": "100nF"},
            {"op": "remove", "path": "/root/res"},
        ],
        attached_refs=[{"ref": "doc:123", "kind": "datasheet"}],
    )
    db_session.add(fb)
    db_session.flush()

    fid = fb.id
    db_session.expire(fb)
    fb2 = db_session.get(FeedbackItem, fid)
    assert len(fb2.suggestion_diff_json) == 2
    assert fb2.suggestion_diff_json[0]["op"] == "add"
    assert fb2.attached_refs[0]["kind"] == "datasheet"


# ---------- 15.5 meta_json → DB 列名 metadata 映射（KnowledgeDoc） ----------

def test_crud_knowledge_doc_meta_json_mapping(db_session):
    """
    KnowledgeDoc: ORM 属性名 meta_json，DB 列名 metadata。
    验证写入能落到 metadata 列，读出能从 metadata 列取。
    """
    meta = {"source_file": "ds.pdf", "page": 12, "tags": ["power", "clock"]}
    doc = KnowledgeDoc(
        title="crud doc",
        content_md="# body",
        meta_json=meta,
    )
    db_session.add(doc)
    db_session.flush()

    did = doc.id
    db_session.expire(doc)
    doc2 = db_session.get(KnowledgeDoc, did)
    assert doc2.meta_json == meta, "meta_json 映射读写失真"

    # 用原生 SQL 直接查 DB 列名 metadata，确保列名真的是 metadata
    row = db_session.execute(
        _sql_text("SELECT metadata FROM knowledge_doc WHERE id = :id"),
        {"id": did},
    ).fetchone()
    assert row is not None
    assert row[0] == meta, f"DB 列 metadata 内容不符: {row[0]}"


# ---------- 15.6 meta_json → DB 列名 metadata 映射（KnowledgeChunk） ----------

def test_crud_knowledge_chunk_meta_json_mapping(db_session):
    """KnowledgeChunk: 同样的 name='metadata' 映射验证"""
    doc = KnowledgeDoc(title="crud doc 6", content_md="# x")
    db_session.add(doc)
    db_session.flush()

    meta = {"chunk_index": 0, "tokens": 512}
    chunk = KnowledgeChunk(
        knowledge_doc_id=doc.id,
        chunk_text="hello world",
        meta_json=meta,
    )
    db_session.add(chunk)
    db_session.flush()

    cid = chunk.id
    db_session.expire(chunk)
    chunk2 = db_session.get(KnowledgeChunk, cid)
    assert chunk2.meta_json == meta

    row = db_session.execute(
        _sql_text("SELECT metadata FROM knowledge_chunk WHERE id = :id"),
        {"id": cid},
    ).fetchone()
    assert row[0] == meta


# ---------- 15.7 RuleCandidate.task_id 读写 ----------

def test_crud_rule_candidate_task_id(db_session):
    """
    RuleCandidate.task_id（Sprint0 预留字段）能读能写。
    注意：SQL 里 task_id 无外键，允许任意 BIGINT 或 NULL。
    """
    # 不填 task_id
    rc1 = RuleCandidate(
        candidate_id="RC-CRUD-007A",
        title="no task",
        description="d",
    )
    db_session.add(rc1)
    db_session.flush()
    assert rc1.task_id is None, "未填 task_id 应为 None"

    # 填 task_id
    rc2 = RuleCandidate(
        candidate_id="RC-CRUD-007B",
        title="with task",
        description="d",
        task_id=12345,
    )
    db_session.add(rc2)
    db_session.flush()
    rc2_id = rc2.id

    db_session.expire(rc2)
    rc2_reloaded = db_session.get(RuleCandidate, rc2_id)
    assert rc2_reloaded.task_id == 12345, "task_id 写入后回读失真"

    # 改 task_id
    rc2_reloaded.task_id = 99999
    db_session.flush()
    db_session.expire(rc2_reloaded)
    rc2_again = db_session.get(RuleCandidate, rc2_id)
    assert rc2_again.task_id == 99999


# ---------- 15.8 server_default 生效：ORM 不赋值也自动填 ----------

def test_crud_server_default_fills_columns(db_session):
    """
    多个表的关键列靠 server_default 填：
      - ReviewResult.category/risk/review_status/is_rule_only/计数
      - ReviewDefect.category/risk/review_status/origin/confidence/feedback_count
      - RuleDefinition.severity/enabled/version
      - FeedbackItem.suggestion_diff_json/attached_refs
    ORM 不显式赋值时应拿到 DB 默认值。
    """
    # --- RuleDefinition ---
    rd = RuleDefinition(
        rule_id="RD-CRUD-008",
        rule_name="defaults test",
        rule_category="POWER",
        rule_yaml="x: 1",
    )
    db_session.add(rd)
    db_session.flush()
    db_session.expire(rd)
    rd2 = db_session.get(RuleDefinition, rd.id)
    assert rd2.severity == "medium"
    assert rd2.enabled is True
    assert rd2.version == "v1.0"

    # --- FeedbackItem ---
    fb = FeedbackItem(feedback_type="knowledge_gap", comment="x")
    db_session.add(fb)
    db_session.flush()
    db_session.expire(fb)
    fb2 = db_session.get(FeedbackItem, fb.id)
    assert fb2.suggestion_diff_json == [], "suggestion_diff_json 默认值应为 []"
    assert fb2.attached_refs == [], "attached_refs 默认值应为 []"


# ---------- 15.9 ReviewResult 的 V1.2 计数默认值 ----------

def test_crud_review_result_v12_defaults(db_session):
    """ReviewResult 的 V1.2 摘要字段与计数默认值应生效"""
    user = User(username="crud_u9", email="c9@test.local")
    db_session.add(user)
    db_session.flush()
    proj = Project(name="crud_proj_9", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()
    case = SchematicCase(project_id=proj.id, case_id="crud_case_9", case_type="A")
    db_session.add(case)
    db_session.flush()
    ir = IRDocument(schematic_case_id=case.id, ir_json={}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()

    rr = ReviewResult(ir_document_id=ir.id, review_output_json={"k": "v"})
    db_session.add(rr)
    db_session.flush()

    db_session.expire(rr)
    rr2 = db_session.get(ReviewResult, rr.id)
    assert rr2.is_rule_only is True
    assert rr2.category == "other"
    assert rr2.risk == "medium"
    assert rr2.review_status == "AI_CONFIRMED"
    assert rr2.total_defects == 0
    assert rr2.ai_confirmed_count == 0
    assert rr2.need_expert_review_count == 0
    assert rr2.low_confidence_count == 0


# ---------- 15.10 关系导航：ORM relationship 走通 ----------

def test_crud_relationship_navigation(db_session):
    """ORM relationship 双向导航能查到对象（不靠手工 FK 拼接）"""
    user = User(username="crud_u10", email="c10@test.local")
    db_session.add(user)
    db_session.flush()

    proj = Project(name="crud_proj_10", owner_id=user.id)
    db_session.add(proj)
    db_session.flush()

    case = SchematicCase(project_id=proj.id, case_id="crud_case_10", case_type="A")
    db_session.add(case)
    db_session.flush()

    ir = IRDocument(schematic_case_id=case.id, ir_json={}, ir_schema_version="v1.0")
    db_session.add(ir)
    db_session.flush()

    # 从 case 导航到 project
    db_session.expire(case)
    case2 = db_session.get(SchematicCase, case.id)
    assert case2.project is not None
    assert case2.project.id == proj.id

    # 从 project 导航到 cases（list）
    db_session.expire(proj)
    proj2 = db_session.get(Project, proj.id)
    assert any(c.id == case.id for c in proj2.schematic_cases)

    # 从 user 导航到 projects
    db_session.expire(user)
    user2 = db_session.get(User, user.id)
    assert any(p.id == proj.id for p in user2.projects)   
"""
Phase-D Schematic IR 测试（单元测试 + PostgreSQL 集成测试）

覆盖内容：
    - 模型实例化、序列化往返、验证器逻辑（单元测试，无数据库依赖）
    - JSONB 字段存储与查询、外键约束、事务隔离、多版本 IR、性能、pgvector 兼容性（集成测试，依赖真实 PostgreSQL）

执行命令：
    # 全部测试
    docker compose exec backend uv run pytest tests/test_ir_schema.py -v

    # 只跑单元测试（无数据库依赖）
    docker compose exec backend uv run pytest tests/test_ir_schema.py -v -k "not pg"

前置条件（仅集成测试需要）：
    1. PostgreSQL 容器已启动：docker compose up -d postgres
    2. 数据库 Schema 已初始化：docker compose exec backend uv run python -m scripts.init_db

⚠️ 重要：集成测试依赖真实 PostgreSQL，不可用 SQLite 替代
⚠️ 重要：禁止 Base.metadata.create_all()，Schema 唯一来源是 init_db
"""

import pathlib
import tempfile
import uuid
import time
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import Session

from app.core.config import settings
from app.persistence.models import (
    IRDocument,
    SchematicCase,
    Project,
    User,
)
from app.ir.schema import (
    SchematicIRDocument,
    EXAMPLE_8031_CASE001,
    PinDirection,
    PinType,
    Component,
    Pin,
    Net,
)
from app.ir.serializer import dump_ir, load_ir
from app.ir.validator import IRSchemaValidator, FloatingCheckResult
from app.ir.service import (
    store_schematic_ir,
    load_schematic_ir,
    load_schematic_ir_by_case,
)


# ============================================================
# Fixtures - 单元测试（无数据库依赖）
# ============================================================
# （单元测试不使用 fixture，原 test_ir_schema.py 中的测试都是独立的）


# ============================================================
# Fixtures - 集成测试（PostgreSQL）
# ============================================================

@pytest.fixture(scope="module")
def pg_engine():
    """PostgreSQL 引擎（模块级，只创建一次）"""
    engine = create_engine(settings.database_url, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def pg_session(pg_engine):
    """
    每个测试函数独立事务，测试后自动回滚

    保证：
        - 测试之间相互隔离
        - 不污染真实数据库
        - 测试可重复执行
    """
    connection = pg_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def test_user(pg_session):
    """创建测试用户（测试后回滚）"""
    user = User(
        username=f"test_user_{uuid.uuid4().hex[:8]}",
        email=f"test_{uuid.uuid4().hex[:8]}@example.com",
    )
    pg_session.add(user)
    pg_session.flush()
    return user


@pytest.fixture(scope="function")
def test_project(pg_session, test_user):
    """创建测试项目（测试后回滚）"""
    project = Project(
        name=f"test_project_{uuid.uuid4().hex[:8]}",
        owner_id=test_user.id,
    )
    pg_session.add(project)
    pg_session.flush()
    return project


@pytest.fixture(scope="function")
def test_schematic_case(pg_session, test_project):
    """
    创建测试用 schematic_case（测试后回滚）

    注意字段命名：
        - case_id: 业务字符串标识（如 'case001'），非主键
        - case_type: 必填（如 'A' 或 'B'）
        - id: 主键（BigInteger）
    """
    case = SchematicCase(
        project_id=test_project.id,
        case_id=f"case_{uuid.uuid4().hex[:8]}",
        case_type="A",
    )
    pg_session.add(case)
    pg_session.flush()
    return case


# ============================================================
# 单元测试（原 test_ir_schema.py 内容）
# ============================================================

# ============ 模型实例化测试 ============
def test_ir_model_8051_example():
    """测试 8051 最小系统样例模型实例化"""
    doc = EXAMPLE_8031_CASE001
    assert doc.case_id == "case001"
    assert doc.ir_schema_version == "v1.0"
    assert len(doc.components) > 0
    assert len(doc.nets) > 0
    
    # created_at 已手动赋值
    assert doc.created_at is not None
    
    # 三语义字段存在（值可以为 None）
    for comp in doc.components:
        assert hasattr(comp, 'intent')
        assert hasattr(comp, 'context')
        assert hasattr(comp, 'constraint')
    
    # MCU 有 40 个引脚
    mcu = next(c for c in doc.components if c.ref == "U1")
    assert len(mcu.pins) == 40
    
    # 引脚类型验证
    vcc_pin = next(p for p in mcu.pins if p.pin_id == "VCC")
    assert vcc_pin.direction == PinDirection.POWER
    assert vcc_pin.pin_type == PinType.POWER
    
    rst_pin = next(p for p in mcu.pins if p.pin_id == "RST")
    assert rst_pin.direction == PinDirection.INPUT
    assert rst_pin.pin_type == PinType.RESET
    
    # SW1 已正确连接
    rst_net = next(n for n in doc.nets if n.net_name == "RST")
    assert "SW1.pin1" in rst_net.connected_pins
    
    gnd_net = next(n for n in doc.nets if n.net_name == "GND")
    assert "SW1.pin2" in gnd_net.connected_pins


def test_pin_enum_types():
    """测试引脚类型枚举"""
    assert PinDirection.INPUT == "input"
    assert PinDirection.OUTPUT == "output"
    assert PinDirection.BIDIRECTIONAL == "bidirectional"
    assert PinDirection.POWER == "power"
    assert PinDirection.GROUND == "ground"
    
    assert PinType.IO == "io"
    assert PinType.ANALOG == "analog"
    assert PinType.POWER == "power"
    assert PinType.GROUND == "ground"
    assert PinType.CLOCK == "clock"
    assert PinType.RESET == "reset"
    assert PinType.OTHER == "other"


# ============ 序列化测试 ============
def test_ir_serialize_roundtrip():
    """JSON序列化-反序列化往返测试"""
    src = EXAMPLE_8031_CASE001
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "ir.json"
        dump_ir(src, f)
        loaded = load_ir(f)
    
    assert loaded.case_id == src.case_id
    assert len(loaded.components) == len(src.components)
    assert loaded.title == src.title


def test_ir_serialize_created_at_preserved():
    """测试 created_at 在序列化往返中保持不变"""
    src = EXAMPLE_8031_CASE001
    original_created_at = src.created_at
    
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "ir.json"
        dump_ir(src, f)
        loaded = load_ir(f)
    
    if original_created_at:
        assert loaded.created_at == original_created_at


# ============ 验证器测试 ============
def test_ir_validator_normal_mode():
    """IR 验证器测试 - 普通模式"""
    doc = EXAMPLE_8031_CASE001
    result = IRSchemaValidator.validate(doc, strict_mode=False)
    
    assert result.is_valid, f"验证失败: {result.errors}"
    
    # 验证摘要包含三语义字段统计
    assert 'semantic_coverage' in result.summary
    semantic = result.summary['semantic_coverage']
    assert semantic['total_components'] == len(doc.components)
    assert semantic['has_intent'] == len(doc.components)
    assert semantic['has_context'] == len(doc.components)


def test_ir_validator_strict_mode():
    """IR 验证器测试 - 严格模式 (Golden Case 验收)"""
    doc = EXAMPLE_8031_CASE001
    result = IRSchemaValidator.validate(doc, strict_mode=True)
    assert result.is_valid, f"严格模式验证失败: {result.errors}"


def test_ir_validator_missing_semantic_normal():
    """验证器：缺失 intent/context 在普通模式下仅为 Warning"""
    # 创建缺少三语义字段的元件
    component = Component(
        ref="U1",
        lib_name="TEST",  # lib_name 必须填写
        pins=[
            Pin(
                pin_id="pin1",
                name="TEST_PIN",
                direction=PinDirection.INPUT,
                pin_type=PinType.OTHER
            )
        ]
        # intent/context 未设置，为 None
    )
    
    doc = SchematicIRDocument(
        case_id="test",
        title="Test",
        description="Test",
        components=[component],
        nets=[]
    )
    
    # 普通模式：应该通过，只有警告
    result = IRSchemaValidator.validate(doc, strict_mode=False)
    assert result.is_valid, "普通模式下不应因缺失 intent/context 而失败"
    
    # 应该包含警告
    assert len(result.warnings) > 0
    assert any("intent" in warn for warn in result.warnings)
    assert any("context" in warn for warn in result.warnings)
    
    # 严格模式：应该失败
    result_strict = IRSchemaValidator.validate(doc, strict_mode=True)
    assert not result_strict.is_valid
    assert any("intent" in err for err in result_strict.errors)
    assert any("context" in err for err in result_strict.errors)


def test_ir_validator_missing_lib_name():
    """验证器：lib_name 缺失应报 Error（规则引擎核心字段）"""
    # 创建缺少 lib_name 的元件
    component = Component(
        ref="U1",
        lib_name="",  # lib_name 为空
        intent="test",
        context="test",
        pins=[
            Pin(
                pin_id="pin1",
                name="TEST_PIN",
                direction=PinDirection.INPUT,
                pin_type=PinType.OTHER
            )
        ]
    )
    
    doc = SchematicIRDocument(
        case_id="test",
        title="Test",
        description="Test",
        components=[component],
        nets=[]
    )
    
    result = IRSchemaValidator.validate(doc, strict_mode=False)
    
    # 应该失败（lib_name 缺失是 Error）
    assert not result.is_valid
    assert any("lib_name" in err for err in result.errors)
    assert any("规则引擎" in err for err in result.errors)


def test_ir_validator_connected_pins_format():
    """验证器：connected_pins 格式验证"""
    doc = SchematicIRDocument(
        case_id="test",
        title="Test",
        description="Test",
        components=[
            Component(
                ref="U1",
                lib_name="TEST",
                pins=[
                    Pin(
                        pin_id="VCC",
                        name="VCC",
                        direction=PinDirection.POWER,
                        pin_type=PinType.POWER
                    )
                ],
                intent="test",
                context="test"
            )
        ],
        nets=[
            Net(
                net_name="VCC",
                connected_pins=["U1.40"]  # 错误：使用了物理编号 40
            )
        ]
    )
    
    result = IRSchemaValidator.validate(doc, strict_mode=False)
    
    # 应该产生警告（pin_id 不匹配）
    assert result.is_valid
    assert any("不存在" in warn for warn in result.warnings)


def test_ir_validator_connected_pins_wrong_format():
    """验证器：connected_pins 格式错误检测"""
    doc = SchematicIRDocument(
        case_id="test",
        title="Test",
        description="Test",
        components=[
            Component(
                ref="U1",
                lib_name="TEST",
                pins=[
                    Pin(
                        pin_id="VCC",
                        name="VCC",
                        direction=PinDirection.POWER,
                        pin_type=PinType.POWER
                    )
                ],
                intent="test",
                context="test"
            )
        ],
        nets=[
            Net(
                net_name="VCC",
                connected_pins=["U1VCC"]  # 错误：缺少点分隔符
            )
        ]
    )
    
    result = IRSchemaValidator.validate(doc, strict_mode=False)
    
    assert not result.is_valid
    assert any("格式错误" in err for err in result.errors)


def test_ir_validator_floating_component_warning():
    """验证器：悬浮元件警告"""
    doc = SchematicIRDocument(
        case_id="test",
        title="Test",
        description="Test",
        components=[
            Component(
                ref="U1",
                lib_name="TEST",
                pins=[
                    Pin(
                        pin_id="pin1",
                        name="pin1",
                        direction=PinDirection.INPUT,
                        pin_type=PinType.OTHER
                    )
                ],
                intent="test",
                context="test"
            )
        ],
        nets=[]  # 没有网络
    )
    
    result = IRSchemaValidator.validate(doc, strict_mode=False)
    
    # 应该通过验证但有警告
    assert result.is_valid
    assert any("悬浮" in warn or "未连接" in warn for warn in result.warnings)


def test_floating_check_result_reuse():
    """测试 FloatingCheckResult 结构复用，避免重复计算"""
    doc = EXAMPLE_8031_CASE001
    
    result = IRSchemaValidator.check_floating_components(doc)
    
    # 验证返回类型
    assert isinstance(result, FloatingCheckResult)
    assert isinstance(result.floating_component_refs, set)
    assert isinstance(result.floating_pin_refs, set)
    
    # 验证统计一致性：floating_components 应等于 floating_component_refs 数量
    validate_result = IRSchemaValidator.validate(doc)
    assert validate_result.summary["floating_components"] == len(result.floating_component_refs)
    assert validate_result.summary["floating_pins"] == len(result.floating_pin_refs)


def test_floating_check_result_with_floating_component():
    """测试悬浮元件检测 - 有悬浮元件时"""
    doc = SchematicIRDocument(
        case_id="test",
        title="Test",
        description="Test",
        components=[
            Component(
                ref="U1",
                lib_name="TEST",
                pins=[
                    Pin(pin_id="pin1", name="pin1", direction=PinDirection.INPUT, pin_type=PinType.OTHER)
                ],
                intent="test",
                context="test"
            ),
            Component(
                ref="U2",
                lib_name="TEST2",
                pins=[
                    Pin(pin_id="pin1", name="pin1", direction=PinDirection.INPUT, pin_type=PinType.OTHER)
                ],
                intent="test",
                context="test"
            )
        ],
        nets=[
            Net(net_name="NET1", connected_pins=["U1.pin1"])  # U2 悬浮
        ]
    )
    
    result = IRSchemaValidator.check_floating_components(doc)
    
    # U2 应该被检测为悬浮
    assert "U2" in result.floating_component_refs
    assert len(result.floating_component_refs) == 1
    assert len(result.warnings) >= 1
    assert any("U2" in warn for warn in result.warnings)


# ============================================================
# 集成测试（原 test_ir_integration_pg.py 内容）
# ============================================================

# ============================================================
# 1. JSONB 存储测试
# ============================================================

def test_ir_jsonb_store(pg_session, test_schematic_case):
    """测试 IR 文档写入 JSONB 字段"""
    doc = EXAMPLE_8031_CASE001

    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )

    assert db_rec.id is not None
    assert db_rec.ir_schema_version == "v1.0"
    assert db_rec.schematic_case_id == test_schematic_case.id


def test_ir_jsonb_field_type(pg_session, test_schematic_case):
    """验证 ir_json 字段的实际类型为 JSONB（PostgreSQL 特有）"""
    doc = EXAMPLE_8031_CASE001
    store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    result = pg_session.execute(
        text("""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'ir_document' AND column_name = 'ir_json'
        """)
    ).scalar()

    assert result == "jsonb", f"ir_json 字段应为 jsonb 类型，实际为 {result}"


def test_ir_jsonb_content_integrity(pg_session, test_schematic_case):
    """验证 JSONB 存储内容完整性（往返不丢失）"""
    doc = EXAMPLE_8031_CASE001

    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    loaded = load_schematic_ir(pg_session, db_rec.id)

    assert loaded.case_id == doc.case_id
    assert loaded.title == doc.title
    assert len(loaded.components) == len(doc.components)
    assert len(loaded.nets) == len(doc.nets)

    mcu_original = next(c for c in doc.components if c.ref == "U1")
    mcu_loaded = next(c for c in loaded.components if c.ref == "U1")
    assert len(mcu_loaded.pins) == len(mcu_original.pins) == 40


# ============================================================
# 2. JSONB 查询测试
# ============================================================

def test_ir_jsonb_query_by_case_id(pg_session, test_schematic_case):
    """通过 JSONB 路径查询 IR 文档"""
    doc = EXAMPLE_8031_CASE001
    store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    result = pg_session.query(IRDocument).filter(
        IRDocument.ir_json['case_id'].astext == "case001"
    ).first()

    assert result is not None
    assert result.ir_json['case_id'] == "case001"
    assert result.ir_json['title'] == "8051最小系统原理图 (STC89C55RC)"


def test_load_schematic_ir_by_case(pg_session, test_schematic_case):
    """测试 load_schematic_ir_by_case 服务函数"""
    doc = EXAMPLE_8031_CASE001
    store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    loaded = load_schematic_ir_by_case(pg_session, "case001")

    assert loaded is not None
    assert loaded.case_id == "case001"
    assert len(loaded.components) == 7


def test_ir_jsonb_query_by_component_count(pg_session, test_schematic_case):
    """通过 JSONB 数组长度查询"""
    doc = EXAMPLE_8031_CASE001
    store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    result = pg_session.execute(
        text("""
            SELECT id, jsonb_array_length(ir_json->'components') AS comp_count
            FROM ir_document
            WHERE ir_json->>'case_id' = 'case001'
        """)
    ).fetchone()

    assert result is not None
    assert result[1] == 7


# ============================================================
# 3. 多版本 IR 管理
# ============================================================

def test_multiple_ir_versions_same_case(pg_session, test_schematic_case):
    """同一 case_id 多个 IR 版本，按时间排序取最新"""
    doc_v1 = EXAMPLE_8031_CASE001

    doc_v2 = doc_v1.model_copy(deep=True)
    doc_v2.title = "8051最小系统 V2"
    doc_v2.created_at = (datetime.now() + timedelta(seconds=1)).isoformat()

    rec_v1 = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc_v1,
        schematic_case_id=test_schematic_case.id,
    )
    rec_v2 = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc_v2,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    assert rec_v1.id != rec_v2.id

    latest = load_schematic_ir_by_case(pg_session, "case001")
    assert latest.title == "8051最小系统 V2"


# ============================================================
# 4. 外键约束测试
# ============================================================

def test_ir_document_foreign_key_constraint(pg_engine):
    """
    测试 ir_document.schematic_case_id 外键约束

    注意：此测试不使用 pg_session fixture,因为需要验证 flush 失败场景，
    而 fixture 的事务控制会与之冲突(SQLAlchemy 在 flush 失败时会自动
    rollback session,导致 fixture 的 transaction 失效)。
    使用独立连接，依赖 with 上下文自动管理资源。
    """
    doc = EXAMPLE_8031_CASE001

    with pg_engine.connect() as conn:
        session = Session(bind=conn)
        try:
            with pytest.raises(Exception) as exc_info:
                store_schematic_ir(
                    session,
                    case_id="case001",
                    ir_doc=doc,
                    schematic_case_id=999999,  # 不存在的 ID
                    auto_commit=False,
                )
                session.flush()  # 触发外键约束检查

            error_msg = str(exc_info.value).lower()
            assert (
                "foreign key" in error_msg
                or "violates" in error_msg
                or "integrity" in error_msg
            ), f"应抛出外键约束错误，实际错误: {exc_info.value}"
        finally:
            session.close()
        # with 语句自动关闭连接，无需显式 rollback


# ============================================================
# 5. 事务隔离与回滚
# ============================================================

def test_transaction_rollback(pg_engine):
    """测试事务回滚，数据不污染数据库"""
    doc = EXAMPLE_8031_CASE001
    unique_case_id = f"rollback_test_{uuid.uuid4().hex[:8]}"

    # 第一个事务：写入后回滚
    conn1 = pg_engine.connect()
    trans1 = conn1.begin()
    session1 = Session(bind=conn1)

    rec_id = None
    try:
        # 需要先创建 User → Project → SchematicCase 才能建 IR（外键约束）
        user = User(
            username=f"rollback_user_{uuid.uuid4().hex[:8]}",
            email=f"rollback_{uuid.uuid4().hex[:8]}@example.com",
        )
        session1.add(user)
        session1.flush()

        project = Project(
            name=f"rollback_project_{uuid.uuid4().hex[:8]}",
            owner_id=user.id,
        )
        session1.add(project)
        session1.flush()

        case = SchematicCase(
            project_id=project.id,
            case_id=unique_case_id,
            case_type="A",
        )
        session1.add(case)
        session1.flush()

        db_rec = store_schematic_ir(
            session1,
            case_id=unique_case_id,
            ir_doc=doc,
            schematic_case_id=case.id,
            auto_commit=False,
        )
        session1.flush()

        rec_id = db_rec.id
        assert rec_id is not None
    finally:
        session1.close()
        trans1.rollback()
        conn1.close()

    # 第二个事务：验证数据不存在
    conn2 = pg_engine.connect()
    session2 = Session(bind=conn2)

    try:
        result = session2.query(IRDocument).filter(IRDocument.id == rec_id).first()
        assert result is None, "回滚后数据不应存在"
    finally:
        session2.close()
        conn2.close()


# ============================================================
# 6. 大 IR 文档性能测试
# ============================================================

def test_large_ir_document_performance(pg_session, test_schematic_case):
    """测试完整 8051 IR 文档（52 引脚）的存储和查询性能"""
    doc = EXAMPLE_8031_CASE001

    # 存储计时
    start = time.time()
    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()
    store_time = time.time() - start

    # 加载计时
    start = time.time()
    loaded = load_schematic_ir(pg_session, db_rec.id)
    load_time = time.time() - start

    # 性能断言（宽松阈值）
    assert store_time < 1.0, f"存储耗时过长: {store_time:.3f}s"
    assert load_time < 1.0, f"加载耗时过长: {load_time:.3f}s"

    # 验证内容
    assert len(loaded.components) == 7
    assert sum(len(c.pins) for c in loaded.components) == 52


# ============================================================
# 7. pgvector 兼容性测试
# ============================================================

def test_ir_and_vector_tables_coexist(pg_session, test_schematic_case):
    """测试 IR 表与 pgvector 表共存无冲突"""
    # 存储 IR
    doc = EXAMPLE_8031_CASE001
    store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    # 验证 pgvector 扩展仍然可用
    result = pg_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    ).fetchone()
    assert result is not None, "pgvector 扩展应仍然启用"

    # 验证 knowledge_chunk 表仍然存在（pgvector 使用）
    insp = inspect(pg_session.bind)
    assert "knowledge_chunk" in insp.get_table_names(schema="public")


# ============================================================
# 8. 字段约束测试
# ============================================================

def test_ir_schema_version_not_null(pg_session, test_schematic_case):
    """测试 ir_schema_version 字段的 NOT NULL 约束"""
    doc = EXAMPLE_8031_CASE001

    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    assert db_rec.ir_schema_version == "v1.0"


def test_ir_document_created_at_auto_generated(pg_session, test_schematic_case):
    """测试 ir_document.created_at 数据库自动生成"""
    doc = EXAMPLE_8031_CASE001
    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()
    pg_session.refresh(db_rec)

    # 数据库层面的 created_at 应该已生成
    assert db_rec.created_at is not None
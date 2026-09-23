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

===== [NEW] v1.0.1 改动说明 =====
本文件针对 schema.py v1.0.1 的以下改动补充测试：
    1. Attribute.value 类型收紧为 Union[int, float, str, bool]
    2. Net.connected_pins 新增 field_validator 格式校验
    3. SchematicIRDocument.created_at 改为 datetime 类型
    4. 示例常量改为 lru_cache 懒加载（get_8051_example）
    5. IRValidationResult.summary 结构化（IRValidationSummary）

⚠️ 影响：EXAMPLE_8031_CASE001 现在指向"函数"而非"文档对象"，
   原有 `doc = EXAMPLE_8031_CASE001` 直接访问属性的写法全部改为
   `doc = get_8051_example()`（函数调用）。
===== [/NEW] =====

===== [NEW] v1.0.1 二次适配说明 =====
validator.py 已同步升级，validate() 返回的 result.summary 现在是
IRValidationSummary 实例（不再是 dict），因此：
    - 原来 `result.summary["xxx"]` 的字典访问方式全部失效
    - 改为 `result.summary.xxx` 属性访问
    - 原先的 ir_schema_version / case_id / component_types / net_types /
      semantic_coverage 分项统计，收敛到 `result.summary.extra` 中
    - floating_components / floating_pins 由 int 变为 List[str]
    - semantic_coverage 由 dict 变为 Optional[float]（综合覆盖率）
===== [/NEW] =====
"""

import pathlib
import tempfile
import uuid
import time
# ===== [NEW] 新增导入 =====
from datetime import datetime, timedelta, date
from typing import get_args
# ===== [/NEW] =====

import pytest
from pydantic import ValidationError
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
    # EXAMPLE_8031_CASE001,  # 原代码：仍可导入（向后兼容别名），但用法变了
    # ===== [NEW] 新增导入：推荐的示例获取方式 =====
    get_8051_example,
    create_8051_example,
    EXAMPLE_8031_CASE001,  # 向后兼容别名（现在是函数，需调用）
    # ===== [/NEW] =====
    PinDirection,
    PinType,
    Component,
    Pin,
    Net,
    # ===== [NEW] 新增导入：本次改动涉及的新模型/字段 =====
    Attribute,
    IRValidationResult,
    IRValidationSummary,
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码：别名现为函数，不能直接当文档用
    # ===== [NEW] 改为调用函数获取文档 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # src = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    src = get_8051_example()
    # ===== [/NEW] =====
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "ir.json"
        dump_ir(src, f)
        loaded = load_ir(f)

    assert loaded.case_id == src.case_id
    assert len(loaded.components) == len(src.components)
    assert loaded.title == src.title


def test_ir_serialize_created_at_preserved():
    """测试 created_at 在序列化往返中保持不变"""
    # src = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    src = get_8051_example()
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
    result = IRSchemaValidator.validate(doc, strict_mode=False)

    assert result.is_valid, f"验证失败: {result.errors}"

    # ===== [NEW] summary 现为 IRValidationSummary 模型，改用属性访问 =====
    # 原代码（字典访问，已失效）：
    # assert 'semantic_coverage' in result.summary
    # semantic = result.summary['semantic_coverage']
    # assert semantic['total_components'] == len(doc.components)
    # assert semantic['has_intent'] == len(doc.components)
    # assert semantic['has_context'] == len(doc.components)
    #
    # 说明：
    #   - 新版 summary.semantic_coverage 是 float（综合覆盖率），不再含分项统计。
    #   - 分项统计在 summary.extra["semantic_coverage_detail"] 中。
    assert result.summary.semantic_coverage is not None
    detail = result.summary.extra["semantic_coverage_detail"]
    assert detail['total_components'] == len(doc.components)
    assert detail['has_intent'] == len(doc.components)
    assert detail['has_context'] == len(doc.components)
    # ===== [/NEW] =====


def test_ir_validator_strict_mode():
    """IR 验证器测试 - 严格模式 (Golden Case 验收)"""
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # ===== [NEW] 说明：U1.40 通过了 Net 层格式校验（点号存在、两侧非空），
    # 但会在 validator 的 validate_nets 中被判为 "pin_id 不存在"，归为 Warning。
    # 因此 Net 构造阶段不会抛错，此测试仍走原逻辑。 =====
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
    # ===== [/NEW] =====


def test_ir_validator_connected_pins_wrong_format():
    """验证器：connected_pins 格式错误检测"""
    # ===== [NEW] Net 构造时 field_validator 会直接拦截格式错误 =====
    # 原先这个测试依赖 IRSchemaValidator 报"格式错误"，
    # 现在格式错误在 Net 实例化阶段就会被 Pydantic 抛出 ValidationError。
    # 因此测试改为断言构造阶段抛错。
    with pytest.raises(ValidationError) as exc_info:
        Net(
            net_name="VCC",
            connected_pins=["U1VCC"]  # 错误：缺少点分隔符
        )
    assert "connected_pins" in str(exc_info.value)
    # ===== [/NEW] =====


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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====

    result = IRSchemaValidator.check_floating_components(doc)

    # 验证返回类型
    assert isinstance(result, FloatingCheckResult)
    assert isinstance(result.floating_component_refs, set)
    assert isinstance(result.floating_pin_refs, set)

    # ===== [NEW] 新版 summary.floating_components 是 List[str]，与旧的 int 不同 =====
    # 原代码（比较对象从 int 变为 list 长度）：
    # validate_result = IRSchemaValidator.validate(doc)
    # assert validate_result.summary["floating_components"] == len(result.floating_component_refs)
    # assert validate_result.summary["floating_pins"] == len(result.floating_pin_refs)
    validate_result = IRSchemaValidator.validate(doc)
    assert len(validate_result.summary.floating_components) == len(result.floating_component_refs)
    assert len(validate_result.summary.floating_pins) == len(result.floating_pin_refs)
    # 内容一致性：集合元素应一一对应
    assert set(validate_result.summary.floating_components) == result.floating_component_refs
    assert set(validate_result.summary.floating_pins) == result.floating_pin_refs
    # ===== [/NEW] =====


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
# [NEW] 新增单元测试 - 针对 schema.py v1.0.1 改动
# ============================================================

# ============ 1. Attribute.value 类型收紧测试 ============
class TestAttributeValueType:
    """测试 Attribute.value 收紧为 Union[int, float, str, bool]"""

    def test_attribute_accepts_int(self):
        """value 接受整数"""
        attr = Attribute(key="resistance", value=10000, unit="Ω")
        assert attr.value == 10000
        assert isinstance(attr.value, int)

    def test_attribute_accepts_float(self):
        """value 接受浮点数"""
        attr = Attribute(key="voltage", value=5.0, unit="V")
        assert attr.value == 5.0
        assert isinstance(attr.value, float)

    def test_attribute_accepts_str(self):
        """value 接受字符串"""
        attr = Attribute(key="frequency", value="11.0592MHz")
        assert attr.value == "11.0592MHz"

    def test_attribute_accepts_bool(self):
        """value 接受布尔值"""
        attr = Attribute(key="is_polarized", value=True)
        assert attr.value is True

    def test_attribute_rejects_list(self):
        """value 拒绝 list（改动前 Any 会接受）"""
        with pytest.raises(ValidationError) as exc_info:
            Attribute(key="bad", value=[1, 2, 3])
        assert "value" in str(exc_info.value)

    def test_attribute_rejects_dict(self):
        """value 拒绝 dict"""
        with pytest.raises(ValidationError) as exc_info:
            Attribute(key="bad", value={"nested": "dict"})
        assert "value" in str(exc_info.value)

    def test_attribute_rejects_none(self):
        """value 拒绝 None（value 是必填字段，非 Optional）"""
        with pytest.raises(ValidationError):
            Attribute(key="bad", value=None)

    def test_attribute_value_type_annotation(self):
        """验证 value 类型注解确实是 Union[int, float, str, bool]"""
        fi = Attribute.model_fields["value"]
        args = set(get_args(fi.annotation))
        expected = {int, float, str, bool}
        assert expected.issubset(args), f"期望类型 {expected}，实际 {args}"


# ============ 2. Net.connected_pins 格式校验器测试 ============
class TestConnectedPinsValidator:
    """测试 Net.connected_pins 的 field_validator"""

    def test_valid_format_simple(self):
        """合法格式：Ref.pin_id"""
        net = Net(net_name="VCC", connected_pins=["U1.VCC"])
        assert net.connected_pins == ["U1.VCC"]

    def test_valid_format_multiple(self):
        """合法格式：多个引用"""
        net = Net(
            net_name="GND",
            connected_pins=["U1.GND", "C1.pin2", "R1.pin2"]
        )
        assert len(net.connected_pins) == 3

    def test_valid_format_pin_id_with_dots(self):
        """合法格式：pin_id 含点号（如 'P1.0'），只要整体只有一个分隔点即可"""
        # 注意：正则用非贪婪 + 整体匹配，"U1.P1.0" 会被判定为包含两个点，
        # 因此这里用 "U1.P1" 表示 ref=U1, pin_id=P1，是合法的
        net = Net(net_name="NET1", connected_pins=["U1.P1"])
        assert net.connected_pins == ["U1.P1"]

    def test_empty_list_is_valid(self):
        """空列表合法（字段可省略）"""
        net = Net(net_name="EMPTY", connected_pins=[])
        assert net.connected_pins == []

    def test_rejects_missing_dot(self):
        """非法：缺少点号"""
        with pytest.raises(ValidationError) as exc_info:
            Net(net_name="BAD", connected_pins=["U1VCC"])
        assert "connected_pins" in str(exc_info.value)

    def test_rejects_empty_ref(self):
        """非法：点号左侧为空"""
        with pytest.raises(ValidationError):
            Net(net_name="BAD", connected_pins=[".VCC"])

    def test_rejects_empty_pin_id(self):
        """非法：点号右侧为空"""
        with pytest.raises(ValidationError):
            Net(net_name="BAD", connected_pins=["U1."])

    def test_rejects_whitespace(self):
        """非法：包含空白字符"""
        with pytest.raises(ValidationError):
            Net(net_name="BAD", connected_pins=["U1 . VCC"])

    def test_rejects_multiple_dots(self):
        """非法：包含多个点号"""
        with pytest.raises(ValidationError):
            Net(net_name="BAD", connected_pins=["U1.P1.0"])

    def test_rejects_non_string_item(self):
        """非法：非字符串项"""
        with pytest.raises(ValidationError):
            Net(net_name="BAD", connected_pins=[123])

    def test_rejects_mixed_valid_invalid(self):
        """非法：列表中混有合法与非法项（整体应失败）"""
        with pytest.raises(ValidationError) as exc_info:
            Net(
                net_name="BAD",
                connected_pins=["U1.VCC", "BADFORMAT", "R1.pin1"]
            )
        assert "BADFORMAT" in str(exc_info.value)

    def test_error_message_contains_guidance(self):
        """错误信息应包含格式说明，便于排错"""
        with pytest.raises(ValidationError) as exc_info:
            Net(net_name="BAD", connected_pins=["NODOT"])
        msg = str(exc_info.value)
        assert "Component.ref" in msg or "pin_id" in msg


# ============ 3. created_at datetime 类型测试 ============
class TestCreatedAtDatetime:
    """测试 SchematicIRDocument.created_at 改为 datetime 类型"""

    def test_default_is_none(self):
        """默认值为 None，不自动生成时间戳"""
        doc = SchematicIRDocument(
            case_id="test",
            title="Test",
            description="Test",
        )
        assert doc.created_at is None

    def test_accepts_datetime_object(self):
        """接受 datetime 对象"""
        now = datetime.now()
        doc = SchematicIRDocument(
            case_id="test",
            title="Test",
            description="Test",
            created_at=now,
        )
        assert isinstance(doc.created_at, datetime)
        assert doc.created_at == now

    def test_accepts_iso_string(self):
        """接受 ISO 字符串（Pydantic 自动解析为 datetime）"""
        iso = "2026-09-23T10:30:00"
        doc = SchematicIRDocument(
            case_id="test",
            title="Test",
            description="Test",
            created_at=iso,
        )
        assert isinstance(doc.created_at, datetime)
        assert doc.created_at.year == 2026
        assert doc.created_at.month == 9
        assert doc.created_at.day == 23

    def test_accepts_iso_string_with_microseconds(self):
        """接受带微秒的 ISO 字符串"""
        iso = "2026-09-23T10:30:00.123456"
        doc = SchematicIRDocument(
            case_id="test",
            title="Test",
            description="Test",
            created_at=iso,
        )
        assert isinstance(doc.created_at, datetime)
        assert doc.created_at.microsecond == 123456

    def test_json_serialization_is_iso(self):
        """JSON 序列化时输出 ISO 格式字符串"""
        now = datetime(2026, 9, 23, 10, 30, 0)
        doc = SchematicIRDocument(
            case_id="test",
            title="Test",
            description="Test",
            created_at=now,
        )
        json_str = doc.model_dump_json()
        assert "2026-09-23T10:30:00" in json_str

    def test_json_roundtrip_preserves_datetime(self):
        """JSON 往返后 created_at 应保持 datetime 类型且值相等"""
        now = datetime(2026, 9, 23, 10, 30, 0)
        doc = SchematicIRDocument(
            case_id="test",
            title="Test",
            description="Test",
            created_at=now,
        )
        json_str = doc.model_dump_json()
        restored = SchematicIRDocument.model_validate_json(json_str)
        assert isinstance(restored.created_at, datetime)
        assert restored.created_at == now

    def test_example_created_at_is_datetime(self):
        """8051 示例中 created_at 应为 datetime 类型"""
        doc = get_8051_example()
        assert isinstance(doc.created_at, datetime)

    def test_rejects_invalid_string(self):
        """拒绝无法解析为 datetime 的字符串"""
        with pytest.raises(ValidationError):
            SchematicIRDocument(
                case_id="test",
                title="Test",
                description="Test",
                created_at="not-a-date",
            )

    def test_field_annotation_is_optional_datetime(self):
        """验证字段类型注解为 Optional[datetime]"""
        fi = SchematicIRDocument.model_fields["created_at"]
        args = get_args(fi.annotation)
        # Optional[datetime] 等价于 Union[datetime, None]
        assert datetime in args
        assert type(None) in args


# ============ 4. lru_cache 懒加载示例测试 ============
class TestExampleLazyLoading:
    """测试示例常量的懒加载与缓存行为"""

    def test_get_example_returns_document(self):
        """get_8051_example() 返回 SchematicIRDocument"""
        doc = get_8051_example()
        assert isinstance(doc, SchematicIRDocument)
        assert doc.case_id == "case001"

    def test_get_example_is_cached(self):
        """两次调用返回同一对象（lru_cache 生效）"""
        doc1 = get_8051_example()
        doc2 = get_8051_example()
        assert doc1 is doc2, "lru_cache 应返回同一实例"

    def test_create_example_returns_new_instance(self):
        """create_8051_example() 每次返回新实例（工厂语义）"""
        doc1 = create_8051_example()
        doc2 = create_8051_example()
        assert doc1 is not doc2, "工厂函数应每次创建新对象"
        # 但内容一致
        assert doc1.case_id == doc2.case_id
        assert len(doc1.components) == len(doc2.components)

    def test_backward_compat_alias_is_callable(self):
        """向后兼容别名 EXAMPLE_8031_CASE001 现在是可调用的函数"""
        assert callable(EXAMPLE_8031_CASE001)
        doc = EXAMPLE_8031_CASE001()
        assert isinstance(doc, SchematicIRDocument)

    def test_lru_cache_info(self):
        """验证 lru_cache 命中统计"""
        # 清空缓存以确保统计准确
        get_8051_example.cache_clear()
        # 首次调用：未命中
        get_8051_example()
        info1 = get_8051_example.cache_info()
        assert info1.misses == 1
        # 再次调用：命中
        get_8051_example()
        info2 = get_8051_example.cache_info()
        assert info2.hits == 1
        # 清理，避免影响其他测试
        get_8051_example.cache_clear()


# ============ 5. IRValidationSummary 结构化测试 ============
class TestIRValidationSummary:
    """测试 IRValidationResult.summary 结构化"""

    def test_default_summary_is_structured(self):
        """默认 summary 是 IRValidationSummary 实例"""
        result = IRValidationResult(is_valid=True)
        assert isinstance(result.summary, IRValidationSummary)

    def test_default_values(self):
        """默认值符合预期"""
        summary = IRValidationSummary()
        assert summary.total_components == 0
        assert summary.total_nets == 0
        assert summary.total_pins == 0
        assert summary.floating_components == []
        assert summary.floating_pins == []
        assert summary.semantic_coverage is None
        assert summary.extra == {}

    def test_construct_with_fields(self):
        """可通过字段构造"""
        summary = IRValidationSummary(
            total_components=7,
            total_nets=5,
            total_pins=52,
            floating_components=["U2"],
            semantic_coverage=1.0,
        )
        assert summary.total_components == 7
        assert summary.floating_components == ["U2"]
        assert summary.semantic_coverage == 1.0

    def test_floating_components_is_list_not_set(self):
        """floating_components 是 list（与 validator 内部 set 区分）"""
        summary = IRValidationSummary(floating_components=["U1", "U2"])
        assert isinstance(summary.floating_components, list)

    def test_summary_extra_for_extensibility(self):
        """extra 字段用于扩展"""
        summary = IRValidationSummary(extra={"custom_metric": 42})
        assert summary.extra["custom_metric"] == 42

    def test_summary_forbids_unknown_fields(self):
        """summary 禁止未知字段（extra='forbid'）"""
        with pytest.raises(ValidationError):
            IRValidationSummary(unknown_field="x")

    def test_result_accepts_summary_instance(self):
        """IRValidationResult 接受 IRValidationSummary 实例"""
        summary = IRValidationSummary(total_components=3)
        result = IRValidationResult(is_valid=True, summary=summary)
        assert result.summary.total_components == 3

    def test_result_accepts_summary_dict(self):
        """IRValidationResult 接受 dict（Pydantic 自动转换）"""
        result = IRValidationResult(
            is_valid=True,
            summary={"total_components": 3, "total_nets": 2},
        )
        assert isinstance(result.summary, IRValidationSummary)
        assert result.summary.total_components == 3

    def test_result_json_roundtrip(self):
        """IRValidationResult JSON 往返"""
        result = IRValidationResult(
            is_valid=True,
            errors=[],
            warnings=["w1"],
            summary=IRValidationSummary(
                total_components=7,
                floating_components=["U2"],
                semantic_coverage=1.0,
            ),
        )
        json_str = result.model_dump_json()
        restored = IRValidationResult.model_validate_json(json_str)
        assert restored.summary.total_components == 7
        assert restored.summary.floating_components == ["U2"]
        assert restored.summary.semantic_coverage == 1.0

    def test_validator_returns_structured_summary(self):
        """验证器返回的 summary 应可结构化访问（与实际 validator 实现对接）"""
        # ===== [NEW] validator 已同步升级，result.summary 直接就是 IRValidationSummary =====
        # 原代码（model_validate 兼容旧 dict 的写法）：
        # doc = get_8051_example()
        # result = IRSchemaValidator.validate(doc, strict_mode=False)
        # summary = IRValidationSummary.model_validate(result.summary)
        # assert summary.total_components >= 0
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        # 直接断言类型，不再需要 model_validate 转换
        assert isinstance(result.summary, IRValidationSummary)
        assert result.summary.total_components >= 0
        # ===== [/NEW] =====

    def test_missing_required_is_valid(self):
        """is_valid 是必填字段"""
        with pytest.raises(ValidationError):
            IRValidationResult()  # 缺 is_valid


# ============================================================
# [NEW] 新增单元测试 - 针对 validator.py 与 IRValidationSummary 的字段对接
# ============================================================

class TestValidatorSummaryFields:
    """测试 validator.validate() 返回的 summary 字段与 IRValidationSummary 对齐"""

    def test_summary_is_irvalidation_summary_instance(self):
        """summary 必须是 IRValidationSummary 实例"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        assert isinstance(result.summary, IRValidationSummary)

    def test_summary_total_fields(self):
        """total_components / total_nets / total_pins 数值正确"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        assert result.summary.total_components == len(doc.components)
        assert result.summary.total_nets == len(doc.nets)
        assert result.summary.total_pins == sum(len(c.pins) for c in doc.components)

    def test_summary_floating_components_is_list(self):
        """floating_components 是 List[str]（不再是 int）"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        assert isinstance(result.summary.floating_components, list)
        assert isinstance(result.summary.floating_pins, list)

    def test_summary_semantic_coverage_is_float(self):
        """semantic_coverage 是 Optional[float]（不再是 dict）"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        cov = result.summary.semantic_coverage
        assert cov is None or isinstance(cov, float)
        # 8051 示例三语义字段 100% 覆盖，应为 1.0
        assert cov == 1.0

    def test_summary_extra_preserves_old_fields(self):
        """原先放在 summary 顶层的字段，现在应保留在 extra 中"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        extra = result.summary.extra
        assert extra["ir_schema_version"] == doc.ir_schema_version
        assert extra["case_id"] == doc.case_id
        assert "component_types" in extra
        assert "net_types" in extra
        assert "semantic_coverage_detail" in extra

    def test_summary_extra_semantic_coverage_detail(self):
        """extra.semantic_coverage_detail 保留原分项统计"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        detail = result.summary.extra["semantic_coverage_detail"]
        assert detail["total_components"] == len(doc.components)
        assert detail["has_intent"] == len(doc.components)
        assert detail["has_context"] == len(doc.components)
        assert detail["has_constraint"] == len(doc.components)

    def test_summary_extra_component_types(self):
        """extra.component_types 统计各 lib_name 数量"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        comp_types = result.summary.extra["component_types"]
        # 8051 示例：STC89C55RC ×1，CRYSTAL ×1，CAP ×2，CAP_ELEC ×1，RES ×1，SW_PB ×1
        assert comp_types.get("STC89C55RC") == 1
        assert comp_types.get("CRYSTAL") == 1
        assert comp_types.get("CAP") == 2

    def test_summary_extra_net_types(self):
        """extra.net_types 统计电源/地/信号网络数量"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        net_types = result.summary.extra["net_types"]
        # 8051 示例：1 个 VCC 电源网络；GND + EA_GND 共 2 个地网络；其余为信号
        assert net_types["power"] == 1
        assert net_types["ground"] == 2

    def test_summary_serializable(self):
        """summary 可序列化为 JSON 并往返"""
        doc = get_8051_example()
        result = IRSchemaValidator.validate(doc, strict_mode=False)
        json_str = result.model_dump_json()
        restored = IRValidationResult.model_validate_json(json_str)
        assert restored.summary.total_components == result.summary.total_components
        assert restored.summary.semantic_coverage == result.summary.semantic_coverage
        assert restored.summary.extra["case_id"] == "case001"


# ============================================================
# 集成测试（原 test_ir_integration_pg.py 内容）
# ============================================================

# ============================================================
# 1. JSONB 存储测试
# ============================================================

def test_ir_jsonb_store(pg_session, test_schematic_case):
    """测试 IR 文档写入 JSONB 字段"""
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====

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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====

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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # doc_v1 = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用，并显式构造两个独立版本 =====
    doc_v1 = get_8051_example()
    # ===== [/NEW] =====

    doc_v2 = doc_v1.model_copy(deep=True)
    doc_v2.title = "8051最小系统 V2"
    # ===== [NEW] created_at 现为 datetime，可直接赋 datetime 对象 =====
    # 原代码：doc_v2.created_at = (datetime.now() + timedelta(seconds=1)).isoformat()
    doc_v2.created_at = datetime.now() + timedelta(seconds=1)
    # ===== [/NEW] =====

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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====

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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====

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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====

    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    assert db_rec.ir_schema_version == "v1.0"


def test_ir_document_created_at_auto_generated(pg_session, test_schematic_case):
    """测试 ir_document.created_at 数据库自动生成"""
    # doc = EXAMPLE_8031_CASE001  # 原代码
    # ===== [NEW] 改为函数调用 =====
    doc = get_8051_example()
    # ===== [/NEW] =====
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


# ============================================================
# [NEW] 集成测试补充 - 针对 created_at datetime 与 JSONB 交互
# ============================================================

def test_ir_jsonb_created_at_datetime_format(pg_session, test_schematic_case):
    """验证 created_at 作为 datetime 存入 JSONB 后仍可正确读取"""
    doc = get_8051_example()

    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    loaded = load_schematic_ir(pg_session, db_rec.id)

    # 往返后 created_at 仍应为 datetime 类型
    assert isinstance(loaded.created_at, datetime)
    # 值应与原始一致（精确到微秒）
    assert loaded.created_at == doc.created_at


def test_ir_jsonb_created_at_nullable(pg_session, test_schematic_case):
    """验证 created_at 为 None 时 JSONB 存储不报错"""
    doc = SchematicIRDocument(
        case_id="no_time_case",
        title="No Time",
        description="created_at is None",
        components=[],
        nets=[],
        # created_at 未设置，默认 None
    )

    db_rec = store_schematic_ir(
        pg_session,
        case_id="no_time_case",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    loaded = load_schematic_ir(pg_session, db_rec.id)
    assert loaded.created_at is None


def test_ir_jsonb_connected_pins_validated_on_load(pg_session, test_schematic_case):
    """验证从 JSONB 加载时 connected_pins 校验器仍生效"""
    doc = get_8051_example()

    db_rec = store_schematic_ir(
        pg_session,
        case_id="case001",
        ir_doc=doc,
        schematic_case_id=test_schematic_case.id,
    )
    pg_session.commit()

    loaded = load_schematic_ir(pg_session, db_rec.id)

    # 所有 connected_pins 应满足 "Ref.pin_id" 格式
    for net in loaded.nets:
        for pin_ref in net.connected_pins:
            assert "." in pin_ref, f"非法 connected_pin: {pin_ref}"
            ref, pin_id = pin_ref.split(".", 1)
            assert ref, f"ref 为空: {pin_ref}"
            assert pin_id, f"pin_id 为空: {pin_ref}"
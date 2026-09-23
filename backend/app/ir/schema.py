"""
Schematic IR Schema v1.0
原理图中间表示标准；Sprint0 Business Freeze 冻结版本
核心实体：Attribute / Pin / Component / Net / SchematicIRDocument

融合特性：
- ✅ 以豆包方案为基线（数据库集成、单元测试、extra="forbid"）
- ✅ 三语义字段 (intent/context/constraint) - DeepSeek
- ✅ 精细引脚类型体系 (PinDirection + PinType) - DeepSeek
- ✅ 统一 Pin 命名 (pin_id + name + number) - 融合
- ✅ 完整的 8051 40 引脚示例 - DeepSeek
- ✅ 评审意见修正 (2026-09-09)
  - connected_pins 格式约定文档化
  - created_at 改为 Optional
  - SW1 网络连接修正
- ✅ 豆包最新意见修正 (2026-09-09)
  - lib_name 在验证器中为 Error（规则引擎核心字段）
  - floating_components 统计复用结构化返回
- ✅ [NEW] 二次优化 (v1.0.1)
  - connected_pins 增加格式校验 (field_validator)
  - Attribute.value 收紧为 Union[int, float, str, bool]
  - created_at 支持 datetime 类型（自动 ISO 序列化）
  - 示例常量改为 lru_cache 懒加载函数
  - IRValidationResult.summary 结构化（新增 IRValidationSummary 模型）
"""

from pydantic import BaseModel, Field, ConfigDict
# ===== [NEW] 新增导入 =====
from pydantic import field_validator  # 字段级校验器（Pydantic v2）
# ===== [/NEW] =====
from typing import Optional, List, Dict, Any
# ===== [NEW] 新增导入 =====
from typing import Union, ClassVar  # ===== [NEW] 新增 ClassVar =====
from datetime import datetime
from functools import lru_cache
import re
# ===== [/NEW] =====
from enum import Enum


# ============ 引脚类型枚举 (DeepSeek) ============
class PinDirection(str, Enum):
    """引脚方向枚举"""
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    POWER = "power"
    GROUND = "ground"


class PinType(str, Enum):
    """引脚功能类型枚举"""
    IO = "io"
    ANALOG = "analog"
    POWER = "power"
    GROUND = "ground"
    CLOCK = "clock"
    RESET = "reset"
    OTHER = "other"


# ============ 核心模型 (豆包基线 + DeepSeek 增强) ============
class Attribute(BaseModel):
    """元件/引脚通用键值属性 """
    key: str = Field(description="属性key，如 resistance, capacitance, frequency")
    # value: Any = Field(description="属性值，支持数字/字符串")  # 原代码：过于宽松
    # ===== [NEW] 收紧 value 类型：排除 list/dict 等不适合当"属性值"的类型 =====
    value: Union[int, float, str, bool] = Field(
        description="属性值，支持数字/字符串/布尔（不含 list/dict）"
    )
    # ===== [/NEW] =====
    unit: Optional[str] = Field(default=None, description="物理单位，如 Ω、pF、MHz")

    model_config = ConfigDict(extra="forbid")


class Pin(BaseModel):
    """
    引脚模型 - 精细类型 + 统一命名

    字段说明：
        pin_id: 引脚唯一标识，如 'P1.0' 或 'pin1'（用于 connected_pins 引用）
        name: 引脚名称，如 'RESET', 'XTAL1', 'VCC'
        number: 物理引脚编号，如 '9', '19', '40'（仅用于显示/参考，不作为连接标识）
        direction: 引脚方向（input/output/bidirectional/power/ground）
        pin_type: 引脚功能类型（io/analog/power/ground/clock/reset/other）
    """
    pin_id: str = Field(description="引脚唯一标识，如 'P1.0' 或 'pin1'")
    name: str = Field(description="引脚名称，如 'RESET', 'XTAL1', 'VCC'")
    number: Optional[str] = Field(default=None, description="物理引脚编号，如 '9', '19', '40'（仅用于显示）")
    direction: PinDirection = Field(description="引脚方向")
    pin_type: PinType = Field(description="引脚功能类型")
    voltage: Optional[float] = Field(default=None, description="电压值")
    attributes: List[Attribute] = Field(default_factory=list, description="引脚附加属性")
    description: Optional[str] = Field(default=None, description="引脚功能描述")

    model_config = ConfigDict(extra="forbid")


class Component(BaseModel):
    """
    原理图元器件模型 - 包含三语义字段 (DeepSeek 核心特性)

    三语义字段说明（Sprint 0 Business Freeze）：
        intent: 功能意图 - 描述该元件在电路中的功能目的
        context: 应用上下文 - 描述该元件所处的电路场景
        constraint: 约束条件 - 描述该元件的关键参数约束

    注意：三语义字段是 Schema 冻结要求（字段必须存在），
    但不是数据冻结要求（值可以为 None）。Golden Case 需要 100% 覆盖，
    普通原理图允许部分元件缺失。

    重要字段说明：
        lib_name: 库器件名称，规则引擎匹配 YAML 规则库的核心字段。
                 验证器中缺失会报 Error，必须填写。
    """
    ref: str = Field(description="位号，如 U1, R1, C1, Y1")
    lib_name: str = Field(
        description="库器件名称，如 STC89C55RC, CRYSTAL, RES, CAP（规则引擎核心匹配字段）"
    )
    value: Optional[str] = Field(default=None, description="器件标称值")
    footprint: Optional[str] = Field(default=None, description="封装类型，如 DIP-40, 0805")
    attributes: List[Attribute] = Field(default_factory=list, description="器件参数属性")
    pins: List[Pin] = Field(default_factory=list, description="引脚列表")

    # ===== 三语义字段 (DeepSeek Sprint 0 冻结要求) =====
    # 注意：字段存在即满足 Schema 冻结，值可以为 None
    intent: Optional[str] = Field(default=None, description="功能意图，如 '系统时钟源'")
    context: Optional[str] = Field(default=None, description="应用上下文，如 '8051外部晶振电路'")
    constraint: Optional[str] = Field(default=None, description="约束条件，如 '频率11.0592MHz，负载电容22pF'")

    model_config = ConfigDict(extra="forbid")


class Net(BaseModel):
    """
    电气网络模型

    connected_pins 格式约定（重要）：
        字符串格式："{Component.ref}.{Pin.pin_id}"
        示例：["U1.VCC", "R1.pin1", "C3.pin1"]

        ⚠️ 注意：使用的是 Pin.pin_id，不是 Pin.number（物理引脚编号）
        正确：U1.VCC（pin_id = "VCC"）
        错误：U1.40（number = "40"，不应使用）

        原因：pin_id 是业务语义标识，稳定且可读；number 仅是物理封装编号，
        同一芯片不同封装可能有不同的 number，使用 pin_id 确保引用稳定性。
    """
    net_name: str = Field(description="网络名称，如 'VCC', 'GND', 'XTAL_OSC'")
    connected_pins: List[str] = Field(
        default_factory=list,
        description="连接的引脚，格式 '{Component.ref}.{Pin.pin_id}'，如 ['U1.VCC', 'R1.pin1']"
    )
    attributes: List[Attribute] = Field(default_factory=list, description="网络属性")

    # 增强字段
    is_power: bool = Field(default=False, description="是否为电源网络")
    is_ground: bool = Field(default=False, description="是否为地网络")
    voltage: Optional[float] = Field(default=None, description="电压值")
    description: Optional[str] = Field(default=None, description="网络描述")

    model_config = ConfigDict(extra="forbid")

    # ===== [NEW] connected_pins 格式校验器 =====
    # 约定格式："{Component.ref}.{Pin.pin_id}"
    # _CONNECTED_PIN_PATTERN = re.compile(r"^[^\s.]+?\.[^\s.]+$")  # 原代码
    # ===== [FIX] 用 ClassVar 声明，避免 Pydantic v2 私有属性封装 =====
    # 说明：
    #   Pydantic v2 会把类里以 `_` 开头的属性默认封装为 ModelPrivateAttr，
    #   导致 `cls._CONNECTED_PIN_PATTERN` 拿到的是包装器而不是 re.Pattern。
    #   显式标注为 ClassVar 后，Pydantic 不再触碰该属性。
    _CONNECTED_PIN_PATTERN: ClassVar[re.Pattern] = re.compile(r"^[^\s.]+?\.[^\s.]+$")
    # ===== [/FIX] =====


    @field_validator("connected_pins")
    @classmethod
    def _validate_connected_pins_format(cls, v: List[str]) -> List[str]:
        bad: List[str] = []
        for item in v:
            if not isinstance(item, str) or not cls._CONNECTED_PIN_PATTERN.match(item):
                bad.append(repr(item))
        if bad:
            raise ValueError(
                "connected_pins 格式必须为 '{Component.ref}.{Pin.pin_id}' "
                f"（点号两侧非空、不含空白），违规项: {', '.join(bad)}"
            )
        return v
    # ===== [/NEW] =====


class SchematicIRDocument(BaseModel):
    """
    顶层原理图 IR 文档对象 (豆包基线，兼容 IRDocument 表)

    注意：created_at 为 Optional，不自动生成时间戳。
    原因：避免 Pydantic model_validate() 反序列化时静默覆盖原始时间。
    示例生成时手动赋值。
    """
    ir_schema_version: str = Field(default="v1.0", description="IR Schema版本，冻结v1.0")
    case_id: str = Field(description="测试用例ID，如 'case001'")
    title: str = Field(description="原理图标题")
    description: str = Field(description="原理图描述")
    # created_at: Optional[str] = Field(default=None, description="创建时间（ISO格式），示例中手动赋值")  # 原代码：字符串
    # ===== [NEW] created_at 改为 datetime 类型 =====
    # 优势：
    #   - Pydantic 原生支持 datetime，自动解析 ISO 字符串
    #   - 序列化时自动输出 ISO 格式（model_dump(mode="json")）
    #   - 语义更强，避免"看起来是字符串、实际是时间"的混淆
    # 兼容性：默认仍是 None，不会自动生成时间戳（避免覆盖原始时间）
    created_at: Optional[datetime] = Field(
        default=None,
        description="创建时间（datetime 类型，序列化时自动 ISO 格式）"
    )
    # ===== [/NEW] =====

    components: List[Component] = Field(default_factory=list, description="全部元器件")
    nets: List[Net] = Field(default_factory=list, description="全部电气网络")
    extra_meta: Dict[str, Any] = Field(default_factory=dict, description="扩展元信息")

    model_config = ConfigDict(extra="forbid")


# ============ 验证结果模型 (DeepSeek) ============
# ===== [NEW] 新增结构化 Summary 模型 =====
class IRValidationSummary(BaseModel):
    """
    IR 验证摘要 - 结构化返回

    将原先松散的 Dict[str, Any] 收紧为明确的字段，
    便于下游消费方类型安全地读取。
    """
    total_components: int = Field(default=0, description="元件总数")
    total_nets: int = Field(default=0, description="网络总数")
    total_pins: int = Field(default=0, description="引脚总数")
    floating_components: List[str] = Field(
        default_factory=list,
        description="悬空元件位号列表（未连接到任何网络的元件）"
    )
    floating_pins: List[str] = Field(
        default_factory=list,
        description="悬空引脚列表，格式 'Ref.pin_id'"
    )
    semantic_coverage: Optional[float] = Field(
        default=None,
        description="三语义字段覆盖率（0.0~1.0），Golden Case 要求 1.0"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="其他扩展统计信息"
    )

    model_config = ConfigDict(extra="forbid")


class IRValidationResult(BaseModel):
    """IR验证结果"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    # summary: Dict[str, Any] = Field(default_factory=dict)  # 原代码：松散字典
    # ===== [NEW] summary 改为结构化模型 =====
    summary: IRValidationSummary = Field(
        default_factory=IRValidationSummary,
        description="结构化验证摘要"
    )
    # ===== [/NEW] =====
# ===== [/NEW] =====


# ============ 完整的 8051 最小系统示例 (DeepSeek) ============
def create_8051_example() -> SchematicIRDocument:
    """
    创建完整的 8051 最小系统示例 IR (包含所有 40 个引脚)

    根据评审意见修正 (2026-09-09)：
        - created_at 手动赋值
        - SW1 已加入 RST 网络（SW1.pin1 连接 RST，SW1.pin2 连接 GND）
    """
    # ===== [NEW] datetime 已在模块顶部导入，函数内不再重复导入 =====
    # from datetime import datetime  # 原代码：函数内导入（已上移至模块顶部）
    # ===== [/NEW] =====

    return SchematicIRDocument(
        case_id="case001",
        title="8051最小系统原理图 (STC89C55RC)",
        description="基于 STC89C55RC 的 8051 最小系统：5V 电源，11.0592MHz 晶振，上电复位电路",
        created_at=datetime.now().isoformat(),  # 手动赋值，避免默认值陷阱
        # ===== [NEW] created_at 现在接受 datetime 或 ISO 字符串（Pydantic 自动解析） =====
        # 上面这行保持不变即可——Pydantic 会自动把 ISO 字符串解析成 datetime
        # 若想更直接，可改为：created_at=datetime.now()
        # ===== [/NEW] =====
        components=[
            # ===== MCU - 完整 40 引脚定义 =====
            Component(
                ref="U1",
                lib_name="STC89C55RC",
                value="STC89C55RC",
                footprint="DIP-40",
                intent="8051微控制器核心，执行程序逻辑",
                context="8051最小系统主控芯片",
                constraint="VCC=5V ±5%，晶振频率11.0592MHz",
                pins=[
                    Pin(pin_id="P1.0", name="P1.0", number="1", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.1", name="P1.1", number="2", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.2", name="P1.2", number="3", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.3", name="P1.3", number="4", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.4", name="P1.4", number="5", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.5", name="P1.5", number="6", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.6", name="P1.6", number="7", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P1.7", name="P1.7", number="8", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="RST", name="RST", number="9", direction=PinDirection.INPUT, pin_type=PinType.RESET),
                    Pin(pin_id="P3.0", name="P3.0", number="10", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.1", name="P3.1", number="11", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.2", name="P3.2", number="12", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.3", name="P3.3", number="13", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.4", name="P3.4", number="14", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.5", name="P3.5", number="15", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.6", name="P3.6", number="16", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P3.7", name="P3.7", number="17", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="XTAL2", name="XTAL2", number="18", direction=PinDirection.OUTPUT, pin_type=PinType.CLOCK),
                    Pin(pin_id="XTAL1", name="XTAL1", number="19", direction=PinDirection.INPUT, pin_type=PinType.CLOCK),
                    Pin(pin_id="GND", name="GND", number="20", direction=PinDirection.POWER, pin_type=PinType.GROUND),
                    Pin(pin_id="P2.0", name="P2.0", number="21", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.1", name="P2.1", number="22", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.2", name="P2.2", number="23", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.3", name="P2.3", number="24", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.4", name="P2.4", number="25", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.5", name="P2.5", number="26", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.6", name="P2.6", number="27", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P2.7", name="P2.7", number="28", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="PSEN", name="PSEN", number="29", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="ALE", name="ALE", number="30", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="EA", name="EA", number="31", direction=PinDirection.INPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="P0.7", name="P0.7", number="32", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.6", name="P0.6", number="33", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.5", name="P0.5", number="34", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.4", name="P0.4", number="35", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.3", name="P0.3", number="36", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.2", name="P0.2", number="37", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.1", name="P0.1", number="38", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="P0.0", name="P0.0", number="39", direction=PinDirection.BIDIRECTIONAL, pin_type=PinType.IO),
                    Pin(pin_id="VCC", name="VCC", number="40", direction=PinDirection.POWER, pin_type=PinType.POWER),
                ]
            ),
            # ===== 晶振 =====
            Component(
                ref="Y1",
                lib_name="CRYSTAL",
                value="11.0592MHz",
                footprint="HC-49S",
                intent="提供系统时钟信号",
                context="8051外部晶振电路",
                constraint="频率11.0592MHz，负载电容22pF",
                pins=[
                    Pin(pin_id="pin1", name="XTAL1", number="1", direction=PinDirection.OUTPUT, pin_type=PinType.CLOCK),
                    Pin(pin_id="pin2", name="XTAL2", number="2", direction=PinDirection.OUTPUT, pin_type=PinType.CLOCK),
                ]
            ),
            # ===== 负载电容 C1 =====
            Component(
                ref="C1",
                lib_name="CAP",
                value="22pF",
                footprint="0805",
                intent="晶振负载电容",
                context="晶振电路匹配电容",
                constraint="22pF ±5%",
                pins=[
                    Pin(pin_id="pin1", name="pin1", number="1", direction=PinDirection.INPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="pin2", name="pin2", number="2", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                ]
            ),
            # ===== 负载电容 C2 =====
            Component(
                ref="C2",
                lib_name="CAP",
                value="22pF",
                footprint="0805",
                intent="晶振负载电容",
                context="晶振电路匹配电容",
                constraint="22pF ±5%",
                pins=[
                    Pin(pin_id="pin1", name="pin1", number="1", direction=PinDirection.INPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="pin2", name="pin2", number="2", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                ]
            ),
            # ===== 复位电容 C3 =====
            Component(
                ref="C3",
                lib_name="CAP_ELEC",
                value="10μF",
                footprint="Radial-5mm",
                intent="复位电路电容，提供上电复位延时",
                context="复位电路",
                constraint="10μF 电解电容",
                pins=[
                    Pin(pin_id="pin1", name="pin1", number="1", direction=PinDirection.INPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="pin2", name="pin2", number="2", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                ]
            ),
            # ===== 复位电阻 R1 =====
            Component(
                ref="R1",
                lib_name="RES",
                value="10kΩ",
                footprint="0805",
                intent="复位电路下拉电阻，保证复位后RST电平为低",
                context="复位电路，配合C3形成RC延时",
                constraint="10kΩ ±1%",
                pins=[
                    Pin(pin_id="pin1", name="pin1", number="1", direction=PinDirection.INPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="pin2", name="pin2", number="2", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                ]
            ),
            # ===== 复位按键 SW1 =====
            Component(
                ref="SW1",
                lib_name="SW_PB",
                value="复位按键",
                footprint="SW_PB",
                intent="手动复位控制，按下时将RST引脚拉低",
                context="复位电路手动触发，用于系统复位",
                constraint="常开按键，按下时导通",
                pins=[
                    Pin(pin_id="pin1", name="pin1", number="1", direction=PinDirection.INPUT, pin_type=PinType.OTHER),
                    Pin(pin_id="pin2", name="pin2", number="2", direction=PinDirection.OUTPUT, pin_type=PinType.OTHER),
                ]
            ),
        ],
        nets=[
            # ===== VCC 电源网络 =====
            Net(
                net_name="VCC",
                connected_pins=["U1.VCC"],
                is_power=True,
                is_ground=False,
                voltage=5.0,
                description="系统电源网络 +5V"
            ),
            # ===== GND 地网络 =====
            Net(
                net_name="GND",
                connected_pins=[
                    "U1.GND",
                    "C1.pin2",
                    "C2.pin2",
                    "C3.pin2",
                    "R1.pin2",
                    "SW1.pin2"  # SW1 的另一端接地，按下时拉低 RST
                ],
                is_power=False,
                is_ground=True,
                voltage=0.0,
                description="系统参考地"
            ),
            # ===== 晶振网络 =====
            Net(
                net_name="XTAL_OSC",
                connected_pins=[
                    "U1.XTAL1",
                    "U1.XTAL2",
                    "Y1.pin1",
                    "Y1.pin2",
                    "C1.pin1",
                    "C2.pin1"
                ],
                is_power=False,
                is_ground=False,
                description="晶振振荡网络"
            ),
            # ===== 复位网络 =====
            Net(
                net_name="RST",
                connected_pins=[
                    "U1.RST",
                    "R1.pin1",
                    "C3.pin1",
                    "SW1.pin1"  # SW1 的一端接 RST，按下时 RST 被拉低
                ],
                is_power=False,
                is_ground=False,
                description="复位信号网络，上电时 C3 充电产生高电平，R1 下拉保证复位后为低"
            ),
            # ===== EA 接地网络 =====
            Net(
                net_name="EA_GND",
                connected_pins=["U1.EA"],
                is_power=False,
                is_ground=True,
                voltage=0.0,
                description="EA引脚接地，选择内部程序存储器"
            ),
        ],
        extra_meta={
            "source": "8051最小系统原理图",
            "design_tool": "基于图片还原",
            "total_components": 7,
            "total_pins": 52,
            "total_nets": 5,
            "power_consumption": "~50mW",
            "semantic_coverage": "100%",  # Golden Case 要求
        }
    )


# ===== 导出的示例常量 =====
# EXAMPLE_8031_CASE001 = create_8051_example()  # 原代码：import 时立即构造
# ===== [NEW] 改为 lru_cache 懒加载函数 =====
# 优点：
#   - import 模块时不再立即构造对象（避免 import 期异常）
#   - 首次调用后缓存，后续调用零成本
#   - 仍可像常量一样使用：get_8051_example()
@lru_cache(maxsize=1)
def get_8051_example() -> SchematicIRDocument:
    """懒加载并缓存 8051 示例文档（首次调用时构造，之后复用同一实例）"""
    return create_8051_example()


# 向后兼容别名：老代码若还引用 EXAMPLE_8031_CASE001，可用函数式访问
# 注意：这不再是"常量"，而是"函数"，需要调用才能拿到对象
EXAMPLE_8031_CASE001 = get_8051_example  # 别名指向函数本身
# ===== [/NEW] =====
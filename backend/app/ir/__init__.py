# app/ir/__init__.py

from app.ir.schema import (
    Attribute,
    Component,
    IRValidationResult,
    IRValidationSummary,
    Net,
    Pin,
    PinDirection,
    PinType,
    SchematicIRDocument,
    create_8051_example,
    # ===== [V1.3 修改] 删除 EXAMPLE_8031_CASE001（别名已从 schema.py 移除） =====
    # [原逻辑 - 保留注释，便于对照回滚]
    # EXAMPLE_8031_CASE001,
    # ===== [/V1.3 修改] =====
    get_8051_example,
)
from app.ir.serializer import dump_ir, load_ir, validate_ir_file
from app.ir.service import load_schematic_ir, load_schematic_ir_by_case, store_schematic_ir
from app.ir.validator import IRSchemaValidator

__all__ = [
    # Schema
    "SchematicIRDocument",
    "Component",
    "Pin",
    "Net",
    "Attribute",
    "PinDirection",
    "PinType",
    "IRValidationResult",
    "IRValidationSummary",
    # ===== [V1.3 修改] 删除 EXAMPLE_8031_CASE001（别名已移除） =====
    # [原逻辑 - 保留注释，便于对照回滚]
    # "EXAMPLE_8031_CASE001",
    # ===== [/V1.3 修改] =====
    "get_8051_example",
    "create_8051_example",
    # Serializer
    "dump_ir",
    "load_ir",
    "validate_ir_file",
    # Service
    "store_schematic_ir",
    "load_schematic_ir",
    "load_schematic_ir_by_case",
    # Validator
    "IRSchemaValidator",
]

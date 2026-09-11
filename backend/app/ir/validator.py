"""
Schematic IR 验证器
提供完整的 IR 文档校验功能，包含元件完整性、网络引用检查

修正记录 (2026-09-09)：
    - intent/context 缺失：降级为 Warning，不阻断验证
    - 增加 strict_mode 参数（用于 Golden Case 验收）
    - 增加 connected_pins 格式验证
    - 增加悬浮元件 Warning（元件存在但未连接任何网络）
    - 增加 pin_id 有效性验证
    - lib_name 缺失提升为 Error（规则引擎匹配核心字段）
    - floating_components 统计复用结构化返回，避免重复遍历
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from app.ir.schema import SchematicIRDocument, IRValidationResult


@dataclass
class FloatingCheckResult:
    """
    悬浮元件检查结果
    
    Attributes:
        warnings: 悬浮元件相关的警告列表
        floating_component_refs: 完全悬浮的元件 ref 集合（所有引脚均未连接）
        floating_pin_refs: 部分悬浮的引脚引用集合（单个引脚未连接）
    """
    warnings: List[str] = field(default_factory=list)
    floating_component_refs: Set[str] = field(default_factory=set)
    floating_pin_refs: Set[str] = field(default_factory=set)


class IRSchemaValidator:
    """IR Schema 验证器 - DeepSeek 完整验证逻辑"""
    
    @staticmethod
    def validate_components(
        ir: SchematicIRDocument, 
        strict_mode: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        验证元件数据完整性
        
        Args:
            ir: IR 文档
            strict_mode: 严格模式，用于 Golden Case 验收
        
        Returns:
            (errors, warnings) 元组
        """
        errors = []
        warnings = []
        
        for comp in ir.components:
            if not comp.ref:
                errors.append("元件缺少 ref")
                continue
            
            # ===== lib_name 缺失：Error（规则引擎匹配核心字段） =====
            if not comp.lib_name:
                errors.append(f"元件 {comp.ref} 缺少 lib_name（规则引擎匹配核心字段，必须填写）")
            
            # ===== 三语义字段验证 (修正：缺失降级为 Warning) =====
            if not comp.intent:
                warnings.append(f"元件 {comp.ref} 缺少 intent (三语义字段，建议 Golden Case 补充)")
            if not comp.context:
                warnings.append(f"元件 {comp.ref} 缺少 context (三语义字段，建议 Golden Case 补充)")
            # constraint 为可选，不做任何提示
            
            # 检查引脚
            if not comp.pins:
                warnings.append(f"元件 {comp.ref} 没有引脚定义（可能为预留位）")
            
            # 检查引脚完整性
            for pin in comp.pins:
                if not pin.pin_id:
                    errors.append(f"元件 {comp.ref} 的引脚缺少 pin_id")
                if not pin.name:
                    warnings.append(f"元件 {comp.ref} 引脚 {pin.pin_id} 缺少 name")
                if not pin.direction:
                    errors.append(f"元件 {comp.ref} 引脚 {pin.pin_id} 缺少 direction")
        
        # strict_mode 下将三语义字段缺失升级为 Error
        if strict_mode:
            for comp in ir.components:
                if not comp.intent:
                    errors.append(f"[strict_mode] 元件 {comp.ref} 缺少 intent（Golden Case 要求 100% 覆盖）")
                if not comp.context:
                    errors.append(f"[strict_mode] 元件 {comp.ref} 缺少 context（Golden Case 要求 100% 覆盖）")
        
        return errors, warnings
    
    @staticmethod
    def validate_nets(
        ir: SchematicIRDocument,
        strict_mode: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        验证网络数据完整性
        
        Args:
            ir: IR 文档
            strict_mode: 严格模式（预留，当前与普通模式行为一致）
        
        Returns:
            (errors, warnings) 元组
        """
        errors = []
        warnings = []
        
        # 构建索引
        all_refs = {comp.ref for comp in ir.components}
        ref_to_pin_ids = {
            comp.ref: {pin.pin_id for pin in comp.pins}
            for comp in ir.components
        }
        
        for net in ir.nets:
            if not net.net_name:
                errors.append("网络缺少 net_name")
                continue
            
            for pin_ref in net.connected_pins:
                # ===== 格式验证 =====
                if '.' not in pin_ref:
                    errors.append(
                        f"网络 {net.net_name} 的引脚引用格式错误: '{pin_ref}'\n"
                        f"  正确格式: '{{Component.ref}}.{{Pin.pin_id}}'\n"
                        f"  示例: 'U1.VCC'"
                    )
                    continue
                
                comp_ref, pin_id = pin_ref.split('.', 1)
                
                # ===== 验证 ref 是否存在 =====
                if comp_ref not in all_refs:
                    errors.append(
                        f"网络 {net.net_name} 引用了不存在的元件 '{comp_ref}'"
                    )
                    continue
                
                # ===== 验证 pin_id 是否存在 =====
                if pin_id not in ref_to_pin_ids.get(comp_ref, set()):
                    warnings.append(
                        f"网络 {net.net_name} 引用了元件 {comp_ref} 中不存在的引脚 '{pin_id}'"
                    )
        
        return errors, warnings
    
    @staticmethod
    def check_floating_components(ir: SchematicIRDocument) -> FloatingCheckResult:
        """
        检查悬浮元件（元件被定义但未连接任何网络）
        
        注意：仅作为 Warning，不阻断验证。
        有些元件可能是预留位（DNP），允许悬浮。
        
        Returns:
            FloatingCheckResult 包含警告列表和悬浮元件统计
        """
        warnings = []
        
        # 收集所有被网络引用的引脚
        referenced_pins: Set[str] = set()
        for net in ir.nets:
            for pin_ref in net.connected_pins:
                if '.' in pin_ref:
                    referenced_pins.add(pin_ref)
        
        floating_component_refs: Set[str] = set()
        floating_pin_refs: Set[str] = set()
        
        # 检查每个元件的每个引脚
        for comp in ir.components:
            floating_pins: List[str] = []
            
            for pin in comp.pins:
                pin_ref = f"{comp.ref}.{pin.pin_id}"
                if pin_ref not in referenced_pins:
                    floating_pins.append(pin.pin_id)
                    floating_pin_refs.add(pin_ref)
            
            if floating_pins and len(floating_pins) == len(comp.pins):
                # 所有引脚都悬浮
                floating_component_refs.add(comp.ref)
                warnings.append(
                    f"元件 {comp.ref} 的所有引脚均未连接任何网络（可能为预留位/DNP）"
                )
            elif floating_pins:
                warnings.append(
                    f"元件 {comp.ref} 的引脚 {', '.join(floating_pins)} 未连接任何网络"
                )
        
        return FloatingCheckResult(
            warnings=warnings,
            floating_component_refs=floating_component_refs,
            floating_pin_refs=floating_pin_refs,
        )
    
    @staticmethod
    def validate(ir: SchematicIRDocument, strict_mode: bool = False) -> IRValidationResult:
        """
        完整验证原理图 IR
        
        Args:
            ir: IR 文档
            strict_mode: 严格模式
                - False（默认）：intent/context 缺失仅为 Warning
                - True：intent/context 缺失为 Error（用于 Golden Case 验收）
        
        Returns:
            IRValidationResult 包含验证结果、错误、警告和统计摘要
        """
        errors = []
        warnings = []
        
        # ===== 1. 验证元件 =====
        comp_errors, comp_warnings = IRSchemaValidator.validate_components(ir, strict_mode)
        errors.extend(comp_errors)
        warnings.extend(comp_warnings)
        
        # ===== 2. 验证网络 =====
        net_errors, net_warnings = IRSchemaValidator.validate_nets(ir, strict_mode)
        errors.extend(net_errors)
        warnings.extend(net_warnings)
        
        # ===== 3. 检查悬浮元件（使用结构化返回，复用计算结果） =====
        floating_result = IRSchemaValidator.check_floating_components(ir)
        warnings.extend(floating_result.warnings)
        
        # ===== 4. 生成摘要统计 =====
        total_pins = sum(len(comp.pins) for comp in ir.components)
        
        # 统计各类型元件
        component_types = {}
        for comp in ir.components:
            lib_name = comp.lib_name or "unknown"
            component_types[lib_name] = component_types.get(lib_name, 0) + 1
        
        # 统计网络类型
        net_types = {
            "power": sum(1 for n in ir.nets if n.is_power),
            "ground": sum(1 for n in ir.nets if n.is_ground),
            "signal": sum(1 for n in ir.nets if not n.is_power and not n.is_ground),
        }
        
        # 三语义字段覆盖率
        total_components = len(ir.components)
        semantic_coverage = {
            "has_intent": sum(1 for c in ir.components if c.intent),
            "has_context": sum(1 for c in ir.components if c.context),
            "has_constraint": sum(1 for c in ir.components if c.constraint),
            "total_components": total_components,
        }
        
        summary = {
            "ir_schema_version": ir.ir_schema_version,
            "case_id": ir.case_id,
            "total_components": total_components,
            "total_nets": len(ir.nets),
            "total_pins": total_pins,
            "component_types": component_types,
            "net_types": net_types,
            "semantic_coverage": semantic_coverage,
            # ===== 复用悬浮元件检查结果，避免重复遍历 =====
            "floating_components": len(floating_result.floating_component_refs),
            "floating_pins": len(floating_result.floating_pin_refs),
        }
        
        return IRValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary,
        )
"""
Schematic IR 序列化/反序列化 + CLI 工具

CLI 命令：
  # 校验 IR 文件（普通模式）
  docker compose exec backend uv run python -m app.ir.serializer --validate ./data/cases/case001/schematic_ir.json
  
  # 校验 IR 文件（严格模式，用于 Golden Case 验收）
  docker compose exec backend uv run python -m app.ir.serializer --validate ./data/cases/case001/schematic_ir.json --strict
  
  # 生成 case001 示例
  docker compose exec backend uv run python -m app.ir.serializer --generate-case001
  
  # 生成到指定目录
  docker compose exec backend uv run python -m app.ir.serializer --generate-case001 --out-dir ./data/cases/case001
"""

import json
import pathlib
import argparse
from typing import Optional

from app.ir.schema import SchematicIRDocument, EXAMPLE_8031_CASE001
from app.ir.validator import IRSchemaValidator


def dump_ir(ir_doc: SchematicIRDocument, output_path: pathlib.Path) -> None:
    """IR 对象导出为 JSON 文件 (豆包)"""
    text = ir_doc.model_dump_json(indent=2, exclude_none=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"✅ IR 已导出: {output_path}")


def load_ir(json_path: pathlib.Path) -> SchematicIRDocument:
    """从 JSON 文件加载 IR 文档，做 Pydantic schema 校验 (豆包)"""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return SchematicIRDocument.model_validate(raw)


def validate_ir_file(file_path: str, strict_mode: bool = False) -> bool:
    """
    校验 IR 文件 schema 合法性
    
    Args:
        file_path: IR JSON 文件路径
        strict_mode: 严格模式
            - False: intent/context 缺失仅警告，不阻断
            - True: intent/context 缺失报错（用于 Golden Case 验收）
    """
    fp = pathlib.Path(file_path)
    
    try:
        doc = load_ir(fp)
        result = IRSchemaValidator.validate(doc, strict_mode=strict_mode)
        
        mode_text = "🔒 严格模式 (Golden Case 验收)" if strict_mode else "📝 普通模式"
        print(f"\n📋 验证文件: {file_path}")
        print(f"模式: {mode_text}")
        print(f"状态: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
        
        # ===== 摘要信息 =====
        print(f"\n📊 摘要:")
        print(f"  - Schema版本: {result.summary.get('ir_schema_version', 'N/A')}")
        print(f"  - Case ID: {result.summary.get('case_id', 'N/A')}")
        print(f"  - 元件数: {result.summary.get('total_components', 0)}")
        print(f"  - 网络数: {result.summary.get('total_nets', 0)}")
        print(f"  - 引脚数: {result.summary.get('total_pins', 0)}")
        print(f"  - 悬浮元件: {result.summary.get('floating_components', 0)}")
        
        # ===== 三语义字段覆盖率 (Sprint 0 验收观测) =====
        semantic = result.summary.get('semantic_coverage', {})
        if semantic:
            total = semantic.get('total_components', 0)
            print(f"\n📝 三语义字段覆盖率 (Sprint 0 验收观测):")
            print(f"  - intent: {semantic.get('has_intent', 0)}/{total} ({semantic.get('has_intent', 0)/total*100:.1f}%)")
            print(f"  - context: {semantic.get('has_context', 0)}/{total} ({semantic.get('has_context', 0)/total*100:.1f}%)")
            print(f"  - constraint: {semantic.get('has_constraint', 0)}/{total} ({semantic.get('has_constraint', 0)/total*100:.1f}%)")
        
        # ===== 错误和警告 =====
        if result.errors:
            print(f"\n❌ 错误 ({len(result.errors)}):")
            for err in result.errors:
                print(f"  - {err}")
        
        if result.warnings:
            print(f"\n⚠️ 警告 ({len(result.warnings)}):")
            for warn in result.warnings:
                print(f"  - {warn}")
        
        return result.is_valid
        
    except Exception as e:
        print(f"❌ IR 校验失败: {e}")
        return False


def generate_case001_sample(output_dir: pathlib.Path) -> None:
    """生成 8051 最小系统样例 schematic_ir.json (豆包)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "schematic_ir.json"
    
    # 使用 DeepSeek 完整示例
    ir_doc = EXAMPLE_8031_CASE001
    dump_ir(ir_doc, out_file)
    
    # 自动验证生成的示例（普通模式）
    result = IRSchemaValidator.validate(ir_doc, strict_mode=False)
    
    print(f"\n📊 生成的示例统计:")
    print(f"  - 元件数: {result.summary.get('total_components', 0)}")
    print(f"  - 引脚数: {result.summary.get('total_pins', 0)}")
    print(f"  - 网络数: {result.summary.get('total_nets', 0)}")
    
    semantic = result.summary.get('semantic_coverage', {})
    total = semantic.get('total_components', 0)
    if total > 0:
        intent_rate = semantic.get('has_intent', 0) / total * 100
        print(f"  - 三语义覆盖率: {intent_rate:.1f}%")
    
    if result.is_valid:
        print(f"\n✅ 示例 IR 验证通过")
    else:
        print(f"\n⚠️ 示例 IR 存在验证问题:")
        for err in result.errors:
            print(f"  - {err}")


def main():
    """CLI 入口 (豆包增强版)"""
    parser = argparse.ArgumentParser(
        description="Schematic IR Schema 工具 (Sprint 0 Phase D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 校验 IR 文件（普通模式，intent/context 缺失仅警告）
  python -m app.ir.serializer --validate ./data/cases/case001/schematic_ir.json
  
  # 校验 IR 文件（严格模式，用于 Golden Case 验收）
  python -m app.ir.serializer --validate ./data/cases/case001/schematic_ir.json --strict
  
  # 生成 case001 示例
  python -m app.ir.serializer --generate-case001
  
  # 生成到指定目录
  python -m app.ir.serializer --generate-case001 --out-dir ./data/cases/case002
        """
    )
    
    parser.add_argument(
        "--validate", 
        type=str, 
        help="校验 IR json 文件路径"
    )
    parser.add_argument(
        "--strict", 
        action="store_true", 
        help="严格模式：intent/context 缺失报错（用于 Golden Case 验收）"
    )
    parser.add_argument(
        "--generate-case001", 
        action="store_true", 
        help="生成 8051 最小系统 case001 样例 IR"
    )
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="./data/cases/case001", 
        help="输出目录 (默认: ./data/cases/case001)"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        success = validate_ir_file(args.validate, strict_mode=args.strict)
        exit(0 if success else 1)
    
    elif args.generate_case001:
        generate_case001_sample(pathlib.Path(args.out_dir))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
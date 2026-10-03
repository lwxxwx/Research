"""
Schematic IR 序列化/反序列化 + CLI 工具

CLI 命令（V1.3：-f 用 $PWD 绝对路径，路径用容器内 /data）：
  # 校验 IR 目录或文件（普通模式）
  docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" \
    exec backend uv run python -m app.ir.serializer --validate /data/cases/case001

  # 校验 IR 目录或文件（严格模式，用于 Golden Case 验收）
  docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" \
    exec backend uv run python -m app.ir.serializer --validate /data/cases/case001 --strict

  # 生成 case001 示例
  docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" \
    exec backend uv run python -m app.ir.serializer --generate-case001

  # 生成到指定目录
  docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" \
    exec backend uv run python -m app.ir.serializer --generate-case001 --out-dir /data/cases/case002

            序列化 (dump)                        反序列化 (load)
  ┌──────────────────────────┐         ┌──────────────────────────┐
  │                          │         │                          │
  │  Python 对象             │         │  Python 对象             │
  │  SchematicIRDocument     │         │  SchematicIRDocument     │
  │   ├─ case_id: "case001"  │         │   ├─ case_id: "case001"  │
  │   ├─ components: [...]   │         │   ├─ components: [...]   │
  │   └─ nets: [...]         │         │   └─ nets: [...]         │
  │                          │         │                          │
  └───────────┬──────────────┘         └───────────▲──────────────┘
              │                                    │
              │  model_dump_json()                 │  model_validate()
              │  （序列化）                         │  （反序列化）
              ▼                                    │
  ┌──────────────────────────┐         ┌──────────────────────────┐
  │                          │         │                          │
  │  JSON 字符串              │  文件   │  Python dict             │
  │  '{"case_id": "case001"…'│ ──────► │  {"case_id": "case001",…}│
  │                          │         │                          │
  └──────────────────────────┘         └──────────────────────────┘
              │                                    ▲
              │  写文件 write_text                  │  json.loads
              │                                    │
              ▼                                    │
  ┌──────────────────────────┐         ┌──────────────────────────┐
  │  schematic_ir.json       │ ──────► │  读文件 read_text         │
  │  (磁盘上的文件)           │         │                          │
  └──────────────────────────┘         └──────────────────────────┘

"""

import argparse
import json
import pathlib

# from app.ir.schema import SchematicIRDocument, EXAMPLE_8031_CASE001  # 原代码：EXAMPLE_8031_CASE001 现为函数别名
# ===== [NEW] 显式导入函数式示例获取入口 =====
from app.ir.schema import SchematicIRDocument, get_8051_example

# ===== [/NEW] =====
from app.ir.validator import IRSchemaValidator


# ===== [NEW] V1.3：支持目录输入的解析辅助函数 =====
def _resolve_ir_path(input_path: str) -> pathlib.Path:
    """
    将 CLI 输入路径解析为具体的 IR JSON 文件路径。

    规则（V1.3）：
        - 若传入的是文件：直接返回该文件
        - 若传入的是目录：拼接 'schematic_ir.json'
        - 若传入的是其它（不存在）：直接返回，由下游 read_text 报错

    背景：方案 §9 的 CLI 传的是目录 '/data/cases/case001'，
         但 load_ir 只接受文件。本函数屏蔽该差异。
    """
    p = pathlib.Path(input_path)
    if p.is_dir():
        return p / "schematic_ir.json"
    return p
# ===== [/NEW] =====


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
        file_path: IR JSON 文件路径，或包含 schematic_ir.json 的目录路径
        strict_mode: 严格模式
            - False: intent/context 缺失仅警告，不阻断
            - True: intent/context 缺失报错（用于 Golden Case 验收）
    """
    # ===== [NEW] V1.3：支持目录输入 =====
    # [原逻辑 - 保留注释，便于对照回滚]
    # fp = pathlib.Path(file_path)
    fp = _resolve_ir_path(file_path)
    # ===== [/NEW] =====

    try:
        doc = load_ir(fp)
        result = IRSchemaValidator.validate(doc, strict_mode=strict_mode)

        mode_text = "🔒 严格模式 (Golden Case 验收)" if strict_mode else "📝 普通模式"
        print(f"\n📋 验证文件: {fp}")  # [NEW] 打印实际解析到的文件路径
        print(f"模式: {mode_text}")
        print(f"状态: {'✅ PASS' if result.is_valid else '❌ FAIL'}")

        # ===== 摘要信息 =====
        print("\n📊 摘要:")
        print(f"  - Schema版本: {doc.ir_schema_version}")
        print(f"  - Case ID: {doc.case_id}")
        print(f"  - 元件数: {result.summary.total_components}")
        print(f"  - 网络数: {result.summary.total_nets}")
        print(f"  - 引脚数: {result.summary.total_pins}")
        print(f"  - 悬浮元件: {len(result.summary.floating_components)}")

        # ===== 三语义字段覆盖率 (Sprint 0 验收观测) =====
        total_components = result.summary.total_components
        if total_components > 0:
            has_intent = sum(1 for c in doc.components if c.intent)
            has_context = sum(1 for c in doc.components if c.context)
            has_constraint = sum(1 for c in doc.components if c.constraint)
            print("\n📝 三语义字段覆盖率 (Sprint 0 验收观测):")
            print(f"  - intent: {has_intent}/{total_components} ({has_intent/total_components*100:.1f}%)")
            print(f"  - context: {has_context}/{total_components} ({has_context/total_components*100:.1f}%)")
            print(f"  - constraint: {has_constraint}/{total_components} ({has_constraint/total_components*100:.1f}%)")

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
    # ===== [NEW] 调用函数获取文档 =====
    ir_doc = get_8051_example()
    # ===== [/NEW] =====
    dump_ir(ir_doc, out_file)

    # 自动验证生成的示例（普通模式）
    result = IRSchemaValidator.validate(ir_doc, strict_mode=False)

    print("\n📊 生成的示例统计:")
    print(f"  - 元件数: {result.summary.total_components}")
    print(f"  - 引脚数: {result.summary.total_pins}")
    print(f"  - 网络数: {result.summary.total_nets}")

    total = result.summary.total_components
    if total > 0:
        has_intent = sum(1 for c in ir_doc.components if c.intent)
        intent_rate = has_intent / total * 100
        print(f"  - 三语义覆盖率: {intent_rate:.1f}%")

    if result.is_valid:
        print("\n✅ 示例 IR 验证通过")
    else:
        print("\n⚠️ 示例 IR 存在验证问题:")
        for err in result.errors:
            print(f"  - {err}")


def main():
    """CLI 入口 (豆包增强版)"""
    parser = argparse.ArgumentParser(
        description="Schematic IR Schema 工具 (Sprint 0 Phase D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 校验 IR 目录或文件（普通模式，intent/context 缺失仅警告）
  python -m app.ir.serializer --validate /data/cases/case001

  # 校验 IR 目录或文件（严格模式，用于 Golden Case 验收）
  python -m app.ir.serializer --validate /data/cases/case001 --strict

  # 生成 case001 示例
  python -m app.ir.serializer --generate-case001

  # 生成到指定目录
  python -m app.ir.serializer --generate-case001 --out-dir /data/cases/case002
        """
    )

    parser.add_argument(
        "--validate",
        type=str,
        help="校验 IR 路径（目录或文件）；目录时自动查找 schematic_ir.json"
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
        # ===== [NEW] V1.3：默认目录改为容器内 /data 挂载点 =====
        # [原逻辑 - 保留注释，便于对照回滚]
        # default="./data/cases/case001",
        default="/data/cases/case001",
        # ===== [/NEW] =====
        help="输出目录 (默认: /data/cases/case001)"
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

"""
IR 持久化服务：写入/读取数据库 ir_document 表
对应 ORM 模型 IRDocument (Phase C)

写入路径：
  Pydantic 对象 (SchematicIRDocument)
       │
       │ ir_doc.model_dump(mode="json")
       ▼
  Python dict (JSON 兼容)
       │
       │ IRDocument(ir_json=...)
       ▼
  SQLAlchemy 对象 (未入库)
       │
       │ session.add + flush/commit
       ▼
  数据库 ir_document 表 (JSONB 列)

读取路径：
  数据库 ir_document 表 (JSONB 列)
       │
       │ session.get / query
       ▼
  SQLAlchemy 对象 (IRDocument)
       │
       │ rec.ir_json  ← SQLAlchemy 自动 json.loads
       ▼
  Python dict
       │
       │ SchematicIRDocument.model_validate(...)
       ▼
  Pydantic 对象 (SchematicIRDocument)
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.ir.schema import SchematicIRDocument
from app.persistence.models import IRDocument, SchematicCase

logger = logging.getLogger(__name__)


def store_schematic_ir(
    session: Session,
    case_id: str,
    ir_doc: SchematicIRDocument,
    schematic_case_id: Optional[int] = None,
    auto_commit: bool = True,
) -> IRDocument:
    """
    保存 SchematicIRDocument 对象存入 ir_document 表 (豆包)

    Args:
        session: 数据库会话
        case_id: 用例 ID（业务字符串，如 'case001'）。
                 V1.3 起：若 schematic_case_id 未提供，会用 case_id 查询
                 schematic_case 表尝试补全关联；查不到时仅 warning，不阻断。
        ir_doc: IR 文档对象
        schematic_case_id: 关联的 schematic_case 表 ID (可选)
        auto_commit: 是否自动提交事务
            - True(默认生产环境): 自动提交事务
            - False(测试环境): 只flush,调用方需手动提交

    Returns:
        保存后的 IRDocument 对象

    V1.3 关联策略（P0-4 选项 B）：
        1. 若显式传入 schematic_case_id，直接使用；
        2. 否则，用 case_id 查 schematic_case 表（取最新一条）；
        3. 查不到时：logger.warning，schematic_case_id 置 None（保持可写）；
           ⚠️ 若希望"找不到 case 就 raise"，改为 raise RuntimeError 即可。
    """
    # ===== [NEW] V1.3：case_id 参数联动补全 schematic_case_id =====
    # [原逻辑 - 保留注释，便于对照回滚]
    # db_obj = IRDocument(
    #     schematic_case_id=schematic_case_id,
    #     ir_json=ir_doc.model_dump(mode="json"),
    #     ir_schema_version=ir_doc.ir_schema_version
    # )
    if schematic_case_id is None and case_id:
        row = (
            session.query(SchematicCase)
            .filter(SchematicCase.case_id == case_id)
            .order_by(SchematicCase.id.desc())
            .first()
        )
        if row is not None:
            schematic_case_id = row.id
        else:
            logger.warning(
                "store_schematic_ir: case_id=%r 在 schematic_case 表中未找到，"
                "ir_document.schematic_case_id 将保持 NULL。",
                case_id,
            )

    db_obj = IRDocument(
        schematic_case_id=schematic_case_id,
        ir_json=ir_doc.model_dump(mode="json"),
        ir_schema_version=ir_doc.ir_schema_version,
    )
    # ===== [/NEW] =====
    session.add(db_obj)
    if auto_commit:
        session.commit()
        session.refresh(db_obj)
    else:
        session.flush()
    return db_obj


def load_schematic_ir(session: Session, ir_doc_id: int) -> SchematicIRDocument:
    """
    从数据库读取 ir_document 记录，解析为 SchematicIRDocument 对象 (豆包)

    Args:
        session: 数据库会话
        ir_doc_id: IR 文档 ID

    Returns:
        SchematicIRDocument 对象

    Raises:
        ValueError: 当 IR 文档不存在时
    """
    rec = session.get(IRDocument, ir_doc_id)
    if rec is None:
        raise ValueError(f"ir_document id={ir_doc_id} not found")
    return SchematicIRDocument.model_validate(rec.ir_json)


def load_schematic_ir_by_case(session: Session, case_id: str) -> Optional[SchematicIRDocument]:
    """
    根据 case_id 加载最新的 IR 文档

    排序策略：按 id 降序(id 自增保证写入顺序)
    不使用 created_at 排序，因为同一事务内 NOW() 返回事务开始时间

    注意：此函数使用 JSONB 查询，仅在 PostgreSQL 中可用。
    SQLite 内存测试中此函数不会被测试。

    Args:
        session: 数据库会话
        case_id: 用例 ID

    Returns:
        SchematicIRDocument 对象，不存在时返回 None
    """
    rec = session.query(IRDocument).filter(
        IRDocument.ir_json['case_id'].astext == case_id
    ).order_by(
        IRDocument.id.desc()
    ).first()

    if rec is None:
        return None
    return SchematicIRDocument.model_validate(rec.ir_json)

# scripts/init_db.py
"""
数据库初始化脚本
⚠️ 重要约束：本脚本仅用于【全新空数据库】初始化，**不能作为迁移脚本**
已有存量数据库禁止重复执行；Sprint‑1引入Alembic处理后续schema变更。
💡 使用提示：
   - 本脚本已支持"数据库已初始化"的友好提示（不会输出错误堆栈）
   - 检测到对象已存在时，会跳过 DDL 执行，直接进行 Schema 校验
   - 如需彻底重建：请先清空数据库（见 README 或 CONTRIBUTING）
Schema唯一事实源：infra/docker/initdb/002_schema.sql
Sprint‑0：建表review_defect，业务层实现JSON解析写入；Sprint‑0不使用该表做查询，Sprint‑1全面启用。
"""
import logging
import pathlib
from sqlalchemy import create_engine, text
from app.core.config import settings
#增加提示导入
from psycopg.errors import DuplicateObject, DuplicateTable
from sqlalchemy.exc import ProgrammingError

logger = logging.getLogger(__name__)

def get_engine():
    return create_engine(settings.database_url, echo=settings.database_echo)

'''
def run_init_sql(engine, sql_file: pathlib.Path):
    """执行单个初始化sql文件"""
    logger.info(f"执行初始化SQL: {sql_file.name}")
    sql_text = sql_file.read_text(encoding="utf-8")
    with engine.connect() as conn:
        conn.execute(text(sql_text))
        conn.commit()
    logger.info(f"✅ {sql_file.name} 执行完成")
'''

def run_init_sql(engine, sql_file: pathlib.Path) -> bool:
    """
    执行单个初始化 SQL 文件
    
    Returns:
        True: 本次执行成功完成了所有 SQL
        False: 检测到数据库已初始化（对象已存在），跳过执行
    """
    logger.info(f"执行初始化SQL: {sql_file.name}")
    sql_text = sql_file.read_text(encoding="utf-8")
    
    try:
        with engine.connect() as conn:
            conn.execute(text(sql_text))
            conn.commit()
        logger.info(f"✅ {sql_file.name} 执行完成")
        return True
    except ProgrammingError as e:
        # 检查是否为"对象已存在"类错误（幂等冲突）
        orig = getattr(e, "orig", None)
        if orig is not None and isinstance(orig, (DuplicateObject, DuplicateTable)):
            # ⚠️ V1.3 修改：明确"整文件跳过"语义 + 指出单表丢失无法重跑补建，
            # 并将重建步骤从 dropdb/createdb 改为 down -v/up（后者才会触发 initdb 全量执行）。
            # 根因：002_schema.sql 为整体脚本，遇首个 DuplicateObject 即整体跳过，
            #      不会继续执行后续 CREATE TABLE IF NOT EXISTS。
            # 详见方案 §19.7。
            # [原逻辑 - 保留注释，便于对照回滚]
            # logger.warning(
            #     f"⚠️ {sql_file.name} 检测到数据库已初始化"
            #     f"（对象已存在: {orig}），跳过执行。"
            #     f"\n    💡 如需重新初始化，请先清空数据库："
            #     f"\n       1. 停 backend: docker compose ... stop backend"
            #     f"\n       2. 删库重建: docker compose ... exec postgres dropdb -U postgres --if-exists schematic_review"
            #     f"\n                   docker compose ... exec postgres createdb -U postgres schematic_review"
            #     f"\n       3. 重新启动: docker compose ... start backend"
            # )
            logger.warning(
                f"⚠️ {sql_file.name} 检测到数据库已初始化"
                f"（对象已存在: {orig}），"
                f"**已跳过整个 {sql_file.name} 文件**的后续所有语句。"
                f"\n    ⚠️ 注意：本脚本为整体执行，遇到首个对象冲突即整体跳过，"
                f"不会继续执行后续 CREATE TABLE IF NOT EXISTS。"
                f"\n    因此，**单张表丢失时无法通过重跑本脚本补建**，"
                f"必须按下方步骤彻底重建（详见方案 §19.7）。"
                f"\n    💡 彻底重建命令（完整命令见方案 §16）："
                f"\n       1. 停服务 + 删 volume（触发 initdb 全量执行）:"
                f"\n          docker compose -f '$PWD/infra/docker/docker-compose.yml' -f '$PWD/infra/docker/docker-compose.dev.yml' down -v"
                f"\n       2. 重启（postgres 首次初始化 volume 时自动执行 initdb）:"
                f"\n          docker compose -f '$PWD/infra/docker/docker-compose.yml' -f '$PWD/infra/docker/docker-compose.dev.yml' up -d --build"
                f"\n       3. 校验（可选，也可由 postgres 容器自动完成）:"
                f"\n          docker compose -f '$PWD/infra/docker/docker-compose.yml' -f '$PWD/infra/docker/docker-compose.dev.yml' exec backend uv run python -m scripts.init_db"
            )
            return False
        # 其他 ProgrammingError 继续抛出
        raise

'''
def init_database():
    engine = get_engine()
    init_dir = pathlib.Path("/app/infra/docker/initdb")

    # ========== 执行所有SQL文件 ==========
    sql_files = sorted([f for f in init_dir.glob("*.sql") if f.is_file()])
    logger.info(f"发现 {len(sql_files)} 个SQL初始化文件")
    for f in sql_files:
        run_init_sql(engine, f)
'''

def init_database():
    engine = get_engine()
    init_dir = pathlib.Path("/infra/docker/initdb")

    # ========== 执行所有SQL文件 ==========
    sql_files = sorted([f for f in init_dir.glob("*.sql") if f.is_file()])
    logger.info(f"发现 {len(sql_files)} 个SQL初始化文件")
    
    all_skipped = True
    for f in sql_files:
        executed = run_init_sql(engine, f)
        if executed:
            all_skipped = False

    if all_skipped:
        logger.info(
            "ℹ️ 所有 SQL 文件均已检测到数据库已初始化，跳过 DDL 执行。"
            "下面进入 Schema 校验环节..."
        )
  
    # ========== 校验环节 ==========
    with engine.connect() as conn:
        # 1. pgvector扩展校验
        ext = conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).fetchone()
        if not ext:
            raise RuntimeError("pgvector extension not enabled!")
        logger.info("✅ pgvector extension active")

        # 2. P0必选10张业务表
        #原代码p0_tables = {'users', 'project', 'schematic_case', 'ir_document',
        #             'rule_definition', 'rule_execution', 'review_result', 'feedback_item',
        #             'knowledge_doc', 'knowledge_chunk'}
        p0_tables = [
            'users', 'project', 'schematic_case', 'ir_document',
            'rule_definition', 'rule_execution', 'review_result', 'feedback_item',
            'knowledge_doc', 'knowledge_chunk'
        ]
        #原代码 rows = conn.execute(text("""
        #SELECT table_name FROM information_schema.tables
        #WHERE table_schema='public' AND table_name IN :table_list
        #"""), {"table_list": tuple(p0_tables)}).fetchall()
        rows = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public' AND table_name = ANY(:table_list)
        """), {"table_list": p0_tables}).fetchall()
        found_p0 = {r.table_name for r in rows}
        #原代码 missing = p0_tables - found_p0
        missing = set(p0_tables) - found_p0
        if missing:
            raise RuntimeError(f"缺少P0必选业务表: {missing}")
        logger.info(f"✅ P0 Database schema v1.2‑enhance ok，共 {len(found_p0)} 张表")

        # 3. 校验 review_defect（新增表，Sprint‑0必须建表）
        defect_table = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public' AND table_name='review_defect'
        """)).fetchone()
        if defect_table:
            logger.info("✅ review_defect 缺陷明细表已创建（支持缺陷粒度追踪）")
        else:
            raise RuntimeError("❌ review_defect 表缺失！请检查 002_schema.sql 是否包含 review_defect 定义。")

        # 4. 校验 review_defect 关键字段
        defect_cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='review_defect'
            AND column_name IN ('defect_id', 'category', 'risk', 'review_status', 'origin')
        """)).fetchall()
        found_defect_cols = {c.column_name for c in defect_cols}
        required_defect_cols = {"defect_id", "category", "risk", "review_status", "origin"}
        if not required_defect_cols.issubset(found_defect_cols):
            raise RuntimeError(f"review_defect 缺失关键字段: {required_defect_cols - found_defect_cols}")
        logger.info("✅ review_defect V1.2 关键字段校验通过")

        # 5. 校验 feedback_item 是否关联 review_defect
        feedback_cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='feedback_item' AND column_name='review_defect_id'
        """)).fetchone()
        if feedback_cols:
            logger.info("✅ feedback_item 已关联 review_defect_id（支持缺陷级反馈）")
        else:
            # Sprint‑0 schema必选字段，缺失直接阻断初始化，不允许仅告警
            raise RuntimeError("❌ feedback_item 缺少 review_defect_id 字段，无法精准关联到缺陷")

        # 6. 校验 review_result V1.2关键字段
        col_rows = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='review_result'
            AND column_name IN ('category', 'risk', 'review_status')
        """)).fetchall()
        found_cols = {c.column_name for c in col_rows}
        required_cols = {"category", "risk", "review_status"}
        if not required_cols.issubset(found_cols):
            raise RuntimeError(f"review_result 缺失V1.2关键字段，缺失：{required_cols - found_cols}")
        logger.info("✅ review_result V1.2关键字段校验通过")

        # 7. feedback_item 6类CHECK约束校验
        check_constraint = conn.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name='feedback_item' AND constraint_type='CHECK'
            AND constraint_name='check_feedback_item_type'
        """)).fetchone()
        if check_constraint:
            logger.info("✅ feedback_item 6类feedback_type CHECK约束已生效")
        else:
            raise RuntimeError("feedback_item 缺少 check_feedback_item_type 约束！")

        # 8. rule_candidates P1表校验
        # ⚠️ V1.3 修改：由 warning 升为 raise。
        # 原因：Sprint0 Phase‑I 反馈闭环依赖 rule_candidates 表，
        #       Sprint0 验收项 9（Feedback 6 Categories）需要该表存在。
        #       P0-1 决策：缺失即阻断，避免 Sprint1 第一步写入时炸。
        # [原逻辑 - 保留注释，便于对照回滚]
        # # 8. rule_candidates P1表仅告警，不阻断
        # p1_rows = conn.execute(text("""
        #     SELECT table_name FROM information_schema.tables
        #     WHERE table_schema='public' AND table_name='rule_candidates'
        # """)).fetchone()
        # if p1_rows:
        #     logger.info("✅ P1表 rule_candidates 已存在（Sprint‑1业务使用）")
        # else:
        #     logger.warning("⚠️ P1 rule_candidates 缺失，Sprint‑0不阻断，Sprint‑1需要。")
        p1_rows = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public' AND table_name='rule_candidates'
        """)).fetchone()
        if p1_rows:
            logger.info("✅ P1表 rule_candidates 已存在（Phase‑I 反馈闭环依赖）")
        else:
            raise RuntimeError(
                "❌ rule_candidates 表缺失！Sprint0 Phase‑I 反馈闭环依赖该表，"
                "请检查 002_schema.sql 是否包含 rule_candidates 定义。"
            )

    logger.info("🎉 数据库初始化全部校验完成")

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=settings.log_level.upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    init_database()
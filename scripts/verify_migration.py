"""在临时库上验证 Alembic upgrade/downgrade 往返，不动开发库。

用法（在项目根）：
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend python /scripts/verify_migration.py --check-table rule_definition_history

退出码：0 成功 / 1 验证失败 / 2 环境异常。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from urllib.parse import urlparse, urlunparse


def _tmp_db_url(base_url: str, tmp_db: str) -> str:
    parsed = urlparse(base_url)
    return urlunparse(parsed._replace(path=f"/{tmp_db}"))


def _run(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _pg_exec(database: str, sql: str, base_url: str) -> tuple[int, str]:
    """psycopg 直连执行一条 SQL（不依赖 psql 客户端）。"""
    import psycopg

    parsed = urlparse(base_url)
    dsn = (
        f"host={parsed.hostname or 'postgres'} "
        f"port={parsed.port or 5432} "
        f"user={parsed.username or 'postgres'} "
        f"password={parsed.password or ''} "
        f"dbname={database}"
    )
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                try:
                    rows = cur.fetchall()
                except psycopg.ProgrammingError:
                    rows = []
        return 0, "\n".join(str(r) for r in rows)
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _table_exists(database: str, table: str, base_url: str) -> bool:
    sql = (
        "SELECT 1 FROM information_schema.tables "
        f"WHERE table_name = '{table}' LIMIT 1"
    )
    code, out = _pg_exec(database, sql, base_url)
    return code == 0 and out.strip() != ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-table", default="")
    parser.add_argument("--tmp-db", default="schematic_review_tmp")
    args = parser.parse_args()

    base_url = os.environ.get("DATABASE_URL", "")
    if not base_url:
        print("[FAIL] DATABASE_URL not set (run inside backend container)")
        return 2

    tmp_url = _tmp_db_url(base_url, args.tmp_db)
    env = {**os.environ, "DATABASE_URL": tmp_url}

    # 0. 清残留（允许不存在）
    _pg_exec("postgres", f"DROP DATABASE IF EXISTS {args.tmp_db};", base_url)

    # 1. 建临时库
    print(f"[1/7] create tmp db: {args.tmp_db}")
    code, out = _pg_exec("postgres", f"CREATE DATABASE {args.tmp_db};", base_url)
    if code != 0:
        print(f"[FAIL] create database: {out}")
        return 1

    # 2. upgrade head
    print("[2/7] upgrade head")
    code, out = _run(["alembic", "upgrade", "head"], env)
    print(out)
    if code != 0:
        return 1

    # 3. current
    print("[3/7] alembic current")
    code, out = _run(["alembic", "current"], env)
    print(out)
    if code != 0 or "(head)" not in out:
        print("[FAIL] not at head after upgrade")
        return 1

    # 4. 表应存在
    if args.check_table:
        print(f"[4/7] check table exists: {args.check_table}")
        if not _table_exists(args.tmp_db, args.check_table, base_url):
            print(f"[FAIL] table missing after upgrade: {args.check_table}")
            return 1
    else:
        print("[4/7] skip table check")

    # 5. downgrade -1
    print("[5/7] downgrade -1")
    code, out = _run(["alembic", "downgrade", "-1"], env)
    print(out)
    if code != 0:
        return 1

    # 6. 表应不存在
    if args.check_table:
        print(f"[6/7] check table dropped: {args.check_table}")
        if _table_exists(args.tmp_db, args.check_table, base_url):
            print(f"[FAIL] table still exists after downgrade: {args.check_table}")
            return 1
    else:
        print("[6/7] skip table check")

    # 7. 再 upgrade
    print("[7/7] upgrade head again")
    code, out = _run(["alembic", "upgrade", "head"], env)
    print(out)
    if code != 0:
        return 1

    # cleanup
    print(f"[cleanup] drop tmp db: {args.tmp_db}")
    _pg_exec("postgres", f"DROP DATABASE {args.tmp_db};", base_url)

    print("[OK] migration round-trip verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
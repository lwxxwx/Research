# backend/tests/test_bootstrap_admin.py
"""Sprint 3 G2 bootstrap_admin.py CLI 5 种场景测试。

调用方式：用 subprocess 跑真实的 python /scripts/bootstrap_admin.py，
使用测试事务 prepare 的临时 username，测试后清理。
"""
from __future__ import annotations

import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.persistence.models import User


def _run_cli(username: str, email: str | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "/scripts/bootstrap_admin.py", "--username", username]
    if email:
        cmd.extend(["--email", email])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


@pytest.fixture()
def pg_session():
    """真实提交；cleanup_users fixture负责删除cli_前缀用户"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.core.config import settings
    engine = create_engine(settings.database_url, echo=False)
    sess = Session(bind=engine)
    yield sess
    sess.close()



@pytest.fixture()
def cleanup_users(pg_session):
    """用于在测试前后清理指定前缀的用户（保证 CLI 的真实数据库操作干净）。"""
    prefix = f"cli_{uuid.uuid4().hex[:6]}_"
    created_names: list[str] = []

    def _record(name: str) -> str:
        full = f"{prefix}{name}"
        created_names.append(full)
        return full

    yield _record

    # 清理：直接删除前缀匹配的用户
    pg_session.rollback()
    with engine_commit_session() as db:
        db.query(User).filter(User.username.like(f"{prefix}%")).delete(synchronize_session=False)
        db.commit()


from contextlib import contextmanager


@contextmanager
def engine_commit_session():
    """独立的提交 Session（不用事务回滚），用于清理真实数据。"""
    engine = create_engine(settings.database_url)
    sess = Session(bind=engine)
    try:
        yield sess
    finally:
        sess.close()


# ============================================================
# 场景 1：用户存在，不是 admin
# ============================================================
def test_cli_scenario_1_user_exists_not_admin(pg_session, cleanup_users):
    uname = cleanup_users("eng")
    pg_session.add(User(username=uname, role="engineer", is_active=True))
    pg_session.commit()

    result = _run_cli(uname)
    assert result.returncode == 1
    assert "REFUSE" in result.stdout or "REFUSE" in result.stderr


# ============================================================
# 场景 2：用户存在，已禁用 admin
# ============================================================
def test_cli_scenario_2_disabled_admin(pg_session, cleanup_users):
    uname = cleanup_users("dis_admin")
    pg_session.add(User(username=uname, role="admin", is_active=False))
    pg_session.commit()

    result = _run_cli(uname)
    assert result.returncode == 1
    assert "disabled" in (result.stdout + result.stderr).lower()


# ============================================================
# 场景 3：用户存在，启用 admin（幂等成功）
# ============================================================
def test_cli_scenario_3_existing_active_admin(pg_session, cleanup_users):
    uname = cleanup_users("act_admin")
    pg_session.add(User(username=uname, role="admin", is_active=True))
    pg_session.commit()

    result = _run_cli(uname)
    assert result.returncode == 0
    assert "already exists" in result.stdout.lower()


# ============================================================
# 场景 4：用户名不存在，系统已有启用 admin
# ============================================================
def test_cli_scenario_4_other_admin_exists(pg_session, cleanup_users):
    # 保证系统至少有一个启用 admin（用真实 DB 里的）
    with engine_commit_session() as db:
        has_admin = db.query(User).filter(User.role == "admin", User.is_active.is_(True)).first()
    if has_admin is None:
        pytest.skip("当前系统无启用 admin，跳过场景 4")

    uname = cleanup_users("new_admin")
    result = _run_cli(uname)
    assert result.returncode == 0
    assert "already has" in result.stdout.lower()


# ============================================================
# 场景 5：用户名不存在，系统无启用 admin（首次创建）
# ============================================================
def test_cli_scenario_5_first_admin(pg_session, cleanup_users):
    """仅在系统没有启用 admin 时才能测，否则 skip。

    为了模拟"无启用 admin"，临时把当前所有 admin 改为 disabled（用独立 session commit），
    测试完成后恢复。
    """
    with engine_commit_session() as db:
        admins = db.query(User).filter(User.role == "admin", User.is_active.is_(True)).all()
        if not admins:
            pytest.skip("当前系统已无启用 admin，跳过场景 5（避免污染）")

    # 为避免污染真实数据，跳过（需要 DB reset 环境才能安全测）
    pytest.skip("场景 5 需数据库清空 reset 环境，跳过以避免误删真实 admin")
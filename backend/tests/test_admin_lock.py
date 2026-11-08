# backend/tests/test_admin_lock.py
"""Sprint 3 G2 管理员锁 A/C 类并发测试。

A 类：正常并发与锁串行化
C 类：锁机制真实性（必须用 PostgreSQL 验证）

B 类（唯一约束冲突恢复）延后到 G6。
"""
from __future__ import annotations

import threading
import time
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.persistence.models import User
from app.services.admin_lock import (
    ADMIN_LOCK_KEY,
    AdminLockTimeoutError,
    acquire_admin_lock,
)


# ============================================================
# A 类：锁串行化
# ============================================================
def test_A1_acquire_lock_twice_no_timeout():
    """A1：同一线程连续 acquire 两次不报错（advisory lock 可重入）。"""
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            from sqlalchemy.orm import Session
            sess = Session(bind=conn)
            acquire_admin_lock(sess)
            acquire_admin_lock(sess)  # 同事务内多次获取是允许的
        finally:
            trans.rollback()


def test_A2_lock_released_after_commit():
    """A2：事务提交后锁自动释放，另一个 Session 可以立即获取。"""
    engine = create_engine(settings.database_url)

    # 连接 1：获取锁后 commit
    with engine.connect() as conn1:
        t1 = conn1.begin()
        sess1 = Session(bind=conn1)
        acquire_admin_lock(sess1)
        t1.commit()

    # 连接 2：短时间内立刻获取，不应超时
    with engine.connect() as conn2:
        t2 = conn2.begin()
        sess2 = Session(bind=conn2)
        acquire_admin_lock(sess2, timeout_ms=2000)
        t2.rollback()


def test_A3_lock_released_after_rollback():
    """A3：事务回滚后锁自动释放。"""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn1:
        t1 = conn1.begin()
        sess1 = Session(bind=conn1)
        acquire_admin_lock(sess1)
        t1.rollback()

    with engine.connect() as conn2:
        t2 = conn2.begin()
        sess2 = Session(bind=conn2)
        acquire_admin_lock(sess2, timeout_ms=2000)
        t2.rollback()


def test_A4_concurrent_acquire_serialized():
    """A4：两个线程并发 acquire，锁串行化后第二个才成功。"""
    engine = create_engine(settings.database_url)
    results: list[str] = []
    barrier = threading.Barrier(2)
    hold_seconds = 1.0

    def worker(name: str, sleep_before: float) -> None:
        time.sleep(sleep_before)
        barrier.wait()
        try:
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    sess = Session(bind=conn)
                    acquire_admin_lock(sess, timeout_ms=8000)
                    results.append(f"{name}:acquired")
                    if name == "A":
                        time.sleep(hold_seconds)
                    results.append(f"{name}:released")
                finally:
                    trans.rollback()
        except AdminLockTimeoutError:
            results.append(f"{name}:timeout")

    t1 = threading.Thread(target=worker, args=("A", 0.0))
    t2 = threading.Thread(target=worker, args=("B", 0.0))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert "A:acquired" in results
    assert "B:acquired" in results
    assert "A:timeout" not in results
    assert "B:timeout" not in results

    idx_a_acq = results.index("A:acquired")
    idx_a_rel = results.index("A:released")
    idx_b_acq = results.index("B:acquired")

    # 两种合法情况：A先拿锁；或者B先拿到锁（操作系统调度随机性）
    if idx_a_acq < idx_b_acq:
        # A先拿到，则必须A释放之后B才能acquire
        assert idx_a_rel < idx_b_acq
    else:
        # B先拿到锁，则B释放后A才能acquire
        idx_b_rel = results.index("B:released")
        assert idx_b_rel < idx_a_acq


# ============================================================
# C 类：锁机制真实性（必须 PostgreSQL）
# ============================================================
def test_C1_lock_key_is_pg_advisory():
    """C1：确认锁 key 是整数，用于 pg_advisory_xact_lock。"""
    assert isinstance(ADMIN_LOCK_KEY, int)
    assert 0 <= ADMIN_LOCK_KEY < 2**63


def test_C2_verify_pg_advisory_xact_lock_exists():
    """C2：确认当前数据库支持 pg_advisory_xact_lock。"""
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        from sqlalchemy import text
        row = conn.execute(
            text("SELECT COUNT(*) FROM pg_proc WHERE proname = 'pg_advisory_xact_lock'")
        ).scalar()
        assert row >= 1

# ============================================================
# B 类：唯一约束冲突恢复（G6 补齐，Sprint3 §2.4）
# ============================================================
# B 类测试策略：
#   - 用 monkeypatch 让 Session.commit 在第 1 次抛 IntegrityError，触发重试路径
#   - 验证重试后按场景判断（不伪装成功，也不死循环）
# B 类测试的公共 import
from app.persistence.db import get_db_session

def _run_bootstrap_cli(username: str) -> int:
    """直接调 bootstrap_admin.main()，返回退出码（不走 subprocess，便于 monkeypatch）。"""
    import sys
    from scripts import bootstrap_admin

    orig_argv = sys.argv
    sys.argv = ["bootstrap_admin.py", "--username", username]
    try:
        return bootstrap_admin.main()
    finally:
        sys.argv = orig_argv


def _cleanup_user(username: str) -> None:
    """清理测试用户（按 username 精确删除）。"""
    from app.persistence.db import get_db_session
    with get_db_session() as db:
        db.query(User).filter(User.username == username).delete(synchronize_session=False)
        db.commit()


def test_B1_integrity_conflict_retry_then_scene4(monkeypatch):
    """B1：第 1 次 commit 抛 IntegrityError → 重试 → 场景 4（已有 admin）→ 返回 0。

    验证：唯一约束冲突后回滚 + 重试 + 重读，按场景判断。
    """
    import uuid as _uuid
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session as ORMSession

    # 1. 造一个活跃 admin
    uname_admin = f"b1_admin_{_uuid.uuid4().hex[:8]}"
    from app.persistence.db import get_db_session
    with get_db_session() as db:
        db.add(User(username=uname_admin, role="admin", is_active=True))
        db.commit()

    # 2. 让第 1 次 commit 抛 IntegrityError
    orig_commit = ORMSession.commit
    call_counter = {"n": 0}

    def _mock_commit(self, *args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise IntegrityError("simulated unique violation", None, Exception("conflict"))
        return orig_commit(self, *args, **kwargs)

    monkeypatch.setattr(ORMSession, "commit", _mock_commit)

    # 3. 跑 CLI（新用户名，重试后应命中场景 4）
    new_uname = f"b1_new_{_uuid.uuid4().hex[:8]}"
    try:
        exit_code = _run_bootstrap_cli(new_uname)
        # 第 1 次 commit 冲突 → 重试 → 第 2 次发现已有 admin → 场景 4 → 0
        assert exit_code == 0
        # 确认没创建新用户
        with get_db_session() as db:
            assert db.query(User).filter(User.username == new_uname).first() is None
    finally:
        monkeypatch.setattr(ORMSession, "commit", orig_commit)
        _cleanup_user(uname_admin)
        _cleanup_user(new_uname)


def test_B2_integrity_conflict_retry_then_scene1(monkeypatch):
    """B2：第 1 次 commit 抛 IntegrityError → 重试 → 场景 1（用户名是 engineer）→ 返回 1。

    验证：冲突后重试，发现用户名已存在且不是 admin，按场景 1 拒绝。
    """
    import uuid as _uuid
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session as ORMSession

    # 1. 造一个 engineer 用户名
    uname_eng = f"b2_eng_{_uuid.uuid4().hex[:8]}"
    from app.persistence.db import get_db_session
    with get_db_session() as db:
        db.add(User(username=uname_eng, role="engineer", is_active=True))
        db.commit()

    # 2. 让第 1 次 commit 抛 IntegrityError
    orig_commit = ORMSession.commit
    call_counter = {"n": 0}

    def _mock_commit(self, *args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise IntegrityError("simulated unique violation", None, Exception("conflict"))
        return orig_commit(self, *args, **kwargs)

    monkeypatch.setattr(ORMSession, "commit", _mock_commit)

    # 3. 跑 CLI（用 engineer 用户名，重试后应命中场景 1）
    try:
        exit_code = _run_bootstrap_cli(uname_eng)
        assert exit_code == 1
    finally:
        monkeypatch.setattr(ORMSession, "commit", orig_commit)
        _cleanup_user(uname_eng)


def test_B3_integrity_conflict_retry_then_scene2(monkeypatch):
    """B3：第 1 次 commit 抛 IntegrityError → 重试 → 场景 2（禁用 admin）→ 返回 1。

    验证：冲突后重试，发现用户名是已禁用 admin，按场景 2 拒绝。
    """
    import uuid as _uuid
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session as ORMSession

    # 1. 造一个禁用的 admin
    uname_dis = f"b3_dis_{_uuid.uuid4().hex[:8]}"
    from app.persistence.db import get_db_session
    with get_db_session() as db:
        db.add(User(username=uname_dis, role="admin", is_active=False))
        db.commit()

    # 2. 让第 1 次 commit 抛 IntegrityError
    orig_commit = ORMSession.commit
    call_counter = {"n": 0}

    def _mock_commit(self, *args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise IntegrityError("simulated unique violation", None, Exception("conflict"))
        return orig_commit(self, *args, **kwargs)

    monkeypatch.setattr(ORMSession, "commit", _mock_commit)

    # 3. 跑 CLI（用禁用 admin 用户名，重试后应命中场景 2）
    try:
        exit_code = _run_bootstrap_cli(uname_dis)
        assert exit_code == 1
    finally:
        monkeypatch.setattr(ORMSession, "commit", orig_commit)
        _cleanup_user(uname_dis)


def test_B4_all_retries_exhausted_returns_2(monkeypatch):
    """B4：场景 5（创建初始 admin）时每次 commit 都抛 IntegrityError → 重试耗尽 → 返回 2。"""
    import uuid as _uuid
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session as ORMSession
    from app.persistence.db import get_db_session

    # 1. 临时禁用所有活跃 admin（finally 恢复）
    with get_db_session() as db:
        active_admin_ids = [u.id for u in db.query(User).filter(User.role == "admin", User.is_active.is_(True)).all()]
        if active_admin_ids:
            db.query(User).filter(User.id.in_(active_admin_ids)).update(
                {User.is_active: False}, synchronize_session=False,
            )
            db.commit()

    orig_commit = ORMSession.commit

    def _always_fail_commit(self, *args, **kwargs):
        raise IntegrityError("simulated unique violation", None, Exception("conflict"))

    monkeypatch.setattr(ORMSession, "commit", _always_fail_commit)

    new_uname = f"b4_{_uuid.uuid4().hex[:8]}"
    try:
        exit_code = _run_bootstrap_cli(new_uname)
        assert exit_code == 2
    finally:
        # 先恢复 commit 方法，再恢复 admin 状态（否则恢复的 commit 也被拦截）
        monkeypatch.setattr(ORMSession, "commit", orig_commit)
        with get_db_session() as db:
            if active_admin_ids:
                db.query(User).filter(User.id.in_(active_admin_ids)).update(
                    {User.is_active: True}, synchronize_session=False,
                )
                db.commit()

    # 确认没创建用户
    with get_db_session() as db:
        assert db.query(User).filter(User.username == new_uname).first() is None
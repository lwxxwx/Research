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
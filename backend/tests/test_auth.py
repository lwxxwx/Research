# backend/tests/test_auth.py
"""Sprint 3 G2 auth API 测试（真实 PostgreSQL，独立事务回滚）。

关键约束：
  - pg_session 只清理 test_auth_ 前缀的用户，不碰 conftest 的 test_admin_global_ / test_eng_global_
  - 不删任何 user（避免 FK 违反 / 破坏 session 级 fixture），改用"临时降级 + 恢复"
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.persistence.models import User


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def pg_session():
    """真实提交；测试后只清理 test_auth_ 前缀的用户。"""
    from sqlalchemy import create_engine
    from app.core.config import settings
    engine = create_engine(settings.database_url, echo=False)
    sess = Session(bind=engine)
    yield sess
    # 只清理本文件创建的用户，不碰 conftest 的 session 级 fixture
    sess.query(User).filter(
        User.username.like("test_auth_%")
    ).delete(synchronize_session=False)
    sess.commit()
    sess.close()


@pytest.fixture()
def admin_user(pg_session: Session):
    uname = f"test_auth_admin_{uuid.uuid4().hex[:6]}"
    user = User(username=uname, role="admin", is_active=True)
    pg_session.add(user)
    pg_session.flush()
    pg_session.commit()
    return user.id, uname


@pytest.fixture()
def engineer_user(pg_session: Session):
    uname = f"test_auth_eng_{uuid.uuid4().hex[:6]}"
    user = User(username=uname, role="engineer", is_active=True)
    pg_session.add(user)
    pg_session.flush()
    pg_session.commit()
    return user.id, uname


@pytest.fixture()
def disabled_user(pg_session: Session):
    uname = f"test_auth_dis_{uuid.uuid4().hex[:6]}"
    user = User(username=uname, role="engineer", is_active=False)
    pg_session.add(user)
    pg_session.flush()
    pg_session.commit()
    return user.id, uname


# ============================================================
# 登录
# ============================================================
def test_login_ok(client, engineer_user):
    _, uname = engineer_user
    resp = client.post("/api/v1/auth/login", json={"username": uname})
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == uname
    assert data["role"] == "engineer"
    assert "user_id" in data


def test_login_user_not_found(client):
    resp = client.post("/api/v1/auth/login", json={"username": "no_such_user_xyz"})
    assert resp.status_code == 401


def test_login_user_disabled(client, disabled_user):
    _, uname = disabled_user
    resp = client.post("/api/v1/auth/login", json={"username": uname})
    assert resp.status_code == 401


# ============================================================
# /auth/me
# ============================================================
def test_me_missing_header(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_me_invalid_header(client):
    resp = client.get("/api/v1/auth/me", headers={"X-User-Id": "abc"})
    assert resp.status_code == 401


def test_me_not_found(client):
    resp = client.get("/api/v1/auth/me", headers={"X-User-Id": "999999"})
    assert resp.status_code == 401


def test_me_disabled(client, disabled_user):
    uid, _ = disabled_user
    resp = client.get("/api/v1/auth/me", headers={"X-User-Id": str(uid)})
    assert resp.status_code == 401


def test_me_ok(client, engineer_user):
    uid, uname = engineer_user
    resp = client.get("/api/v1/auth/me", headers={"X-User-Id": str(uid)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == uid
    assert data["username"] == uname
    assert data["role"] == "engineer"


# ============================================================
# 用户列表（admin only）
# ============================================================
def test_list_users_admin_ok(client, admin_user):
    uid, _ = admin_user
    resp = client.get("/api/v1/auth/users", headers={"X-User-Id": str(uid)})
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_list_users_engineer_403(client, engineer_user):
    uid, _ = engineer_user
    resp = client.get("/api/v1/auth/users", headers={"X-User-Id": str(uid)})
    assert resp.status_code == 403


# ============================================================
# 创建用户（admin only）
# ============================================================
def test_create_user_admin_ok(client, admin_user):
    uid, _ = admin_user
    uname = f"new_eng_{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/api/v1/auth/users",
        headers={"X-User-Id": str(uid)},
        json={"username": uname, "role": "engineer"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["username"] == uname
    assert data["role"] == "engineer"
    assert data["is_active"] is True


def test_create_user_engineer_403(client, engineer_user):
    uid, _ = engineer_user
    resp = client.post(
        "/api/v1/auth/users",
        headers={"X-User-Id": str(uid)},
        json={"username": "should_fail", "role": "engineer"},
    )
    assert resp.status_code == 403


def test_create_user_duplicate(client, admin_user):
    uid, uname = admin_user
    resp = client.post(
        "/api/v1/auth/users",
        headers={"X-User-Id": str(uid)},
        json={"username": uname, "role": "engineer"},
    )
    assert resp.status_code == 409


# ============================================================
# PATCH / DELETE（admin only）
# ============================================================
def test_patch_role_admin_ok(client, admin_user, engineer_user):
    admin_id, _ = admin_user
    target_id, _ = engineer_user
    resp = client.patch(
        f"/api/v1/auth/users/{target_id}",
        headers={"X-User-Id": str(admin_id)},
        json={"role": "reviewer"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "reviewer"


def test_patch_explicit_null_422(client, admin_user, engineer_user):
    admin_id, _ = admin_user
    target_id, _ = engineer_user
    resp = client.patch(
        f"/api/v1/auth/users/{target_id}",
        headers={"X-User-Id": str(admin_id)},
        json={"role": None},
    )
    assert resp.status_code == 422


def test_delete_soft_disable_ok(client, admin_user, engineer_user):
    admin_id, _ = admin_user
    target_id, _ = engineer_user
    resp = client.delete(
        f"/api/v1/auth/users/{target_id}",
        headers={"X-User-Id": str(admin_id)},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["is_active"] is False


def test_last_admin_protection(client, admin_user, pg_session):
    """唯一的 admin 不能自我降级。

    关键：不能删其它 admin（会触发 FK 违反 / 破坏 conftest 的 session 级 fixture），
    改为临时降级为 engineer，测试结束恢复。
    """
    admin_id, _ = admin_user

    # 1. 其它 active admin 临时降级为 engineer（不删，避免 FK 违反）
    other_admins = pg_session.query(User).filter(
        User.id != admin_id,
        User.role == "admin",
        User.is_active.is_(True),
    ).all()
    original_roles = {u.id: u.role for u in other_admins}
    for u in other_admins:
        u.role = "engineer"
    pg_session.commit()

    try:
        # 2. 此时 admin_id 是唯一 active admin，尝试自我降级
        resp = client.patch(
            f"/api/v1/auth/users/{admin_id}",
            headers={"X-User-Id": str(admin_id)},
            json={"role": "engineer"},
        )
        assert resp.status_code == 422, resp.text
        assert "最后一个启用管理员" in resp.json()["detail"]
    finally:
        # 3. 恢复其它 admin
        for uid, original_role in original_roles.items():
            pg_session.query(User).filter(User.id == uid).update(
                {User.role: original_role}, synchronize_session=False
            )
        pg_session.commit()
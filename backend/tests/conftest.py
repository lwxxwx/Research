# backend/tests/conftest.py
"""Sprint 3 G3 全局测试 fixture。

提供：
  - test_admin_user:  session 级 admin 用户（X-User-Id 用）
  - test_engineer_user: session 级 engineer 用户
  - client: TestClient，自动带 admin 的 X-User-Id
  - client_as_engineer: TestClient，自动带 engineer 的 X-User-Id
  - client_no_auth: 不带 X-User-Id 的 TestClient
  - client_with_invalid_uid: 带非法 X-User-Id 的 TestClient（用于负面测试）
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.persistence.db import get_db_session
from app.persistence.models import User


@pytest.fixture(scope="session")
def test_admin_user():
    """session 级 admin，测试结束删除。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import User, FeedbackItem
    import uuid

    uname = f"test_admin_global_{uuid.uuid4().hex[:6]}"
    with get_db_session() as db:
        user = User(username=uname, role="admin", is_active=True)
        db.add(user)
        db.flush()
        uid = user.id
        db.commit()
    yield {"id": uid, "username": uname}

    # teardown：先清除feedback_item外键引用，再删除用户，规避FK约束报错
    with get_db_session() as db:
        # 将该用户创建的反馈记录 created_by 置NULL，解除外键依赖
        db.query(FeedbackItem).filter(FeedbackItem.created_by == uid).update(
            {FeedbackItem.created_by: None}, synchronize_session=False
        )
        db.commit()
        # 再删除测试用户
        db.query(User).filter(User.id == uid).delete(synchronize_session=False)
        db.commit()



@pytest.fixture(scope="session")
def test_engineer_user():
    uname = f"test_eng_global_{uuid.uuid4().hex[:6]}"
    with get_db_session() as db:
        user = User(username=uname, role="engineer", is_active=True)
        db.add(user)
        db.flush()
        uid = user.id
        db.commit()
    yield {"id": uid, "username": uname}

    from app.persistence.models import FeedbackItem, ReviewResult, RuleCandidate, ReviewDefect
    with get_db_session() as db:
        # 逐表解除 FK 引用（未来 G3-B/G3-C 会引用这些表）
        db.query(FeedbackItem).filter(FeedbackItem.created_by == uid).update(
            {FeedbackItem.created_by: None}, synchronize_session=False
        )
        db.query(ReviewResult).filter(ReviewResult.created_by == uid).update(
            {ReviewResult.created_by: None}, synchronize_session=False
        )
        db.query(RuleCandidate).filter(RuleCandidate.reviewed_by == uid).update(
            {RuleCandidate.reviewed_by: None}, synchronize_session=False
        )
        db.query(ReviewDefect).filter(ReviewDefect.expert_decision_by == uid).update(
            {ReviewDefect.expert_decision_by: None}, synchronize_session=False
        )
        db.commit()
        db.query(User).filter(User.id == uid).delete(synchronize_session=False)
        db.commit()


@pytest.fixture()
def client(test_admin_user):
    """默认 admin 身份。"""
    c = TestClient(app)
    c.headers["X-User-Id"] = str(test_admin_user["id"])
    yield c
    c.headers.pop("X-User-Id", None)


@pytest.fixture()
def client_as_engineer(test_engineer_user):
    c = TestClient(app)
    c.headers["X-User-Id"] = str(test_engineer_user["id"])
    yield c
    c.headers.pop("X-User-Id", None)


@pytest.fixture()
def client_no_auth():
    return TestClient(app)


@pytest.fixture()
def client_with_invalid_uid():
    c = TestClient(app)
    c.headers["X-User-Id"] = "not_an_integer"
    return c
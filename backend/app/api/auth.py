# backend/app/api/auth.py
"""Sprint 3 G2 认证与用户管理 API。

端点：
  POST   /auth/login                  身份模拟登录（dev/test + prod 内网）
  GET    /auth/me                     当前用户信息
  GET    /auth/users                  用户列表（admin）
  POST   /auth/users                  创建用户（admin）
  PATCH  /auth/users/{user_id}        更新角色/启用状态（admin，含最后管理员保护）
  DELETE /auth/users/{user_id}        软禁用（admin，含最后管理员保护）

关键设计（决策已冻结）：
  - 第一阶段不交付 JWT / 密码登录，登录后由前端持有 user_id 并随请求带 X-User-Id。
  - 所有需要 admin 的端点使用 Depends(require_roles(ROLE_ADMIN)) 强制角色。
  - PATCH 请求用 model_fields_set 判断"字段是否存在"，避免显式 null 被误判为"不更新"。
  - 所有写操作中，Service 层已封装锁顺序、最后管理员保护、审计日志。
  - 端点不调用 db.commit()，由 get_db_session 统一 commit / rollback。
  - 禁止在事务块内嵌慢外部调用（D1）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import CurrentUser, get_current_user, require_roles, ROLE_ADMIN
from app.api.schemas import (
    AuthMeResponse,
    LoginRequest,
    LoginResponse,
    UserCreateRequest,
    UserListItem,
    UserListResponse,
    UserPatchRequest,
)
from app.persistence.db import get_db_session
from app.services import user_service


router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================
# POST /auth/login —— 身份模拟
# ============================================================
@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """username-only 身份模拟登录。

    校验用户存在且 is_active=true，返回 user_id / username / role。
    前端拿到 user_id 后自行保存，后续请求带 X-User-Id。
    """
    with get_db_session() as db:
        user = user_service.get_user_by_username(db, body.username)
    if user is None:
        raise HTTPException(status_code=401, detail=f"用户 {body.username} 不存在")
    if not user["is_active"]:
        raise HTTPException(status_code=401, detail=f"用户 {body.username} 已禁用")
    return LoginResponse(
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
    )


# ============================================================
# GET /auth/me —— 当前用户信息
# ============================================================
@router.get("/me", response_model=AuthMeResponse)
def me(current: CurrentUser = Depends(get_current_user)) -> AuthMeResponse:
    return AuthMeResponse(
        user_id=current.id,
        username=current.username,
        role=current.role,
        is_active=current.is_active,
    )


# ============================================================
# GET /auth/users —— 用户列表（admin）
# ============================================================
@router.get("/users", response_model=UserListResponse)
def list_users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _: CurrentUser = Depends(require_roles(ROLE_ADMIN)),
) -> UserListResponse:
    with get_db_session() as db:
        result = user_service.list_users(db, limit=limit, offset=offset)
    return UserListResponse(
        total=result["total"],
        items=[UserListItem(**item) for item in result["items"]],
    )


# ============================================================
# POST /auth/users —— 创建用户（admin）
# ============================================================
@router.post("/users", response_model=UserListItem, status_code=201)
def create_user(
    body: UserCreateRequest,
    current: CurrentUser = Depends(require_roles(ROLE_ADMIN)),
) -> UserListItem:
    try:
        with get_db_session() as db:
            created = user_service.create_user(
                db,
                username=body.username,
                email=body.email,
                role=body.role,
                actor_id=current.id,
            )
    except user_service.UsernameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return UserListItem(**created)


# ============================================================
# PATCH /auth/users/{user_id} —— 更新 role / is_active（admin）
# ============================================================
@router.patch("/users/{user_id}", response_model=UserListItem)
def patch_user(
    user_id: int,
    body: UserPatchRequest,
    current: CurrentUser = Depends(require_roles(ROLE_ADMIN)),
) -> UserListItem:
    """更新用户 role / is_active。

    关键：用 model_fields_set 判断"字段是否存在"。
      - 字段不存在 → 不更新
      - 字段存在且为 None → 422（不允许显式 null）
      - 字段存在且非 None → 用请求值
    """
    # 显式 null 拒绝
    fields_provided = body.model_fields_set
    if "role" in fields_provided and body.role is None:
        raise HTTPException(status_code=422, detail="role 不允许显式 null")
    if "is_active" in fields_provided and body.is_active is None:
        raise HTTPException(status_code=422, detail="is_active 不允许显式 null")

    # 空 PATCH（什么都没给）→ 直接返回当前状态
    if not fields_provided:
        with get_db_session() as db:
            user = user_service.get_user_by_id(db, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"用户 id={user_id} 不存在")
        return UserListItem(**user)

    # 构造只包含实际提供的字段的 patch
    patch: dict = {}
    if "role" in fields_provided:
        patch["role"] = body.role
    if "is_active" in fields_provided:
        patch["is_active"] = body.is_active

    try:
        with get_db_session() as db:
            updated = user_service.update_user(
                db,
                user_id=user_id,
                patch=patch,
                actor_id=current.id,
            )
    except user_service.UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        # LastAdminProtectionError 属于业务异常，转 422
        # 注意：这里用宽泛捕获避免 import 循环，具体类型见 admin_lock
        from app.services.admin_lock import LastAdminProtectionError
        if isinstance(exc, LastAdminProtectionError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise
    return UserListItem(**updated)


# ============================================================
# DELETE /auth/users/{user_id} —— 软禁用（admin）
# ============================================================
@router.delete("/users/{user_id}", response_model=UserListItem)
def soft_delete_user(
    user_id: int,
    current: CurrentUser = Depends(require_roles(ROLE_ADMIN)),
) -> UserListItem:
    try:
        with get_db_session() as db:
            updated = user_service.soft_delete_user(
                db,
                user_id=user_id,
                actor_id=current.id,
            )
    except user_service.UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        from app.services.admin_lock import LastAdminProtectionError
        if isinstance(exc, LastAdminProtectionError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise
    return UserListItem(**updated)
## 四、bootstrap admin（首次部署）

```powershell
# 1. 创建初始 admin（CLI，5 种场景）
docker compose exec backend python /scripts/bootstrap_admin.py --username admin
# 预期：OK: created admin 'admin' (id=XXX)

# 2. 验证 users 表
docker compose exec postgres psql -U postgres -d schematic_review -c "SELECT id, username, role, is_active FROM users;"
# 预期：1 行（admin / admin / t）

# 3. 二次执行（幂等验证）
docker compose exec backend python /scripts/bootstrap_admin.py --username admin
# 预期：OK: admin 'admin' already exists (id=XXX)
```

## 五、启动验证（后端 + 前端）

```powershell
# 1. 健康检查
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/healthz" -Method GET
# 预期：status=ok

# 2. 登录验证
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/login" -Method POST -Body '{"username":"admin"}' -ContentType "application/json"
# 预期：{user_id: 3232, username: admin, role: admin}

# 3. 用 admin 调 /auth/me
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/me" -Method GET -Headers @{ "X-User-Id" = "3232" }
# 预期：{user_id: 3232, username: admin, role: admin, is_active: true}

# 4. 前端可访问性验证
# 4.1 开发模式（Vite dev server，端口 5173）
cd frontend && npm run dev
# 另开终端：
Invoke-WebRequest -Uri "http://localhost:5173/" -UseBasicParsing
# 预期：200 + HTML 包含 "原理图评审系统"

# 4.2 生产模式（Docker + Nginx，端口 3000）
# Sprint 3 legacy 批次④（D23）已修复：镜像由 frontend/Dockerfile 多阶段构建，COPY dist 进 Nginx。
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" build frontend
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" up -d frontend
Invoke-WebRequest -Uri "http://localhost:3000/" -UseBasicParsing
# 预期：200 + HTML 含 "电子设计EDA原理图评审与演进系统"
# 深链验证：http://localhost:3000/cases 应返回 200（SPA fallback）
```

## 六、冒烟测试（端到端）

冒烟测试需覆盖两条前端路径：
- **开发模式 5173**（vite dev server，热更）：日常开发验证
- **生产镜像 3000**（Dockerfile 构建产物）：上线前验证

先按 5173 完成下表 6 步；上线前用 3000 再跑一次 1、2、6 步（登录 / 菜单 / 登出），确认打包制品行为一致。

| # | 步骤 | 预期 |
|---|---|---|
| 1 | 未登录访问 `/cases` | 自动跳转 `/login`，localStorage 无 `user_id` |
| 2 | 输入 admin 登录 | 跳转 `/cases`，顶部展示 5 个菜单，`localStorage.user_id = 3232` |
| 3 | 点击 case001 卡片「单次评审」 | 跳转 `/tasks/{id}` 详情页，缺陷列表显示 2 条缺陷 |
| 4 | 点击顶部「任务列表」 | 列表中出现刚才创建的任务记录 |
| 5 | 在详情页对缺陷提交反馈，类型选 `knowledge_gap` | Network 面板中 `POST /feedbacks` 响应包含 `knowledge_gap_ref: "GAP-..."` |
| 6 | 点击右上角「退出」 | 跳回 `/login`，`localStorage.user_id` 被清空 |

## 七、CI 验证

推送前确认 GitHub Actions 全绿：

```powershell
# 触发推送
git push

# 观察 GitHub Actions
# - backend job：656 passed（-m "not integration"）；若批次④ #8 合入后基线变化，以当次实际为准
# - frontend job：npm run build 成功
```

CI 环境变量（`.github/workflows/ci.yml`）：

| 变量 | 值 |
|---|---|
| `APP_ENV` | `test` |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/schematic_review` |
| `SECRET_KEY` | `ci-test-secret-key-0123456789abcdef` |
| `DATA_DIR` | `/home/runner/work/Research/Research/data` |
| `LLM_PROVIDER` | `mock` |
| `RAG_EMBEDDING_BACKEND` | `zhipu` |

## 八、故障排查

| 现象 | 排查 |
|---|---|
| healthz 连不上 | 查后端日志：`docker compose logs backend --tail 50`；看是否有 `ImportError` / `ValidationError` |
| `DATABASE_URL` 在容器内部不能使用 localhost | `APP_ENV=prod` 时禁止 localhost，用 `postgres` 服务名 |
| `ImportError: cannot import name 'XXX'` | 代码回滚 / 文件丢失，用 `git status` 看；恢复文件 |
| Multiple head revisions | `alembic heads` 查当前 head 数；多个 head 要合并（`alembic merge -m "merge" <rev1> <rev2>`） |
| 前端 401 跳登录死循环 | `http.ts` 的 `handleUnauthorized` 已加 `window.location.pathname !== "/login"` 防重复 |
| 前端 403 跳 `/403` 页 | `router/index.ts` 的 `beforeEach` 已处理；检查 `meta.roles` 配置 |
| `docker compose build frontend` 报 `dial tcp [2a03:2880:...]` | Docker Hub IPv6 不通；用 `docker.1ms.run` 反代 pull + tag 后重试，或配 Docker Desktop 代理 / registry mirror |

## 九、Sprint 3 回滚预案

如果部署后发现严重问题，按以下顺序回滚：

1. **回滚应用版本**：`git revert <commit>` 或 `git checkout <上一个稳定 tag>`
2. **回滚 Migration**（仅临时验证环境）：`alembic downgrade -1`

> ⚠️ **生产环境禁止 downgrade**：`review_result.created_by` / `rule_candidates.reviewed_by` 等新增列一旦回滚，审计数据丢失。生产回退应结合「应用版本回退 + 备份恢复」。

## 十、部署后验证清单（打勾）

- [ ] `alembic current` 显示 `s3_b2_feedback_soft_delete (head)`
- [ ] `users` 表至少有 1 个 admin 用户
- [ ] `healthz` 返回 `status=ok`
- [ ] admin 登录成功，`/auth/me` 返回 `role=admin`
- [ ] 前端 `/cases` 页面正常渲染 5 个菜单
- [ ] 冒烟测试 6 步全过
- [ ] GitHub Actions 全绿
- [ ] 生产镜像 3000 端口：`http://localhost:3000/` 返回真实前端（含 `电子设计EDA原理图评审与演进系统`），`http://localhost:3000/cases` 深链返回 200
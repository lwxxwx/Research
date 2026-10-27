# Schematic Design Review · H 阶段前端（Web MVP）

> Sprint 1 阶段 H 的最小可运行 Web MVP。
> 技术栈：Vue 3 + Vite 5 + TypeScript 5 + Tailwind CSS 3 + Pinia + Vue Router 4。

## 快速开始

### 1. 前置条件

| 工具 | 版本 |
|---|---|
| Node.js | >= 18（推荐 20 LTS 或 22 LTS） |
| npm | >= 9 |
| 后端 | `docker compose up`（FastAPI 8000 端口） |

### 2. 安装依赖

```powershell
cd frontend
npm install
```

### 3. 启动开发服务器

```powershell
npm run dev
```

浏览器访问 `http://localhost:5173/cases`。

**Vite proxy** 会将 `/api` 转发到后端 `http://localhost:8000`（见 `vite.config.ts`）。

### 4. 类型检查

```powershell
cd frontend
npm run type-check
```

### 5. 生产构建（I 阶段用，H 阶段不需跑）

```powershell
cd frontend
npm run build
# 产物在 frontend/dist/
```

## 目录结构

```text
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json / tsconfig.app.json / tsconfig.node.json
├── tailwind.config.js / postcss.config.js
├── .env.development / .env.production
└── src/
    ├── main.ts                 # 应用入口
    ├── App.vue                 # 根组件（导航栏 + RouterView）
    ├── style.css               # Tailwind 全局样式
    ├── vite-env.d.ts           # ImportMetaEnv 类型
    ├── router/index.ts         # 4 路由
    ├── stores/task.ts          # Pinia store
    ├── types/                  # TS 类型（与后端契约对齐）
    │   ├── case.ts
    │   ├── evidence.ts
    │   ├── defect.ts
    │   ├── task.ts
    │   └── feedback.ts
    ├── api/                    # API 客户端
    │   ├── http.ts             # fetch 封装（baseURL + 20 秒超时）
    │   ├── cases.ts
    │   ├── review.ts
    │   └── feedback.ts
    └── pages/
        ├── CaseListPage.vue    # ✅ Step 5 完整实现
        ├── TaskListPage.vue    # ⏳ Step 5 之后
        ├── TaskDetailPage.vue  # ⏳ Step 5 之后
        └── NotFoundPage.vue    # ✅ 404
```

## API 契约

**核对文档**：`H 阶段 API 契约核对文档.md`（12 章 + 附录 A/B/C，80 字段，0 ❌ / 3 ⚠️）。

**7 端点**：

| Method | Path | 用途 |
|---|---|---|
| GET | `/api/v1/cases` | Case 列表 |
| POST | `/api/v1/review/tasks` | 发起评审（约 20 秒） |
| GET | `/api/v1/review/tasks/{task_id}` | 任务详情 |
| GET | `/api/v1/review/tasks` | 任务列表 |
| GET | `/api/v1/review/tasks/{task_id}/defects` | 缺陷明细 |
| POST | `/api/v1/feedbacks` | 提交反馈 |
| GET | `/api/v1/healthz` | 健康检查（前端不直接用） |

## 20 秒超时语义（关键）

| 场景 | 行为 |
|---|---|
| `POST /review/tasks` 20 秒内成功 | 跳转 `/tasks/{task_id}` |
| 20 秒超时（`TIMEOUT_ERROR`） | **超时 ≠ 失败**，显示"任务可能已在后台创建" + "去任务列表确认"链接，**禁止自动重试** |
| 网络错误 / 4xx / 5xx（非超时） | 显示错误 + 保留"发起评审"按钮（可重试） |
| loading 期间 | 按钮 disabled，**禁止重复提交** |

## 不做清单（H 阶段明确不做）

| 不做 | 原因 | 归属 |
|---|---|---|
| 文件上传 UI | Sprint1_融合版 §11.5 | 不在 Sprint 1 |
| 登录 / 权限 | Sprint1_融合版 §11.5 | 不在 Sprint 1 |
| 规则管理后台 | Sprint1_融合版 §11.5 | 不在 Sprint 1 |
| category/risk 彩色标签（复杂样式） | V1.2 §S2.5 | Sprint 2（H 阶段文字徽标） |
| suggestion_diff 对比 UI | V1.2 §S2.6 | Sprint 2（H 阶段 JSON 文本域） |
| rule_candidate 表单 | V1.2 §S2.6 | Sprint 2（H 阶段 JSON 文本域） |
| TaskDetail 顶部摘要卡片（EVC / AI_NER / EAR） | V1.2 §S2.7 | Sprint 2（Sprint 1 的 `report.summary` 无这 3 字段） |
| 前端单测 / E2E 自动化 | 1.5 人日约束 | H 阶段不做（手动验收） |
| SSR / SSG | 无 SEO 需求 | 不做 |
| 前端 CI | 1.5 人日约束 | I 阶段做 |

## 8 个异常场景手动验收

| # | 场景 | 验收方式 | 预期 UI |
|---|---|---|---|
| 1 | 后端不可用（网络错误） | 停后端容器 | "加载 Case 列表失败" + 重试按钮 |
| 2 | 评审请求超时（20 秒） | 后端临时 sleep 30 秒 | "请求超时" amber 提示 |
| 3 | **超时后不自动重复创建任务** | 同 #2，观察无自动重试 | 只显示"去任务列表确认"，**无重试按钮** |
| 4 | 任务不存在或详情接口 404 | 访问不存在的 task_id | "任务不存在或未完成落库" + 返回链接 |
| 5 | 任务执行失败（`status=failed`） | 找 failed 任务 | 显示 `errors` 列表 |
| 6 | 报告为空或缺陷列表为空 | 找 pending / 无缺陷任务 | "报告未生成" / "无缺陷" |
| 7 | 反馈 JSON 格式错误 | 手动传非法 JSON 到 `suggestion_diff` | 前端拦截（JSON.parse 失败）→ "JSON 格式错误" |
| 8 | 反馈提交遇到后端验证错误或网络错误 | 传非法 `feedback_type` / 停后端 | "参数错误" / "网络错误" |

## 已知问题

- **npm audit: esbuild dev server 漏洞（moderate）**
  - 影响：仅在 `npm run dev` 时存在，本地开发（localhost）风险极低
  - 不影响：`npm run build` 产物（生产用 Nginx 托管 dist/）
  - 处理：接受（不跑 `npm audit fix --force`，避免破坏 Vite 5 技术栈）
  - 后续：Sprint 2 或 I 阶段评估升级 Vite（vite 6+ 修复此漏洞，但需要 Vue 3.6+/Tailwind 4）

## 参考文档

- `H 阶段 API 契约核对文档.md`（12 章 + 附录，80 字段冻结）
- `backend/app/api/schemas.py`（后端 Pydantic 权威来源）
- `backend/app/services/task_service.py`（service 返回 dict 权威来源）
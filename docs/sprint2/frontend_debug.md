# Sprint 2 开发与调试问题记录（Debugging Notes）

> **用途**：沉淀 Sprint 2 开发过程中遇到的典型 Bug、环境问题及排查思路，供后续开发、联调及运维部署参考。
> **归档日期**：2026-10-29
> **最后更新**：2026-10-30（补充 S2-H（四）指标看板及 Docker 构建/依赖同步排障）

---

## 目录
1. [S2-H（二）：批量评审交付记录](#1-s2-h二批量评审2026-10-30)
2. [Vite HMR 失效 / 前端代码更新不生效](#2-vite-hmr-失效--前端代码更新不生效)
3. [API 请求返回 HTML（被 Vite 路由拦截）](#3-api-请求返回-html被-vite-路由拦截)
4. [组件数据不响应 / 样式不更新](#4-组件数据不响应--样式不更新)
5. [后端 Pydantic 422 校验报错（Extra inputs are not permitted）](#5-后端-pydantic-422-校验报错extra-inputs-are-not-permitted)
6. [指标数据缺失与 UI 组件消失](#6-指标数据缺失与-ui-组件消失)
7. [提交反馈后"无反应"的错觉](#7-提交反馈后无反应的错觉)
8. [Sprint 1 规则候选 ID 包含特殊连字符](#8-sprint-1-规则候选-id-包含特殊连字符)
9. [前后端超时限制不一致导致批量任务失败](#9-前后端超时限制不一致导致批量任务失败)
10. [Vue 模板标签未闭合导致编译失败](#10-vue-模板标签未闭合导致编译失败)
11. [开发者调试利器：全链路 Console 日志](#11-开发者调试利器全链路-console-日志)
12. [FastAPI 启动报错 python-multipart 缺失](#12-fastapi-启动报错-python-multipart-缺失)
13. [前端上传文件报错 Unexpected token 'F', "[object FormData]"](#13-前端上传文件报错-unexpected-token-f-object-formdata)
14. [S2-H（四）：指标看板前后端联调与交付](#14-s2-h四指标看板前后端联调与交付2026-10-30)
15. [Docker 构建环境与依赖同步排障（S2-H 基础设施）](#15-docker-构建环境与依赖同步排障s2-h-基础设施)
16. [前端 Vue/TypeScript 编译与构建排障（Build 模式）](#16-前端-vuetypescript-编译与构建排障build-模式)
17. [附：版本与适用范围](#17-附版本与适用范围)

---

## 1. S2-H（二）：批量评审（2026-10-30）

> 状态：✅ 完成
> 依据：`Sprint2_详细执行计划_V1.0.md` §14 S2-H 行

### 交付内容

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/api/schemas.py` | 追加 `BatchReviewRequest` / `BatchReviewItem` / `BatchReviewResponse` |
| 2 | `backend/app/api/review_tasks.py` | 新增 `POST /review/tasks/batch` 接口，限制最大 5 个，异常隔离不中断 |
| 3 | `frontend/src/api/review.ts` | `createTask` 超时由 20s 调整为 30s（给 LLM 留足时间，避免批量时大面积 TIMEOUT） |
| 4 | `frontend/src/pages/CaseListPage.vue` | 增加卡片多选、底部悬浮操作栏、全屏批量进度遮罩 |

### 验收证据

| 项 | 结果 |
|---|---|
| 前端多选限制 | ✅ 超过 5 个弹窗拦截 |
| 批量执行 | ✅ 进度条实时显示（1/3 → 3/3） |
| 批量结果 | ✅ 3 个 Case 全部成功，无 TIMEOUT |
| 跳转 | ✅ 跑完自动跳转到任务列表 |

### 踩坑与修正

| 坑 | 现象 | 修正 |
|---|---|---|
| 批量时部分 Case 报 `TIMEOUT` | 前端单次请求超时 20 秒，但实际 LLM 跑 20~25 秒，导致批量前半段被截断 | `review.ts` 的 `createTask` 超时调整为 30 秒；批量执行文案提示预计总耗时，避免用户焦虑 |
| 模板嵌套标签未闭合 | 替换代码时报 `Element is missing end tag` | 全量覆盖 `CaseListPage.vue`，修复标签嵌套 |

---

## 2. Vite HMR 失效 / 前端代码更新不生效

### 现象
修改了 Vue 组件或 `vite.config.ts` 后，浏览器页面没有发生变化，或者请求路径被错误转发。

### 根因
1. Vite 的热更新（HMR）在处理配置文件（如 `vite.config.ts`、`http.ts`）时，无法静默热替换，必须重启。
2. 浏览器存在强缓存，继续读取旧版的 `.js`/`.vue` chunk。
3. 开启了多个 `npm run dev` 进程，导致端口冲突或代理失效。

### 解决与排查
1. **万能终结法**：在终端 `Ctrl+C` 停掉 Vite，运行 `taskkill /F /IM node.exe` 清理残留 Node 进程，重新 `npm run dev`。
2. 浏览器必须按 `Ctrl + F5` 强制刷新。
3. 开发期在代码里增加 `console.log`，或者查看 Network 面板的 `Initiator` 确认发起了新请求。

---

## 3. API 请求返回 HTML（被 Vite 路由拦截）

### 现象
前端发起 `GET /api/v1/review/tasks/xxx`，但在 Network 面板中，该请求的 `Type` 显示为 `document`，`Response` 是一段 `<!doctype html>`，导致 JSON 解析失败，页面数据无法渲染。

### 根因
Vite 没有配置代理（Proxy），导致 Vite Dev Server 把 `/api` 请求当成了前端路由（`/api/...` 页面），直接 Fallback 返回了 `index.html`。

### 解决
1. 在 `frontend/vite.config.ts` 中配置 `server.proxy`：
```typescript
server: {
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```
2. 修改完配置后，必须重启 Vite。
3. 排查 `frontend/.env.development` 中的 `VITE_API_BASE_URL` 是否为 `/api/v1`，禁止在开发环境写死 `http://localhost:8000`。

---

## 4. 组件数据不响应 / 样式不更新

### 现象 A：徽标颜色点完不变化
点击「专家确认」按钮，后端返回 200，但页面徽标依然是灰色的「待判定」。

**根因**：`ExpertDecisionBadge.vue` 中使用了 `const badge = DECISION_MAP[props.decision]`，这是非响应式的静态对象，props 改变时不会重新计算。

**解决**：使用 `computed` 包裹。
```typescript
const badge = computed(() => DECISION_MAP[props.decision] ?? defaultBadge);
```

### 现象 B：外部列表状态不同步
内部面板点击确认后，外面折叠状态的徽标没有变绿。

**根因**：Vue 3 对于数组内部对象的替换，有时响应式 Proxy 不会强触发 v-for 更新。

**解决**：在父组件 `TaskDetailPage.vue` 更新 defects 数组时，使用 splice 替换，而不是直接赋值索引。
```typescript
defects.value.splice(idx, 1, updatedDefect);
```

---

## 5. 后端 Pydantic 422 校验报错（Extra inputs are not permitted）

### 现象
前端提交「新规则候选」反馈时，抛出 422 错误：`"loc": ["body", "rule_candidate", "severity_hint"], "msg": "Extra inputs are not permitted"` 以及缺少 `rule_id`、`rule_name` 等字段。

### 根因
Sprint 1 的 `RuleCandidate` 模型定义非常严格（11 个字段），而 S2-E 前端出于用户体验只设计了轻量表单（标题/描述/严重度）。前端直接透传 `{ title, description, severity_hint }` 导致与后端契约冲突。

### 解决
在 API 边界层（`frontend/src/api/feedback.ts`）做一层映射适配，把轻量表单转化为完整的后端模型结构：
```typescript
if (body.rule_candidate) {
  const rc: any = body.rule_candidate;
  rc.severity = rc.severity_hint || "medium";
  delete rc.severity_hint;
  rc.rule_id = "AUTO-GEN";
  rc.rule_name = rc.title;
  rc.category = "other";
  rc.rule_basis = "待补充";
}
```

---

## 6. 指标数据缺失与 UI 组件消失

### 现象
后端接口返回了 `summary_metrics`，但前端页面顶部的「任务摘要指标」卡片直接消失不渲染。

### 根因
1. 后端 `summary_metrics.evidence_coverage` 返回的是 `null`（Sprint 1 遗留，当时未存该列）。
2. 前端 `TaskSummaryCard.vue` 外壳使用了 `<div v-if="metrics">`，一旦某些边界情况导致整个对象为 `null`，或者被错误拦截，整个卡片就会从 DOM 中移除。
3. 另外，业务规则要求 N/A 时不能显示 0%。

### 解决
1. 去掉 `TaskSummaryCard` 外壳的 `v-if`，让卡片常驻。
2. 内部指标单元格判断：如果值为 `null`，则显示 `N/A`。
3. 对于样本不足的 `EAR` 和 `AI_NER`，需显示 `N/A · 样本不足`（基于 `sample_info` 判断），坚守 `metrics_definition.md` v3.2 的口径，绝不向用户展示伪造的 `0%`。

---

## 7. 提交反馈后"无反应"的错觉

### 现象
点击提交反馈后，页面瞬间似乎回到了原样，用户以为提交失败。

### 根因
1. 请求确实成功了（Network 200）。
2. 父组件 `TaskDetailPage` 监听了 `@success`，在成功瞬间触发 `toggleFeedback()` 把整个反馈面板折叠了。
3. 面板一折叠，内部的「提交成功绿框」也跟着被销毁了。

### 解决与后续
目前属于正常逻辑。只要 Network 返回 200 即可确认数据已落库。
优化方向：未来可以在父组件用全局 Toast 提示，或者提交成功后延迟 1 秒再折叠面板，给用户一个明确的视觉反馈。

---

## 8. Sprint 1 规则候选 ID 包含特殊连字符

### 现象
在 S2-G 的前端页面请求详情 `GET /api/v1/rule-candidates/RC-20261029-...` 时，可能会报 404 或找不到。查看控制台 Network，发现 URL 变成了 `RC%E2%80%80%91...`（一堆乱码字符）。

### 根因
Sprint 1 的 `_make_candidate_id()` 在生成候选 ID 时，使用了一个特殊的非 ASCII 连字符（`RC‑...`，看起来像减号，但实际上是全角或非断行连字符）。浏览器和 HTTP 协议对 URL 中的非 ASCII 字符会进行编码，导致路径匹配失败。

### 解决
1. **前端兜底**：在 `frontend/src/api/rule_candidate.ts` 中，使用 `encodeURIComponent(id)` 对 ID 进行编码，确保特殊字符能正确传给后端。
2. **后续治理**：Sprint 3 需要修复 `rule_evolution_service.py` 的 `_make_candidate_id()`，强制使用标准 ASCII 的短横线 `-`（U+002D），清理存量脏数据。

---

## 9. 前后端超时限制不一致导致批量任务失败

### 现象
在 S2-H 批量评审时，勾选 5 个 Case 串行执行，前端弹窗提示部分 Case 报 TIMEOUT 失败。

### 根因
1. 前端 `http.ts` 里的 `DEFAULT_TIMEOUT_MS` 写死为 20 秒。
2. 单个 Case 的 LLM 耗时如果在 20~25 秒之间，前端就会单方面断开连接并抛出 TIMEOUT。
3. 因为批量是串行调用，所以导致后面的请求大面积报错。

### 解决
1. 在 `frontend/src/api/review.ts` 的 `createTask` 中，将单次请求的超时时间显式调整为 30 秒，给 LLM 留足时间。
2. 前端批量遮罩层上提示"系统正在串行执行，预计耗时 XX 秒，请勿关闭页面"，安抚用户等待情绪。

---

## 10. Vue 模板标签未闭合导致编译失败

### 现象
修改 `CaseListPage.vue` 或 `TaskDetailPage.vue` 的嵌套 div 时，保存后页面白屏，终端报错 `[plugin:vite:vue] Element is missing end tag`。

### 根因
使用替换的方式增加嵌套结构（如在外层包一个 flex 容器）时，由于只替换了开头 `<div>`，没同步移动结尾的 `</div>`，导致标签不配对。

### 解决
1. 遇到该报错，直接在终端看报错指向的 `.vue` 文件及行号。
2. 找到对应的行，手动补齐 `</div>`。
3. 或者直接全量覆盖对应的 `.vue` 文件，避免局部替换产生配对误差。

---

## 11. 开发者调试利器：全链路 Console 日志

为了快速定位前后端交互问题，S2-D/E/G/H 阶段在关键路径埋了 DEV 环境的日志：
- `http.ts`：打印 `[API 请求开始]`、`[API 成功]`、`[API 失败]`，并附带耗时和 payload。
- `ExpertDecisionPanel.vue`：打印 `[专家判定]` 按钮点击和接口回包。
- `TaskDetailPage.vue`：打印 `[页面状态更新]` 数组渲染。

**建议**：在后续的 S2-H（RAG 管理）中，继续保持这种「关键请求+关键状态变更」的日志规范，利用浏览器的 Console 面板过滤 `[API` 或 `[State` 即可迅速排障。

---

## 12. FastAPI 启动报错 python-multipart 缺失

### 现象
新增了处理文件上传的接口（`@router.post("/upload")`），重启后端时容器直接崩溃退出，日志报错：

```
RuntimeError: Form data requires "python-multipart" to be installed.
```

### 根因分析
1. **FastAPI 底层依赖缺失**
   FastAPI 在解析 `UploadFile` 或 `Form` 表单数据时，底层依赖 `python-multipart` 库；Sprint 0/1 阶段没有文件上传需求，所以 `pyproject.toml` 里没有这个依赖，接口一挂载即崩溃。

2. **Docker 缓存机制（层缓存导致新依赖没进镜像）**
   修改 `pyproject.toml` 加上 `python-multipart>=0.0.9` 后，单纯执行 `docker compose up -d --build` 依然报错。根因在于：Docker 为加速构建，会按 Dockerfile 指令逐层比对，**只要前面的指令（拉取基础镜像、复制源码、安装系统依赖等）没变，就复用上一层的缓存**，导致执行依赖安装的 `uv sync` 步骤被整体跳过。因此新依赖根本没有进入镜像，运行时环境里依然没有 `python-multipart`。

3. **UV 的环境隔离与锁文件（uv.lock）滞后**
   - 通过 `docker exec` 进去用 `uv pip install python-multipart` 临时装包、`uv pip list` 也能看到，但 **uvicorn 启动用的是 `/opt/venv` 虚拟环境**；如果临时包被装到了系统 Python 或其它环境（位置不对），它依然找不到，uvicorn 进程照样 import 失败。
   - 更致命的是：项目依赖由 `uv.lock` 锁定。只改 `pyproject.toml` 而不跑 `uv lock`，锁文件里没有新记录；即使跑 `uv sync`，若加了 `--frozen`，它会严格按旧锁文件执行，**甚至把环境中临时多出来的包（刚才手动装的）卸载掉**，导致改动反复失效。

**小结**：表面是「缺一个库」，实际是 **Docker 层缓存 + uv 锁文件/虚拟环境隔离** 两层机制叠加。只改配置或临时装包都不够，必须让锁文件、虚拟环境与镜像三层同步刷新，才能让 `python-multipart` 真正进入 uvicorn 运行环境。

### 解决思路与最终步骤
1. 打开 `backend/pyproject.toml`，在 `dependencies` 中添加 `"python-multipart>=0.0.9"`。
2. **核心解决**：进入容器重新生成锁文件并同步环境。
```bash
docker compose exec backend uv lock
docker compose exec backend uv sync
```
3. 重启后端容器（`restart backend`）。
4. 进容器验证：`docker compose exec backend python -c "import multipart; print(multipart.__file__)"`，必须能输出版本路径。

---

## 13. 前端上传文件报错 Unexpected token 'F', "[object FormData]"

### 现象
在 Web 端点击「上传 Markdown」选择文件时，页面出现红色报错：

```
Unexpected token 'F', "[object FormData]" is not valid JSON
```

### 根因分析
1. 前端 `http.ts` 封装了一个全局的 `request` 函数，里面默认设置了请求头 `"Content-Type": "application/json"`。
2. 当传入 `FormData` 对象时，浏览器发现 Content-Type 被手动写死了，导致它无法自动生成带 boundary 的 `multipart/form-data` 头。
3. `fetch` 底层把 `FormData` 强制转成了字符串 `"[object FormData]"` 发送给后端。后端返回了 400/422，前端尝试用 `JSON.parse` 去解析这个非 JSON 字符串，直接抛出语法错误。

### 解决思路与最终代码
1. **动态判断请求头**：在 `http.ts` 的 `request` 函数中，判断如果 `options.body` 是 `FormData` 实例，则绝对不设置 Content-Type 头。
2. **修复日志解析报错**：在开发日志打印处，如果是 FormData，不要去 `JSON.parse`，直接打印 `[FormData]`。
```typescript
// http.ts 修复片段
const headers: Record<string, string> = {
  ...(options.headers as Record<string, string> || {}),
};
if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
  headers["Content-Type"] = "application/json";
}
```

---

## 14. S2-H（四）：指标看板前后端联调与交付（2026-10-30）

> 状态：✅ 完成

### 交付内容
- **后端**：`dashboard_service.py`（纯聚合逻辑）、`schemas.py`（Pydantic 契约）、`dashboard.py`（`GET /api/v1/dashboard/summary`）。
- **前端**：`DashboardPage.vue`（5 块区域）、`api/dashboard.ts`、`types/dashboard.ts`、路由挂载。
- **契约冻结**：返回 `scope`、`metrics`、`rule_candidates`、`knowledge_gaps` 四个顶层对象，不设时间范围，独立统计规则/缺口。
- **聚合口径**：EVC 取 done 任务的算术平均；EAR/AI_NER 全局池复用 `benchmark_service`；Seed 双路径检测。

### 踩坑与修正

| 坑 | 现象 | 机理与修正 |
|---|---|---|
| 空文件引发 TS2305 | `vue-tsc` 报 `Module '"@/api/dashboard"' has no exported member 'getDashboardSummary'` | 抓取文件时保存为空。补全 `export function getDashboardSummary` 恢复正常 |
| 测试代码时间戳错位 | `test_db_schema.py::test_trigger_rule_definition_updated_at_refresh` 报 `assert t2 > t1` 失败，且 t2 竟然早于 t1 | 机理：原测试直接读取 flush 后的 ORM 内存对象 `rd.updated_at` 作为 t1，而 t2 是 `expire()` 后重读的 DB 真实值。两者采样口径不同，且受 SQLAlchemy 对象状态影响导致比对错乱。修正：在获取 t1 前也加上 `expire()` + `get()`，让两端都从 DB 取值 |
| 看板数据校验无数据 | 初次渲染时 EVC/AI_NER/EAR 全是 0% 或页面崩溃 | 严格遵循指标 v3.2：无样本时返回 `null` 并在前端渲染 `N/A · 样本不足`；去掉上层 `v-if`，防止组件消失 |

---

## 15. Docker 构建环境与依赖同步排障（S2-H 基础设施）

### 现象
执行 `bootstrap.ps1` 时，在 `docker compose up -d --build` 阶段直接报错退出；或即使构建成功，进容器后 `import pytest` 报 `ModuleNotFoundError`。

### 根因分析（四大叠加）
1. **网络超时（Download Timeout）**
   `Dockerfile` 执行 `uv sync --frozen` 时，访问 `files.pythonhosted.org` 出现随机超时或下载卡死（重试 3 次耗时 126 秒后报错）。属于网络链路问题，与代码无关。
2. **Dev 依赖被二次卸载（venv 无 pytest）**
   `Dockerfile` 的 final 阶段原本 `COPY --from=builder /opt/venv_prod /opt/venv`，而 `venv_prod` 是 `--no-dev` 构建的，天然不含 pytest。本地开发跑测试必须用 venv，导致 pytest 永远缺失。
3. **脚本强行"精确同步"（致命杀手）**
   `bootstrap.ps1` 和 `dev.ps1` 的启动逻辑中带了 `uv sync`（不带 `--extra dev`）。`uv sync` 默认是「精确同步」，它会把当前 venv 里属于 extra dev 的依赖（pytest 等）全部卸载（Uninstalled），导致构建阶段千辛万苦装上的 dev 依赖瞬间灰飞烟灭。
4. **源被污染导致 lock 篡改**
   在容器内跑 `uv lock` 时，由于容器网络指向了不可信的索引（或是存在冲突的 index-url 环境变量），生成了如 `httpcore2`、`python-dateutil 3.0.0` 这样不存在的依赖包，并写回了宿主机的 `uv.lock`。

### 解决思路与最终步骤
1. **修正 Dockerfile 容灾与 dev 环境**
   - 在 runtime 的 ENV 块加上 `UV_HTTP_TIMEOUT=120` / `UV_HTTP_RETRIES=5` 解决网络抖动。
   - 将 final 阶段的 `COPY --from=builder /opt/venv_prod` 改为 `COPY --from=builder /opt/venv`，保证本地开发有 pytest。
   - 增加构建期防呆：`RUN python -c "import pytest; ..."`，一旦缺失构建直接失败，阻断污染镜像。
   - 注意：ENV 的多行续行中间禁止插入注释行，否则 Docker 会报语法错误。需把注释统提至 ENV 块上方。
2. **清理锁文件与脚本**
   - 恢复被污染的 `uv.lock`（`git checkout -- backend/uv.lock`），移除异常的 pythonhosted 镜像源。
   - 从 `bootstrap.ps1` 和 `dev.ps1` 中彻底删除 `uv sync` 的代码块。启动脚本不应干涉 venv 的精确依赖状态，依赖只由镜像构建期负责。
3. **强制刷新镜像**
   - 运行 `docker compose ... up -d --build`。观察日志，若需重装，使用 `uv sync --extra dev`。

---

## 16. 前端 Vue/TypeScript 编译与构建排障（Build 模式）

### 现象
运行 `npm run type-check`（`vue-tsc --noEmit`）0 错误，但运行 `npm run build`（`vue-tsc -b && vite build`）直接失败，报 9 个错误。

### 根因分析
`-b` 是 Project References 的 Build 模式。`vue-tsc --noEmit` 仅做快速静态类型检查，而 `vue-tsc -b` 会检查所有子项目（包括不常改的 `vite.config.ts`），并将 TS6133（变量未使用）等 lint 级规则视为致命错误终止构建。

### 解决思路与最终步骤
1. **清理历史遗留的未使用变量/引入**
   - `TaskSummaryCard.vue`：删除 `const props =`，直接 `defineProps`。
   - `CaseListPage.vue`：删除未使用的 `const resp =`，改为 `await createTask(cid)`。
   - `RuleCandidatePage.vue`：移除 `computed` 和 `AlertCircle` 的 import。
   - `TaskListPage.vue`：移除 `TaskListItem` 的 import。
   - `TaskDetailPage.vue`：将报错 `Cannot assign to 'taskId'` 的 `const` 定义改为 `let`。
2. **补齐 Node 类型声明**
   - 在 `vite.config.ts` 中用到了 `path` 和 `__dirname`，需安装依赖：`npm install -D @types/node`。
3. **结果**
   - 重新运行 `npm run build`，1988 modules transformed，构建成功，产物输出至 `dist/`。

---

## 17. 附：版本与适用范围

- 前端：Vue 3 + Vite 5 + TypeScript
- 后端：FastAPI + Pydantic + uv
- 基础设施：Docker Compose
- 适用范围：Sprint 2 开发联调阶段（S2-D / S2-E / S2-G / S2-H 为主）
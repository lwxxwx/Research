# Sprint 2 开发与调试问题记录（Debugging Notes）

> **用途**：沉淀 Sprint 2 开发过程中遇到的典型 Bug、环境问题及排查思路，供后续开发、联调及运维部署参考。
> **归档日期**：2026-10-29
> **最后更新**：2026-11-02（补充 S2-F07 / S2-F15 交付及全链路恢复记录）

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
17. [S2-F07：Case 管理与单 Case Benchmark 前后端联调](#17-s2-f07case-管理与单-case-benchmark-前后端联调2026-11-02)
18. [S2-F15：技术基线快照与全链路恢复](#18-s2-f15技术基线快照与全链路恢复2026-11-02)
19. [S2-F15-Fix：CI/CD 流水线环境适配与排障](#19-s2-f15-fixcicd-流水线环境适配与排障2026-11-02)
20. [S2-Git-Fix：Git 历史分叉治理与分支清理](#20-s2-git-fixgit-历史分叉治理与分支清理2026-11-02)
21. [附：版本与适用范围](#21-附版本与适用范围)

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

## 17. S2-F07：Case 管理与单 Case Benchmark 前后端联调（2026-11-02）

> 状态：✅ 完成
> 交付：`CaseDetailPage.vue`（资产预览 + Benchmark 运行 + 历史表格）、`CaseListPage.vue`（卡片跳详情 + 独立多选）。
> 核心难点：在同一个卡片上同时实现“点击主体跳转”和“点击多选阻断跳转”，以及跨框架默认样式的 UI 适配。

### 踩坑与修正

| 坑 | 现象 | 排查思路与机理 | 最终修正 |
|---|---|---|---|
| **多选与跳转事件冒泡** | 点击卡片右上角的 checkbox，预期是勾选，结果直接跳转到了 Case 详情页。 | 初始尝试用绝对定位 `absolute top-3 right-3` 加 `@click.stop` 隔离。但由于卡片是 `relative` 且子元素有 `flex` 撑开，绝对定位层由于没有显式宽度或层级被底部的按钮盖住，事件冒泡到了外层的卡片 `@click` 上。 | **重构卡片内部布局**：彻底放弃绝对定位。用 Flex 布局把卡片顶部划分为左右两栏：左侧（`@click="goToCaseDetail"`），右侧（`@click.stop="toggleSelect"`）。物理隔离点击区域。 |
| **原生 checkbox 在白色背景下“隐形”** | 切换为原生 `<input type="checkbox">` 后，视觉上看不见方框，只有点击后才会出现蓝色对勾。 | Chrome/Edge 在特定系统主题下，原生的 unchecked checkbox 边框非常浅，甚至完全透明，在白色卡片背景中丢失了视觉存在感。 | 外层包裹 `<label>`，加上明显的灰色背景（`bg-gray-50`）和边框（`border-2 border-gray-300`），强行“画”出一个清晰的方块。 |
| **胶囊按钮多选交互体验不佳** | 原生 checkbox 的点击区域太小，且没有悬停反馈，不符合现代 Web 交互习惯。 | 原生控件受限于浏览器自身的 UI 引擎，无法做到像 GitHub 或 Figma 那样“点击标签块变色选中”的效果。 | 不再使用 `<input>`，改回 `@lucide/vue` 的 `<CheckSquare>` / `<Square>` 图标。将其包裹在一个带圆角、带背景色的 `<div>` 里，绑定 `@click.stop="toggleSelect"`。 |
| **图标文字组合的样式抖动** | 把图标和文字“选择/已选择”放在一起时，选中状态切换会导致卡片宽度抖动或换行。 | 图标和文字的长度不同（“选择”二字变“已选择”三字），在没有固定宽度的 flex 容器中，会挤压其他元素。 | 给该容器统一设置 `px-2 py-1` 的内边距，并使用 Tailwind 的 `bg-blue-50 text-blue-700` 与 `bg-gray-50 text-gray-500` 做色彩区分，确保布局稳定。 |
| **前端 API 引入路径报错** | `npm run build` 报错 `Cannot find module '@/api/case'` 和 `'@/types/cases'`。 | 在写 `CaseDetailPage.vue` 时，按照直觉引入了单数的 `api/case` 和复数的 `types/cases`，但实际文件命名恰好相反。 | 修正 `CaseDetailPage.vue` 的 import：`api` 用复数 `@/api/cases`，`types` 用单数 `@/types/case`。 |
| **F05 字段在 ORM 中丢失导致回归测试失败** | 跑全量回归时，`test_db_schema.py::test_orm_columns_match_db` 报 `rule_candidates` 的 `db_only` 多出 5 个 `publish_*` 字段。 | F05 和 F07 跨天开发时，工作区发生了一些未预期的代码丢失（或回滚）。`models.py` 里关于 F05 的 5 个发布字段不翼而飞，但 DB 里已经有了这些列。 | 通过 `git show 53878c7:backend/alembic/versions/s2_f05_rule_publish.py > ...` 恢复了丢失的 Migration 文件，并手动在 `models.py` 的 `RuleCandidate` 类里补回了对应的 5 个 SQLAlchemy 字段定义。 |

### 核心认知与总结

1. **Vue 事件穿透的终极防线**：当遇到“点击 A 却触发了 B 的事件”时，如果 CSS 层级（`z-index`）难以排查，最稳妥的方案是**修改 DOM 树结构**，用正常的文档流（Flex/Grid）将点击区域物理隔开，而不是依赖 `absolute` 悬空浮在表面。
2. **UI 控件的视觉存在感**：不要盲目信任原生 `<input>` 的默认样式。在自定义组件中，给交互元素增加明显的边框、背景色和 `hover` 状态，能大幅降低用户的学习成本。
3. **跨功能模块的持久化验证**：在进行新的迁移（如 F07）前，务必跑一次全量测试（如 `test_orm_columns_match_db`）。它可以帮你提前发现历史代码（F05）是否因为误操作而丢失了字段定义，防止带病提交。

## 18. S2-F15：技术基线快照与全链路恢复（2026-11-02）

> 状态：✅ 完成
> 用途：把 Sprint 2 的最终状态固化为不可变的基线快照，作为 Sprint 3 (G1–G6) 起步的唯一事实地基。
> 核心难点：F05/F06/F07 的代码在工作区被意外回滚，导致后端启动崩溃、接口 404，需要逐个精准恢复。

### 一、事故背景与根因

**现象**：执行 `scripts/demo_sprint2.ps1` 时，后端在 `docker compose restart backend` 后直接崩溃（`基础连接已经关闭`），`healthz` 连不上。

**根因链条**（一次连锁回滚）：

在 F07 收尾之后，工作区被某次 git 操作（可能是 `git checkout` 或 `git reset`）错误回滚，导致 **F05/F06/F07 三个阶段的部分文件丢失**。具体表现：

| 阶段 | 丢失的文件 | 后端启动报错 |
|---|---|---|
| F05 | `backend/alembic/versions/s2_f05_rule_publish.py`、`backend/app/api/rule_candidates.py` 的 `/publish` 端点、`backend/app/services/rule_evolution_service.py` 的 3 个异常类 + `publish_rule_candidate` 方法、`backend/app/api/schemas.py` 的 2 个 Pydantic 模型 | `ImportError: cannot import name 'RulePublishConflictError'` |
| F06 | `backend/app/api/knowledge_gaps.py`、`backend/app/services/knowledge_gap_service.py`、`backend/app/api/__init__.py` 的 `knowledge_gaps` 路由注册、`backend/app/api/schemas.py` 的 4 个 Pydantic 模型 | `ImportError: cannot import name 'KnowledgeGapListResponse'` |
| F07 | `backend/app/persistence/models.py` 的 `RuleCandidate` 5 个 F05 发布字段 | `AssertionError: 列级不一致: {'rule_candidates': {'db_only': ['publish_*', ...]}}` |

**关键观察**：`git status` 里这些文件**没有显示 `D`（deleted）状态**，而是直接"不存在"。原因是工作区文件被物理删除，但 git index 里仍然保留了 F05/F06/F07 commit 的记录，所以 `git status` 看不到删除动作。

### 二、排查思路（逐步缩小范围）

**第 1 步：从报错位置反向追踪缺失模块**

先后端日志出现 `ImportError: cannot import name 'PublishRuleRequest' from 'app.api.schemas'`。立刻能定位到 `schemas.py` 缺少 F05 的 Pydantic 模型。

此时**不能直接 `git checkout 53878c7 -- schemas.py`**，因为 `schemas.py` 同时承载 F05 / F06 / F07 三个阶段的模型（共 10+ 个），直接 checkout 会丢失 F06/F07 的模型定义，导致新的 ImportError。

**第 2 步：从 commit 历史找回原始定义**

用 `git show 53878c7:backend/app/api/schemas.py | Select-String "PublishRuleRequest" -Context 2,10` 找到 F05 commit 里的原始模型定义，然后**手工追加到当前工作区的 `schemas.py` 末尾**（不能整体覆盖）。

**第 3 步：同一方法检查其他文件**

依次检查 `rule_candidates.py` / `rule_evolution_service.py` / `knowledge_gaps.py` / `knowledge_gap_service.py` / `api/__init__.py`：

```powershell
git show 53878c7:backend/app/api/rule_candidates.py | Select-String "publish"
git show 03ddf34:backend/app/api/knowledge_gaps.py | Select-String "router"
```

发现 rule_candidates.py / knowledge_gaps.py / api/__init__.py 在 commit 里是完整的，但工作区不存在（或缺少关键端点），于是用 `git checkout <commit> -- <path>` 直接恢复（这些文件 F05/F06 commit 之后没被 F07 改过，所以可以整体恢复）。

**第 4 步：反复重启后端验证**

每恢复一批文件，就 `docker compose restart backend` + healthz 检查。每次报错位置会往上推移（从 rule_candidates.py → rule_evolution_service.py → knowledge_gaps.py），直到 healthz 返回 status: ok。

**第 5 步：跑全量测试**

`docker compose exec backend uv run pytest -m "not integration" -v`。发现 test_orm_columns_match_db 挂，说明 models.py 的 RuleCandidate 也丢了 F05 字段，手工补回 5 个 publish_* 字段。

### 三、具体恢复方法（可复用清单）

#### A. 从 commit 恢复整个文件

```powershell
# F05 的文件（F05 commit 之后没被 F07 改过，安全）
git checkout 53878c7 -- backend/app/api/rule_candidates.py
git checkout 53878c7 -- backend/app/services/rule_evolution_service.py
git checkout 53878c7 -- backend/alembic/versions/s2_f05_rule_publish.py

# F06 的文件（F06 commit 之后没被 F07 改过，安全）
git checkout 03ddf34 -- backend/app/api/knowledge_gaps.py
git checkout 03ddf34 -- backend/app/services/knowledge_gap_service.py
git checkout 03ddf34 -- backend/app/api/__init__.py
```

注意：`api/__init__.py` 必须先确认 F06 之后没被 F07 改过：

```powershell
git log --oneline 03ddf34..HEAD -- backend/app/api/__init__.py
```

无输出才能 checkout。

#### B. 手工补回共享文件的模型定义

`schemas.py` 同时被 F05/F06/F07 改过，不能用 git checkout（会丢后续改动），必须在文件末尾手工追加。

**F05 部分（从 53878c7 提取）：**

```python
class PublishRuleRequest(BaseModel):
    idempotency_key: str = Field(..., min_length=8, max_length=128)

class RulePublishResponse(BaseModel):
    candidate_id: str
    publish_status: Literal["publishing", "published", "publish_failed"]
    published_version: str | None = None
    publish_error: str | None = None
    idempotent_replay: bool = False
    benchmark_regression: dict[str, Any] | None = None
```

**F06 部分（从 03ddf34 提取）：**

```python
class KnowledgeGapItem(BaseModel):
    gap_id: str
    title: str
    description: str | None = None
    status: str
    case_id: str | None = None
    task_id: str | None = None
    defect_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    resolved_at: str | None = None
    resolved_by: int | None = None

class KnowledgeGapDetail(KnowledgeGapItem):
    source_feedback: KnowledgeGapFeedbackRef | None = None
    all_feedbacks: list[KnowledgeGapFeedbackRef] = Field(default_factory=list)

class KnowledgeGapListResponse(BaseModel):
    total: int
    items: list[KnowledgeGapItem]
    limit: int
    offset: int

class UpdateGapStatusRequest(BaseModel):
    status: Literal["open", "in_progress", "resolved", "wontfix"]
    resolved_by: int | None = None
```

**F07 部分（models.py 的 RuleCandidate 5 字段 + 3 约束）：**

```python
# RuleCandidate 类内
publish_status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="not_published")
published_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
published_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
publish_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
publish_idempotency_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

# __table_args__ 内
CheckConstraint("publish_status IN ('not_published','publishing','published','publish_failed')", name="check_rule_candidates_publish_status"),
Index("idx_rule_candidates_publish_status", "publish_status"),
Index("uniq_rule_candidates_publish_idempotency_key", "publish_idempotency_key", unique=True, postgresql_where=text("publish_idempotency_key IS NOT NULL")),
```

### 四、PowerShell 特殊坑点

#### A. > 重定向写 UTF-16 LE

`git show ... > file` 在 PowerShell 5.1 中默认写 UTF-16 LE with BOM，Python 读取时崩溃：

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

修复：改用 `git checkout <commit> -- <path>`，从 git 对象库直接写工作区（UTF-8 无 BOM）。

#### B. 中文乱码导致语法错误

PowerShell 5.1 用系统编码（GBK）解码 UTF-8 脚本，脚本里 `Print-Step "阶段 1/3"` 变成 `Print-Step "闃舵 1/3"`，解析器报"意外的标记"。

修复：脚本里所有中文字符串改为英文（`Print-Step "Phase 1/3"`），或者脚本文件保存为 UTF-8 with BOM（但这也可能引起其他问题，最稳的是纯英文）。

#### C. Invoke-WebRequest 安全警告

每次 POST 都弹 `安全警告: 脚本执行风险`，需手动按 Y。

修复：加 `-UseBasicParsing` 参数，跳过 HTML 解析引擎。

#### D. Invoke-RestMethod 错误信息被吞

404 / 422 时只返回 `远程服务器返回错误: (404) 未找到`，看不到后端返回的 detail。

修复：改用 Invoke-WebRequest + `$_.Exception.Response.GetResponseStream()` 提取错误 body：

```powershell
try {
    $resp = Invoke-WebRequest -Uri "..." -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
} catch {
    $stream = $_.Exception.Response.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($stream)
    $errorBody = $reader.ReadToEnd()
    throw "HTTP $($_.Exception.Response.StatusCode.value__): $errorBody"
}
```

#### E. alembic current 的 stdout 噪声

`alembic current` 会把 `INFO [alembic.runtime.migration] ...` 日志打到 stdout（不是 stderr），导致 `2>$null` 屏蔽不掉，Out-String 后的内容无法被正则精准匹配。

修复：直接查 DB 的 alembic_version 表，绕开命令输出：

```python
from sqlalchemy import text
from app.persistence.db import get_db_session
with get_db_session() as db:
    row = db.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    print(row[0] if row else "unknown")
```

### 五、核心认知与总结

1. **工作区丢失代码的排查方法论**
   - 先看 git status——如果没显示 D，说明文件被"物理删除"，但 git 索引里还有 commit 记录
   - 用 `git show <commit>:<path>` 确认 commit 里的原始内容
   - 用 `git log <commit>..HEAD -- <path>` 确认该文件后续是否被修改过，决定能否 git checkout
   - 共享文件（如 schemas.py）必须手工追加，不能整体覆盖

2. **PowerShell 5.1 的 5 大坑**
   > 重定向写 UTF-16 LE、中文乱码、Invoke-WebRequest 安全警告、Invoke-RestMethod 吞错误 body、alembic current 的 stdout 噪声。

3. **Demo 脚本必须自动清理**
   否则每次跑都会污染 /data/rules/ 和 DB，导致 test_rules_loaded_exactly_ten 等测试失败。清理逻辑要放在 try/finally 或主流程末尾，确保异常时也清理。

4. **静态快照的三个必填字段**
   - Alembic Version 必须是具体的 revision 名（如 s2_f07_case_benchmark），不能是 unknown 或 `<base>`
   - Database Type 必须是 postgresql / mysql / sqlite，不能带 +driver 后缀
   - Test Baseline 必须是"数字 + passed + 数字 + skipped"的汇总行，不能是单条 tests/test_xxx PASSED [ 0%]

5. **跨阶段恢复的依赖顺序**
   从 F05 开始，每恢复一层就重启后端 + healthz 验证，再进下一层。一次性恢复所有文件反而容易引入新错误（比如把 F07 的 schemas.py 改动覆盖掉）。

## 19. S2-F15-Fix：CI/CD 流水线环境适配与排障（2026-11-02）

> 状态：✅ 完成
> 核心难点：本地开发环境（Docker Compose）与 CI 环境（GitHub Actions Runner）在**网络连通、目录挂载、数据状态**上的巨大差异。

### 一、坑点与修复全过程

#### 坑 1：缺少 PostgreSQL 服务 & 环境变量

**现象**：CI 的 `backend` job 在收集测试阶段直接崩溃，报 `ImportError` 或 `ValidationError`。

**根因**：GitHub Runner 默认没有启动数据库。`Settings` 里的必填字段（`secret_key` / `database_url` / `data_dir` / `llm_api_key`）也全为空。

**解决**：在 `ci.yml` 中引入 `services.postgres` 容器，并补全所有 `env` 变量。

#### 坑 2：数据库主机名校验拦截了 CI 环境

**现象**：CI 尝试用 `localhost` 连数据库，被 `config.py` 拦截报错 `DATABASE_URL 在容器内部不能使用 localhost`。

**根因**：本地开发时，容器间互信用 `postgres` 服务名；但 CI Runner 是宿主机环境，访问映射端口必须用 `localhost`。

**解决**：把 `config.py` 的 `check_db_host` 校验器改为**仅针对 `APP_ENV == "prod"` 生效**，`test` 和 `dev` 环境放行 `localhost`。

#### 坑 3：硬编码绝对路径 `/data` 导致文件找不到 / 无权限

**现象**：`test_benchmark.py` 报 `FileNotFoundError: /data/cases/...`；`test_demo.py` 报 `PermissionError: /out`。

**根因**：项目历史悠久，大量测试代码（尤其是 Sprint 0/1 时期的）硬编码了容器内的绝对路径（`/data` 和 `/out`）。但 CI 里的项目在 `/home/runner/work/...` 下。

**解决**：在 CI 的 `steps` 中加一步 `cp -r`：

```yaml
      - name: Create symlinks for CI compatibility
        run: |
          sudo mkdir -p /data /out
          sudo chmod -R 777 /data /out
          sudo cp -r $(pwd)/data/* /data/ 2>/dev/null || true
          sudo cp -r $(pwd)/out/* /out/ 2>/dev/null || true
```

#### 坑 4：Embedding 维度不匹配

**现象**：`test_rag_seed.py` 报 `RuntimeError: 后端 fake 期望 1536 维，DB 当前 1024 维`。

**根因**：数据库在迁移脚本 `c1_embedding_dim_1024` 里写死了 `vector(1024)`（为了适配智谱），但 CI 默认的 `RAG_EMBEDDING_BACKEND` 是 `fake`（1536 维）。

**解决**：在 `ci.yml` 的 `env` 里显式指定 `RAG_EMBEDDING_BACKEND: "zhipu"`，与数据库维度对齐。

#### 坑 5：测试断言过度依赖"全库为空"的历史状态

**现象**：`test_rule_evolution_list_candidates` 断言 `len(accepted) == 0` 失败，实际查出 1 条 Sprint 1 遗留的历史数据。

**根因**：测试用了独立事务的 `pg_session`，但数据库本身不会因为测试回滚就抹掉历史数据。

**解决**：移除 `assert len(empty) == 0` 这种脆弱断言，改为 `assert out.rule_candidate_ref not in accepted_ids`（只验证刚创建的候选不在里面即可）。

### 二、核心认知与总结

1. **测试用例必须是"自封闭"的**：绝对不能假设"数据库里除了我造的数据没有其他数据"。查列表时，要么通过 `candidate_id` 精确匹配，要么只断言增量。
2. **`config.py` 的校验逻辑必须区分环境**：任何"禁止 localhost"的硬性拦截，都必须放到 `APP_ENV == "prod"` 分支里。否则会卡死本地和 CI。
3. **隔离"容器绝对路径"**：长期项目里硬编码的 `/data` 很难一次性全改，在 CI 环境用 `sudo cp -r` 做兼容是最快的工程手段。

## 20. S2-Git-Fix：Git 历史分叉治理与分支清理（2026-11-02）

> 状态：✅ 完成
> 核心难点：处理因项目早期初始化不一致导致的 `entirely different commit histories` 问题，在保留本地完整代码的前提下，安全地将远端 `main` 覆盖为最新成果。

### 一、坑点与修复全过程

#### 坑 1：GitHub 无法创建 PR（提示历史分叉）

**现象**：在 GitHub 上尝试创建 Pull Request 合并 `feature/sprint1-core-review` 到 `main` 时，页面提示 `There isn't anything to compare`，并指出两个分支是 `entirely different commit histories`。

**根因**：本地仓库的初始根节点与远端 `main` 完全不同。可能是因为早期在 GitHub 网页创建了初始仓库，而本地是重新 `git init` 拉起了一套全新历史。后续开发（Sprint 0 ~ 2）全在本地进行，导致这两条时间线毫无交集。

**解决**：
1. 既然本地分支包含了从 Sprint 0 到 Sprint 2 的所有正确代码，且远端 `main` 已经过时。直接放弃传统的 PR 合并。
2. 在本地执行 `git checkout -B main`，将本地 `main` 分支指针强制重置到当前最新的 commit。
3. 执行 `git push -f origin main`，强制覆盖远端 `main` 的历史，让两条时间线彻底对齐。

#### 坑 2：本地遗留的“化石分支”过多

**现象**：执行 `git branch -a` 时，发现本地除了 `main` 之外，还残留着 `develop`、`feature/schema-a-flat-mounts`、`feature/schema-b-symlink`、`sprint0/base` 等一堆废弃分支。

**根因**：历史迭代过程中，各种实验性分支没有及时清理。

**解决**：在确认这些分支的代码都已合入当前主干后，执行 `git branch -D develop feature/schema-a-flat-mounts feature/schema-b-symlink sprint0/base`，强制清理。

### 二、核心认知与总结

1. **Git 历史是可以重塑的**：遇到没有共同祖先的分支（`entirely different commit histories`），只要确定其中一条包含了所有最终代码，直接用 `git push -f` 覆盖另一条是最快、最安全的做法，不要尝试去“合并”（大概率会产生海量冲突）。
2. **定期清理分支**：保持 `git branch -a` 输出只有 `main` 和一个当前开发分支，能极大降低心智负担。
3. **主干分支对齐的意义**：对于交付级项目，`main` 分支必须包含最完整、CI 全绿的所有代码。强制覆盖后，所有后续 CI 和 PR 都将基于这个正确的基线运行。

## 21. 附：版本与适用范围

- 前端：Vue 3 + Vite 5 + TypeScript
- 后端：FastAPI + Pydantic + uv
- 基础设施：Docker Compose
- 适用范围：Sprint 2 开发联调阶段（S2-D / S2-E / S2-G / S2-H 为主）
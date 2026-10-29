# Sprint 2 开发与调试问题记录（Debugging Notes）

> **用途**：沉淀 Sprint 2 开发过程中遇到的典型 Bug、环境问题及排查思路，供后续开发、联调及运维部署参考。
> **归档日期**：2026-10-29

---

## 目录
1. [Vite HMR 失效 / 前端代码更新不生效](#1-vite-hmr-失效--前端代码更新不生效)
2. [API 请求返回 HTML（被 Vite 路由拦截）](#2-api-请求返回-html被-vite-路由拦截)
3. [组件数据不响应 / 样式不更新](#3-组件数据不响应--样式不更新)
4. [后端 Pydantic 422 校验报错（Extra inputs are not permitted）](#4-后端-pydantic-422-校验报错extra-inputs-are-not-permitted)
5. [指标数据缺失与 UI 组件消失](#5-指标数据缺失与-ui-组件消失)
6. [提交反馈后"无反应"的错觉](#6-提交反馈后无反应的错觉)
7. [开发者调试利器：全链路 Console 日志](#7-开发者调试利器全链路-console-日志)

---

## 1. Vite HMR 失效 / 前端代码更新不生效

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

## 2. API 请求返回 HTML（被 Vite 路由拦截）

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

## 3. 组件数据不响应 / 样式不更新

**现象 A：徽标颜色点完不变化**
点击「专家确认」按钮，后端返回 200，但页面徽标依然是灰色的「待判定」。

**根因**：`ExpertDecisionBadge.vue` 中使用了 `const badge = DECISION_MAP[props.decision]`，这是非响应式的静态对象，props 改变时不会重新计算。

**解决**：使用 `computed` 包裹。
```typescript
const badge = computed(() => DECISION_MAP[props.decision] ?? defaultBadge);
```

**现象 B：外部列表状态不同步**
内部面板点击确认后，外面折叠状态的徽标没有变绿。

**根因**：Vue 3 对于数组内部对象的替换，有时响应式 Proxy 不会强触发 v-for 更新。

**解决**：在父组件 `TaskDetailPage.vue` 更新 defects 数组时，使用 splice 替换，而不是直接赋值索引。
```typescript
defects.value.splice(idx, 1, updatedDefect);
```

---

## 4. 后端 Pydantic 422 校验报错（Extra inputs are not permitted）

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

## 5. 指标数据缺失与 UI 组件消失

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

## 6. 提交反馈后"无反应"的错觉

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

## 7. 开发者调试利器：全链路 Console 日志

为了快速定位前后端交互问题，S2-D/E 阶段在关键路径埋了 DEV 环境的日志：
- `http.ts`：打印 `[API 请求开始]`、`[API 成功]`、`[API 失败]`，并附带耗时和 payload。
- `ExpertDecisionPanel.vue`：打印 `[专家判定]` 按钮点击和接口回包。
- `TaskDetailPage.vue`：打印 `[页面状态更新]` 数组渲染。

**建议**：在后续的 S2-G（规则候选审核）、S2-H（导出与批量）中，继续保持这种「关键请求+关键状态变更」的日志规范，利用浏览器的 Console 面板过滤 `[API` 或 `[State` 即可迅速排障。

---

> **附：版本与适用范围**
> - 前端：Vue 3 + Vite + TypeScript
> - 后端：FastAPI + Pydantic
> - 适用范围：Sprint 2 开发联调阶段（S2-D / S2-E 为主）

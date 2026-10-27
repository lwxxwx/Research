# H 阶段 API 契约核对文档

> 本文档基于 `backend/app/api/schemas.py` 与 `backend/app/services/task_service.py` 的真实源码，逐个核对 H 前端需要的 6 个类型字段。
> 核对完成前，不写 `types/*.ts`。

## 文档结构（共 12 章）

| 章 | 标题 | 内容 |
|---|---|---|
| 1 | 核对来源与范围 | 2 个源码文件 + 6 个类型 |
| 2 | `CaseItem` / `CaseListResponse` 对照 | 6 + 3 字段 |
| 3 | `TaskDetailResponse` 对照 | 7 字段 + 2 个 ⚠️ |
| 4 | `TaskListItem` / `TaskListResponse` 对照 | 6 + 3 字段 + 1 个 ⚠️ |
| 5 | `Defect` 对照 | 16 字段 |
| 6 | `Evidence` 对照 | 6 字段 |
| 7 | `FeedbackCreateAPI` 对照 | 8 字段 + 3 个嵌套类型 |
| 8 | `FeedbackResponse` 对照 | 11 字段 |
| 9 | 汇总表（6 类型 × 字段数） | 0 ❌ / 3 ⚠️ |
| 10 | 3 个 ⚠️ 的 H 前端处理 | 类型窄化 + 注释 |
| 11 | `report.defects` vs `GET /defects` 关键差异 | `DefectFromReport` / `DefectFromAPI` 分开 |
| 12 | 核对结论 + 下一步 | 3 件确认事项 |

---

## 第 1 章：核对来源与范围

### 1.1 核对来源（真实源码）

| 来源 | 内容 |
|---|---|
| `backend/app/api/schemas.py` | 8 个 Pydantic 模型：`HealthResponse` / `CreateTaskRequest` / `CreateTaskResponse` / `TaskDetailResponse` / `TaskListItem` / `TaskListResponse` / `DefectsResponse` / `CaseItem` / `CaseListResponse` / `FeedbackCreateAPI` / `FeedbackResponse` |
| `backend/app/services/task_service.py` | 7 个返回 dict 函数：`create_task` / `get_task` / `list_tasks` / `list_tasks_filtered` / `get_task_defects` / `submit_task_feedback` / `list_cases` |

**说明**：`list_tasks`（旧函数）**不在 H 前端使用**（用 `list_tasks_filtered`），不核对。

### 1.2 核对范围（6 个类型）

| # | 类型 | 对应端点 | 核对章 |
|---|---|---|---|
| 1 | `CaseItem` + `CaseListResponse` | `GET /api/v1/cases` | 第 2 章 |
| 2 | `TaskDetailResponse` | `GET /api/v1/review/tasks/{task_id}` | 第 3 章 |
| 3 | `TaskListItem` + `TaskListResponse` | `GET /api/v1/review/tasks` | 第 4 章 |
| 4 | `Defect`（`get_task_defects` 返回） | `GET /api/v1/review/tasks/{task_id}/defects` | 第 5 章 |
| 5 | `Evidence`（`Defect.evidence[]` 内部） | 同上 | 第 6 章 |
| 6 | `FeedbackCreateAPI` + `FeedbackResponse` | `POST /api/v1/feedbacks` | 第 7–8 章 |

### 1.3 核对方式

**逐字段对照**，每行 6 列：

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|

- `✅`：字段名 + 类型 + 可空性完全一致
- `⚠️`：字段存在但类型需窄化 / 语义需澄清（不阻塞开发，加注释）
- `❌`：字段缺失 / 类型冲突（阻塞开发，必须先解决）

**核对完成前，不写 `types/*.ts`。**

### 1.4 本章结论

- 核对来源：2 个真实源码文件已确认
- 核对范围：6 个类型（对应 6 个端点）
- 核对方式：6 列对照 + ✅ / ⚠️ / ❌ 标记


## 第 2 章：`CaseItem` / `CaseListResponse` 对照

### 2.1 后端定义

**Pydantic**（`schemas.py::CaseItem`）：

```python
class CaseItem(BaseModel):
    """GET /api/v1/cases 列表项（透出 list_cases 的 item）。"""
    case_id: str
    case_class: str
    title: str | None = None
    description: str | None = None
    calibration_status: str | None = None
    tags: list[str] = Field(default_factory=list)
```

**service 返回**（`task_service.list_cases()` 的 `items.append(...)`）：

```python
{
    "case_id": case_id,           # str（strip 后）
    "case_class": case_class,     # str（"A" / "B"）
    "title": raw.get("title"),    # Any（info.yaml 原值，可能 None / str / 其他）
    "description": raw.get("description"),  # Any
    "calibration_status": raw.get("calibration_status"),  # Any
    "tags": _normalize_tags(raw.get("tags")),  # list[str]（已归一化）
}
```

**service 的校验逻辑**（`list_cases`）：

- **`case_id`**：必须是 `str` 且 `strip()` 非空，否则跳过 + 记 errors
- **`case_class`**：先 `isinstance(str)` → 再 `strip()` 非空 → 再 `in ("A", "B")`，任一不满足跳过 + 记 errors
- **`title` / `description` / `calibration_status`**：直接 `raw.get()`，**不做类型校验**（info.yaml 写非 str 会 Pydantic 校验失败 → 500，这是后端 G3 的边界，不是 H 前端问题）
- **`tags`**：`_normalize_tags` 统一为 `list[str]`

### 2.2 `CaseItem` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `case_id` | `str` | 否 | `case_id` | `string` | ✅ |
| `case_class` | `str` | 否 | `case_class` | `"A" \| "B"` | ✅（service 只返回 A/B，窄化） |
| `title` | `str \| None` | 是 | `title` | `string \| null` | ✅ |
| `description` | `str \| None` | 是 | `description` | `string \| null` | ✅ |
| `calibration_status` | `str \| None` | 是 | `calibration_status` | `string \| null` | ✅ |
| `tags` | `list[str]` | 否（默认 `[]`） | `tags` | `string[]` | ✅ |

**核对结果**：6/6 ✅，0 ⚠️，0 ❌。

### 2.3 `CaseListResponse` 字段对照

**Pydantic**（`schemas.py::CaseListResponse`）：

```python
class CaseListResponse(BaseModel):
    """GET /api/v1/cases 响应体。"""
    total: int
    items: list[CaseItem]
    errors: list[str]
```

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `total` | `int` | 否 | `total` | `number` | ✅ |
| `items` | `list[CaseItem]` | 否 | `items` | `CaseItem[]` | ✅ |
| `errors` | `list[str]` | 否 | `errors` | `string[]` | ✅ |

**核对结果**：3/3 ✅，0 ⚠️，0 ❌。

### 2.4 补充说明

**说明 1：`title` / `description` / `calibration_status` 的实际类型是 `Any`**

`list_cases` 里 `raw.get("title")` 从 YAML 读，可能是 `str` / `int` / `None`。Pydantic `CaseItem` 声明 `str | None`，如果 info.yaml 写了 `title: 123`（int），Pydantic 会 **500 校验失败**。

**H 前端处理**：TS 按 `string | null`，**不处理后端 500 场景**（info.yaml 资产错误是后端 G3 的边界，Sprint 1 已冻结）。

**说明 2：`case_class` 的 `"A" | "B"` 窄化是安全的**

`list_cases` 显式校验 `case_class not in ("A", "B")` → 跳过 + errors。所以 `CaseItem.case_class` **只会是 `"A"` 或 `"B"`**，窄化安全。

**说明 3：`tags` 已归一化，不会是 `null`**

`_normalize_tags` 对 `None` / `str` / `list` / `tuple` 都返回 `list[str]`（`None` → `[]`）。所以 `CaseItem.tags` **一定是 `string[]`**，不会是 `null`。

**说明 4：`errors` 是"坏 Case 说明"，不是请求错误**

`CaseListResponse.errors` 是 `list_cases` 扫描时跳过的坏 Case 说明（如 `"case_bad: info.yaml 解析失败: ..."`），**不是 HTTP 错误**。H 前端：
- 正常情况：`errors = []`，显示 10 个 Case
- 有坏 Case：`errors` 非空，**仍显示有效 Case + 顶部提示"有 N 个 Case 资产异常"**（可选，H 阶段可简化只显示有效 Case）

### 2.5 本章 TS 类型（基于核对结果，Day 1 上午写入 `types/case.ts`）

```typescript
// frontend/src/types/case.ts

/** GET /api/v1/cases 列表项（对齐 CaseItem） */
export interface CaseItem {
  case_id: string;
  case_class: "A" | "B";
  title: string | null;
  description: string | null;
  calibration_status: string | null;
  tags: string[];
}

/** GET /api/v1/cases 响应体（对齐 CaseListResponse） */
export interface CaseListResponse {
  total: number;
  items: CaseItem[];
  errors: string[];
}
```

**注意**：

- `case_class` 窄化为 `"A" | "B"`（service 保证）
- `tags` 是 `string[]`（service 已归一化）
- `title` / `description` / `calibration_status` 是 `string | null`（Pydantic 声明，info.yaml 非 str 会 500，H 前端不处理）
- `errors` 是坏 Case 说明，不是 HTTP 错误

### 2.6 本章结论

| 项 | 结果 |
|---|---|
| `CaseItem` 字段 | 6/6 ✅，0 ⚠️，0 ❌ |
| `CaseListResponse` 字段 | 3/3 ✅，0 ⚠️，0 ❌ |
| 类型窄化 | `case_class: "A" \| "B"`（安全） |
| 后端边界 | `title` 等非 str 会 500（H 前端不处理） |
| TS 类型 | 已给（`types/case.ts`） |

## 第 3 章：`TaskDetailResponse` 对照

### 3.1 后端定义

**Pydantic**（`schemas.py::TaskDetailResponse`）：

```python
class TaskDetailResponse(BaseModel):
    """GET /api/v1/review/tasks/{task_id} 响应体（透出 get_task 的返回）。"""
    task_id: str
    status: Literal["done", "failed", "pending"]
    case_id: str | None
    report: dict[str, Any] | None
    errors: list[str]
    created_at: str | None
    defects_count: int
```

**service 返回**（task_service.get_task() 的 return {...}）：


```python
{
    "task_id": rr.task_id,           # str
    "status": status,                # "done" / "failed"（不会返回 "pending"）
    "case_id": case_id,              # str | None
    "report": report_json,           # dict（review_output_json，至少 {} 但不会 None）
    "errors": errors,                # list[str]
    "created_at": rr.created_at.isoformat() if rr.created_at else None,  # str | None
    "defects_count": len(defects),   # int
}
```

**service 的状态判定逻辑**（`get_task`）：

- `review_result` 不存在 → 返回 `None` → API 层转 **404**
- `defects` 非空 → `status = "done"`
- `defects` 为空 + `errors` 非空 → `status = "failed"`
- `defects` 为空 + `errors` 为空 → `status = "done"`（空报告）

**关键**：`status` **只会是 `"done"` 或 `"failed"`**，**不会返回 `"pending"`**。`"pending"` 场景对应 `get_task` 返回 `None`，API 层转 404。

### 3.2 `TaskDetailResponse` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `task_id` | `str` | 否 | `task_id` | `string` | ✅ |
| `status` | `Literal["done", "failed", "pending"]` | 否 | `status` | `"done" \| "failed"` | ⚠️ service 实际只返回 `done` / `failed`，`pending` 不会出现 |
| `case_id` | `str \| None` | 是 | `case_id` | `string \| null` | ✅ |
| `report` | `dict[str, Any] \| None` | 是（Pydantic），实际非 None | `report` | `ReviewReport \| Record<string, never> \| null` | ⚠️ 实际至少 `{}`，不会 `null` |
| `errors` | `list[str]` | 否 | `errors` | `string[]` | ✅ |
| `created_at` | `str \| None` | 是 | `created_at` | `string \| null` | ✅ |
| `defects_count` | `int` | 否 | `defects_count` | `number` | ✅ |

**核对结果**：5/7 ✅，2/7 ⚠️（都是"语义澄清"，不是字段缺失），0 ❌。


### 3.3 ⚠️ 偏差 1：`status` **不会返回 `"pending"`**

`get_task` 的逻辑：

```python
if rr is None:
    return None        # ← task 不存在时返回 None，不是 {status: "pending"}
# ...
if defects:
    status = TASK_STATUS_DONE      # "done"
elif errors:
    status = TASK_STATUS_FAILED    # "failed"
else:
    status = TASK_STATUS_DONE      # "done"
```

**`status` 只有 `done` / `failed`**。`TaskDetailResponse.status` 的 `Literal["done", "failed", "pending"]` 里的 `pending` **实际不会出现**。

**H 前端处理**：

- TS 类型：`status: "done" | "failed"`（**去掉 `pending`**，或保留但注释"实际不返回"）
- 逻辑：`status === "done"` 显示报告；`status === "failed"` 显示 errors

**`pending` 怎么表示？** `get_task` 返回 `None` → API 层 `HTTPException(404)`。**"pending" 不存在，只有 404 或 done/failed**。

### 3.4 ⚠️ 偏差 2：`report` **实际非 None**（Pydantic 声明 `dict | None`）

`get_task` 里 `report_json = rr.review_output_json or {}`，**至少是 `{}`**（空 dict），不会是 `None`。

但 `TaskDetailResponse.report: dict[str, Any] | None` 允许 `None`——这是 Pydantic 的"防御性声明"，实际 `get_task` 不会返回 `None`。

**H 前端处理**：

- TS 类型：`report: ReviewReport | Record<string, never> | null`（兼容 `{}` 和 `null`）
- 逻辑：`report === null || Object.keys(report).length === 0` → 显示"报告为空"

### 3.5 `report` 的内部结构（H 前端需定义 `ReviewReport` TS 类型）

`report` 是 `review_output_json`（`ReviewReport.model_dump(mode="json")`），结构来自 `app/schemas/report.py::ReviewReport`：

```python
class ReviewReport:
    task_id: str
    report_id: str
    defects: list[Defect]           # ← 注意：report.defects 是 Defect 列表
    summary: ReviewReportSummary
    errors: list[str]
```

**`ReviewReportSummary`**：

```python
class ReviewReportSummary:
    evidence_coverage: float | None
    review_status_distribution: ReviewStatusDistribution
```

**`ReviewStatusDistribution`**：

```python
class ReviewStatusDistribution:
    AI_CONFIRMED: int
    NEED_EXPERT_REVIEW: int
    LOW_CONFIDENCE: int
```

**H 前端 `ReviewReport` TS 类型**：

```typescript
export interface ReviewReport {
  task_id: string;
  report_id: string;
  defects: DefectFromReport[];   // 与 GET /defects 的 items 不同（见第 11 章）
  summary: ReviewReportSummary;
  errors: string[];
}

export interface ReviewReportSummary {
  evidence_coverage: number | null;
  review_status_distribution: {
    AI_CONFIRMED: number;
    NEED_EXPERT_REVIEW: number;
    LOW_CONFIDENCE: number;
  };
}
```

**⚠️ 重要**：`report.defects` 里的 `Defect` 是 `ReviewReport.model_dump()` 的快照，**只有 10 字段 + `review_status` / `origin` / `rule_id` / `expert_decision` / `synthesized`**，**没有** `expert_decision_updated_at` / `expert_decision_by`（G1 只给 `get_task_defects` 补了，没给 `report.defects` 补）。

**H 前端处理**：

- `report.defects` 用 `DefectFromReport`（15 字段，无 `expert_decision_updated_at` / `_by`）
- `GET /defects` 的 `items` 用 `DefectFromAPI`（16 字段，有 `expert_decision_updated_at` / `_by`）
- **两个类型分开定义**，不要混用（详见第 11 章）


### 3.6 本章结论

| 项 | 结果 |
|---|---|
| `TaskDetailResponse` 字段 | 5/7 ✅，2/7 ⚠️，0 ❌ |
| ⚠️ 1 | `status` 无 `"pending"`（service 只返回 `done` / `failed`） |
| ⚠️ 2 | `report` 实际非 `None`（至少 `{}`，判断空用 `Object.keys(report).length === 0`） |
| 新增类型 | `ReviewReport` + `ReviewReportSummary`（`report` 字段内部结构） |
| 关键发现 | `report.defects` 与 `GET /defects` 字段不同，需分 `DefectFromReport` / `DefectFromAPI`（详见第 11 章） |
| 后端边界 | `status="pending"` 不存在，task 不存在时 404 |


## 第 4 章：`TaskListItem` / `TaskListResponse` 对照

### 4.1 后端定义

**Pydantic**（`schemas.py::TaskListItem`）：

```python
class TaskListItem(BaseModel):
    """GET /api/v1/review/tasks 列表项（透出 list_tasks_filtered 的 item）。"""
    task_id: str
    case_id: str | None
    case_class: str | None
    defects_count: int
    review_status: str
    created_at: str | None
```

**Pydantic**（`schemas.py::TaskListResponse`）：


```python
class TaskListResponse(BaseModel):
    """GET /api/v1/review/tasks 响应体。"""
    total: int
    items: list[TaskListItem]
    errors: list[str]
```

**service 返回**（`task_service.list_tasks_filtered()` **的 **`filtered_items.append(...)`）：


```python
{
    "task_id": rr.task_id,           # str
    "case_id": case_id,              # str | None
    "case_class": item_case_class,   # "A" / "B" / None
    "defects_count": rr.total_defects,  # int
    "review_status": rr.review_status,  # str
    "created_at": rr.created_at.isoformat() if rr.created_at else None,  # str | None
}
```

**service 的语义**（`list_tasks_filtered`）：

- **`case_id`**：从 `SchematicCase` outerjoin 拿到，可能为 `None`（`schematic_case_id` 为 NULL 或关联 case 不存在）
- **`case_class`**：`_infer_case_class(case_id)` 推断，**只返回 `"A"` / `"B"` / `None`**
- **`defects_count`**：来自 `review_result.total_defects`（**Pred 数**，不是 GT 数，G 阶段决策点 5）
- **`review_status`**：来自 `review_result.review_status`（DB 存三态）
- **不含 `status` 字段**（G2 决策 1：status 过滤不实现）


### 4.2 `TaskListItem` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `task_id` | `str` | 否 | `task_id` | `string` | ✅ |
| `case_id` | `str \| None` | 是 | `case_id` | `string \| null` | ✅ |
| `case_class` | `str \| None` | 是 | `case_class` | `"A" \| "B" \| null` | ✅（service 只返回 A/B/None） |
| `defects_count` | `int` | 否 | `defects_count` | `number` | ✅ |
| `review_status` | `str` | 否 | `review_status` | `"AI_CONFIRMED" \| "NEED_EXPERT_REVIEW" \| "LOW_CONFIDENCE"` | ⚠️ Pydantic 声明 `str`，需窄化 |
| `created_at` | `str \| None` | 是 | `created_at` | `string \| null` | ✅ |

**核对结果**：5/6 ✅，1/6 ⚠️（`review_status` 需窄化），0 ❌。

### 4.3 ⚠️ `review_status` 是 `str`，不是三态联合

Pydantic `TaskListItem.review_status: str`（**不是 `Literal`**），所以**编译时不校验**。但 `rr.review_status` 来自 DB（`review_result.review_status`），**实际值**是 `AI_CONFIRMED` / `NEED_EXPERT_REVIEW` / `LOW_CONFIDENCE` 三态。

**H 前端处理**：

- TS 类型：`review_status: "AI_CONFIRMED" | "NEED_EXPERT_REVIEW" | "LOW_CONFIDENCE"`（窄化，便于徽标组件匹配）
- 如果后端返回未知值（DB 脏数据），徽标组件 fallback 到灰色"未知"

### 4.4 `TaskListResponse` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `total` | `int` | 否 | `total` | `number` | ✅ |
| `items` | `list[TaskListItem]` | 否 | `items` | `TaskListItem[]` | ✅ |
| `errors` | `list[str]` | 否 | `errors` | `string[]` | ✅ |

**核对结果**：3/3 ✅，0 ⚠️，0 ❌。

**⚠️ 注意**：`list_tasks_filtered` 的 `items` **不含 `status` 字段**（G2 决策 1）。H 前端列表页**不显示任务状态**（done/failed），只显示 `review_status`（三态质量）。这是**设计意图**，不是遗漏。

### 4.5 本章 TS 类型（基于核对结果，Day 1 上午写入 `types/task.ts`）

```typescript
// frontend/src/types/task.ts

/** 三态质量状态（与 review_result.review_status 对齐） */
export type ReviewStatus =
  | "AI_CONFIRMED"
  | "NEED_EXPERT_REVIEW"
  | "LOW_CONFIDENCE";

/** GET /api/v1/review/tasks 列表项（对齐 TaskListItem） */
export interface TaskListItem {
  task_id: string;
  case_id: string | null;
  case_class: "A" | "B" | null;
  defects_count: number;
  review_status: ReviewStatus;
  created_at: string | null;
}

/** GET /api/v1/review/tasks 响应体（对齐 TaskListResponse） */
export interface TaskListResponse {
  total: number;
  items: TaskListItem[];
  errors: string[];
}
```

**注意**：

- `review_status` 窄化为 `ReviewStatus`（DB 存三态，Pydantic 声明 `str` 未窄化，前端窄化后便于徽标组件匹配）
- `case_class` 窄化为 `"A" | "B" | null`（service 保证只返回这三种）
- **`items` 不含 `status` 字段**（G2 决策 1），列表页不显示 done/failed
- `defects_count` 是 **Pred 数**（系统报了几条），不是 GT 数（G 阶段决策点 5）

### 4.6 本章结论

| 项 | 结果 |
|---|---|
| `TaskListItem` 字段 | 5/6 ✅，1/6 ⚠️，0 ❌ |
| `TaskListResponse` 字段 | 3/3 ✅，0 ⚠️，0 ❌ |
| ⚠️ 1 | `review_status` 是 `str`，前端窄化为 `ReviewStatus` 三态联合 |
| 设计意图 | `items` 不含 `status` 字段（G2 决策 1），列表页不显示 done/failed |
| 语义提醒 | `defects_count` 是 Pred 数，不是 GT 数（G 阶段决策点 5） |
| TS 类型 | 已给（`types/task.ts`） |

## 第 5 章：`Defect` 对照（`GET /api/v1/review/tasks/{task_id}/defects`）

### 5.1 后端定义

**service 返回**（`task_service.get_task_defects()` 的 `return [...]`）：

```python
{
    "defect_id": d.defect_id,                                      # str
    "category": d.category,                                        # str
    "location": d.location,                                        # dict
    "component": d.component,                                      # str | None
    "net": d.net,                                                  # str | None
    "risk": d.risk,                                                # str（low/medium/high/critical）
    "evidence": d.evidence,                                        # list[dict]
    "root_cause": d.root_cause,                                    # str
    "suggestion": d.suggestion,                                    # str
    "confidence": d.confidence,                                    # float
    "review_status": d.review_status,                              # str（三态）
    "origin": d.origin,                                            # str（rule / ai_discovered / hybrid）
    "rule_id": d.rule_id,                                          # str | None
    "expert_decision": d.expert_decision,                          # str（四态）
    "expert_decision_updated_at": (...),                           # str | None（ISO 或 None）
    "expert_decision_by": d.expert_decision_by,                    # int | None
}
```

**API 层**（`schemas.py::DefectsResponse`）：

```python
class DefectsResponse(BaseModel):
    """GET /api/v1/review/tasks/{task_id}/defects 响应体。

    service.get_task_defects 返回 list[dict]，API 层用 list[dict] 透传，
    不重复定义 16 字段（避免与 service 层 schema 漂移）。
    """
    items: list[dict[str, Any]]
```

**说明**：API 层用 `list[dict[str, Any]]` 透传，**Pydantic 不校验 16 字段**（避免与 service 层 schema 漂移）。字段的权威来源是 `task_service.get_task_defects()` 返回的 dict。

**service 的语义**（`get_task_defects`）：

- 按 `task_id` 查 `review_result`（最新一条），不存在返回 `[]`
- 按 `review_result_id` 查 `review_defect`，按 `id` 升序
- `expert_decision_updated_at` 是 `datetime.isoformat()` 或 `None`
- `expert_decision_by` 是 `int` 或 `None`


### 5.2 `Defect` 字段对照（16 字段）

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `defect_id` | `str` | 否 | `defect_id` | `string` | ✅ |
| `category` | `str` | 否 | `category` | `string` | ✅ |
| `location` | `dict` | 否 | `location` | `Record<string, unknown>` | ✅ |
| `component` | `str \| None` | 是 | `component` | `string \| null` | ✅ |
| `net` | `str \| None` | 是 | `net` | `string \| null` | ✅ |
| `risk` | `str` | 否 | `risk` | `"low" \| "medium" \| "high" \| "critical"` | ✅（DB 存四态，窄化） |
| `evidence` | `list[dict]` | 否 | `evidence` | `Evidence[]` | ✅（详见第 6 章） |
| `root_cause` | `str` | 否 | `root_cause` | `string` | ✅ |
| `suggestion` | `str` | 否 | `suggestion` | `string` | ✅ |
| `confidence` | `float` | 否 | `confidence` | `number` | ✅ |
| `review_status` | `str` | 否 | `review_status` | `ReviewStatus`（三态） | ✅（窄化） |
| `origin` | `str` | 否 | `origin` | `"rule" \| "ai_discovered" \| "hybrid"` | ✅（窄化） |
| `rule_id` | `str \| None` | 是 | `rule_id` | `string \| null` | ✅ |
| `expert_decision` | `str` | 否 | `expert_decision` | `ExpertDecision`（四态） | ✅（窄化） |
| `expert_decision_updated_at` | `str \| None` | 是 | `expert_decision_updated_at` | `string \| null` | ✅ |
| `expert_decision_by` | `int \| None` | 是 | `expert_decision_by` | `number \| null` | ✅ |

**核对结果**：16/16 ✅，0 ⚠️，0 ❌。


### 5.3 窄化的 3 个枚举字段

`risk` / `review_status` / `origin` / `expert_decision` 的 DB 存储是 `String`，但实际值是固定枚举。前端窄化为联合类型，便于类型收窄和徽标组件匹配。

#### 5.3.1 `risk`（四态）

**DB 存储**：`String(16)`，默认 `'medium'`

**实际值**（来自 `app/rules/engine.py::RuleResult.severity` 和 `app/schemas/report.py::Defect.risk` 校验）：

```typescript
export type Risk = "low" | "medium" | "high" | "critical";
```

**H 前端用途**：

- `DefectCard` 的 `risk` 徽标（H 阶段文字徽标，Sprint 2 再加颜色）
- `TaskDetailPage` 的缺陷列表排序（可选）

#### 5.3.2 `review_status`（三态）

**DB 存储**：`String(32)`，默认 `'AI_CONFIRMED'`

**实际值**（来自 `app/schemas/report.py::ReviewStatus`）：

```typescript
export type ReviewStatus =
  | "AI_CONFIRMED"
  | "NEED_EXPERT_REVIEW"
  | "LOW_CONFIDENCE";
```

**H 前端用途**：

- `StatusBadge` 三色徽标（绿 / 黄 / 红）
- `TaskListPage` 的 `review_status` 筛选

**注意**：`review_status` 是 **AI 输出质量状态**，**不是专家确认状态**。专家确认状态是 `expert_decision`（见 5.3.4）。

#### 5.3.3 `origin`（三态）

**DB 存储**：`String(32)`，默认 `'rule'`

**实际值**（来自 `app/schemas/report.py::DiscoveryOrigin`）：

```typescript
export type DiscoveryOrigin = "rule" | "ai_discovered" | "hybrid";
```

**H 前端用途**：

- `DefectCard` 的 `origin` 标签（"规则发现" / "AI 发现" / "混合"）
- 帮助专家区分"规则已命中的确定性缺陷"和"AI 提出的候选缺陷"

#### 5.3.4 `expert_decision`（四态）

**DB 存储**：`String(32)`，默认 `'pending'`（前置条件 §3.2）

**实际值**（来自 `app/schemas/report.py::ExpertDecision`）：

```typescript
export type ExpertDecision =
  | "pending"
  | "confirmed"
  | "rejected"
  | "needs_verification";
```

**H 前端用途**：

- `ExpertDecisionBadge` 四态徽标（灰 / 绿 / 红 / 黄）
- **H 阶段只展示，不修改**（`POST /feedbacks` 不改 `expert_decision`，只记反馈历史；`expert_decision` 的更新是 S2 的 API）

**关键区别**：

| 字段 | 语义 | 谁能改 |
|---|---|---|
| `review_status` | AI 输出质量状态（自动判定） | 后端 Parser 自动判定 |
| `expert_decision` | 专家最终决定（人工判定） | **H 阶段只读**（S2 才有 API 改） |


### 5.4 `DefectsResponse` 字段对照

**Pydantic**（`schemas.py::DefectsResponse`）：

```python
class DefectsResponse(BaseModel):
    items: list[dict[str, Any]]
```


| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `items` | `list[dict[str, Any]]` | 否 | `items` | `DefectFromAPI[]` | ✅（Pydantic 透传，字段由 `get_task_defects` 保证） |

**核对结果**：1/1 ✅，0 ⚠️，0 ❌。

**注意**：`items` 的 Pydantic 类型是 `list[dict[str, Any]]`，**没有严格 schema 校验**。H 前端用 `DefectFromAPI[]` 窄化，**假设后端返回 16 字段**（由 `get_task_defects` 保证）。如果后端返回缺字段（不应发生），前端运行时 `undefined`，需在 `DefectCard` 里做防御（`defect.component ?? "-"` 等）。


### 5.5 本章 TS 类型（基于核对结果，Day 1 上午写入 `types/defect.ts`）

```typescript
// frontend/src/types/defect.ts

/** 风险等级（DB 存四态） */
export type Risk = "low" | "medium" | "high" | "critical";

/** 三态质量状态（与 review_result.review_status 对齐） */
export type ReviewStatus =
  | "AI_CONFIRMED"
  | "NEED_EXPERT_REVIEW"
  | "LOW_CONFIDENCE";

/** 发现来源（DB 存三态） */
export type DiscoveryOrigin = "rule" | "ai_discovered" | "hybrid";

/** 专家最终决定（DB 存四态） */
export type ExpertDecision =
  | "pending"
  | "confirmed"
  | "rejected"
  | "needs_verification";

/** 单条证据（详见第 6 章） */
export interface Evidence {
  type: "rag_ref" | "ir_ref";
  source?: string;
  section?: string;
  reason?: string;
  detail?: string;
  source_type?: string;
}

/** GET /api/v1/review/tasks/{task_id}/defects 列表项（对齐 get_task_defects 返回） */
export interface DefectFromAPI {
  defect_id: string;
  category: string;
  location: Record<string, unknown>;
  component: string | null;
  net: string | null;
  risk: Risk;
  evidence: Evidence[];
  root_cause: string;
  suggestion: string;
  confidence: number;
  review_status: ReviewStatus;
  origin: DiscoveryOrigin;
  rule_id: string | null;
  expert_decision: ExpertDecision;
  expert_decision_updated_at: string | null;
  expert_decision_by: number | null;
}

/** GET /api/v1/review/tasks/{task_id}/defects 响应体 */
export interface DefectsResponse {
  items: DefectFromAPI[];
}
```

**注意**：

- 4 个枚举字段（`risk` / `review_status` / `origin` / `expert_decision`）窄化为联合类型
- `DefectFromAPI` 是 **16 字段**（`get_task_defects` 保证），与 `DefectFromReport`（15 字段，第 11 章）分开
- `evidence` 用第 6 章的 `Evidence` 类型
- `location` 用 `Record<string, unknown>`（后端是 JSONB，结构不固定）
- `expert_decision_updated_at` / `expert_decision_by` 是 **H 阶段只读**的元信息（S2 才有 API 改 `expert_decision`）


### 5.6 本章结论

| 项 | 结果 |
|---|---|
| `Defect` 字段 | 16/16 ✅，0 ⚠️，0 ❌ |
| `DefectsResponse` 字段 | 1/1 ✅，0 ⚠️，0 ❌ |
| 枚举窄化 | `risk` / `review_status` / `origin` / `expert_decision` 全部窄化为联合类型 |
| 关键发现 | `DefectFromAPI`（16 字段）与 `DefectFromReport`（15 字段）必须分开定义（第 11 章） |
| H 阶段边界 | `expert_decision` 只读，`POST /feedbacks` 不改它 |
| TS 类型 | 已给（`types/defect.ts`） |
| 防御性处理 | `DefectCard` 里对可空字段做 `?? "-"` 防御（后端返回缺字段不应发生，但防御成本低） |


## 第 6 章：`Evidence` 对照

### 6.1 后端定义

**来源**：`app/schemas/report.py::Evidence`（Pydantic 模型），`Defect.evidence` 是 `list[Evidence]`，序列化后存 `review_defect.evidence`（JSONB）。

```python
class Evidence(BaseModel):
    """
    单条证据。type 决定哪些字段必填：

    - type="rag_ref"：source / section / reason 必填，各长度 >= 10
    - type="ir_ref" ：detail 必填，长度 >= 5

    source_type 为可选扩展（datasheet / reference_design / application_note 等），
    对应 app.rag.sources.DocumentType；不强制。
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["rag_ref", "ir_ref"]

    # rag_ref 专用（type=rag_ref 时必填）
    source: str | None = None
    section: str | None = None
    reason: str | None = None

    # ir_ref 专用（type=ir_ref 时必填）
    detail: str | None = None

    # 可选扩展
    source_type: str | None = None
```

**关键约束**（`Evidence._check_required_by_type`）：

- `type="rag_ref"`：`source` / `section` / `reason` **必填**（S1-B 修正后只校验非空，长度不足交 Parser 降级）
- `type="ir_ref"`：`detail` **必填**
- `source_type`：可选（`datasheet` / `reference_design` / `application_note` 等）
- `extra="forbid"`：不允许额外字段

### 6.2 `Evidence` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `type` | `Literal["rag_ref", "ir_ref"]` | 否 | `type` | `"rag_ref" \| "ir_ref"` | ✅ |
| `source` | `str \| None` | 是（`rag_ref` 时必填） | `source` | `string \| null` | ✅ |
| `section` | `str \| None` | 是（`rag_ref` 时必填） | `section` | `string \| null` | ✅ |
| `reason` | `str \| None` | 是（`rag_ref` 时必填） | `reason` | `string \| null` | ✅ |
| `detail` | `str \| None` | 是（`ir_ref` 时必填） | `detail` | `string \| null` | ✅ |
| `source_type` | `str \| None` | 是 | `source_type` | `string \| null` | ✅ |

**核对结果**：6/6 ✅，0 ⚠️，0 ❌。


### 6.3 条件必填语义（按 `type` 分流）

`Evidence` 的字段可空性**取决于 `type`**，TS 类型无法直接表达"条件必填"，H 前端需运行时判断。

#### 6.3.1 `type="rag_ref"` 时

- `source` / `section` / `reason` **必有值**（Pydantic 校验非空）
- `detail` 通常为 `null`（不填）
- `source_type` 可选

**H 前端渲染**：

```vue
<template v-if="evidence.type === 'rag_ref'">
  <div class="text-sm text-gray-600">
    <span class="font-medium">来源：</span>{{ evidence.source }}
  </div>
  <div class="text-sm text-gray-600">
    <span class="font-medium">章节：</span>{{ evidence.section }}
  </div>
  <div class="text-sm text-gray-600">
    <span class="font-medium">依据：</span>{{ evidence.reason }}
  </div>
</template>
```

#### 6.3.2 `type="ir_ref"` 时

- `detail` **必有值**（Pydantic 校验非空）
- `source` / `section` / `reason` 通常为 `null`
- `source_type` 通常为 `null`

**H 前端渲染**：

```vue
<template v-else-if="evidence.type === 'ir_ref'">
  <div class="text-sm text-gray-600">
    <span class="font-medium">IR 引用：</span>
    <code class="text-xs">{{ evidence.detail }}</code>
  </div>
</template>
```

#### 6.3.3 兜底（不应发生）

如果 `type` 是未知值（DB 脏数据），或 `rag_ref` 缺字段（Pydantic 校验失败不应落库），H 前端**兜底显示原始 JSON**：

```vue
<template v-else>
  <pre class="text-xs">{{ JSON.stringify(evidence, null, 2) }}</pre>
</template>
```

### 6.4 本章 TS 类型（基于核对结果，Day 1 上午写入 `types/evidence.ts`）

```typescript
// frontend/src/types/evidence.ts

/** 证据类型（Pydantic Literal） */
export type EvidenceType = "rag_ref" | "ir_ref";

/**
 * 单条证据（对齐 app/schemas/report.py::Evidence）。
 *
 * 条件必填语义（TS 无法直接表达，运行时按 type 判断）：
 *   - type="rag_ref"：source / section / reason 必有值
 *   - type="ir_ref" ：detail 必有值
 *   - source_type 可选
 */
export interface Evidence {
  type: EvidenceType;
  // rag_ref 专用
  source?: string | null;
  section?: string | null;
  reason?: string | null;
  // ir_ref 专用
  detail?: string | null;
  // 可选扩展
  source_type?: string | null;
}
```

**注意**：

- `type` 是联合类型 `"rag_ref" | "ir_ref"`，不是 `string`
- 字段用 `?: ... | null` 表达"可空"（Pydantic 默认 `None`，序列化后是 `null`）
- 条件必填由**运行时 `evidence.type` 判断**（第 6.3 节的 Vue 模板）
- `source_type` 是可选扩展，H 阶段可不用（Sprint 3 才细化 `DocumentType`）

### 6.5 本章结论

| 项 | 结果 |
|---|---|
| `Evidence` 字段 | 6/6 ✅，0 ⚠️，0 ❌ |
| 条件必填 | `type="rag_ref"` 时 `source`/`section`/`reason` 必有值；`type="ir_ref"` 时 `detail` 必有值（运行时判断） |
| 兜底处理 | 未知 `type` 或缺字段时显示原始 JSON（不应发生，防御性处理） |
| TS 类型 | 已给（`types/evidence.ts`） |
| H 阶段边界 | `source_type` 可选，H 阶段可不展示（Sprint 3 细化） |

## 第 7 章：`FeedbackCreateAPI` 对照（`POST /api/v1/feedbacks` 请求体）

### 7.1 后端定义

**Pydantic**（`schemas.py::FeedbackCreateAPI`）：

```python
class FeedbackCreateAPI(BaseModel):
    """POST /api/v1/feedbacks 请求体。

    【与 service 层 FeedbackCreate 的区别】
      - service 层含 review_result_id / review_defect_id 外键
      - API 层用业务标识 task_id + defect_id，端点内翻译为外键
      - suggestion_diff / rule_candidate 复用 service 层模型（选项 A）
    """
    task_id: str = Field(..., min_length=1)
    defect_id: str = Field(..., min_length=1)
    feedback_type: Literal[
        "correct_defect",
        "false_positive",
        "false_negative",
        "suggestion_update",
        "knowledge_gap",
        "new_rule_candidate",
    ]
    expert_suggestion: str | None = None
    suggestion_adopted: Literal["yes", "partial", "no"] | None = None
    suggestion_diff: list[SuggestionDiff] | None = None
    rule_candidate: RuleCandidate | None = None
    created_by: int | None = None
```

**嵌套类型来源**（`app/schemas/feedback.py`）：

- `SuggestionDiff`：`old: str` / `new: str`（2 字段）
- `RuleCandidate`：11 字段（`rule_id` / `rule_name` / `category` / `severity` / `rule_basis` / `applicable_condition` / `check_logic` / `suggestion` / `title` / `description` / `evidence_refs`）

### 7.2 `FeedbackCreateAPI` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `task_id` | `str`（min 1） | 否 | `task_id` | `string` | ✅ |
| `defect_id` | `str`（min 1） | 否 | `defect_id` | `string` | ✅ |
| `feedback_type` | `Literal[6 类]` | 否 | `feedback_type` | `FeedbackType`（见 7.3） | ✅ |
| `expert_suggestion` | `str \| None` | 是 | `expert_suggestion` | `string \| null` | ✅ |
| `suggestion_adopted` | `Literal["yes", "partial", "no"] \| None` | 是 | `suggestion_adopted` | `"yes" \| "partial" \| "no" \| null` | ✅ |
| `suggestion_diff` | `list[SuggestionDiff] \| None` | 是 | `suggestion_diff` | `SuggestionDiff[] \| null` | ✅ |
| `rule_candidate` | `RuleCandidate \| None` | 是 | `rule_candidate` | `RuleCandidate \| null` | ✅ |
| `created_by` | `int \| None` | 是 | `created_by` | `number \| null` | ✅ |

**核对结果**：8/8 ✅，0 ⚠️，0 ❌。

### 7.3 `feedback_type` 枚举（6 类）

**后端**（`schemas.py::FeedbackCreateAPI.feedback_type`）：

```python
feedback_type: Literal[
    "correct_defect",
    "false_positive",
    "false_negative",
    "suggestion_update",
    "knowledge_gap",
    "new_rule_candidate",
]
```

**TS 类型**：

```typescript
export type FeedbackType =
  | "correct_defect"
  | "false_positive"
  | "false_negative"
  | "suggestion_update"
  | "knowledge_gap"
  | "new_rule_candidate";
```

**6 类语义**（来自 `app/schemas/feedback.py::FeedbackType`）：

| 值 | 含义 | H 前端场景 |
|---|---|---|
| `correct_defect` | 专家确认缺陷成立 | 缺陷正确，建议可用 |
| `false_positive` | 误报 | 系统报错了 |
| `false_negative` | 漏检 | 系统没报，专家发现了 |
| `suggestion_update` | 修改建议 | 缺陷对，但建议不准 |
| `new_rule_candidate` | 新规则候选 | 建议加规则 |
| `knowledge_gap` | 知识缺失 | 缺 Datasheet / 参考资料 |

**H 前端 `FeedbackPanel`**：6 类单选（radio），根据类型动态显示/隐藏 `expert_suggestion` / `suggestion_adopted` / `suggestion_diff` / `rule_candidate` 字段（详见第 8 章）。

### 7.4 `SuggestionDiff` 嵌套类型

**后端**（`app/schemas/feedback.py::SuggestionDiff`）：

```python
class SuggestionDiff(BaseModel):
    """suggestion_update 场景：新旧建议对比条目"""
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    old: str
    new: str
```

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `old` | `str` | 否 | `old` | `string` | ✅ |
| `new` | `str` | 否 | `new` | `string` | ✅ |

**核对结果**：2/2 ✅，0 ⚠️，0 ❌。

**约束**：`extra="forbid"`，不允许额外字段；`str_strip_whitespace=True`，自动去首尾空格。

**H 前端 `FeedbackPanel`**：`suggestion_update` 类型时显示"旧建议 / 新建议"两列表单，可添加多条。H 阶段简化为 **JSON 文本域**（直接输入 `[{"old": "...", "new": "..."}]`），`JSON.parse` 后传。


### 7.5 `RuleCandidate` 嵌套类型（11 字段）

**后端**（`app/schemas/feedback.py::RuleCandidate`）：

```python
class RuleCandidate(BaseModel):
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    # --- 规则 YAML 必填字段（rule_format.md §1）---
    rule_id: str
    rule_name: str
    category: str
    severity: str
    rule_basis: str

    # --- 规则 YAML 必填字段，但草稿阶段专家可省略（系统填占位）---
    applicable_condition: dict | None = None
    check_logic: dict | None = None
    suggestion: str | None = None

    # --- rule_candidates ORM 冗余字段 ---
    title: str | None = None
    description: str | None = None
    evidence_refs: list[dict] | None = None
```

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `rule_id` | `str` | 否 | `rule_id` | `string` | ✅ |
| `rule_name` | `str` | 否 | `rule_name` | `string` | ✅ |
| `category` | `str` | 否 | `category` | `string` | ✅ |
| `severity` | `str` | 否 | `severity` | `string` | ✅ |
| `rule_basis` | `str` | 否 | `rule_basis` | `string` | ✅ |
| `applicable_condition` | `dict \| None` | 是 | `applicable_condition` | `Record<string, unknown> \| null` | ✅ |
| `check_logic` | `dict \| None` | 是 | `check_logic` | `Record<string, unknown> \| null` | ✅ |
| `suggestion` | `str \| None` | 是 | `suggestion` | `string \| null` | ✅ |
| `title` | `str \| None` | 是 | `title` | `string \| null` | ✅ |
| `description` | `str \| None` | 是 | `description` | `string \| null` | ✅ |
| `evidence_refs` | `list[dict] \| None` | 是 | `evidence_refs` | `Record<string, unknown>[] \| null` | ✅ |

**核对结果**：11/11 ✅，0 ⚠️，0 ❌。

**约束**：`extra="forbid"`，不允许额外字段；`str_strip_whitespace=True`。

**H 前端 `FeedbackPanel`**：`new_rule_candidate` / `false_negative` 类型时显示 `rule_candidate` 表单。H 阶段简化为 **JSON 文本域**（直接输入 11 字段 JSON），`JSON.parse` 后传。


### 7.6 本章 TS 类型（基于核对结果，Day 1 上午写入 `types/feedback.ts`）

```typescript
// frontend/src/types/feedback.ts

/** 6 类反馈类型（Pydantic Literal） */
export type FeedbackType =
  | "correct_defect"
  | "false_positive"
  | "false_negative"
  | "suggestion_update"
  | "knowledge_gap"
  | "new_rule_candidate";

/** 修改建议对比条目（对齐 SuggestionDiff） */
export interface SuggestionDiff {
  old: string;
  new: string;
}

/** 规则候选草稿（对齐 RuleCandidate，11 字段） */
export interface RuleCandidate {
  rule_id: string;
  rule_name: string;
  category: string;
  severity: string;
  rule_basis: string;
  applicable_condition?: Record<string, unknown> | null;
  check_logic?: Record<string, unknown> | null;
  suggestion?: string | null;
  title?: string | null;
  description?: string | null;
  evidence_refs?: Record<string, unknown>[] | null;
}

/** POST /api/v1/feedbacks 请求体（对齐 FeedbackCreateAPI） */
export interface FeedbackCreateAPI {
  task_id: string;
  defect_id: string;
  feedback_type: FeedbackType;
  expert_suggestion?: string | null;
  suggestion_adopted?: "yes" | "partial" | "no" | null;
  suggestion_diff?: SuggestionDiff[] | null;
  rule_candidate?: RuleCandidate | null;
  created_by?: number | null;
}
```

**注意**：

- `feedback_type` 窄化为 6 类联合，与 `FeedbackPanel` 的 radio 选项一一对应
- `suggestion_diff` / `rule_candidate` 是嵌套类型，H 阶段用 **JSON 文本域** 输入，`JSON.parse` 后传
- `RuleCandidate` 11 字段，5 个必填（`rule_id` / `rule_name` / `category` / `severity` / `rule_basis`），6 个可选
- `SuggestionDiff` 2 字段（`old` / `new`），`extra="forbid"` 前端不传额外字段


### 7.7 本章结论

| 项 | 结果 |
|---|---|
| `FeedbackCreateAPI` 字段 | 8/8 ✅，0 ⚠️，0 ❌ |
| `SuggestionDiff` 字段 | 2/2 ✅，0 ⚠️，0 ❌ |
| `RuleCandidate` 字段 | 11/11 ✅，0 ⚠️，0 ❌ |
| `feedback_type` 枚举 | 6 类，窄化为联合类型 |
| 嵌套类型约束 | `extra="forbid"`，不允许额外字段 |
| H 阶段简化 | `suggestion_diff` / `rule_candidate` 用 JSON 文本域，`JSON.parse` 后传 |
| TS 类型 | 已给（`types/feedback.ts`） |

## 第 8 章：`FeedbackResponse` 对照（`POST /api/v1/feedbacks` 响应体）

### 8.1 后端定义

**Pydantic**（`schemas.py::FeedbackResponse`）：

```python
class FeedbackResponse(BaseModel):
    """POST /api/v1/feedbacks 响应体（对齐 FeedbackOut）。

    【约束 3】created_at 用 datetime 类型，由 FastAPI 自动序列化为 ISO 字符串，
    不手动 .isoformat()。
    """
    id: int
    task_id: str | None = None
    defect_id: str | None = None
    review_result_id: int | None = None
    review_defect_id: int | None = None
    feedback_type: str
    expert_suggestion: str | None = None
    suggestion_diff: list[dict[str, Any]] = Field(default_factory=list)
    rule_candidate_ref: str | None = None
    created_by: int | None = None
    created_at: datetime
```

**来源**（`feedbacks.py` 端点的 `return FeedbackResponse(...)`，映射自 `FeedbackOut.model_dump()`）：

- `id` / `task_id` / `defect_id` / `review_result_id` / `review_defect_id`：来自 `FeedbackOut` 的对应字段（`task_id` / `defect_id` 是 JOIN 回填的业务标识）
- `feedback_type`：`FeedbackOut.feedback_type`（`FeedbackType` 枚举，序列化为 `str`）
- `expert_suggestion` / `suggestion_diff` / `rule_candidate_ref` / `created_by` / `created_at`：来自 `FeedbackOut` 对应字段
- `created_at` 是 `datetime`，FastAPI 自动序列化为 ISO 8601 字符串（如 `"2026-10-26T07:13:10.290704Z"`）


### 8.2 `FeedbackResponse` 字段对照

| 后端字段 | 后端类型 | 可空 | TS 字段 | TS 类型 | 核对 |
|---|---|---|---|---|---|
| `id` | `int` | 否 | `id` | `number` | ✅ |
| `task_id` | `str \| None` | 是 | `task_id` | `string \| null` | ✅ |
| `defect_id` | `str \| None` | 是 | `defect_id` | `string \| null` | ✅ |
| `review_result_id` | `int \| None` | 是 | `review_result_id` | `number \| null` | ✅ |
| `review_defect_id` | `int \| None` | 是 | `review_defect_id` | `number \| null` | ✅ |
| `feedback_type` | `str` | 否 | `feedback_type` | `FeedbackType`（6 类） | ✅（窄化） |
| `expert_suggestion` | `str \| None` | 是 | `expert_suggestion` | `string \| null` | ✅ |
| `suggestion_diff` | `list[dict]` | 否（默认 `[]`） | `suggestion_diff` | `SuggestionDiff[]` | ✅ |
| `rule_candidate_ref` | `str \| None` | 是 | `rule_candidate_ref` | `string \| null` | ✅ |
| `created_by` | `int \| None` | 是 | `created_by` | `number \| null` | ✅ |
| `created_at` | `datetime` | 否 | `created_at` | `string`（ISO 8601） | ✅（FastAPI 序列化） |

**核对结果**：11/11 ✅，0 ⚠️，0 ❌。


### 8.3 字段说明与 H 前端使用

#### 8.3.1 `id`（反馈 ID）

**来源**：`FeedbackOut.id`（`feedback_item.id`，DB 自增主键）

**H 前端用途**：提交成功后显示"反馈已提交（ID: {id}）"，用于用户追溯。

#### 8.3.2 `task_id` / `defect_id`（业务标识，JOIN 回填）

**来源**：`FeedbackOut.task_id`（`review_result.task_id`）/ `FeedbackOut.defect_id`（`review_defect.defect_id`），由 `feedback_service._build_out` JOIN 回填。

**H 前端用途**：确认反馈挂在正确的任务和缺陷下。可用于"反馈后跳转回原缺陷卡片"的锚点。

**注意**：`defect_id` 是**带 `task_id` 前缀**的完整 ID（如 `RUN-xxx-DEF-RULE-CLOCK_001`），与 `FeedbackCreateAPI.defect_id` 一致。

#### 8.3.3 `review_result_id` / `review_defect_id`（外键）

**来源**：`FeedbackOut.review_result_id` / `review_defect_id`（DB 外键）。

**H 前端用途**：H 阶段**不直接使用**（前端用业务标识 `task_id` / `defect_id`），仅调试时参考。

#### 8.3.4 `feedback_type`（6 类枚举）

**来源**：`FeedbackOut.feedback_type`（`FeedbackType` 枚举，Pydantic 序列化为 `str`）。

**H 前端用途**：回显用户提交的反馈类型。窄化为 `FeedbackType` 联合类型。

#### 8.3.5 `expert_suggestion` / `suggestion_diff`

**来源**：`FeedbackOut.expert_suggestion` / `suggestion_diff`。

**H 前端用途**：回显用户提交的建议。`suggestion_diff` 是 `[]`（默认）或 `SuggestionDiff[]`，H 阶段不做 UI 对比（Sprint 2 做），仅存。

#### 8.3.6 `rule_candidate_ref`（规则候选 ID）

**来源**：`FeedbackOut.rule_candidate_ref`（`feedback_item.rule_candidate_ref`，指向 `rule_candidates.candidate_id`）。

**H 前端用途**：当 `feedback_type="new_rule_candidate"` 或 `false_negative` 时，后端会生成规则候选草稿，`rule_candidate_ref` 是候选 ID（如 `RC-20261026-ABC123`）。H 阶段显示"已生成规则候选：{rule_candidate_ref}"（可选）。

**注意**：`rule_candidate_ref` 为 `None` 时表示"未生成规则候选"（如 `correct_defect` / `false_positive` / `knowledge_gap` 类型不生成候选）。

#### 8.3.7 `created_by` / `created_at`

**来源**：`FeedbackOut.created_by`（`users.id`）/ `created_at`（`feedback_item.created_at`）。

**H 前端用途**：
- `created_by`：H 阶段可不展示（Sprint 1 无用户系统）
- `created_at`：显示"提交时间：{created_at}"，格式 ISO 8601

### 8.4 本章 TS 类型（基于核对结果，Day 1 上午追加到 `types/feedback.ts`）

```typescript
// frontend/src/types/feedback.ts（追加）

/** POST /api/v1/feedbacks 响应体（对齐 FeedbackResponse） */
export interface FeedbackResponse {
  id: number;
  task_id: string | null;
  defect_id: string | null;
  review_result_id: number | null;
  review_defect_id: number | null;
  feedback_type: FeedbackType;
  expert_suggestion: string | null;
  suggestion_diff: SuggestionDiff[];
  rule_candidate_ref: string | null;
  created_by: number | null;
  /** FastAPI 序列化为 ISO 8601 字符串 */
  created_at: string;
}
```


**注意**：

- `feedback_type` 窄化为 `FeedbackType`（6 类联合）
- `suggestion_diff` 是 `SuggestionDiff[]`（默认 `[]`，不会是 `null`）
- `created_at` 是 `string`（FastAPI 自动序列化 `datetime` 为 ISO 8601）
- `rule_candidate_ref` 可能为 `None`（非 `new_rule_candidate` / `false_negative` 类型）


### 8.5 本章结论

| 项 | 结果 |
|---|---|
| `FeedbackResponse` 字段 | 11/11 ✅，0 ⚠️，0 ❌ |
| `feedback_type` 枚举 | 窄化为 6 类联合 |
| `created_at` 序列化 | FastAPI 自动转 ISO 8601 字符串 |
| `rule_candidate_ref` 语义 | 仅 `new_rule_candidate` / `false_negative` 可能非 `None` |
| H 前端使用 | 提交成功后显示 `id` + `rule_candidate_ref`（可选） |
| TS 类型 | 已给（`types/feedback.ts` 追加） |


## 第 9 章：汇总表（6 类型 × 字段数）

### 9.1 全部类型核对汇总

| 类型 | 字段数 | ✅ | ⚠️ | ❌ | 所在章 |
|---|---|---|---|---|---|
| `CaseItem` | 6 | 6 | 0 | 0 | 2.2 |
| `CaseListResponse` | 3 | 3 | 0 | 0 | 2.3 |
| `TaskDetailResponse` | 7 | 5 | 2 | 0 | 3.2 |
| `TaskListItem` | 6 | 5 | 1 | 0 | 4.2 |
| `TaskListResponse` | 3 | 3 | 0 | 0 | 4.4 |
| `Defect`（`get_task_defects`） | 16 | 16 | 0 | 0 | 5.2 |
| `DefectsResponse` | 1 | 1 | 0 | 0 | 5.4 |
| `Evidence` | 6 | 6 | 0 | 0 | 6.2 |
| `FeedbackCreateAPI` | 8 | 8 | 0 | 0 | 7.2 |
| `SuggestionDiff` | 2 | 2 | 0 | 0 | 7.4 |
| `RuleCandidate` | 11 | 11 | 0 | 0 | 7.5 |
| `FeedbackResponse` | 11 | 11 | 0 | 0 | 8.2 |
| **合计** | **80** | **77** | **3** | **0** | — |

**总计**：**0 ❌，3 ⚠️**。所有字段都真实存在，3 个 ⚠️ 都是"类型窄化 / 语义澄清"，不是字段缺失。


### 9.2 3 个 ⚠️ 汇总

| # | 类型 | 字段 | ⚠️ 内容 | 处理章 |
|---|---|---|---|---|
| 1 | `TaskDetailResponse` | `status` | service 实际只返回 `done` / `failed`，不会返回 `pending` | 第 3.3 节 |
| 2 | `TaskDetailResponse` | `report` | Pydantic 声明 `dict \| None`，实际至少 `{}`，不会是 `null` | 第 3.4 节 |
| 3 | `TaskListItem` | `review_status` | Pydantic 声明 `str`，DB 存三态，需前端窄化 | 第 4.3 节 |

**3 个 ⚠️ 的共同点**：**不阻塞开发**，都是"Pydantic 声明宽泛 / 实际值更窄"的情况。H 前端按"实际值"窄化 TS 类型 + 加注释即可。

### 9.3 按端点汇总

| 端点 | 类型 | 字段数 | ✅ | ⚠️ | ❌ |
|---|---|---|---|---|---|
| `GET /api/v1/cases` | `CaseListResponse` | 3 | 3 | 0 | 0 |
| `POST /api/v1/review/tasks` | `CreateTaskResponse` | 3 | 3 | 0 | 0 |
| `GET /api/v1/review/tasks/{task_id}` | `TaskDetailResponse` | 7 | 5 | 2 | 0 |
| `GET /api/v1/review/tasks` | `TaskListResponse` | 3 | 3 | 0 | 0 |
| `GET /api/v1/review/tasks/{task_id}/defects` | `DefectsResponse` | 1 | 1 | 0 | 0 |
| `POST /api/v1/feedbacks` | `FeedbackResponse` | 11 | 11 | 0 | 0 |

**说明**：`CreateTaskResponse` 第 9 章未单独核对（H 前端 `CaseListPage` 发起评审时用），字段与 `TaskDetailResponse` 的前 3 字段一致（`task_id` / `status` / `defects_count`），无额外 ⚠️。

### 9.4 本章结论

| 项 | 结果 |
|---|---|
| 核对类型总数 | 10 个（含嵌套类型 `SuggestionDiff` / `RuleCandidate`） |
| 字段总数 | 80 |
| ✅ | 77（96.25%） |
| ⚠️ | 3（3.75%，都是"窄化/澄清"，不阻塞） |
| ❌ | **0** |
| 结论 | **所有字段真实存在，无缺失、无冲突，可以开始写 `types/*.ts`** |


## 第 10 章：3 个 ⚠️ 的 H 前端处理

### 10.1 ⚠️ 1：`TaskDetailResponse.status` 无 `"pending"`

**问题**：Pydantic 声明 `Literal["done", "failed", "pending"]`，但 service 实际只返回 `done` / `failed`。

**H 前端处理**：

```typescript
// types/task.ts

/**
 * 任务状态（对齐 TaskDetailResponse.status）。
 * 注意：后端 Pydantic 声明允许 "pending"，但 get_task 实际只返回 "done" / "failed"。
 * "pending" 场景 = task 不存在 → API 层返回 404。
 */
export type TaskStatus = "done" | "failed";
```


**逻辑**：

```typescript
// TaskDetailPage.vue
if (task.status === "done") {
  // 显示报告 + 缺陷列表
} else if (task.status === "failed") {
  // 显示 errors 列表
}
// 无 "pending" 分支（404 已在 getTask 调用处捕获）
```

### 10.2 ⚠️ 2：`TaskDetailResponse.report` 实际非 `None`

**问题**：Pydantic 声明 `dict[str, Any] | None`，实际 `get_task` 里 `report_json = rr.review_output_json or {}`，**至少 `{}`**，不会 `null`。

**H 前端处理**：

```typescript
// types/task.ts

export interface TaskDetail {
  // ...
  /**
   * 报告（对齐 TaskDetailResponse.report）。
   * 注意：后端 Pydantic 声明允许 null，但实际至少 {}（空 dict）。
   * 判断"报告为空"用：report === null || Object.keys(report).length === 0
   */
  report: ReviewReport | Record<string, never> | null;
}
```

**逻辑**：

```typescript
// TaskDetailPage.vue
function hasReport(report: TaskDetail["report"]): boolean {
  if (report === null) return false;
  return Object.keys(report).length > 0;
}
```

### 10.3 ⚠️ 3：`TaskListItem.review_status` 是 `str`，需窄化

**问题**：Pydantic 声明 `TaskListItem.review_status: str`（未窄化），但 DB `review_result.review_status` 实际存三态（`AI_CONFIRMED` / `NEED_EXPERT_REVIEW` / `LOW_CONFIDENCE`）。

**H 前端处理**：

```typescript
// types/task.ts

/** 三态质量状态（与 review_result.review_status 对齐） */
export type ReviewStatus =
  | "AI_CONFIRMED"
  | "NEED_EXPERT_REVIEW"
  | "LOW_CONFIDENCE";

export interface TaskListItem {
  // ...
  review_status: ReviewStatus;
}
```

**逻辑**（`StatusBadge` 组件防御未知值）：

```typescript
// components/StatusBadge.vue
const STATUS_MAP: Record<string, { label: string; color: string }> = {
  AI_CONFIRMED: { label: "AI 已确认", color: "green" },
  NEED_EXPERT_REVIEW: { label: "需专家复核", color: "yellow" },
  LOW_CONFIDENCE: { label: "低置信", color: "red" },
};

const badge = STATUS_MAP[props.status] ?? {
  label: "未知",
  color: "gray",  // 兜底：DB 脏数据 / 新状态
};
```

**关键**：前端窄化是"声明意图"，运行时仍要兜底未知值（DB 脏数据不应导致页面崩溃）。


### 10.4 3 个 ⚠️ 的统一处理原则

| 原则 | 说明 |
|---|---|
| **TS 类型窄化** | 按"实际值"窄化（`status: "done" \| "failed"`、`review_status: ReviewStatus`），不按 Pydantic 声明的宽泛类型 |
| **注释说明** | 每个窄化处加注释，说明"Pydantic 声明 X，实际 Y，原因 Z" |
| **运行时兜底** | 未知值 fallback（`StatusBadge` 灰色"未知"），不崩溃 |
| **不阻塞开发** | 3 个 ⚠️ 都不需要后端改，前端处理即可 |
| **不改变契约** | 窄化是"前端声明"，不影响后端 Pydantic 契约（后端仍允许宽泛值） |

**结论**：3 个 ⚠️ 全部可在 H 前端内部消化，不需要动后端代码，也不需要改 Pydantic Schema。


## 第 11 章：`report.defects` vs `GET /defects` 关键差异

### 11.1 差异总览

`TaskDetailResponse.report.defects` 和 `GET /defects` 的 `items` **都是缺陷列表**，但来源和字段不同：

| 维度 | `report.defects` | `GET /defects` 的 `items` |
|---|---|---|
| 来源 | `review_result.review_output_json.defects`（**落库快照**） | `review_defect` 表（**DB 当前状态**） |
| 序列化时机 | `feedback_save` 节点落库时（`ReviewReport.model_dump(mode="json")`） | `get_task_defects` 查询时实时序列化 |
| 字段数 | **15** | **16** |
| `expert_decision_updated_at` | ❌ 无 | ✅ 有 |
| `expert_decision_by` | ❌ 无 | ✅ 有 |
| `synthesized` | ✅ 有（S1-B 新增） | ❌ 无（G1 未补） |


### 11.2 `report.defects` 的字段清单（15 字段）

`report.defects` 是 `ReviewReport.model_dump(mode="json")` 的 `defects` 数组，每条来自 `app/schemas/report.py::Defect`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `defect_id` | `str` | 缺陷 ID（带 `task_id` 前缀） |
| `category` | `str` | 分类 |
| `location` | `dict` | 位置 |
| `component` | `str \| None` | 元件 |
| `net` | `str \| None` | 网络 |
| `risk` | `str` | 风险等级 |
| `evidence` | `list[Evidence]` | 证据 |
| `root_cause` | `str` | 根因 |
| `suggestion` | `str` | 建议 |
| `confidence` | `float` | 置信度 |
| `review_status` | `str` | 三态质量 |
| `origin` | `str` | 来源（rule / ai_discovered / hybrid） |
| `rule_id` | `str \| None` | 规则 ID |
| `expert_decision` | `str` | 专家决定（四态） |
| `synthesized` | `bool` | 代码补齐标记（S1-B 新增） |

**共 15 字段**。**没有** `expert_decision_updated_at` / `expert_decision_by`（G1 只给 `get_task_defects` 补了，没给 `report.defects` 补）。


### 11.3 `GET /defects` 的字段清单（16 字段）

`GET /defects` 的 `items` 来自 `task_service.get_task_defects()`，从 `review_defect` 表查询：

| 字段 | 类型 | 说明 |
|---|---|---|
| `defect_id` | `str` | 缺陷 ID（带 `task_id` 前缀） |
| `category` | `str` | 分类 |
| `location` | `dict` | 位置 |
| `component` | `str \| None` | 元件 |
| `net` | `str \| None` | 网络 |
| `risk` | `str` | 风险等级 |
| `evidence` | `list[dict]` | 证据 |
| `root_cause` | `str` | 根因 |
| `suggestion` | `str` | 建议 |
| `confidence` | `float` | 置信度 |
| `review_status` | `str` | 三态质量 |
| `origin` | `str` | 来源 |
| `rule_id` | `str \| None` | 规则 ID |
| `expert_decision` | `str` | 专家决定（四态） |
| `expert_decision_updated_at` | `str \| None` | 判定时间（ISO 8601，G1 补） |
| `expert_decision_by` | `int \| None` | 判定人（G1 补） |

**共 16 字段**。**没有** `synthesized`（G1 未给 `get_task_defects` 补）。

### 11.4 字段差异对照表

| 字段 | `report.defects`（15） | `GET /defects`（16） | 差异 |
|---|---|---|---|
| `defect_id` | ✅ | ✅ | — |
| `category` | ✅ | ✅ | — |
| `location` | ✅ | ✅ | — |
| `component` | ✅ | ✅ | — |
| `net` | ✅ | ✅ | — |
| `risk` | ✅ | ✅ | — |
| `evidence` | ✅ | ✅ | — |
| `root_cause` | ✅ | ✅ | — |
| `suggestion` | ✅ | ✅ | — |
| `confidence` | ✅ | ✅ | — |
| `review_status` | ✅ | ✅ | — |
| `origin` | ✅ | ✅ | — |
| `rule_id` | ✅ | ✅ | — |
| `expert_decision` | ✅ | ✅ | — |
| `synthesized` | ✅ | ❌ | **仅在 `report.defects`** |
| `expert_decision_updated_at` | ❌ | ✅ | **仅在 `GET /defects`** |
| `expert_decision_by` | ❌ | ✅ | **仅在 `GET /defects`** |

**差异总结**：
- `report.defects` **多** `synthesized`（S1-B 新增的 provenance 标记）
- `GET /defects` **多** `expert_decision_updated_at` / `expert_decision_by`（G1 补的专家判定元信息）
- 其余 14 字段**完全一致**


### 11.5 H 前端处理：两个类型分开定义

**结论**：H 前端必须定义**两个不同的 `Defect` 类型**，不能混用。

#### 11.5.1 `DefectFromReport`（15 字段）

```typescript
// types/defect.ts

/**
 * 来自 report.defects（ReviewReport.model_dump 快照）。
 * 注意：无 expert_decision_updated_at / expert_decision_by（G1 只给 GET /defects 补了）。
 * 有 synthesized（S1-B 新增的 provenance 标记）。
 */
export interface DefectFromReport {
  defect_id: string;
  category: string;
  location: Record<string, unknown>;
  component: string | null;
  net: string | null;
  risk: Risk;
  evidence: Evidence[];
  root_cause: string;
  suggestion: string;
  confidence: number;
  review_status: ReviewStatus;
  origin: DiscoveryOrigin;
  rule_id: string | null;
  expert_decision: ExpertDecision;
  synthesized: boolean;
}
```

#### 11.5.2 `DefectFromAPI`（16 字段）

```typescript
// types/defect.ts

/**
 * 来自 GET /review/tasks/{task_id}/defects（review_defect 表当前状态）。
 * 有 expert_decision_updated_at / expert_decision_by（G1 补）。
 * 无 synthesized（G1 未给 get_task_defects 补）。
 */
export interface DefectFromAPI {
  defect_id: string;
  category: string;
  location: Record<string, unknown>;
  component: string | null;
  net: string | null;
  risk: Risk;
  evidence: Evidence[];
  root_cause: string;
  suggestion: string;
  confidence: number;
  review_status: ReviewStatus;
  origin: DiscoveryOrigin;
  rule_id: string | null;
  expert_decision: ExpertDecision;
  expert_decision_updated_at: string | null;
  expert_decision_by: number | null;
}
```

### 11.6 H 前端用哪个类型

| 场景 | 用哪个类型 | 理由 |
|---|---|---|
| `TaskDetailPage` 渲染缺陷列表 | **`DefectFromAPI`** | 优先调 `GET /defects`，拿 16 字段（含 `expert_decision_updated_at` / `_by`），`DefectCard` 显示完整信息 |
| `report.defects` 快照对照 | `DefectFromReport` | 仅在"报告快照 vs DB 当前状态"对比时用（H 阶段可不用，Sprint 2 才用） |
| `DefectCard` 组件 props | **`DefectFromAPI`** | 统一用最全字段，避免 `expert_decision_updated_at` / `_by` 缺失 |

**H 阶段决策**：

- **`TaskDetailPage` 优先调 `GET /defects`**，用 `DefectFromAPI[]` 渲染 `DefectCard`
- **不读 `report.defects`**（避免缺 `expert_decision_updated_at` / `_by` 导致 `ExpertDecisionBadge` 显示不全）
- `report` 仅用于 `report.summary`（`review_status_distribution`）+ `report.errors`
- `DefectFromReport` 类型**保留定义但不使用**（Sprint 2 可能用，H 阶段不删）


### 11.7 为什么不能只用一个类型

**潜在 bug 场景**（如果只用 `DefectFromAPI` 或只用 `DefectFromReport`）：

| 场景 | 后果 |
|---|---|
| 只用 `DefectFromAPI` 渲染 `report.defects` | 运行时 `expert_decision_updated_at` 是 `undefined`（`report.defects` 没这字段），`ExpertDecisionBadge` 显示 `"undefined"` |
| 只用 `DefectFromReport` 渲染 `GET /defects` | 运行时 `synthesized` 是 `undefined`，`DefectCard` 无法区分"LLM 输出"和"代码补齐" |
| 混用（部分字段从 `report` 取，部分从 `GET /defects` 取） | 需要前端做"字段合并"，逻辑复杂，容易漏字段 |

**正确做法**：

- **两个类型分开定义**（`DefectFromReport` 15 字段 / `DefectFromAPI` 16 字段）
- **H 阶段只用一个**（`DefectFromAPI`，因为 `GET /defects` 字段最全）
- **不合并两个类型**（避免 `Object.assign` 类的字段合并逻辑）

**后端优化方向**（Sprint 2 可选）：
- 给 `report.defects` 补 `expert_decision_updated_at` / `expert_decision_by`（`_persist_review_report` 时同步写）
- 给 `GET /defects` 补 `synthesized`（`get_task_defects` 序列化时加）
- 两个接口字段统一后，H 前端可以合并成一个 `Defect` 类型

**H 阶段决策**：**不推动后端改**（Sprint 1 已冻结），前端用 `DefectFromAPI` 单类型，`DefectFromReport` 定义保留不用。

### 11.8 本章结论

| 项 | 结果 |
|---|---|
| 字段差异 | `report.defects`（15 字段）与 `GET /defects`（16 字段）差 3 个字段：`synthesized` 只在 report；`expert_decision_updated_at` / `expert_decision_by` 只在 GET /defects |
| H 前端类型 | **两个类型分开定义**：`DefectFromReport`（15）/ `DefectFromAPI`（16） |
| H 阶段使用 | **只用 `DefectFromAPI`**（`GET /defects` 字段最全），`report.defects` 不读 |
| `report` 用途 | 仅用 `report.summary` + `report.errors`（不读 `report.defects`） |
| 潜在 bug 规避 | 不混用、不合并两个类型，避免运行时 `undefined` |
| 后端优化方向 | Sprint 2 可选：给 `report.defects` 补 2 字段、给 `GET /defects` 补 `synthesized`，统一后前端可合并 |
| 本章关键发现 | 这是本次核对**最重要的发现**，不是字段缺失，而是"两个接口的字段集不完全一致" |

## 第 12 章：核对结论 + 下一步

### 12.1 核对结论

| 项 | 结果 |
|---|---|
| 核对来源 | `backend/app/api/schemas.py` + `backend/app/services/task_service.py`（真实源码） |
| 核对类型 | 10 个（含嵌套 `SuggestionDiff` / `RuleCandidate`） |
| 字段总数 | 80 |
| ✅ | 77（96.25%） |
| ⚠️ | 3（3.75%，都是"窄化/澄清"，不阻塞） |
| ❌ | **0** |
| 关键发现 | `report.defects`（15 字段）vs `GET /defects`（16 字段）字段集不同，需分 `DefectFromReport` / `DefectFromAPI` |
| 结论 | **所有字段真实存在，无缺失、无冲突，可以开始写 `types/*.ts`** |

### 12.2 3 个 ⚠️ 的处理（已定稿）

| # | 类型 | 字段 | 处理 |
|---|---|---|---|
| 1 | `TaskDetailResponse` | `status` | TS 窄化为 `"done" \| "failed"`，无 `"pending"` |
| 2 | `TaskDetailResponse` | `report` | TS 类型 `ReviewReport \| Record<string, never> \| null`，判断空用 `Object.keys(report).length === 0` |
| 3 | `TaskListItem` | `review_status` | TS 窄化为 `ReviewStatus` 三态联合，运行时兜底未知值 |

### 12.3 类型清单（Day 1 上午写入 4 个文件）

| 文件 | 类型 | 字段数 |
|---|---|---|
| `types/case.ts` | `CaseItem` / `CaseListResponse` | 6 + 3 |
| `types/task.ts` | `TaskStatus` / `TaskDetail` / `ReviewStatus` / `TaskListItem` / `TaskListResponse` | 1 类型别名 + 7 + 1 类型别名 + 6 + 3 |
| `types/defect.ts` | `Risk` / `ReviewStatus` / `DiscoveryOrigin` / `ExpertDecision` / `Evidence` / `DefectFromAPI` / `DefectFromReport` / `DefectsResponse` | — + — + — + — + 6 + 16 + 15 + 1 |
| `types/feedback.ts` | `FeedbackType` / `SuggestionDiff` / `RuleCandidate` / `FeedbackCreateAPI` / `FeedbackResponse` | 6 + 2 + 11 + 8 + 11 |


### 12.4 开工前置条件确认清单

写代码前，以下 5 项已确认（用户逐条确认）：

| # | 项 | 确认结果 |
|---|---|---|
| 1 | API baseURL 统一 `/api/v1` + 代理转发 | ✅ 开发 Vite proxy / 生产 Nginx，`api/*.ts` 只传相对路径 |
| 2 | 字段核对以真实源码为准 | ✅ 基于 `schemas.py` + `task_service.py`，80 字段逐核对 |
| 3 | 20 秒超时语义 | ✅ 前端等待上限，超时 ≠ 失败，禁止自动重试，引导去任务列表 |
| 4 | 任务详情加载顺序 + 失败/空报告/空缺陷状态 | ✅ 先 `getTask` 判 404/status，再 `getDefects`，各状态 UI 明确 |
| 5 | 8 个异常场景手动验收 | ✅ 后端不可用 / 超时 / 超时不重试 / 404 / failed / 空报告空缺陷 / 反馈 JSON 错 / 反馈验证或网络错 |

### 12.5 下一步（等用户确认本核对文档后执行）

1. **用户确认本核对文档（第 1–12 章）**：确认 3 个 ⚠️ 处理 + `DefectFromReport`/`DefectFromAPI` 分开 + 8 异常场景清单
2. **我给 Day 1 上午代码**：
   - `frontend/package.json` + `vite.config.ts` + `.env.development` + `.env.production`
   - `frontend/src/api/http.ts`（baseURL `/api/v1` + 20 秒超时 + `TIMEOUT` 错误类型）
   - `frontend/src/types/case.ts` + `task.ts` + `defect.ts` + `feedback.ts`（基于核对结果，含 3 个 ⚠️ 注释）
   - `frontend/src/api/cases.ts` + `review.ts` + `feedback.ts`（6 函数）
   - `frontend/src/pages/CaseListPage.vue`（骨架）
   - `frontend/README.md`（不做清单 + 8 异常场景验收清单）
3. **用户跑 `npm install && npm run dev`**，验证 `http://localhost:5173/cases` 能列 10 个 Case

### 12.6 待确认事项（3 件）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 第 1–12 章核对结论 | ⏳ 待用户确认 |
| 2 | `DefectFromReport` / `DefectFromAPI` 分开定义（第 11 章） | ⏳ 待用户确认 |
| 3 | 8 个异常场景清单（第 2.4 节） | ⏳ 待用户确认 |

**回这 3 点，我给 Day 1 上午代码。**


## 附：核对文档输出完毕

本文档共 12 章，覆盖：

| 章 | 内容 | 状态 |
|---|---|---|
| 1 | 核对来源与范围 | ✅ |
| 2 | `CaseItem` / `CaseListResponse` 对照 | ✅ |
| 3 | `TaskDetailResponse` 对照（2 ⚠️） | ✅ |
| 4 | `TaskListItem` / `TaskListResponse` 对照（1 ⚠️） | ✅ |
| 5 | `Defect` 对照（16 字段 + 4 枚举窄化） | ✅ |
| 6 | `Evidence` 对照（6 字段 + 条件必填） | ✅ |
| 7 | `FeedbackCreateAPI` 对照（8 字段 + 3 嵌套类型） | ✅ |
| 8 | `FeedbackResponse` 对照（11 字段） | ✅ |
| 9 | 汇总表（80 字段，0 ❌ / 3 ⚠️） | ✅ |
| 10 | 3 个 ⚠️ 的 H 前端处理 | ✅ |
| 11 | `report.defects` vs `GET /defects` 关键差异 | ✅ |
| 12 | 核对结论 + 下一步 | ✅ |

**核对结论**：**0 ❌，3 ⚠️（都是"窄化/澄清"，不阻塞开发），所有 80 字段真实存在**。

**下一步**：等用户确认第 12.6 节的 3 件事，我给 Day 1 上午代码。


## 附录 A：核对文档使用说明

### A.1 文档定位

本核对文档是 **H 阶段（Web MVP）开发前的"接口契约冻结记录"**，基于真实源码逐字段核对，不是计划中的 TypeScript 类型推演。

### A.2 文档使用方式

| 阶段 | 使用方式 |
|---|---|
| Day 1 上午（写 `types/*.ts`） | 按第 2–8 章逐字段对照写 TS 类型，第 10 章处理 3 个 ⚠️，第 11 章分开 `DefectFromReport` / `DefectFromAPI` |
| Day 1 下午（写页面/组件） | 按第 12.4 节加载顺序 + 第 2.4 节 8 异常场景写 UI 逻辑 |
| Day 2（联调） | 按第 12.4 节 8 异常场景逐个手动验收 |
| 后续 Sprint 2 | 第 11.7 节"后端优化方向"可作为 Sprint 2 的输入 |

### A.3 文档变更规则

- **本核对文档已冻结**：字段核对结果不再改（除非后端 Schema 变更）
- **后端 Schema 变更时**：更新本核对文档对应章节 + 重新核对受影响的 TS 类型
- **H 阶段开发中发现的字段问题**：先记录到"H 阶段问题清单"，不在核对文档里直接改

### A.4 文档与代码的关系

| 文档 | 代码 | 关系 |
|---|---|---|
| 本核对文档 | `frontend/src/types/*.ts` | 文档是"字段契约"，代码是"实现"，代码必须与文档一致 |
| `backend/app/api/schemas.py` | 本核对文档 | 后端 Pydantic 是"权威来源"，本核对文档是"前端视角的映射" |
| `backend/app/services/task_service.py` | 本核对文档 | 后端 service 返回 dict 是"实际数据"，本核对文档核对"声明 vs 实际"的差异 |


## 附录 B：H 前端开工前 5 项处理方案回顾

### B.1 5 项处理方案（已定稿）

| # | 项 | 处理方案 |
|---|---|---|
| 1 | API baseURL 与端点路径拼接 | `VITE_API_BASE_URL=/api/v1`（开发/生产一致），Vite proxy 开发 / Nginx 生产转发 `/api`，`api/*.ts` 只传 `/review/tasks` 等相对路径 |
| 2 | 真实后端 Schema 核对 | 基于 `schemas.py` + `task_service.py`，80 字段逐核对，0 ❌ / 3 ⚠️ |
| 3 | 20 秒超时语义 | 前端等待上限，不是后端执行时限；超时 ≠ 失败；禁止自动重试；引导去任务列表 |
| 4 | 任务详情加载顺序及状态 | 先 `getTask`（404 / status / report），再 `getDefects`；failed 显示 errors；report 空显示"报告为空"；defects 空显示"无缺陷" |
| 5 | 超时重复提交 + 异常场景 | 超时后禁止自动重试，显示"去任务列表"链接；8 个异常场景手动验收 |


### B.2 5 项之间的依赖关系

```text
1. baseURL 拼接规则
   ↓（决定 api/*.ts 的路径写法）
2. 字段核对
   ↓（决定 types/*.ts 的字段定义）
3. 20 秒超时语义
   ↓（决定 http.ts 的 AbortController 逻辑）
4. 加载顺序及状态
   ↓（决定 TaskDetailPage.vue 的分支逻辑）
5. 异常场景验收
   ↓（决定 smoke test 清单 + FeedbackPanel 错误处理）
```

**5 项全部定稿后，才开始写代码**。


## 附录 C：核对文档完

### C.1 文档全貌

| 部分 | 内容 | 章/节 |
|---|---|---|
| 正文 | 核对来源与范围 | 第 1 章 |
| 正文 | 6 个类型逐字段对照 | 第 2–8 章 |
| 正文 | 汇总表（80 字段，0 ❌ / 3 ⚠️） | 第 9 章 |
| 正文 | 3 个 ⚠️ 的 H 前端处理 | 第 10 章 |
| 正文 | `report.defects` vs `GET /defects` 关键差异 | 第 11 章 |
| 正文 | 核对结论 + 下一步 | 第 12 章 |
| 附录 | 文档使用说明 | 附录 A |
| 附录 | 5 项处理方案回顾 | 附录 B |

### C.2 核对文档状态

| 项 | 状态 |
|---|---|
| 文档状态 | ✅ 已冻结（字段核对结果不再改） |
| 待用户确认 | 第 12.6 节 3 件事（核对结论 / 2 类型分开 / 8 异常场景） |
| 下一步 | 用户确认后，给 Day 1 上午代码 |

### C.3 核对文档输出完毕

**本核对文档（第 1–12 章 + 附录 A/B/C）已全部输出完毕。**

**下一步等你确认第 12.6 节的 3 件事**：

1. 第 1–12 章核对结论（80 字段，0 ❌ / 3 ⚠️）
2. `DefectFromReport` / `DefectFromAPI` 分开定义（第 11 章）
3. 8 个异常场景清单（第 2.4 节 / 第 12.4 节）

**回这 3 点，我给 Day 1 上午代码**（`package.json` + `vite.config.ts` + `.env.*` + `http.ts` + `types/*.ts` + `api/*.ts` + `CaseListPage.vue` 骨架 + `README.md`）。

---

**核对文档已全部输出完毕（第 1–12 章 + 附录 A/B/C）。等你确认第 12.6 节的 3 件事，我给 Day 1 上午代码。**



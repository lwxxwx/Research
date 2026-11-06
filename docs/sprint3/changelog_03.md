### 9.5b 坑 6：API 层 DTO 漏传字段导致 `knowledge_gap_ref` 为空

**现象**：
测试发现，`POST /api/v1/feedbacks` 提交 `knowledge_gap` 类型反馈后，DB 中 `knowledge_gap` 和 `gap_feedback_link` 均成功写入，但 API 响应中的 `knowledge_gap_ref` 字段始终为空。

**排查思路（逐步缩小范围）**：
1. **查 DB**：确认 `gap_feedback_link` 表有记录（`feedback_id=2528` 对应 `gap_id=188`），数据层无问题。
2. **查 Service 层**：在 `_build_out` 增加调试日志，确认 `db_feedback.id` 是整数（2528/2529），且独立 Session 手工 SQL 能查出 `GAP-...`，说明查询逻辑正确。
3. **最终定位**：检查 API 路由 `feedbacks.py` 发现，`FeedbackResponse` 组装时**根本没有传** `knowledge_gap_ref`，且 `FeedbackResponse` schema **也没有定义该字段**。

**根因**：
`FeedbackOut`（Service DTO）有 `knowledge_gap_ref`，但 `FeedbackResponse`（API DTO）未同步新增字段，路由组装时也未传值。三层链路（Service → API Schema → 路由）缺失两层。

**修复**：
1. `schemas.py` 的 `FeedbackResponse` 增加 `knowledge_gap_ref: str | None = None`。
2. `feedbacks.py` 的 `return FeedbackResponse(...)` 增加 `knowledge_gap_ref=out.knowledge_gap_ref`。

**结果**：
修复后 API 响应正常返回 `GAP-20261106-5708C0`，`knowledge_gap` 总数正确递增。

**教训**：
1. **"Service 层有值"不等于"API 响应有值"**。手动验证必须看 `Invoke-RestMethod` 的最终响应，而不是 Service 函数的返回对象。
2. **DTO 字段必须逐层对齐**：新增字段必须同时检查 `Service DTO` → `API Schema` → `路由组装` 三处。
3. **全链路 grep 是终极大法**：遇到字段静默丢失，直接 `grep` 字段名，缺哪补哪。
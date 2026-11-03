# Sprint 3 G2 前置核查报告 (Audit)

> 由 `scripts/g2_pre_check.py` 自动生成。
> 对应 `Sprint3_详细执行计划_V1.0.md` 附录 F。

## 自动核查结果

| 核查项 | 结论 |
| :--- | :--- |
| 1. `deps.py` 现状 | ❌ 缺失 | 是否已有 get_current_user: 否 |
| 2. `db.py` 现状 | ❌ 缺失 | 是否已有 get_db_session: 否 |
| 3. 数据库类型确认 | PostgreSQL (通过 docker-compose.dev.yml 确认) |
| 4. `s2a_schema_updates` revision | `未知` |
| 5. API 端点清单 | 共 0 个端点:未找到端点 |
| 6. submit_task_feedback 生成逻辑 | 含 rule_candidate 触发: 否 |
| 7. feedback_service 写 created_by | 含 created_by: 否 |
| 8. update_expert_decision 写 expert_decision_by | 含 expert_decision_by: 否 |
| 10 & 13. 脚本 uv sync 清理 | dev.ps1 含 uv sync: 否(已清理) | bootstrap.ps1 含 uv sync: 否(已清理) |
| 11. users 表行数 | 0 (根据 baseline_snapshot.md) |
| 12. 候选 ID 连字符修复 | 暂不修（Sprint3 不做），需确认前端有 encodeURIComponent 兜底 |
| 14. demo_sprint2.ps1 只读入口 | 需手工确认（检查脚本是否存在 --readonly 或类似参数） |
| 15. CI 配置 | ❌ 缺失 | 需确认 APP_ENV 是否为 test |

## 需手工确认的事项
1. [ ] **只读审计**：确认 `demo_sprint2.ps1` 是否有 `--readonly` 参数。
2. [ ] **基线重测**：执行 `pytest -m "not integration" -v`，确认基线与 568 passed 无意外偏差。
3. [ ] **Case 归属字段**：检查 `schematic_case` 表是否有 `created_by` 或 `owner_id`。
4. [ ] **Benchmark 数据边界**：人工评估 `/cases/{id}/benchmark` 是否修改正式数据（修改则需审计）。

---
**下一步**：将本报告发给 AI 助手确认，随后开始 G2 编码。
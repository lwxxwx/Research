# Sprint 2 复盘报告（Retrospective）

> **文档性质**：Sprint 2（2026-10-27 ~ 2026-11-02）的完整复盘，总结交付物、经验教训、技术债与 Sprint 3 输入。
> **归档日期**：2026-11-02
> **封板 commit**：`8073329`（`docs(s2-f15): append F15 records + fix frontend_debug TOC`）

---

## 一、交付总览

### 1.1 功能交付（15 项 F 功能，100% 完成）

| 阶段 | 功能 | 状态 | Commit 摘要 |
|---|---|---|---|
| F01 | 任务状态与筛选 | ✅ | S2-C |
| F02 | 专家判定（PATCH） | ✅ | S2-B |
| F03 | 真实指标与 Seed 排除 | ✅ | S2-C |
| F04 | 规则候选管理 | ✅ | S2-G |
| **F05** | **正式规则发布状态机** | ✅ | `53878c7`（恢复） |
| **F06** | **知识缺口 Backlog** | ✅ | `03ddf34`（恢复） |
| **F07** | **Case 管理 + 单 Case Benchmark** | ✅ | `fc40829` + `18155f1` |
| F08 | Markdown 报告导出 | ✅ | S2-H（一） |
| F09 | 批量评审 | ✅ | S2-H（二） |
| F10 | Benchmark CSV 导出 | ✅ | S2-I |
| F11 | RAG 知识库管理 | ✅ | S2-H（三） |
| F12 | 指标看板 | ✅ | S2-H（四） |
| F13 | CI 与资产治理 | ✅ | S2-I |
| F14 | vLLM 兼容验证（占位） | ✅ | S2-I |
| **F15** | **Demo + 技术基线快照** | ✅ | `8ff5b02` |

### 1.2 三条业务线闭环（全部验证通过）

1. **评审质量闭环**：建任务 → 详情 → 专家判定
2. **规则演进闭环**：反馈 → 候选 → 批准 → 发布 → 回归
3. **知识反馈闭环**：knowledge_gap 反馈 → KnowledgeGap 落库 → Backlog API

### 1.3 技术基线（封板时）

| 项 | 结果 |
|---|---|
| Git HEAD | `8073329` |
| 全量非 integration 测试 | **568 passed, 1 skipped, 24 deselected, 0 failed** |
| Bench 基线 | `tp=16 / fn=0 / fp ≤ 2`（S2-A 冻结） |
| Alembic 版本 | `s2_f07_case_benchmark` |
| 数据库类型 | `postgresql` |
| 前端构建 | `1990 modules transformed`，build 成功 |
| 业务闭环 | 4 条全部通过（见 `baseline_snapshot.md`） |

### 1.4 交付物清单

- **代码**：`backend/app/` / `frontend/src/` / `scripts/` 全部功能实现
- **文档**：`docs/sprint2/changelog.md` / `frontend_debug.md` / `baseline_snapshot.md` / `baseline_evidence/`
- **Demo**：`scripts/demo_sprint2.ps1`（三阶段自动化）
- **远端**：`origin/feature/sprint1-core-review`（推送成功，829 objects）

---

## 二、关键里程碑

| 日期 | 阶段 | 核心事件 |
|---|---|---|
| 2026-10-27 | S2-A Day 1 | 基线核查 10 项，发现 5 项差异，修订计划 5 处 |
| 2026-10-28 | S2-A Day 2 | `s2a_schema_updates` Migration（`task_status` / `rule_candidates` benchmark 字段 / `knowledge_gap` 一对多） |
| 2026-10-28 | S2-B / S2-C | 专家判定 PATCH + 真实指标聚合 |
| 2026-10-29 | S2-D / S2-E / S2-G | 前端专家判定 + 摘要卡片 + 规则候选审核 |
| 2026-10-30 | S2-H / S2-I | 报告导出 + 批量评审 + RAG 管理 + 指标看板 + CI 预研 |
| 2026-11-01 | F05 / F06 | 规则发布状态机 + 知识缺口 Backlog |
| 2026-11-02 | F07 / F15 | Case 管理 + Demo + 基线快照 + 全链路恢复 |

---

## 三、踩坑与经验教训

### 3.1 最大教训：工作区回滚导致 F05/F06 代码丢失

**现象**：F07 收尾后执行 `demo_sprint2.ps1`，后端启动直接崩溃（`ImportError`），接口 404，甚至数据库层出现「列级不一致」。

**根因**：F05/F06 收尾后，工作区被某次 git 操作错误回滚，导致：
- `backend/alembic/versions/s2_f05_rule_publish.py` 丢失
- `backend/app/api/rule_candidates.py` 的 `/publish` 端点丢失
- `backend/app/services/rule_evolution_service.py` 的 3 个异常类 + `publish_rule_candidate` 方法丢失
- `backend/app/api/knowledge_gaps.py` 和 `knowledge_gap_service.py` 丢失
- `backend/app/api/__init__.py` 的路由注册丢失
- `backend/app/api/schemas.py` 的 F05（2 个）和 F06（4 个）Pydantic 模型丢失
- `backend/app/persistence/models.py` 的 `RuleCandidate` 5 个 F05 发布字段丢失

**关键点**：`git status` 里这些文件**没显示 `D`（deleted）状态**，因为它们被物理删除，但 git index 里仍保留了 F05/F06 commit 的记录，所以 git 看不到删除动作。

**修复过程**（每一步都触发新的 ImportError，逐层往上推）：
1. `schemas.py` 缺 `PublishRuleRequest` → 手工补 2 个 F05 模型
2. `rule_candidates.py` 缺 `RulePublishConflictError` → `git checkout 53878c7 -- ...`
3. `schemas.py` 缺 `KnowledgeGapListResponse` → 手工补 4 个 F06 模型
4. `knowledge_gaps.py` 不存在 → `git checkout 03ddf34 -- ...`
5. `api/__init__.py` 缺 `knowledge_gaps` 路由注册 → `git checkout 03ddf34 -- ...`
6. `models.py` 缺 `RuleCandidate` 5 字段 → 手工补

**教训**：
1. **共享文件（`schemas.py`）不能整体 checkout**，因为同一文件承载多个阶段的模型，直接 checkout 会丢后续改动，必须手工追加。
2. **每恢复一层就重启后端 + `healthz` 验证**，不要一次性恢复所有文件。
3. **必须用 `git show <commit>:<path>` 确认原始内容**，用 `git log <commit>..HEAD -- <path>` 确认文件后续是否被修改过，再决定能否 `git checkout`。
4. **`git status` 的 `D` 不是唯一信号**——文件被物理删除但 git index 仍有记录时，`git status` 完全看不到。

### 3.2 第二教训：PowerShell 5.1 的 5 大坑

| 坑 | 现象 | 修复 |
|---|---|---|
| **`>` 重定向写 UTF-16 LE** | `git show ... > file` 后用 Python 读取崩溃 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff` | 改用 `git checkout <commit> -- <path>` |
| **中文乱码** | PowerShell 5.1 用 GBK 解码 UTF-8 脚本，`Print-Step "阶段"` 报 `意外的标记"闃舵"` | 脚本内中文字符串全部改为英文 |
| **`Invoke-WebRequest` 安全警告** | 每次 POST 都弹 `安全警告: 脚本执行风险`，需手动按 `Y` | 加 `-UseBasicParsing` 参数 |
| **`Invoke-RestMethod` 吞错误 body** | 404 / 422 时只返回 `远程服务器返回错误`，看不到 `detail` | 改用 `Invoke-WebRequest` + `$_.Exception.Response.GetResponseStream()` |
| **`alembic current` 的 stdout 噪声** | `INFO [alembic.runtime.migration] ...` 打到 stdout，`2>$null` 屏蔽不掉，`Out-String` 后无法正则匹配 | 直接查 DB 的 `alembic_version` 表 `version_num` 字段 |

**核心经验**：在 PowerShell 5.1 环境下，任何涉及"多行脚本 + 命令输出解析 + 编码"的场景，一律用「临时 Python 脚本 + 直接查 DB」绕开，不要用 `python -c` + PowerShell here-string。

### 3.3 第三教训：Demo 脚本必须自动清理

**现象**：第一次跑 `demo_sprint2.ps1` 后，`test_rules.py::test_rules_loaded_exactly_ten` 失败，因为 `data/rules/` 多了 `DEMO_RULE_*.yaml`（每次发布测试都会生成新的）。

**根因**：Demo 脚本的 `Closure 2/4`（规则演进）通过 `POST /rule-candidates/{id}/publish` 真实发布规则到 `data/rules/`，但脚本没有清理机制。

**修复**：Demo 主流程末尾加 `Cleanup` 步骤：

```python
with get_db_session() as db:
    db.query(RuleDefinition).filter(RuleDefinition.rule_id.like("DEMO_RULE_%")).delete(...)
    db.query(RuleCandidate).filter(RuleCandidate.candidate_id.like("RC-DEMO-%")).delete(...)
    db.commit()

for f in Path("/data/rules").glob("DEMO_RULE_*.yaml"):
    f.unlink()
```

**教训**：任何"写数据"的脚本，必须配"清理"逻辑，且要放在 try/finally 或主流程末尾，确保异常时也清理。

### 3.4 第四教训：F05 的"批准 ≠ 发布"必须守住

**背景**：S2 冻结规则 "批准（accepted）只改状态，publish 才写 data/rules/ + DB 同步"。这是为了避免"批准即合入"的权限滥用风险。

**实现要点**：
1. `accepted` 状态只是数据库状态更新，不影响 `data/rules/`。
2. `publish_rule_candidate` 是独立方法，状态机 `accepted → publishing → published | publish_failed`。
3. 文件原子替换 + DB 同步 + 引擎验证（`load_rule_definitions`）全部成功后，才落 `published`。
4. 失败时 `_rollback_publish` 恢复文件 + 落 `publish_failed`，返回 200 + `publish_status="publish_failed"`（方案 A，前端可重试）。
5. 幂等 Key（`publish_idempotency_key`）避免重复发布。

**教训**：业务状态机不能省。哪怕只是为了"区分批准和发布"，也要独立字段 + 独立方法 + 独立异常类型。否则一旦业务上需要"撤销批准"或"发布后回滚"，系统就崩了。

### 3.5 第五教训：F07 的卡片交互冲突

**现象**：CaseListPage 卡片的"多选"和"跳详情"事件冲突，点击复选框却跳转到了详情页。

**根因**：最初尝试用 `absolute top-3 right-3` 定位复选框 + `@click.stop` 阻断冒泡。但 absolute 层被外层 flex 容器或底部按钮的 z-index 遮挡，点击事件冒泡到了卡片主体的 `@click`。

**修复**：放弃绝对定位，用 Flex 布局物理隔离点击区域——卡片顶部左侧是"标题 + 描述"（`@click="goToCaseDetail"`），右侧是"状态胶囊 + 复选框"（`@click.stop="toggleSelect"`）。

**教训**：Vue 事件穿透的终极防线是修改 DOM 树结构，而不是依赖 CSS 层级。UI 控件的视觉存在感也不能盲目信任原生 `<input>` 默认样式，要加边框、背景、hover 状态。

---

## 四、技术债与遗留项

| 项 | 影响 | 归属 | 优先级 |
|---|---|---|---|
| Sprint 1 候选 ID 特殊连字符 | RC‑2026... 含 U+2011 非 ASCII 字符，URL 编码后 404，前端 encodeURIComponent 兜底 | Sprint 3 修复 `_make_candidate_id()` 用 ASCII `-` | P1 |
| 发布后 Benchmark 回归未自动跑 | `publish_rule_candidate` 的 `benchmark_regression` 字段恒为 None | Sprint 3 或后续迭代补 | P1 |
| `scripts/rollback_rule.py`（F05 CLI） | 已实现（git checkout 恢复），但未在 `baseline_snapshot.md` 验证过 | Sprint 3 首次验证 | P2 |
| F07 的 `full_review` 模式未做浏览器验收 | 只测了 `rule_only`（不耗 LLM），`full_review`（20-30s）未跑过 | Sprint 3 首次全流程验证 | P2 |
| CI 平台未最终确认 | 当前只有 `ci.yml` 占位，未实际跑通 CI 流水线 | Sprint 3 落地 | P2 |
| `rule_candidates` 的 `_make_candidate_id()` 特殊字符 | 与 Sprint 1 候选 ID 特殊连字符重复 | Sprint 3 | P1 |
| `test_llm_zhipu.py` 中 1 个 skipped（minilm 未注册） | 不影响功能，minilm 是占位 | Sprint 3 决定是否补注册 | P3 |
| vLLM 未实装 | 只有占位（`VLLMProvider._do_complete` 抛 `NotImplementedError`） | Sprint 3 视 GPU 部署进展 | P3 |

---

## 五、Sprint 3 输入（V5.5 G2 前置核查）

`baseline_snapshot.md` 已明确 Sprint 3 的 4 个 G2 前置缺口：

| 缺口 | 当前状态 | G2 要做的事 |
|---|---|---|
| Authentication | `get_current_user` 不存在 | 新建 `app/core/auth.py`（`get_current_user` / `require_roles` / `get_optional_user`） |
| Audit Fields | `ReviewResult.created_by` / `RuleCandidate.reviewed_by` 不存在 | 通过 Migration `s2j_g2_roles.py` 加字段 + FK |
| Initial Admin | users 表 0 条记录 | 通过 `scripts/bootstrap_admin.py` CLI 创建初始 admin |
| Lock Mechanism | DB 是 postgresql，未选锁实现 | 用 `pg_advisory_xact_lock` 实现管理员并发保护 |

`baseline_snapshot.md` 已经确认的事实（G2 可直接用）：
- Git HEAD：`8073329`
- 测试基线：`568 passed, 1 skipped, 24 deselected, 0 failed`
- Alembic 版本：`s2_f07_case_benchmark`
- 数据库类型：`postgresql`
- users 表记录数：`0` 条
- 审计字段存在性：`created_by` / `reviewed_by` 不存在
- `get_current_user`：不存在

V5.5 附录 C 的 11 项核查，还需要补的几项：

| # | 核查项 | 状态 |
|---|---|---|
| 1 | `backend/app/api/deps.py` 完整内容 | ⚠️ 待补 |
| 2 | `backend/app/persistence/db.py` 的 `get_db_session` 实现 | ⚠️ 待补 |
| 3 | 项目数据库类型 | ✅ 已确认 postgresql |
| 4 | `s2a_schema_updates.py` 的 `revision` 变量值 | ✅ 已知 `s2a_schema_updates` |
| 5 | `app/api/` 全部 `@router.` 端点清单 | ⚠️ 待补 |
| 6 | `task_service.submit_task_feedback` 是否真的调用候选/缺口生成逻辑 | ⚠️ 待补 |
| 7 | `feedback_service.submit_feedback` 是否接受并写入 `created_by` | ⚠️ 待补 |
| 8 | `task_service.update_expert_decision` 是否接受 `updated_by` 并写入 `expert_decision_by` | ⚠️ 待补 |
| 9 | 当前 Git HEAD 与实际 pytest 结果 | ✅ 已在 `baseline_snapshot.md` |
| 10 | `dev.ps1` 的 start/restart 分支 `uv sync` 是否已删除 | ⚠️ 待补 |
| 11 | 历史 `users` 表内是否存在承担管理员角色的账号 | ✅ 已确认 users 表 0 条 |

**G2 的第一步**：补完 1 / 2 / 5 / 6 / 7 / 8 / 10 这 7 项代码事实，然后开始 `s2j_g2_roles.py` Migration。

---

## 六、总结与展望

### 6.1 关键成就

1. Sprint 2 全部 15 项功能完成，Sprint 2 里程碑达成。
2. 三条业务线闭环全部验证通过，产品可用性达到 MVP 标准。
3. 技术基线固化为 `baseline_snapshot.md`，Sprint 3 起步有据可依。
4. Demo 脚本自动化（`demo_sprint2.ps1`），后续回归测试可一键跑通。

### 6.2 关键教训

1. 工作区回滚是最大的隐患，Git 操作必须谨慎，`git checkout` / `git reset` 前必须确认影响范围。
2. PowerShell 5.1 的编码和输出解析是高频坑点，能绕则绕。
3. Demo 脚本必须自动清理，否则污染基线。
4. UI 事件隔离优先用 DOM 结构（Flex/Grid）而非 CSS 层级。
5. 业务状态机不能省，哪怕只是"批准 vs 发布"的区分。

### 6.3 Sprint 3 展望

Sprint 3 的核心是 V5.5 G2–G6：四角色权限隔离 + 端到端业务闭环。
- **G2**：Migration + `auth.py` + `bootstrap_admin.py` + 用户管理 API
- **G3**：业务路由挂鉴权 + 对象级数据范围（IDOR 测试）
- **G4**：前端登录 + 路由守卫 + 菜单/按钮权限
- **G5**：审计字段写入 + 反馈闭环 E2E
- **G6**：完整回归 + 负向安全测试 + 部署检查清单

Sprint 3 的起步输入已经就绪（`baseline_snapshot.md` + 本复盘），只需要补齐 7 项代码事实，就可以开 `s2j_g2_roles.py` Migration。

---

## 七、致谢与感言

Sprint 2 从 2026-10-27 到 2026-11-02，连续 7 天，中间经历了工作区回滚、ImportError 连环、Demo 脚本反复调试，最终 **568 passed、0 failed** 封板。

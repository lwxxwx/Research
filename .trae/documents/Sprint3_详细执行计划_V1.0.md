# 角色权限与端到端业务闭环实施方案（G1–G6 施工图 · 最终稿 V1.0）

> **文档性质**：实施路线图，不含具体代码。
>
> **修订日期**：2026-11-03（V1.0 评审定稿修订：C1/C2 决策定案 + 评审意见全部落地，版本标识保持 V1.0）
>
> **V1.0 最终修订点**：
> 1. 修正 §2.1 对历史 users 数据与 `DEFAULT 'engineer'` 的技术表述：不再使用“默认覆盖”措辞，明确必须验证 Migration 对既有记录的实际填充值，不得依赖未经验证的数据库默认行为。
> 2. 修正 §9 `POST /auth/login` 权限矩阵：明确 dev/test 的 username-only 身份模拟与 prod 正式认证策略由 §1 环境边界统一约束。
> 3. 除上述两项一致性修订外，不新增 P0/P1 功能范围，不改变 G1–G6 阶段边界、权限矩阵、锁机制、指标口径和既有验收要求。
>
> **V1.0（Sprint2 基线核查增补，版本标识保持不变）**：
> 本文档在 V1.0 基础上，对照 Sprint2 全套已封板技术文档（`changelog.md`、`frontend_debug.md`、`retrospective.md`、`baseline_audit.md`、`demo_sprint2.ps1`、`Sprint2_详细执行计划_V1.0.md`）做了一次系统性核对，把 Sprint2 历史踩坑、遗留项、测试/CI/Demo 基础设施反推为 Sprint3 的落地约束与验收条件，**全部为补充约束、核查项、验收条件与文档对齐，不新增、不改变任何 P0/P1 功能范围与 G1–G6 阶段边界**。本次增补点汇总：
> - 附录 C 前置核查由 11 项扩充为 **13 项**（新增候选 ID 特殊连字符核查、脚本 uv sync 清理核查），并修正第 8 项为「expert_decision_by 由后端注入、禁止前端传入」；
> - §2.1 Migration 增加 dry-run / ORM import / 双向回滚 / 基线对比验收约束；
> - §2.2/§2.3 补充 PowerShell 脚本编码坑、真实 PostgreSQL 并发测试、结构化审计日志字段约定；
> - §3.3 增加 `Depends(get_current_user)` autobegin 事务冲突的验收测试场景；
> - §7 明确 4 个审计字段一律由后端注入、Pydantic Schema 禁止前端传入；
> - §9 补充 `GET /cases/{case_id}` 对象级权限与 `POST /review/tasks/batch` 归属校验；
> - §8 补充前端踩坑（Vite 重启、数组响应式、空值渲染回归、localStorage 不可信）；
> - 新增 **第 12 条：CI 与 Demo 脚本适配**（CI 环境约束、PostgreSQL 并发测试、demo 脚本 RBAC 闭环 Phase 4）；
> - 附录 B 一致性检查由 29 项扩充为 **35 项**；附录 D 追加 Sprint2 遗留 P1/P2 技术债；新增 **附录 F：G2 开发启动前置 Checklist**。
>
> **V1.0（专家评审修订，版本标识保持不变）**：
> 本版经两位专家评审后修订，采纳全部意见，**保留既有 G2–G6 实施安排，不新增 G7、不新增业务模块、不为未来扩展提前设计复杂架构**。本次修订汇总：
> - **P0（必须）**：① 删除与 Sprint3 功能范围矛盾描述、统一范围；② 测试基线统一口径（`562` 标注为历史中间快照、`568` 为冻结封板基线，标记待复核、以冻结版本重测）；③ 附录 B 验收状态区分「已验证 / 待验证 / 不适用」；④ 明确 JWT 最小必要安全要求（仅作未来阶段范围边界，不纳入 G2–G6 实现）；
> - **P1（建议）**：⑤ 补齐 API 路由矩阵实际业务接口；⑥ Case 数据归属依据现有数据结构确定；⑦ 数据库迁移增加真实 PostgreSQL 迁移验证；⑧ Demo 清理改用临时数据库 / Schema；⑨ CI 数据库连接按真实执行环境确认；
> - **P2（范围优化）**：⑩ 删除 Vue 数组操作绝对化限制；⑪ 技术债与扩展性任务明确移入下一阶段；
> - **P0 统一验收**：⑫ 新增「第 13 条：Sprint 3 统一验收」章。
>
> **V1.0（评审定稿修订 · 2026-11-03，实质性规则定案，版本标识保持 V1.0）**：
> 本版完成《Sprint3_方案评审意见与待确认修改清单.md》全部意见落地，并按项目负责人决策定案两个核心事项。本轮为**实质性规则定稿**（非纯文字整理），修订记录见附录 A。本次定案与修订：
> - **C1 定案（方案 B）**：第一阶段**不交付 JWT / 密码登录**，当前最重要目标是把三条业务线端到端跑通；第一阶段正式部署采用**基于角色的身份体系（username-only / X-User-Id）+ 可信内网部署**，生产环境**解除「无 JWT 即启动失败」约束**；正式 JWT / 密码登录统一移入下一阶段（Sprint4 及以后）。已同步修订：总体原则第 8 条、§1 环境边界、§9、§12.1、§13.1。
> - **C2 定案（PostgreSQL）**：Sprint2、CI、第一阶段目标部署数据库**全部为 PostgreSQL**；锁实现确定为 `pg_advisory_xact_lock`；MySQL / SQLite 仅作**未来扩展占位**，不作为正式施工要求；附录 C #3 改为核查连接配置、数据库版本及各环境一致性。已同步修订：§2.1、§2.3、§2.4、§12.1、§13、附录 B 第 17/32 项、附录 C #3、附录 F。
> - **评审 P0/P1/P2 落地**：P0-3（附录 B 第 32 项拆分）、P0-4（基线验收口径）、P0-5（验收证据要求）；P1-1（Case 权限统一）、P1-2（dev/test 身份模拟流程）、P1-3（Benchmark 边界）、P1-4（反馈链路判定）、P1-5（Migration 回滚环境）、P1-6（Demo 只读边界）、P1-7（路由矩阵精确化）、P1-8（验收证据字段）；P2-1（版本历史清晰化）、P2-2（附录 D 去重）、P2-3（Vue 数组口径保留）、P2-4（技术债不纳入 Sprint3）。
>
> **第一阶段业务演进目标（C1 方案 B 下验收的业务主线）**：本阶段所有角色设置与权限均服务于以下三条业务线的端到端演进，作为验收的业务判据：
> 1. **规则演进闭环**：候选提出 → Benchmark → 审核 → 人工合入正式规则；
> 2. **知识反馈闭环**：专家反馈 → EAR / AI_NER → 知识缺口或新规则候选；
> 3. **评审质量闭环**：原理图评审 → 证据完整性 → 专家判定 → 全局 Dashboard。
>
> **版本号接续策略（按项目负责人确认）**：本方案版本标识保持 **V1.0**，不升级为 V1.1 及更高正式版本号；若后续仍需修改，版本号**回退至 V0.x 序列接续**（在 V0.x 区间递增），V1.0 保留给最终真正冻结的版本。详见附录 A。

---

## 总体原则

1. 复用现有 `users` 表，不另建用户体系。
2. 渐进式改造：G2 只加字段与基础设施，G3 才给业务路由加依赖，每阶段独立可测。
3. 后端强制鉴权：前端隐藏只是体验，安全边界在后端 `Depends`。
4. 不改指标口径：EAR / AI_NER / EVC 计算、样本门槛、Seed 排除、空值语义一律不动。
5. 批准不等于发布：Rule Candidate 审核只改状态，正式规则人工合入。
6. 旧数据兼容：517 条历史 `review_result` 的 `created_by` 留 NULL，不强制回填。
7. 审计字段不因物理删除丢失：用户软禁用优先，外键删除策略按审计保留原则设置。
8. **第一阶段不交付 JWT / 密码登录（C1 方案 B 定案）**：第一阶段身份体系基于角色（username-only / X-User-Id），生产环境以可信内网 / 受控网络部署；`assigned_to` 也不在本阶段。正式 JWT / 密码登录、复杂 Token 管理统一移入下一阶段（见 §1、附录 D）。
9. **目标数据库确定为 PostgreSQL（C2 定案）**：锁、Migration、CI、并发测试、最终验收统一基于 PostgreSQL；MySQL / SQLite 仅作未来扩展占位。

---

## 第 1 条：环境边界（隔离与身份约束）

**APP_ENV 失败关闭（fail-closed）**：
- `APP_ENV` 必须显式设置，取值仅允许 `dev` / `test` / `prod`；缺失或取值非法 → 应用启动失败，**不默认回退 dev**。
- 该机制在各环境均生效，是启动阶段的第一道边界。

**第一阶段身份体系（C1 方案 B 定案）**：
- 第一阶段**不交付 JWT / 密码登录**，统一采用**基于角色的身份体系**：`username-only` 身份 + `X-User-Id`，身份对应数据库中存在且 `is_active=true` 的用户，角色决定权限范围。
- `username-only / X-User-Id` 是第一阶段（开发 / 测试 / 生产内网）的身份机制；正式 JWT / 密码登录在下一阶段替换之。
- 角色（admin / engineer / reviewer / rule_admin）即第一阶段身份体系的核心载体，服务于三条业务线演进（见首页业务目标）。

**环境身份约束表**：

| 环境 | 允许的身份通道 | 约束 |
|---|---|---|
| dev | ✅ username-only / ✅ X-User-Id | 仅限本地隔离开发，禁止暴露外网 |
| test | ✅ username-only / ✅ X-User-Id | 隔离测试环境，禁止连接生产数据 |
| prod（第一阶段，JWT 未交付） | ✅ username-only / ✅ X-User-Id | **仅限可信内网 / 受控网络部署**，禁止暴露公网；应用**可正常启动**（解除“无 JWT 即 fail-closed”）；is_active=false 用户鉴权一律 401 |
| prod（下一阶段，JWT 已交付） | ❌ username-only / ❌ X-User-Id；✅ 正式 JWT | 完全移除 X-User-Id，使用正式 JWT 认证 |

> **V1.0 修订（评审 P0-1 方案 B 定案）**：原方案「prod 在 JWT 未完成时启动即 fail-closed、无任何身份通道」与「第一阶段不交付 JWT」矛盾。定案后第一阶段 prod 以内网可信部署 + 基于角色身份体系可正常启动；正式 JWT 移入下一阶段。
> **prod 内网部署说明**：第一阶段 prod 仅在可信内网 / 受控网络内运行，依赖网络层隔离保障安全；对外正式发布必须待下一阶段 JWT / 密码登录完成后方可开放公网。

**身份有效性补充约束**：
- `is_active=false` 的用户，无论携带任何请求头，鉴权一律返回 401，不能获得任何有效身份。
- 未认证用户不能通过其他接口获得身份信息或角色权限。
- 不允许前端通过修改 localStorage 中的用户 ID / 角色绕过后端认证（后端 Depends 是最终边界）。

**dev / test 身份模拟完整流程（评审 P1-2）**：

```text
dev / test：
POST /auth/login
    -> username-only 身份模拟
    -> 后端校验用户存在且 is_active=true
    -> 建立测试/开发身份
    -> 后续请求携带约定的 X-User-Id
    -> 后端每个受保护 API 重新校验用户状态与当前角色（Depends）

prod（第一阶段）：
同上，基于角色身份体系 + 可信内网部署
    -> 后端校验用户存在、启用、角色，执行对象级数据范围

prod（下一阶段）：
正式登录（JWT / 密码）
    -> 正式认证机制
    -> 后续请求携带正式凭证
    -> 后端校验凭证、用户状态和当前角色
```

**身份模拟调用约定（评审 P1-2）**：
1. dev/test 的模拟身份入口仅在对应环境启用，`APP_ENV` 非 dev/test 时不得加载或开放模拟通道；
2. 模拟身份必须对应数据库中存在且 `is_active=true` 的用户，否则按 401 / 未授权处理；
3. dev/test 模拟身份通道不得在 prod 被加载或开放（prod 第一阶段仅内网，仍需校验身份与角色）；
4. `X-User-Id` 不能与正式 JWT 同时作为身份来源；第一阶段未引入 JWT，故 X-User-Id 为唯一身份载体；
5. 前端 localStorage 中的 `role`、`user_id` 不作为后端可信身份，一律以后端 `GET /auth/me` 返回为准；
6. 所有受保护 API 最终都由后端依赖项完成身份与角色校验。

**下一阶段升级方向（JWT / 密码登录，不纳入本阶段）**：password_hash + JWT + 刷新 + 登出；移除 X-User-Id 与前端 localStorage 的 `x-user-id`。

> **JWT 最小必要安全要求（评审 P0-4，仅作下一阶段范围边界，不纳入第一阶段 G2–G6 实现）**：
> 下一阶段实装正式 JWT 时，至少满足以下**最小**要求；未明确要求的复杂安全机制（多端登录踢出、黑名单、非对称双证书轮换等）不在当前范围：
> 1. 密码以单向哈希存储（推荐 bcrypt / argon2），禁止明文；
> 2. Access Token 必须校验签名与过期时间，服务端不信任客户端传入的任意用户标识；
> 3. `GET /auth/me` 以 Token 内可信身份为准，前端 localStorage 不再作为权限来源；
> 4. 登出使服务端 Token 失效（刷新 / 吊销语义）。

---

## 第 2 条：Migration 与 Bootstrap Admin CLI

### 2.1 Migration（单文件 `s2j_g2_roles.py`）

只做结构变更，不含任何 INSERT：

| 表 | 变更 | 约束 / 说明 |
|---|---|---|
| users | 新增 role | NOT NULL DEFAULT 'engineer'，CHECK role IN ('admin','engineer','reviewer','rule_admin') |
| users | 新增 is_active | NOT NULL DEFAULT true |
| review_result | 新增 created_by | FK users.id ON DELETE SET NULL，允许 NULL（517 条历史数据不回填） |
| rule_candidates | 新增 reviewed_by | FK users.id ON DELETE SET NULL，允许 NULL |
| 索引 | idx_users_role、idx_review_result_created_by | — |

- `down_revision` 指向 `s2a_schema_updates.py` 的 revision（具体值待核实）。
- 初始管理员由独立 CLI 创建，Migration 不插入任何用户。
- `downgrade` 必须可回滚（drop 列 / 索引）。

**【V0.3-P1 保留：Migration 前置历史用户预检查】**

> 执行 alembic upgrade 前，必须人工/脚本完成如下预核查，作为 G2 前置核查环节：
1. 统计现有 `users` 表总记录数；
2. 校验用户名、email 唯一性；
3. 梳理存量账号业务用途（实际核查内容，包括是否存在系统维护账号、是否曾承担管理员职责等；**不预设任何虚构场景**）；
4. 说明既有数据迁移策略：新增 `role NOT NULL DEFAULT 'engineer'` 后，必须明确验证 Migration 对**既有 users 记录**的实际填充值，不得仅假定或依赖未经验证的数据库 DEFAULT 行为。本方案的目标初始化策略是既有用户统一初始化为 `engineer`。**待核查**：历史 `users` 表内是否确实存在承担管理员角色的账号（是否存在的结论来自附录 C 核查项 3 / 11，当前不预设）；若核查发现此类账号，必须在升级前确认其角色迁移策略，并在升级完成后通过正式管理员流程验证或修正 role。
5. 第一个 admin 账号，升级完成后只能通过 `bootstrap_admin.py` CLI 生成；
6. 升级完成校验：role / is_active 字段、CHECK 约束、外键、索引全部生效。

> downgrade 行为约定：downgrade 仅做**结构回滚删除新增列与索引**；不恢复新增列内业务数据；回滚前需要确认应用版本已经降级。

**【Sprint2 基线核查增补：Migration 执行与验收约束】**

> 继承 Sprint2 迁移踩坑经验，`s2j_g2_roles.py` 执行与验收须满足：
>
> 1. 执行 `alembic upgrade head --sql` 做 **dry-run**，人工核对全部 SQL 后再实际执行迁移，不得直接盲跑 `upgrade head`；
> 2. 迁移前后执行 **ORM import 校验**：`python -c "from app.persistence.models import User, ReviewResult, RuleCandidate; print('OK')"`，避免 ORM 字段定义错误导致迁移期运行时崩溃；
> 3. 完整验证 **`downgrade → upgrade head` 双向回滚**：回滚后恢复 Sprint2 原始 schema，不破坏 Sprint2 存量业务数据；
> 4. **目标数据库确认为 PostgreSQL（C2 定案）**：迁移与回滚验证必须基于**真实 PostgreSQL 实例**完成，不能以 SQLite 或内存库替代；重点验证新增列 / CHECK 约束 / 外键 / 索引在 PostgreSQL 上真实生效；
> 5. G2 执行 Migration 前，先运行 `demo_sprint2.ps1` **只读静态审计**部分获取基线快照（不写入业务数据，见 P1-6）；迁移完成后对比 `baseline_snapshot.md`，确认数据库类型、原始记录数无异常变更；
> 6. **前置事实**：Sprint2 冻结基线 `users` 表为 0 行，G2 Migration 完成后仍为 0 行，**不自动生成任何 admin 账号**；初始管理员只能通过 `bootstrap_admin.py` CLI 创建（§2.2）。

**【V1.0 修订（评审 P1-5）：Migration 双向回滚仅限临时验证环境】**

> 1. **双向回滚（`downgrade → upgrade head`）测试仅在临时 PostgreSQL 数据库或可丢弃测试环境执行**，不得在正式 / 共享开发库上做回滚演练；
> 2. 回滚前后检查 Sprint2 原有业务数据和 schema，确认无异常变更；
> 3. **明确认知**：`created_by` / `reviewed_by` 等新增列一旦随 downgrade 删除，其内的审计数据随之丢失，再次 `upgrade` 无法自动恢复原值；
> 4. **生产环境不得把删除新增列的 downgrade 当作常规回滚手段**；生产回退应结合应用版本回退、向前兼容迁移和备份恢复策略执行。

### 2.2 Bootstrap Admin CLI（`scripts/bootstrap_admin.py`，容器内 `/scripts/bootstrap_admin.py`）

**参数**：
- `--username`（必填）
- `--email`（可选）
- `--i-understand`（生产必填；仅防误操作，不代表认证或权限验证）

**五种判断场景（严格按顺序执行）**：

| 顺序 | 场景 | 行为 | 返回码 |
|---|---|---|---|
| 1 | 用户名存在，角色不是 admin | 提示角色冲突，不修改、不创建 | 1 |
| 2 | 用户名存在，是已禁用 admin | 提示管理员已禁用，不自动启用 | 1 |
| 3 | 用户名存在，是启用 admin | 提示管理员已存在，幂等成功 | 0 |
| 4 | 用户名不存在，系统已有启用 admin | 提示已有启用管理员，不创建 | 0 |
| 5 | 用户名不存在，系统无启用 admin | 创建初始 admin，打印 user_id | 0 |

**补充规则**：
- 判断顺序必须严格按表格执行。
- 不得将已禁用 admin 自动启用。
- 不得将普通用户自动提升为 admin。
- 已有启用 admin 时，新增管理员必须通过现有 admin 的 `POST /auth/users` 接口完成。
- 用户管理 API 仍须遵守最后管理员保护规则。
- 同一命令重复执行时，如果指定用户已经是启用 admin，应幂等成功。

**【Sprint2 基线核查增补：CLI 被 PowerShell 脚本调用的编码坑】**

> `bootstrap_admin.py` 会被 PowerShell 封装脚本（参考 `demo_sprint2.ps1`）调用，须满足：
> - **禁止在 PowerShell 脚本中使用 `> / >>` 重定向捕获 CLI 输出**（PowerShell 5.1 会生成 UTF-16-LE BOM，Python 读取会崩溃）；一律 `python script.py` 直接执行并捕获 stdout/stderr；
> - CLI 的标准输出应为纯 UTF-8 文本，不输出 BOM、不做 `Write-Host` 中文重定向；
> - 供 PowerShell 判断的返回码必须为整型且稳定（0/1，见五种场景表），不得依赖解析中英文 stdout 文本。

### 2.3 并发初始化保护与数据库锁实现（P1 / P0）

**【C2 定案：目标数据库确定为 PostgreSQL】**

> **目标数据库已确认（C2 定案）**：Sprint2、CI、第一阶段目标部署数据库均为 **PostgreSQL**。锁实现确定为 PostgreSQL 事务级 Advisory Lock `pg_advisory_xact_lock`（事务提交 / 回滚自动释放），需在 G2 验证事务绑定与跨连接 / 跨进程语义。
> **MySQL / SQLite 仅作未来扩展占位**，不作为正式施工要求，不纳入第一阶段锁实现 / 并发测试范围，避免开发误以为需要兼容多数据库。

**锁实现（正式，PostgreSQL）**：
- 统一使用 `pg_advisory_xact_lock`（事务级 Advisory Lock），事务提交 / 回滚自动释放。
- 固定锁 key 集中定义，CLI 与所有管理员变更操作共用同一 key（与 §3.3 / §3.4 同一把锁）。
- 需在 G2 验证事务绑定、跨连接 / 跨进程语义、锁超时行为。

> 硬性约束：**未完成 PostgreSQL 锁机制的功能 + 并发测试验证，不得验收管理员并发保护相关功能**。该锁机制与 §3.3 的管理员变更锁为同一把锁，统一覆盖：CLI 初始化、API 管理员降级、API 管理员软禁用、未来所有会增减启用管理员的入口。需要考虑连接池、事务异常中断、锁超时场景。（锁实现细节与 §3.4 共用，不在 §3.4 重复展开；API 侧锁的使用见 §3.2 / §3.3。）

- 管理员状态检查与初始管理员创建必须并发安全，多个 CLI 实例不能同时基于"系统无启用管理员"创建。
- 必须使用事务级互斥机制；禁止仅用进程内 Python Lock / 线程锁 / 全局变量（多进程、多容器不共享）。
- 锁的获取、用户名检查、启用管理员检查、创建、提交必须在同一事务内；不得拆事务；不得在提交前释放事务级锁。
- 优先复用第 3 条统一管理员变更互斥锁与固定锁 key，使 CLI 与用户管理 API 串行化。

获取锁后必须重新读取数据库状态，不能沿用取锁前的查询结果：
- 另一个 CLI 已创建启用管理员 → 后取得锁者按场景 4 处理。
- 相同用户名并发 → 后取得锁者在锁内重读，命中场景 3 幂等成功；**正常路径不应发生唯一约束冲突（锁已串行化）**。

**唯一约束冲突恢复（防御性设计，单独测试）**：
1. **仅捕获用户名唯一约束冲突；其他 IntegrityError（外键违规、CHECK 失败等）按真实异常处理，不回滚重试。**
2. 完整回滚原事务 → 不得继续使用失效的事务 / Session 状态。
3. 开启新事务 → 重新取得同一把统一管理员变更互斥锁。
4. 重新读取：用户名是否存在、该用户 role、该用户 is_active、系统是否有启用 admin。
5. 按五种场景重新判断；只有确认命中幂等成功才返回成功，否则按真实异常失败退出。
6. 不得把唯一约束异常直接视为幂等成功；不得回滚后无锁重读。

### 2.4 并发测试要求（A / B / C 三类）

> **V1.0 修订（评审 C3 确认 + P0-3）**：A / C 类在 G2 完成并验收；**B 类（唯一约束冲突恢复）延后至 G6 完整回归阶段补齐验证**，G2 不因 B 类未测而阻塞管理员并发保护验收。

测试必须验证最终数据库状态，而不仅是 CLI 返回码：
- **A 类（正常并发，锁串行化）**：不同用户名同时初始化（最多一个启用 admin）；相同用户名同时初始化（幂等成功，不预期唯一约束冲突）；一个完成后另一个执行（场景 4）；CLI 与 API 并发（统一互斥，无状态竞争）。
- **B 类（唯一约束冲突恢复，防御性）**：通过受控手段构造冲突，验证回滚 → 新事务 → 重新加锁 → 重读 → 按场景判断；冲突后重查发现非 admin 用户按场景 1 退出；发现已有其他启用 admin 按场景 4 退出；状态仍不合法则失败退出，不伪装成功。**【V1.0 修订：B 类延后至 G6 验证**，G2 仅做 A / C 类；避免 G2 范围膨胀。】
- **C 类（锁机制真实性）**：必须用 PostgreSQL（目标部署数据库）验证，SQLite / Mock 不能单独作为通过依据。

> **V1.0 修订（评审 P0-3，测试类别状态拆分）**：并发测试按类别分别记录计划阶段与通过证据（见附录 B 第 32 项）。

**【V0.3-P1 保留：CLI/API 管理员操作审计要求】**

> G2 阶段不实现独立审计表，但必须输出结构化审计日志（stdout + 应用日志文件）；字段：操作来源 [cli/api]、操作者 system/actor_id、被操作用户 id、变更前后 role、变更前后 is_active、时间戳。
> 覆盖场景：CLI 初始化管理员；API 创建用户；API 修改角色；API 软禁用/启用用户。
> 注意：`review_result.created_by`、`rule_candidates.reviewed_by` 是业务归属字段，不等于用户管理操作审计。

**【Sprint2 基线核查增补：并发测试真实性 + 审计日志字段细化】**

> 1. §2.4 的 A/B/C 三类并发测试**必须使用真实 PostgreSQL 数据库完成**，禁止仅以 SQLite / Mock 作为通过依据（Sprint2 大量踩坑为 mock 与真实数据库行为不一致）；
> 2. 锁超时异常需输出结构化审计日志，**不得伪装业务成功**；锁超时视为基础设施异常并告警；
> 3. 审计日志字段落地建议（对齐 Sprint2 demo 脚本审计口径）：`操作来源[cli/api]`、`操作者 system/actor_id`、`被操作用户 id`、`变更前 role`、`变更后 role`、`变更前 is_active`、`变更后 is_active`、`时间戳`；写入 stdout + 应用日志文件，字段以 KV 形式输出便于检索。

---

## 第 3 条：最后管理员保护（P0 核心）

### 3.1 触发条件与业务规则

触发条件：目标用户当前是启用状态的 admin，且本次操作会减少启用管理员数量。

- 目标用户不是启用 admin → 不触发保护。
- 是启用 admin 且 `active_admin_count <= 1` → 拒绝（`LastAdminProtectionError`）。判断用 `<= 1`，不是 `< 1`。
- 是启用 admin 且 `active_admin_count >= 2` → 允许。

**PATCH 场景表**（"是否触发保护"= 是否进入最后管理员保护判断，实际是否拒绝还取决于管理员数量）：

| 当前状态 | PATCH 请求 | 是否触发保护 |
|---|---|---|
| 启用 admin | 只改 email / username | 否 |
| 启用 admin | 只改 role=engineer | 是 |
| 启用 admin | 只改 is_active=false | 是 |
| 启用 admin | role=engineer 且 is_active=false | 是 |
| 启用 admin | role=admin 且 is_active=true | 否 |
| 已禁用 admin | 改 email / 其他非状态字段 | 否 |
| 启用 engineer | 改 role=reviewer | 否 |
| 启用 engineer | 改 is_active=false | 否 |

### 3.2 统一 Service 保护机制

以下操作必须复用同一套 Service 层保护机制，不得在路由层分散实现、不得绕过：

- `PATCH /auth/users/{id}`：admin 降级。
- `DELETE /auth/users/{id}`：启用 admin 软禁用。
- PATCH 同时修改 role 与 is_active。
- 未来任何减少启用管理员数量的入口。

### 3.3 事务所有权与锁顺序（关键）

**锁获取顺序（不得改变）**：

```text
1. 确定事务边界并进入同一个数据库事务
2. 首先取得统一的管理员变更互斥锁
3. 再锁定目标用户行
4. 查询当前启用管理员数量
5. 判断是否触发最后管理员保护
6. 更新用户
7. 提交事务
8. 事务级锁随事务结束自动释放
```

禁止先锁目标用户再取全局锁（死锁风险）。

**【V0.3-P0 保留：事务隔离级别与快照一致性约束】**

> 风险说明：若事务隔离级别为 REPEATABLE-READ，事务开启瞬间生成快照；即便后续拿到互斥锁，count 查询仍然读取旧快照，会绕过最后管理员保护。
1. 管理员变更相关事务，**优先使用 READ COMMITTED 隔离级别**，保证拿到互斥锁之后，count 查询读取已经提交的最新数据库状态。
2. 如果项目全局不能使用 READ COMMITTED，则管理员变更业务可**使用独立 Session 显式设置隔离级别**（不复用普通请求 Session）。
   > **表述澄清（V0.4）**：独立 Session 是"隔离级别设置的显式载体"，**不是规避事务一致性 / 拆分事务的手段**。若采用独立 Session，关键锁、数量查询、用户更新必须全部使用**同一个独立 Session 与同一事务**，且保持"锁 → 查询 → 更新"顺序；不允许借此把锁与更新拆到两个事务。
   > **验证要求（V0.5）**：独立 Session 是否真正应用了 READ COMMITTED，必须有可观测验证（如测试中查询 `SHOW transaction_isolation`，或在连接建立事件上断言），不能只依赖代码中写了 `isolation_level="READ COMMITTED"`。
3. `get_current_user` 查询和管理员变更业务必须明确 Session 事务所有权；不能依赖旧事务快照做 `active_admin_count` 判断。
4. 必须在真实 PostgreSQL 并发场景下验证保护逻辑，不能仅单元测试验证锁对象获取。

**SQLAlchemy Session 事务所有权要求**：
- `Depends(get_current_user)` 可能已通过同一 Session 触发 autobegin，Service 再调 `with db.begin()` 可能报 `InvalidRequestError: A transaction is already begun`。
- 禁止在已开启事务的 Session 上未经判断直接嵌套 `with db.begin()`。
- 禁止为规避冲突把 advisory lock 与用户更新拆到两个事务；禁止提交前释放事务级锁；禁止锁释放后才做数量判断或更新。
- 全局锁 + 行锁 + 数量查询 + 更新必须在同一事务。
- 若采用独立 Session，关键锁、查询、更新必须全部使用同一个独立 Session 与同一事务。
- 具体实现以 G2 前置核查（附录 C）结果为准；不得为修事务问题改变业务判断。

> **Sprint2 基线核查增补（autobegin 事务冲突验收测试场景）**：
> `Depends(get_current_user)` 可能提前触发 Session autobegin，service 层再次调用 `with db.begin()` 会抛 `InvalidRequestError: A transaction is already begun`。
> 具体化测试用例（加入 G2 单元/集成测试清单，真实 PostgreSQL 下验证）：
> - **用例 1（正常路径）**：`GET /auth/me` 鉴权通过后，直接调 `PATCH /auth/users/{id}` 降级一个启用 admin（且非最后一个），断言返回 200、DB 中 role 已更新，**且全程不抛 `InvalidRequestError`**；
> - **用例 2（受保护路径）**：`GET /auth/me` 鉴权通过后，调 `PATCH` 试图降级唯一启用 admin，断言返回 `LastAdminProtectionError`（业务异常），**不抛 `InvalidRequestError`**，DB 状态未变更；
> - **用例 3（session 复用）**：同一请求内先触发一次 `get_current_user` 查询，再进入管理员变更 Service，断言 Service 能正确判断 Session 事务状态、不复用已开启事务快照、不重复 `with db.begin()`。

> 补充锁超时要求：管理员变更互斥锁必须设置合理锁等待超时；超时抛出独立锁超时异常，区分业务异常 `LastAdminProtectionError`；锁超时视为基础设施异常，告警，不伪装业务成功。

### 3.4 锁机制与数据库类型（PostgreSQL 定案）

- 目标数据库已确认为 PostgreSQL，锁实现确定为 `pg_advisory_xact_lock`（详见 §2.3），此处不重复展开。
- MySQL / SQLite 为未来扩展占位，不作为第一阶段实现范围。
- 固定锁 key 集中定义，CLI 与所有管理员变更操作共用同一 key。

### 3.5 PATCH 部分更新与显式 null

| 请求情况 | 处理方式 |
|---|---|
| 未提供 role | 保留数据库原 role |
| 未提供 is_active | 保留数据库原 is_active |
| 提供合法 role / is_active | 用请求值参与最终状态计算 |
| 显式 role=null | HTTP 422，不进入 Service |
| 显式 is_active=null | HTTP 422，不进入 Service |

- role、is_active 数据库层均 NOT NULL；Schema 允许字段缺省、不允许显式 null。
- 字段存在性判断用 `model_fields_set` / `exclude_unset=True` 或等效机制；不得用 `exclude_none=True` 作为唯一依据（可能掩盖显式 null）。
- 校验失败不进入 Service，不执行数量检查或数据库更新。
- 只更新本次实际提供且通过校验的字段。

**最终状态计算**：

```text
effective_role       = patch.role       若请求提供 role，否则 current_role
effective_is_active  = patch.is_active  若请求提供 is_active，否则 current_is_active

was_active_admin     = current_role == "admin"      AND current_is_active is True
will_be_active_admin = effective_role == "admin"    AND effective_is_active is True
should_protect       = was_active_admin AND NOT will_be_active_admin
```

`should_protect == True` 时才检查 `active_admin_count <= 1`，成立则抛 `LastAdminProtectionError`。

**伪代码示意**：

```python
def update_user_role(db, user_id, patch_data, actor_id):
    with <事务上下文>:                      # 事务所有权遵循 3.3
        <取得统一管理员变更互斥锁>
        <锁定目标用户行>
        target = <查询目标用户>
        if target is None:
            raise NotFound

        current_role, current_is_active = target.role, target.is_active
        effective_role = patch_data["role"] if "role" in patch_data else current_role
        effective_is_active = (
            patch_data["is_active"] if "is_active" in patch_data else current_is_active
        )
        was_active_admin = current_role == "admin" and current_is_active is True
        will_be_active_admin = effective_role == "admin" and effective_is_active is True

        if was_active_admin and not will_be_active_admin:
            if <查询启用管理员数量> <= 1:
                raise LastAdminProtectionError("不能降级或停用最后一个启用管理员")

        <只更新本次 PATCH 实际提供的字段>
```

> 伪代码仅表达业务逻辑；事务上下文、Session 复用、数据库锁实现以 G2 核查为准。

### 3.6 PATCH 输入测试

| 输入 | 预期 |
|---|---|
| `{}` | 不修改 role / is_active（空 PATCH 行为以现有 API 契约为准，待核实） |
| `{"role": "engineer"}` | 改 role，is_active 保持 |
| `{"is_active": false}` | 改 is_active，role 保持 |
| `{"role": null}` / `{"is_active": null}` | 422，Service 不被调用，数据库不变（不仅看返回码） |
| `{"role": "unknown"}` | 422，不执行更新 |
| `{"role": "admin", "is_active": true}` | 按最终状态规则判断 |
| `{"role": "engineer", "is_active": false}` | 按最后管理员保护规则判断 |

### 3.7 并发与部分更新测试

| 场景 | 预期 |
|---|---|
| 仅一个启用 admin，只 PATCH is_active=false / role=engineer / 两者同时 | 拒绝 |
| 两个启用 admin，PATCH 其中一个降级 | 允许，最终保留一个启用 admin |
| 仅 PATCH email | 不触发保护，email 正常更新 |
| PATCH 未提供 role / is_active | 原值不变 |
| 并发 PATCH 不同管理员 | 统一互斥锁生效，最终至少保留一个启用 admin |

必须用真实 PostgreSQL 验证最终 DB 状态；SQLite / Mock 不能单独作为依据。

---

## 第 4 条：审计字段与用户软禁用

- 不提供物理删除用户 API；`DELETE /auth/users/{id}` 语义为软禁用（is_active=false）。

**外键删除策略**：

| 字段 | 策略 |
|---|---|
| review_result.created_by | ON DELETE SET NULL |
| rule_candidates.reviewed_by | ON DELETE SET NULL |
| review_defect.expert_decision_by | NO ACTION（不动 DB） |
| feedback_item.created_by | NO ACTION（不动 DB） |

**审计查询口径**：

```text
用户禁用：
  is_active=false，用户记录保留，审计查询仍可 join 出 username，
  并可显示"用户已禁用"状态。

用户物理删除：
  对 ON DELETE SET NULL 字段，外键置 NULL，前端显示"已删除用户"。
  对 NO ACTION 字段，仍存在引用记录时数据库拒绝物理删除。

因此系统不提供物理删除用户 API，软禁用是唯一正常停用方式。
```

**【V0.3-P1 保留：用户软禁用完整行为定义】**

1. 用户设置 `is_active=false`，**立即对后续所有请求生效；现有会话不再具备任何权限**。
2. 用户历史创建任务、反馈、专家判定记录全部保留，不删除、不迁移。
3. 历史审计展示：仍然展示原用户名，附加标记「用户已禁用」。
4. 用户重新启用：恢复身份权限；**不修改任何历史业务记录的归属字段**。
5. 用户名全局唯一；被禁用的用户名**不允许被新账号复用**，保障审计可追溯。

---

## 第 5 条：认证模块结构（防循环导入，仅模块结构）

| 文件 | 职责 | 导入约束 |
|---|---|---|
| `app/core/auth.py`（新建） | 定义 ROLE_*、get_current_user、require_roles、get_optional_user | 依赖 `app.persistence.db.get_db_session`；不 import `app.api.*` |
| `app/api/deps.py` | re-export get_current_user / require_roles | — |
| `app/api/auth.py`（新建） | 登录 / me / 用户管理端点 | 只 import `app.core.auth` 与 `app.persistence` |

`require_roles(*roles)` 返回内部 `_dependency(current_user = Depends(get_current_user))`，角色不符 → 403。

（环境 / 身份 / APP_ENV 相关约束见第 1 条。第一阶段身份基于 username-only / X-User-Id，不涉及 JWT。）

---

## 第 6 条：G5 指标口径（冻结，不可改）

唯一真值：`metrics_definition.md` v3.2；实现以 `benchmark_service.compute_real_ear()` / `compute_real_ai_ner()` / `compute_sample_info()` / `calc_case_evc()` 为准（函数名待核实）。

**`[SEED]` 排除适用性**：
- **适用**：真实 EAR、真实 AI_NER、依赖反馈样本量的 sample_info（如 ear_sample_size）。
- **不适用**：EVC（GT 侧证据完整率，与反馈无关）。

**EAR**：N_adopted / N_defect_with_feedback；分母仅含采纳相关反馈（correct_defect / suggestion_update / false_positive / false_negative），排除 `[SEED]`；`>= 20` 才非 NULL。禁止把反馈数等同 EAR 样本数。

**AI_NER**：以 v3.2 与 `compute_real_ai_ner()` 实际实现为准；分别核对 AI 缺陷数、样本量、分子、分母四个量；不得把 AI 缺陷数等同样本量；门槛 10、去重、专家确认影响以冻结定义与代码为准；文档与代码不一致先记录差异。

**EVC**：仅对 `task_status='done'` 且 `evidence_coverage IS NOT NULL` 的任务取算术平均，不按缺陷数加权；无有效样本为 None。

**空值与前端展示**：
- 无有效样本 → None → N/A
- 样本不足（EAR < 20、AI_NER < 10）→ None → N/A · 样本不足（可附样本量 / 门槛）
- EVC 无有效样本 → None → N/A
- **绝不把 None 渲染成 0%**

**G5 反馈生成验证**：由 reviewer 执行（提交 false_negative 验 Rule Candidate、提交 knowledge_gap 验 KnowledgeGap、验 gap_feedback_link）；不得为验收开放 engineer 反馈权限。若业务逻辑未自动生成候选 / 缺口，按 §10 判定规则处理（见 P1-4）。

---

## 第 7 条：审计字段与后端注入

- `feedback_item.created_by`、`review_defect.expert_decision_by` 当前均为 Optional[int]、FK users.id、允许 NULL、无删除策略。
- 本阶段不改 DB 结构；保持 NO ACTION。
- 需核实：`feedback_service.submit_feedback` 是否接受并写入 created_by；`task_service.update_expert_decision` 是否接受 updated_by 并写入 expert_decision_by（附录 C / D）。

**【Sprint2 基线核查增补：审计字段一律后端注入，禁止前端传入】**

> G5 审计字段写入统一原则：**4 个审计字段全部由后端从 `get_current_user` 填充当前登录用户，Pydantic Schema 直接排除该字段，不接受 request body 传入**，避免 IDOR / 身份伪造风险。逐项约定：

| 字段 | 写入接口 | 后端注入约定 |
|---|---|---|
| `review_result.created_by` | `POST /review/tasks`、`POST /review/tasks/batch` | 后端自动填充当前登录用户 id；禁止前端传入 created_by |
| `rule_candidates.reviewed_by` | `POST /rule-candidates/{id}/approve` / `reject` | 后端填充登录用户 id；前端不可传入 |
| `feedback_item.created_by` | `POST /feedbacks` | 后端自动注入 actor_id；不允许前端传 created_by |
| `review_defect.expert_decision_by` | `PATCH /review/tasks/{id}/defects/{defect_id}` | **彻底移除请求体中 `expert_decision_by` 入参，后端直接取当前登录用户**；有登录用户写用户 id，无登录用户保留 NULL |

> 注：G5 验收明确**禁止**上述 4 个字段由前端 request body 携带；`expert_decision_by` 在 Sprint2 允许 null，Sprint3 有登录用户即写用户 id，无登录用户保留 null。

---

## 第 8 条：前端启动与权限展示

- 启动调 `GET /auth/me` 刷新状态：401 清身份跳 `/login`；403 保留身份显示提示；网络错误不清身份。
- 不依赖 localStorage 过期 role 做权限判断。
- 菜单按角色显示，按钮按角色禁用 / 隐藏；右上角显示用户名 + 退出。
- 后端 Depends 是最终安全边界，前端隐藏仅为体验。

**【Sprint2 基线核查增补：前端踩坑与约束】**

> 继承 Sprint2 `frontend_debug.md` 前端调试经验，G4 开发须遵守：
> 1. **Vite HMR / 代理配置**：修改 `vite.config.ts` 代理配置后**建议完全重启 Vite**，避免 HMR 未生效导致联调误判；
> 2. **数组响应式更新（V1.0 保留现有口径，评审 P2-3）**：修改用户 / 任务缺陷状态时，优先用 `splice` / 整行替换等**确保触发 Vue 响应式**的写法；**不建议直接改数组索引**（Sprint2 曾遇数组索引直接赋值不触发更新的情况），但不作绝对化强制，以实际可触发更新为准，不提升为本阶段硬性架构规范；
> 3. **空值渲染约束回归**：指标卡片 `TaskSummaryCard`、Dashboard 看板，**严格禁止把 null 渲染成 0%**；`N/A`、`N/A · 样本不足` 语义与 Sprint2 完全一致；G4 前端验收必须回归该部分 UI，权限切换后指标空值语义不变；
> 4. **localStorage 不可信**：禁止把 role / user_id 作为可信来源存储于 localStorage 并据此判断权限；一律以后端 `GET /auth/me` 返回值为准。

---

## 第 9 条：权限矩阵（完整，按实际路由精确化）

> **V1.0 修订（评审 P1-7）**：本表以「HTTP 方法 + 路径 + 允许角色 + 对象级数据范围」列示，清理 `*` 通配组合写法。**最终端点清单以附录 C #5 清点的真实 `@router.` 为准**；下表为方案级对齐，不臆造未确认端点，不遗漏实际暴露端点。以下路由中标注 `（待核查）` 的，须在 G2 前置核查确认后据实落地，不得默认成立。

| HTTP 方法 | 路径 | 允许角色 | 对象级数据范围 |
|---|---|---|---|
| GET | /healthz | 公开 | — |
| POST | /auth/login | dev/test：身份模拟；prod：以第 1 条环境身份策略为准 | 仅已有且启用用户 |
| GET | /auth/me | 登录用户 | 当前用户 |
| GET | /auth/users | admin | 全部用户 |
| POST | /auth/users | admin | 新建用户（最后管理员保护适用） |
| PATCH | /auth/users/{id} | admin | 目标用户（最后管理员保护适用） |
| DELETE | /auth/users/{id} | admin | 目标用户（软禁用，最后管理员保护适用） |
| GET | /cases | 所有登录用户 | 见下方 Case 数据范围规则（P1-1） |
| GET | /cases/{case_id} | 所有登录用户 | 见下方 Case 数据范围规则（P1-1） |
| POST | /cases/{case_id}/benchmark | 所有登录用户（角色边界待核查，见 P1-3） | 见下方 Case 数据范围规则 |
| GET | /cases/{case_id}/benchmarks | 所有登录用户 | 见下方 Case 数据范围规则 |
| POST | /review/tasks | engineer / admin | created_by=me 写入 |
| POST | /review/tasks/batch | engineer / admin | 批量任务 created_by=me 写入（归属校验见下） |
| GET | /review/tasks | admin / engineer / reviewer | engineer 仅 created_by=me；reviewer/admin 全部 |
| GET | /review/tasks/{id} | admin / engineer / reviewer | engineer 仅自己任务；reviewer/admin 全部 |
| GET | /review/tasks/{id}/defects | admin / engineer / reviewer | engineer 仅自己任务；reviewer/admin 全部 |
| GET | /review/tasks/{id}/export | admin / engineer / reviewer | engineer 仅自己任务；reviewer/admin 全部 |
| PATCH | /review/tasks/{id}/defects/{defect_id} | reviewer / admin | reviewer 全部（仅评审/反馈/判定字段，不改归属字段）；admin 全部 |
| POST | /feedbacks | reviewer / admin（不开放 engineer） | 后端注入 created_by |
| POST | /rule-candidates/{id}/approve | rule_admin / admin | 目标候选（审核字段 reviewed_by 后端注入） |
| POST | /rule-candidates/{id}/reject | rule_admin / admin | 目标候选 |
| GET | /rule-candidates | rule_admin / admin | 全部候选 |
| PATCH | /rule-candidates/{id} | rule_admin / admin | 目标候选 |
| GET/POST | /knowledge-docs | rule_admin / admin | 全部文档（端点待附录 C #5 确认） |
| GET/POST | /knowledge-gaps | rule_admin / admin | 全部缺口（端点待附录 C #5 确认） |
| GET | /dashboard/summary | admin / rule_admin / reviewer | 全局指标 |

> 注：`GET/POST /rule-candidates/{id}/approve、/reject` 在原文中组合写法易被误读为 approve/reject 同时支持 GET/POST；定稿后按上表**分别列出实际存在的 POST 路由**，具体以附录 C #5 实际端点为准。

**数据范围**：engineer 只看 created_by=me；reviewer 第一阶段看全部；admin 全部；rule_admin 不自动获得任务详情权。

**【V1.0 修订（评审 P1-1）：Case 列表 / 详情 / Benchmark 对象级权限统一】**

> 依据现有 Case 数据结构统一规则，不新增 Case 所有权模型。核查以附录 C #5（端点清单 + 现有 ORM 模型）为准，不得臆测：

| Case 数据结构 | 列表规则（GET /cases） | 详情规则（GET /cases/{case_id}） | Benchmark 接口（POST /benchmark、GET /benchmarks） |
|---|---|---|---|
| Case 有明确归属字段（created_by / owner_id） | engineer 仅看自己的 Case；reviewer/admin 全部；rule_admin 按接口权限 | engineer 仅看自己的 Case；reviewer/admin 全部；rule_admin 按接口权限 | 与列表/详情一致的范围（基于 case_id 的访问范围一致，不遗漏） |
| Case 没有归属字段 | 沿用现有所有登录用户可见规则 | 沿用现有所有登录用户可见规则 | 沿用现有所有登录用户可见规则 |

> 关键：**不只要保护详情接口，还要对基于 `case_id` 创建 / 读取关联数据的接口（benchmark）应用相同访问范围**，避免列表泄露元数据而详情拒绝的不一致。

**【Sprint2 基线核查增补：批量评审接口归属校验】**

- 批量任务创建后 `created_by` 必须设置为**当前登录用户**（后端注入，见 §7）；
- engineer 不能通过 batch 接口创建他人归属任务；创建者归属校验必须后端实现，不能只靠前端限制；
- G3 需对该接口叠加角色鉴权 + 归属校验。

**【V1.0 修订（评审 P1-3）：Benchmark 接口角色与数据影响边界（待 G2 核查）】**

> 以下问题在 G2 前置核查（附录 C #5 / #6）核实现有业务定义后落地，**不额外设计新的角色或权限系统**：
> - engineer 是否可以对所有 Case 创建 Benchmark？（按 P1-1 Case 归属规则决定）
> - reviewer / rule_admin 是否可以创建 Benchmark？（按现有业务定义与 §9 上表）
> - Benchmark 执行结果是否影响正式指标或正式审核结果？（若影响，须纳入指标口径冻结范围；若仅为试算，不影响）
> - 创建 Benchmark 是否产生需要审计的记录？（按现有业务数据模型确定，不新增审计表）
> 上述结论记入 G2 核查记录，落地前不得默认成立。

**易漏点**：batch 每个子任务写 created_by；export 先校验归属；列表过滤与数据范围过滤叠加。

**【V0.3-P1 保留：对象级授权 / IDOR 验收约束】**

> 所有涉及 task_id / defect_id 的接口，除角色鉴权之外，**必须叠加对象级数据范围校验**，不能仅依赖列表过滤；重点接口：
> - GET /review/tasks/{id}
> - GET /review/tasks/{id}/defects
> - GET /review/tasks/{id}/export
> - PATCH /review/tasks/{id}/defects/{defect_id}
> 测试覆盖场景：

| 测试场景 | engineer | reviewer | admin | rule_admin |
|---|---|---|---|---|
| 查看自己任务 | 允许 | 允许 | 允许 | 按接口权限 |
| 查看其他 engineer 任务 | 拒绝 | 允许 | 允许 | 拒绝 |
| 直接通过 ID 访问他人任务 | 拒绝 | 允许 | 允许 | 拒绝 |
| 直接修改他人缺陷 | 拒绝 | 允许 | 允许 | 拒绝 |
| 导出他人任务 | 拒绝 | 允许 | 允许 | 拒绝 |

> **reviewer PATCH 边界澄清**：reviewer 第一阶段数据范围为全部任务，故 PATCH 任何任务的缺陷均允许；但该 PATCH 权限**仅作用于评审 / 反馈 / 判定相关字段（defect 状态、专家反馈等），不得通过 PATCH 改变任务或缺陷的归属字段（created_by 等）**，归属字段不在此接口可写范围。engineer 无论对他人还是自己的缺陷，均无 PATCH 权限。

> 额外校验点：
> 1. defect_id 必须校验归属所属 task，不能只校验 defect 自身；
> 2. engineer 不能依靠查询参数绕过数据过滤；
> 3. export 导出权限校验必须后端实现，不能只前端隐藏按钮；
> 4. 已禁用用户，全部接口返回鉴权失败。

> **V0.3-P2 标记（后续迭代）**：四角色完整职责文档、admin 是否隐式继承 reviewer/rule_admin 业务权限；记录附录 D 待迭代。

---

## 第 10 条：反馈生成链路核实与判定规则

需核实（附录 C / D）：`task_service.submit_task_feedback` 是否真的调用候选 / 缺口生成逻辑（rule_candidates / knowledge_gap / gap_feedback_link 是否有写入路径）。

**【V1.0 修订（评审 P1-4）：反馈链路缺失时的判定规则】**

> 避免出现「G5 核心业务链路未实现，却仍通过最终验收」。按以下规则判定：

| 情形 | 处理 |
|---|---|
| 现有代码已具备候选 / 缺口生成链路 | G5 完成权限接入、审计字段注入、端到端验证 |
| 现有代码缺失，但该链路属于第一阶段已确认业务需求 | 必须补齐完成第一阶段闭环所需的最小实现（不膨胀为完整生成系统） |
| 产品负责人确认该链路并非第一阶段必要功能 | 从本阶段验收范围移除，并记录明确的范围决策到评审记录 |
| 任何情形 | **不得为了通过测试临时开放 engineer 反馈权限，或伪造候选 / 缺口数据** |

> 第一阶段三条业务线（规则演进 / 知识反馈 / 评审质量）以端到端跑通为验收判据（见首页业务目标）；反馈生成链路是「知识反馈闭环」的关键环节，若判定为第一阶段必要功能则必须补齐最小实现。

---

## 第 11 条：测试基线（历史结果 vs 正式 Git HEAD 基线）

**历史已知结果（尚未构成 G2 正式基线）**：
- **562 passed, 1 skipped, 24 deselected, 0 failed（约 3 分钟）**：为 Sprint2 **历史中间快照**，早于最终封板。
- **568 passed, 1 skipped, 24 deselected**：为 Sprint2 **冻结封板基线**（用户已确认固定，待重测确认）。
- 1 skipped：`test_llm_zhipu.py::test_embedding_factory_minilm_returns_instance`（minilm 未注册到 `_PROVIDER`），正常跳过。
- 1 warning：Starlette 内部 PendingDeprecationWarning: Please use import python_multipart，Sprint 2 暂不修改依赖。
- 具体 Git HEAD 若未确认，不得补造 commit hash。

> **V1.0 修订（评审 P0-4）：测试基线验收口径**
> `562` 与 `568` 的差异来自不同阶段快照（新增用例后封板）。G2 正式基线以**冻结 Git HEAD 真实重测结果**为准，**不以某个数字作为唯一验收标准**。重测须记录完整 Git HEAD、工作区状态、测试命令、passed / failed / skipped / deselected、warnings、执行日期；若与历史记录不一致，必须记录真实差异并完成原因分析。Sprint3 最终回归时，Sprint2 原有测试不得出现未解释的新增失败；不得为对齐历史数字修改测试或跳过失败项。`568 passed, 1 skipped, 24 deselected` 保留为「待重测确认的历史封板记录」。

**G2 正式基线（进入编码前必须执行并记录）**：

```bash
git rev-parse HEAD
git status --short
pytest -m "not integration" -v      # 或项目真实测试命令
```

至少记录：Git HEAD（完整 hash）、工作区状态（clean 或列出未提交修改）、测试命令、passed / failed / skipped / deselected、warnings、执行日期。

**基线有效性原则**：
1. 必须对应明确 Git HEAD。
2. 工作区有未提交修改必须明确记录，不能描述成纯 HEAD 基线。
3. 基线测试失败不得直接开始 G2，先记录并判断是否既有问题。
4. G2 结束后用新 Git HEAD 相同口径回归。
5. 不得为对齐历史数字修改测试或跳过失败项。

**【V0.3-P1 保留：G1-G6 阶段总验收表】**

| 阶段 | 核心交付产物 | 进入下一阶段前置条件 |
|---|---|---|
| G1 | 方案评审完成，现状梳理核查 | V1.0 定稿确认，启动附录 C 前置核查 |
| G2 | Migration；auth 底座；bootstrap cli；用户管理接口；单元/并发测试 | Migration 双向回滚正常（临时库验证）；锁机制 A/C 类验证通过；auth 全套单元测试全部通过；基线重测回归通过 |
| G3 | 业务路由挂载鉴权；对象级数据范围；IDOR 测试 | 权限矩阵全部接口验收；IDOR 越权测试全部通过；存量用例回归 0 失败 |
| G4 | 前端登录、路由守卫、菜单按钮权限 | type-check / build 无报错；前后端权限行为对齐；401/403 处理符合约定 |
| G5 | 审计字段写入；反馈闭环；规则候选、知识缺口；指标看板 E2E 验证 | reviewer 完整端到端闭环；EAR/AI_NER/EVC 指标口径冻结不变；样本不足 N/A 语义正确 |
| G6 | 完整回归、负向安全测试、B 类并发测试、部署检查清单 | 全部测试用例通过；B 类并发测试补齐；benchmark 基线不变；CI 部署预案完备 |

---

## 第 12 条：CI 与 Demo 脚本适配（Sprint2 基线核查增补）

**【Sprint2 基线核查增补：本条为新增章节，继承 Sprint2 基础设施，不改变任何 P0/P1 功能范围】**

### 12.1 CI 环境适配

> 参考 Sprint2 CI 排坑经验，**目标数据库已确认为 PostgreSQL（C2 定案）**：
> 1. **CI 数据库连接按真实执行环境确认（评审 P1-9）**：CI 中数据库连接方式（localhost / service / 外部实例）以 Sprint2 实际 CI 配置为基准确认，不在本版臆测；prod 环境才拦截 localhost，test/dev 放行；Sprint3 新增 auth 模块、管理员锁后，`ci.yml` 必须继承现有 CI 脚本的连接逻辑；
> 2. CI 测试环境 `APP_ENV` **必须设置为 `test`**，**绝对不允许设置为 `prod`**；第一阶段 prod 以内网部署 + 基于角色身份，不开放模拟通道给 CI（详见 §1）；
> 3. CI 中 **PostgreSQL 作为 service** 跑**全部锁并发测试（§2.4 A/C，B 类延后 G6）**；**不要在 CI 只跑 sqlite/mock 单元测试**；
> 4. 测试基线：Sprint3 开发完成后，以冻结 Git HEAD 真实重测基线为准（详见 §11 P0-4）；新增 auth 相关测试后更新非 integration 测试基线，**不允许 Sprint2 原有用例失败**；
> 5. §11 G1–G6 验收表补充 CI 条件：每个阶段完成时，CI 后端 + 前端 job 必须全绿。

### 12.2 Demo 脚本升级需求（`demo_sprint2.ps1`）

> G6 完成后升级 demo 脚本，新增 **Phase 4：RBAC 权限闭环**。**G2 前置检查只做静态只读审计，不跑完整 RBAC 业务闭环；G6 完整阶段才跑完整 RBAC Demo**。Phase 4 闭环用例：
> 1. 用 `bootstrap_admin.py` CLI 创建初始 admin；
> 2. admin 调 `POST /auth/users` 创建 engineer、reviewer、rule_admin 账号；
> 3. 模拟 engineer 登录：验证不能访问他人任务、不能调用反馈接口；
> 4. 模拟 reviewer 登录：验证可评审缺陷，不能修改用户管理；
> 5. 模拟 rule_admin：验证规则候选权限；
> 6. 触发最后管理员保护：尝试停用最后一个 admin，接口返回业务异常（`LastAdminProtectionError`）；
> 7. 验证 `is_active=false` 禁用账号访问全部接口返回 401；
> 8. **Demo 数据隔离（评审 P1-8）**：demo 执行使用**临时数据库 / Schema**（如独立 demo schema 或一次性临时库），**不直接写生产或开发共享库**；demo 结束后丢弃临时库 / Schema，不污染业务数据；现有 `demo_sprint2.ps1` 中依赖业务库的做法在 Phase 4 升级时一并改为临时库 / Schema；
> 9. demo 执行完成后输出更新后的 `baseline_snapshot.md`，追加 RBAC 事实：auth 模块是否存在、admin 账号数量、锁验证状态。

**【V1.0 修订（评审 P1-6）：G2 前置检查与 G6 完整 Demo 的只读边界】**

> - **G2 前置检查**：只执行静态审计和只读快照，**不执行用户、任务、规则等业务数据写入，不执行清理业务数据**；生成 `baseline_snapshot.md` 不得污染 Sprint2 历史数据库或共享开发数据库；
> - **G6 完整 Demo**：使用临时数据库 / Schema，完成 RBAC 业务闭环后丢弃临时环境；
> - 如果现有 `demo_sprint2.ps1` 没有独立只读入口，**先核实脚本行为，再决定采用何种只读审计方式**；不得假设它天然只读。

---

## 第 13 条：Sprint 3 统一验收（V1.0 定稿）

> **本验收章为 Sprint 3 最终统一验收入口**，G2–G6 各阶段产出在此统一核对，避免分阶段验收口径不一致。验收以「是否达成 + 是否具备证据」为判断，不做范围扩张。

### 13.1 验收范围

- 仅验收本方案 G2–G6 既定 P0/P1 功能范围；**不验收**本方案明确标注「下一阶段 / 后续迭代 / 不纳入」的内容（见附录 D 技术债、§1 JWT 下一阶段要求、附录 C #12 候选 ID 修复）。
- 不新增 G7，不新增业务模块，不为未来扩展提前设计复杂架构。
- **业务判据（C1 方案 B）**：三条业务线（规则演进 / 知识反馈 / 评审质量）端到端跑通为验收的业务主线，角色与权限设置服务于该演进（见首页业务目标）。

### 13.2 统一验收清单

> **V1.0 修订（评审 P0-5 / P1-8）**：状态枚举统一为 **待验证 / 已通过 / 未通过 / 不适用（须注明理由）**；每一项「已通过」必须有对应证据（测试报告 / CI Job 记录 / Migration 执行日志 / 真实 PostgreSQL 并发测试结果 / API 权限测试报告 / 前端构建结果 / Demo 执行记录）；无证据不得标记为已通过。

| # | 验收项 | 负责阶段 | 通过标准 | 状态 | 证据 / 记录位置 |
|---|---|---|---|---|---|
| 1 | G2 Migration 双向回滚 | G2 | 临时 PostgreSQL 执行成功，存量数据符合预期 | 待验证 | Migration 测试报告 / 执行日志 |
| 2 | 管理员并发保护 | G2/G6 | A/C 类在 G2、B 类在 G6 测试通过 | 待验证 | 并发测试报告 |
| 3 | 最后管理员保护 | G2/G6 | 真实 PostgreSQL 并发下正确 | 待验证 | API / 数据库测试报告 |
| 4 | 权限矩阵 + 对象级 IDOR | G3 | 所有实际路由和对象级访问测试通过 | 待验证 | API 权限测试报告 |
| 5 | 前端权限行为 | G4 | 登录、路由、菜单、按钮及 401/403 行为符合约定 | 待验证 | Build / E2E 记录 |
| 6 | 审计与反馈闭环 | G5 | 后端注入审计字段，三条业务线闭环通过 | 待验证 | E2E 测试记录 |
| 7 | 冻结基线与回归 | G2/G6 | 冻结版本已重测并记录完整信息，新增回归失败均已解释和处理 | 待验证 | pytest 报告 |
| 8 | CI 与最终 Demo | G6 | CI 全绿，临时环境 Demo（含 B 类并发测试）完整通过 | 待验证 | CI / Demo 记录 |

> 上表「证据 / 记录位置」由项目组在验收时填写；负责人字段按项目分工在验收记录中补充。

### 13.3 验收通过条件

- 上述 8 项全部「已通过」且均有证据；任一未通过须记录缺口、原因与处理计划，不得以「文档已写」代替实际验证。
- 验收基于真实 PostgreSQL 与真实前端构建，SQLite / Mock 不作为独立通过依据（§2.4 / §3.7 / §2.1）。

---

## 附录 A：V1.0 版本历史与修订摘要

> **V1.0（评审定稿修订 · 2026-11-03，实质性规则定案）**——本轮为**实质性规则修订**，因 C1（方案 B）与 C2（PostgreSQL）定案改变了生产环境约束与数据库目标，属实质变更；版本标识保持 V1.0，本表明确记录各轮，避免多版本混淆。
>
> **版本号接续策略（按项目负责人确认）**：本方案文档版本标识保持 **V1.0**，**不升级为 V1.1 及更高正式版本号**。若后续仍需修改，版本号**回退至 V0.x 序列接续**（在 V0.x 区间递增），**V1.0 保留给最终真正冻结、不再改动的版本**。

| # | 修改点 | 位置 | 类型 |
|---|---|---|---|
| A1 | C1 方案 B 定案：第一阶段不交付 JWT/密码登录；prod 解除「无 JWT 即 fail-closed」，改为可信内网部署 + 基于角色身份体系；JWT 移入下一阶段 | 总体原则 §8、§1、§9、§12.1、§13.1 | 实质性规则定案 |
| A2 | C2 定案：目标数据库确认为 PostgreSQL；锁确定 pg_advisory_xact_lock；MySQL/SQLite 降为未来扩展占位 | §2.1、§2.3、§2.4、§3.4、§12.1、附录 B、附录 C | 实质性规则定案 |
| A3 | 首页新增三条业务线演进目标作为第一阶段业务判据 | 首页 | 业务目标明确 |
| A4 | §1 补充 dev/test 身份模拟完整流程与调用约定 | §1 | P1-2 |
| A5 | §2.1 明确 Migration 双向回滚仅限临时验证环境、生产不得用 downgrade 回滚、审计数据降级丢失 | §2.1 | P1-5 |
| A6 | §2.4 并发测试 A/C 在 G2、B 类延后 G6，并按类别拆分状态与证据 | §2.4、附录 B #32 | P0-3 / C3 |
| A7 | §9 权限矩阵改为「HTTP 方法+路径+角色+对象级范围」精确列示，清理通配符 | §9 | P1-7 |
| A8 | §9 Case 列表/详情/Benchmark 权限按现有数据结构统一 | §9 | P1-1 |
| A9 | §9 Benchmark 角色与数据影响边界标为 G2 待核查，不新增角色系统 | §9 | P1-3 |
| A10 | §10 反馈生成链路缺失时判定规则（补齐最小实现 / 移出范围须记录决策） | §10 | P1-4 |
| A11 | §11 测试基线验收口径改为「冻结 Git HEAD 重测为准，不以数字为唯一标准」 | §11、§12.1、§13.2、附录 F | P0-4 |
| A12 | §12.2 / P1-6 G2 前置检查只读边界与 G6 完整 Demo 临时库隔离 | §12.2 | P1-6 |
| A13 | §13 统一验收：状态枚举 + 证据要求 + 负责阶段/通过标准/证据列 | §13 | P0-5 / P1-8 |
| A14 | 附录 D 清理与 §7 重复的审计待办，仅保留增强项；技术债明确不纳入 Sprint3 | 附录 D | P2-2 / P2-4 |
| A15 | 附录 C #3 改为核查 PostgreSQL 连接配置/版本/环境一致性 | 附录 C | C2 |

**V1.0（专家评审修订）历史**：
| # | 修改点 | 位置 | 类型 |
|---|---|---|---|
| B1 | 统一 JWT 范围矛盾描述（本版经 C1 定案进一步收敛） | 首页 / §1 | 评审修订 |
| B2 | 测试基线 562/568 统一口径 | §11 | 评审修订 |
| B3 | 附录 B 状态区分已验证/待验证/不适用 | 附录 B | 评审修订 |
| B4 | 补充 JWT 最小必要安全要求（下一阶段边界） | §1 | 评审修订 |
| B5 | API 路由矩阵补齐 | §9 | 评审修订 |
| B6 | Case 归属依据数据结构确定 | §9 | 评审修订 |
| B7 | 真实 PostgreSQL 迁移验证 | §2.1 | 评审修订 |
| B8 | Demo 清理改用临时库/Schema | §12.2 | 评审修订 |
| B9 | CI 数据库连接按真实环境确认 | §12.1 | 评审修订 |
| B10 | Vue 数组操作软化为宽松口径 | §8 | 评审修订 |
| B11 | 技术债移入下一阶段 | 附录 D | 评审修订 |
| B12 | 新增第 13 条统一验收章 | §13 | 评审修订 |

**V1.0（Sprint2 基线核查增补）历史**：
| # | 修改点 | 位置 | 类型 |
|---|---|---|---|
| C1 | 附录 C 扩充至 13 项（候选 ID 连字符、uv sync），修正第 8 项 | 附录 C | 增补 |
| C2 | §2.1 Migration dry-run/ORM import/双向回滚/基线对比 | §2.1 | 增补 |
| C3 | §2.2/§2.3 PowerShell 编码坑、真实 PostgreSQL 并发、审计日志 | §2.2/§2.3 | 增补 |
| C4 | §3.3 autobegin 事务冲突测试场景 | §3.3 | 增补 |
| C5 | §7 审计字段后端注入 | §7 | 增补 |
| C6 | §9 case 详情/batch 归属校验 | §9 | 增补 |
| C7 | §8 前端踩坑 | §8 | 增补 |
| C8 | 新增第 12 条 CI 与 Demo 适配 | §12 | 增补 |
| C9 | 附录 B 扩至 35 项、附录 D 技术债、附录 F Checklist | 附录 B/D/F | 增补 |

**V1.0（相对 V0.5）最终一致性修订**：
| # | 修改点 | 位置 | 类型 |
|---|---|---|---|
| D1 | 修正历史 users 与 DEFAULT 'engineer' 技术表述 | §2.1 | 一致性修订 |
| D2 | POST /auth/login 权限矩阵区分 dev/test 与 prod | §9 | 一致性修订 |
| D3 | 方案定位最终稿、冻结范围 | 首页 / §11 | 版本冻结 |

**V0.5 / V0.4 / V0.3 既定修订（V1.0 完整保留）**：环境矩阵表头语义、同一把锁说明、附录 C 扩至 11 项、login 503 表述、独立 Session 可观测验证、环境边界独立成章、prod fail-closed、锁实现表合并、独立 Session 非规避手段、历史用户待核查、reviewer PATCH 边界、事务隔离级别、多数据库锁对照表、Migration 预检查、审计日志、软禁用定义、IDOR 矩阵、阶段验收表、P2 标记。——（详见前版修订记录，本版不再重复展开。）

---

## 附录 B：35 项一致性检查

> **验收状态说明（评审 P0-3）**：本表「结果」列区分三种状态——
> - ✅ **已验证**：结论已由 Sprint2 封板文档或已完成核查确认；
> - ⏳ **待验证**：结论依赖 G2 编码 / 真实数据库测试后才能确认，当前为方案约定，不作为已达成事实；
> - ➖ **不适用**：与当前 G2–G6 范围无关（如 JWT 相关，Sprint3 不实现，仅作下一阶段范围说明）。

| # | 检查项 | 结果 |
|---|---|---|
| 1 | CLI 五种判断场景和优先级未改变 | ✅ |
| 2 | CLI 与用户管理 API 使用同一互斥机制 | ✅ |
| 3 | CLI 锁、检查、创建、提交仍在同一事务 | ✅ |
| 4 | CLI 唯一约束冲突后先完整回滚 | ✅ |
| 5 | CLI 恢复流程重新开启事务并重新获取同一把锁 | ✅ |
| 6 | CLI 恢复流程重新读取数据库状态 | ✅ |
| 7 | 不把唯一约束冲突直接当成幂等成功 | ✅ |
| 8 | 仅捕获用户名唯一约束冲突，其他 IntegrityError 不重试 | ✅ |
| 9 | 相同用户名并发由锁串行化，不预期唯一约束冲突 | ✅ |
| 10 | PATCH 未提供字段保留数据库原值 | ✅ |
| 11 | PATCH 显式 null 返回 422，不进入 Service | ✅ |
| 12 | PATCH Schema 允许缺省但不允许 role / is_active 为 null | ✅ |
| 13 | PATCH 字段存在性判断与有效值判断没有混淆 | ✅ |
| 14 | 最后管理员保护基于更新前与更新后的最终状态 | ✅ |
| 15 | 最后管理员保护的锁、数量查询和更新仍在同一事务 | ✅ |
| 16 | SQLAlchemy Session autobegin 风险要求未删减 | ✅ |
| 17 | 目标数据库确认为 PostgreSQL；锁确定为 pg_advisory_xact_lock；MySQL/SQLite 仅作未来扩展占位 | ✅（C2 定案） |
| 18 | SET NULL / NO ACTION 策略未改变 | ✅ |
| 19 | [SEED] 排除适用范围已恢复（EAR / AI_NER / sample_info，不含 EVC） | ✅ |
| 20 | None → N/A、样本不足 → N/A · 样本不足 约定已恢复，EVC 不得显示 0% | ✅ |
| 21 | 四角色权限矩阵 + IDOR 对象级授权 + reviewer PATCH 边界 | ✅ |
| 22 | G1-G6 阶段边界未改变，阶段验收总表 | ✅ |
| 23 | 全文没有把待核实的代码事实写成已实现事实 | ⏳ |
| 24 | 事务隔离级别快照一致性约束；独立 Session 不得规避事务一致性；可观测验证 | ⏳ |
| 25 | 环境边界：第一阶段 prod 内网部署 + 基于角色身份，无正式 JWT；JWT 移入下一阶段 | ✅（C1 定案） |
| 26 | Migration 历史用户预检查待核查、不预设场景；软禁用完整行为 | ✅ |
| 27 | 管理员操作审计日志输出要求 | ✅ |
| 28 | 历史 users 数据迁移验证既有记录实际填充值 | ✅ |
| 29 | POST /auth/login 权限矩阵与第 1 条环境身份边界一致，无跨环境歧义 | ✅ |
| 30 | G5 审计字段均由后端注入，接口 schema 禁止前端传入 | ✅ |
| 31 | POST /review/tasks/batch 批量接口 created_by 自动设置登录用户 | ✅ |
| 32 | PostgreSQL 并发锁测试按类别记录（见下拆分表） | ⏳（A/C 在 G2，B 类延后 G6） |
| 33 | bootstrap_admin.py 幂等行为符合五种判断场景 | ⏳ |
| 34 | 最后管理员保护在 REPEATABLE-READ 隔离级别下用独立 Session + READ COMMITTED，count 不被快照绕过（有可观测验证） | ⏳ |
| 35 | CI 环境 APP_ENV=test；第一阶段 prod 内网部署不开放模拟通道 | ⏳（C9 待实际 CI 确认） |

> **附录 B 第 32 项拆分（评审 P0-3）**：

| 测试类别 | 计划阶段 | 当前方案状态 | 通过证据 |
|---|---|---|---|
| A 类：正常并发与锁串行化 | G2 | 待验证 | 真实 PostgreSQL 并发测试报告 |
| C 类：锁机制真实性、跨连接/跨进程行为 | G2 | 待验证 | 真实 PostgreSQL 测试报告 |
| B 类：唯一约束冲突恢复 | G6 | 待验证 | 受控冲突恢复测试报告 |

---

## 附录 C：G2 前置核查清单（13 项 + 评审补充核查）

> 以下核查项必须在 G2 编码前基于真实代码 / 数据库 / CI 配置完成，不得预设结论；其中标注「C2/C5/C6/C7/C8/C9」的对应评审确认问题清单，在 G2 记录结论后落地，落地前不得默认成立。

| # | 核查项 | 用途 | 对应 |
|---|---|---|---|
| 1 | `backend/app/api/deps.py` 完整内容 | 决定 get_current_user 放置与 re-export | — |
| 2 | `backend/app/persistence/db.py` 的 get_db_session 实现 | 决定管理员变更 Service 事务所有权 | — |
| 3 | PostgreSQL 连接配置、数据库版本及各环境是否一致 | 决定锁 / Migration / CI / 验收统一基于 PostgreSQL | C2 |
| 4 | `s2a_schema_updates.py` 的 revision 变量值 | 决定新 Migration down_revision | — |
| 5 | `app/api/` 全部 @router. 端点清单（含 Case 归属字段、Benchmark 接口、rule-candidates approve/reject、knowledge 端点） | 权限矩阵 / Case 归属 / Benchmark 边界据实落地 | C5/C6 |
| 6 | `task_service.submit_task_feedback` 是否真的调用候选/缺口生成逻辑 | 确认 rule_candidates / knowledge_gap / gap_feedback_link 写入路径 | C7 |
| 7 | `feedback_service.submit_feedback` 是否接受并写入 created_by | 审计字段写入核实 | — |
| 8 | `task_service.update_expert_decision` 是否已接受 updated_by 并写入 expert_decision_by | 审计字段写入核实；expert_decision_by 由后端填充，禁止前端传入 | — |
| 9 | 当前 Git HEAD 与实际 pytest -m "not integration" 结果 | 正式基线起点 | C4 |
| 10 | `dev.ps1` 的 start/restart 分支 uv sync 是否已删除 | 防止 dev 依赖被卸载 | — |
| 11 | 历史 `users` 表内是否存在承担管理员角色的账号 | §2.1 预检查第 4 项取真实结论；当前 G1 已核实 users 表 0 条，需确认仍成立 | — |
| 12 | 候选 ID 特殊连字符（`_make_candidate_id`） | **前置事实：Sprint3 暂不修**；仅确认前端已用 `encodeURIComponent(id)` 编码兜底；修复列入附录 D | — |
| 13 | `dev.ps1` / `bootstrap.ps1` 脚本内已删除无参数 `uv sync` | 避免启动时卸载 dev 依赖 | — |
| 14 | `demo_sprint2.ps1` 是否存在可独立执行的只读审计入口 | G2 前置只读检查方式；若无非只读，先核实脚本行为再决定只读审计方式 | C8 |
| 15 | CI 数据库连接方式及 APP_ENV=test 是否已在实际 CI 中确认 | CI 连接逻辑继承与环境约束确认 | C9 |

---

## 附录 D：待核实事实 + P2 延后迭代清单

**待核实事实（同附录 C，追加）**
- 空 PATCH `{}` 的现有 API 契约行为（是否允许、返回什么）。
- `rule_evolution_service` / `feedback_service` 中候选与缺口生成的实际调用链是否存在于 submit_task_feedback 路径（对应附录 C #6 / C7）。
- 历史 `users` 表内是否存在承担管理员角色的账号（已并入附录 C #11）。

**P2，本最终稿不开发，后续迭代文档再补充，不膨胀本施工图：**
1. 四角色完整业务职责文档；明确 admin 是否隐式继承 reviewer/rule_admin 业务权限。
2. 审计字段的进一步增强：审计事务原子性、系统自动生成记录与人工操作记录的来源标记、独立审计查询和管理能力。
3. 扩充全套细分测试用例清单：锁超时、管理员异常事务退出、inactive 用户、身份伪造等。
4. 复杂 Token 管理（Refresh Token 轮换、多端登录、黑名单、分布式撤销）——随 JWT 下一阶段一并实现。
5. 多租户、组织级数据隔离、完整角色权限配置后台。

> **V1.0 修订（评审 P2-2）**：正文 §7 已明确为第一阶段要求的审计事项（4 个审计字段后端注入、batch 任务 created_by、禁止前端传入 actor_id、expert_decision_by 不接受请求体）**不再重复列为下一阶段待办**；附录 D 仅保留真正属于下一阶段的增强项。

**【V1.0 修订（评审 P2-4）：下一阶段技术债明确不纳入 Sprint 3】**

> 以下事项**统一移入下一阶段（Sprint4 及以后）迭代处理**，本施工图不做任何提前实现，避免范围膨胀。判断原则：只有影响第一阶段产品安全性、正确性、稳定性或验收的事项才进入 Sprint3。

| # | 事项 | 优先级 | 计划阶段 |
|---|---|---|---|
| 1 | 修复 `_make_candidate_id()` 生成标准 ASCII 短横线，解决候选 ID 特殊连字符 URL 编码 404 | P1 | 下一阶段 |
| 2 | 发布后 `benchmark_regression` 自动运行回归 benchmark（当前字段恒为 null） | P1 | 下一阶段 |
| 3 | `scripts/rollback_rule.py` CLI 规则回滚完整验收 | P2 | 下一阶段 |
| 4 | Case Benchmark `full_review` 模式浏览器端完整验收 | P2 | 下一阶段 |
| 5 | vLLM 完整实装（硬件条件具备才落地） | P3 | 下一阶段 |
| 6 | `test_llm_zhipu.py` skipped 测试 minilm 注册（对应 §11 基线 1 skipped 项） | P3 | 下一阶段 |
| 7 | 复杂 Token 管理（Refresh 轮换 / 多端登录 / 黑名单 / 分布式撤销） | P1 | 下一阶段 |
| 8 | 完整四角色职责与权限配置后台 | P2 | 下一阶段 |
| 9 | 多租户、组织级数据隔离 | P2 | 下一阶段 |
| 10 | 独立审计查询管理系统 | P2 | 下一阶段 |

> 以上事实未核实前，不得写成"已实现"。

---

## 附录 E：声明

1. 本文档只修改了方案文档，没有修改源码、数据库或配置文件，没有执行 Migration，没有运行测试。
2. 本文档未开始 G2 开发；审核通过前不动代码。
3. 所有"待核实"事项保持待核实状态，不以推测代替实际代码检查。
4. V0.1 / V0.2 / V0.3 / V0.4 / V0.5 / 各轮 V1.0 修订的有效设计约定保留；本版为评审定稿修订，不改变既定施工范围。
5. EAR / AI_NER / EVC 冻结口径、样本门槛、Seed 排除、空值语义未改变。
6. 四角色权限矩阵、G1-G6 阶段边界未改变；C1（方案 B）与 C2（PostgreSQL）为定案决策，已在对应章节落地。

---

## 附录 F：G2 开发启动前置 Checklist（综合 Sprint2 基线 + 附录 C + 评审定稿修订）

> G2 编码开始前必须逐项完成并记录结果，写入基线审计报告（结合 `baseline_audit.md` 做法）：

1. [ ] Git HEAD 确认：Sprint2 封板 commit，切分支 `feature/sprint3-rbac`；
2. [ ] 运行 `demo_sprint2.ps1` **只读审计**入口（若无只读入口，先核实脚本行为），产出 `baseline_snapshot.md`，确认 PostgreSQL、users 表 0 行、无 auth 模块；
3. [ ] 附录 C #1–#15 项完成核查并记录结论；
4. [ ] 确认 `s2a_schema_updates.py` revision 值，确定 G2 Migration `down_revision`；
5. [ ] 确认 `app/api/deps.py`、`app/persistence/db.py` get_db_session 实现；确认 Session autobegin 行为；
6. [ ] 确认 `dev.ps1` / `bootstrap.ps1` 不再有无参 `uv sync`；
7. [ ] 确认 `task_service.submit_task_feedback`、`feedback_service.submit_feedback`、`task_service.update_expert_decision` 代码实现，确认审计字段写入入口；
8. [ ] 清点全部 `@router.` 端点清单，核对权限矩阵（含 Case 归属字段、benchmark、batch、export、dashboard、rule-candidates approve/reject）；
9. [ ] 确认空 PATCH `{}` API 契约行为；
10. [ ] 以冻结 Git HEAD 重测测试基线并记录完整信息（不以 568 数字为唯一标准，见 §11 P0-4）；
11. [ ] 确认 case 表是否存在创建者归属字段（供 §9 P1-1 判定 Case 数据范围）；
12. [ ] 确认 Benchmark 接口是否修改正式业务数据 / 影响正式指标（供 §9 P1-3 落地）；
13. [ ] 确认反馈生成链路是否存在（供 §10 P1-4 判定）；
14. [ ] 确认 `demo_sprint2.ps1` 只读审计入口（供 §12.2 P1-6 落地）；
15. [ ] 确认 CI 数据库连接方式及 APP_ENV=test（供 §12.1 P1-9 继承）。

---

## 下一步

1. V1.0 已完成「Sprint2 基线核查增补 + 专家评审修订 + 评审定稿修订（C1/C2 定案）」并冻结，不再继续增加 P0/P1 设计条款（见首页修订说明与附录 A）。
2. 先补齐附录 C 的 15 项代码 / 数据库 / CI 事实核查，所有待核实结论必须以真实代码、数据库和配置为准（C3–C10 在 G2 执行期逐项验证）。
3. 以冻结 Git HEAD 重测测试基线并记录完整信息（见 §11 P0-4）。
4. 然后按 G2 → G3 → G4 → G5 → G6 顺序开发，每阶段 commit，暂不 push。
5. 全部完成后按「第 13 条：Sprint 3 统一验收」逐项验收（须附证据），再统一整理 git 历史并 push 触发 CI；CI 首次可能因清华源不可达失败，预案是给 uv sync 加 `UV_INDEX_URL`。

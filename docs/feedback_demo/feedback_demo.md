# 14. Phase I‑J · Feedback & Demo
> 文档版本：**V1.17**
> 更新日期：2026‑09‑19
> Sprint‑0边界约束（严格遵守，不越界到Sprint‑1/Sprint‑2）
> 1. Sprint‑0**不接入LangGraph knowledge_retrieve节点**，Demo直接调用retriever.py；LangGraph节点接入为Sprint‑1核心任务。
> 2. Sprint‑0**不实现Web前端Feedback UI**；全部验证通过Service/CLI/pytest/DB完成；Feedback Web UI移交Sprint‑2。
> 3. Sprint‑0 `rule_candidate`仅自动生成**proposed草稿YAML，禁止直接合并正式rules库；草稿必须人工编辑，候选规则benchmark校验逻辑放到Sprint‑1**。
> 4. Sprint‑0不引入Alembic迁移；数据库Schema唯一事实来源`infra/docker/initdb/002_schema.sql`；Alembic迁移列入Sprint‑1技术债务TD‑007。
> 5. V1.16迭代新增修复点：
>    - 修复SQLAlchemy DetachedInstanceError；session内预拷贝主键id到普通int变量；
>    - 修复`docker compose exec -T`无PTY模式stdout换行被压扁；业务日志落容器`payload_run.log`，stdout仅输出单行JSON；
>    - 修正RuleResult真实字段映射：`hit_message / rule_suggestion / severity / evidence_ir_refs`；移除臆测字段`message/suggestion/risk/location`；
>    - 修正RetrievalResult模型：元数据全部平层一级属性`source_type/source_title/source_section/snippet`，不存在`.meta`/`.metadata`嵌套字典；
>    - PowerShell5.1异常捕获优化：规避docker.exe RemoteException；使用`$LASTEXITCODE`判断子进程退出码；
>    - 补充IDE(Trae)索引缓存问题说明：`out/`目录被gitignore过滤，会出现部分产物侧边栏不可见，磁盘文件真实存在。
> 6. **V1.17迭代新增修复点：**
>    - `scripts/demo_sprint0.ps1`升级至V1.17；Step4容器环境诊断彻底移除ps1内部嵌套bash heredoc/PYEOF代码；复用`demo_feedback_payload.py --diagnose`CLI分支；消除`U+FEFF`、`wanted PYEOF delimited by end‑of‑file`语法报错；
>    - Step6手工调试提示优化：移除过时黄色Warning告警，替换英文说明；明确提示仅为复制粘贴参考命令，脚本不会自动执行；
>    - test_demo.py：`prepare_out_dir`执行前执行`shutil.rmtree(OUT_DIR)`清空历史产物，防止旧文件导致断言误通过；`pg_session`维持`scope="module"`，增加完整契约注释，**强制完整运行整个模块，禁止单用例挑选执行**；细粒度隔离测试下沉`test_feedback_rule.py`；
>    - rule_evolution_service.py：增加evidence_refs `source/section/reason`三要素最小assert结构化校验；在校验通过后再传入RuleCandidate ORM构造，满足V1.2 §3.2契约；
>    - demo_feedback_payload.py location修复为V1.2 §3.1完整结构：`sheet:AUTO‑DEMO`、`path=rule_id`、`coords:{x:0,y:0}`，`ir_refs`子字段存储原始evidence_ir_refs列表；Sprint‑1待办替换AUTO‑DEMO mock占位；
>    - DeepSeek评审权衡记录：`prepare_out_dir shutil.rmtree`✅采纳；`pg_session scope="module改为function"`❌拒绝；拒绝根因：test_demo.py是顺序E2E冒烟，用例之间共享DB测试数据；scope=function会每条用例事务回滚，后续用例读取不到前置数据直接失败；缓解依靠fixture契约注释+文档执行约束。

## Phase‑I Feedback闭环（P0）
### I.1 Phase‑I目标
1. 完整落地V1.2增强版6类`feedback_type`枚举：
`correct_defect / false_positive / false_negative / suggestion_update / new_rule_candidate / knowledge_gap`。
2. DDL增量扩展存量`feedback_item`表字段；新建`rule_candidates`草稿表，配套索引。
3. `FeedbackService`：完成反馈入参校验、`feedback_item`落库；根据feedback_type分发业务逻辑：
   - `false_negative` / `new_rule_candidate`：自动生成`proposed`状态`rule_candidate`草稿；
   - `knowledge_gap`：记录知识缺口，支持导出backlog JSON；
   - `correct_defect / false_positive / suggestion_update`：仅落库，不生成候选规则。
4. `RuleEvolutionService`：提供CLI入口：`--list` / `--export-yaml` / `--export-knowledge-gap`；**新增`--output‑json <path>`参数，支持容器内直接写出JSON文件，规避Windows docker‑exec stdout GBK编码损坏**；供Demo脚本调用。
5. 新增pytest集成测试`tests/test_feedback_rule.py`，全覆盖6类反馈、rule_candidate草稿生成、YAML语法校验、knowledge_gap导出。

### I.2 Phase‑I 文件清单
#### 🆕全新创建文件
1. `backend/app/schemas/__init__.py`
2. `backend/app/schemas/feedback.py`：FeedbackType枚举、SuggestionDiff、FeedbackCreate、FeedbackOut
3. `backend/app/schemas/report.py`：ReviewStatus三态枚举（AI_CONFIRMED/NEED_EXPERT_REVIEW/LOW_CONFIDENCE）
4. `backend/app/services/feedback_service.py`：反馈提交主服务，落库+分发rule演进
5. `scripts/demo_sprint0.ps1`：扩展Phase‑I反馈提交、rule‑candidate产物导出逻辑；**V1.17加固版本**：
   - 删除powershell here‑string内嵌大段Python代码，彻底规避Markdown复制引入U+2011全角破折号隐形字符；
   - 全部JSON/YAML产物改为**容器内部Python直接落盘**，使用`docker compose cp`二进制拷贝文件回宿主机；彻底规避Windows PowerShell5.1 docker‑exec stdout管道GBK破坏性转码乱码；
   - 修复`docker compose exec -T`换行压扁问题：payload业务日志写入容器`payload_run.log`，ps1执行`cat`读取日志打印控制台，再cp拷贝回宿主机；stdout仅保留单行JSON；
   - PowerShell异常捕获优化：执行docker子进程临时切换`ErrorActionPreference=Continue`，使用`$LASTEXITCODE`判断退出码，规避docker.exe RemoteException；无论payload成功失败均打印完整`payload_run.log`堆栈；
   - 增加脚本退出码校验，子脚本失败直接exit终止，不继续执行后续脏逻辑；
   - **V1.17变更：Step4诊断重构，移除ps1内嵌套bash heredoc/PYEOF；调用`demo_feedback_payload.py --diagnose`做容器环境诊断，不再新增独立diag_env.py文件；Step6手工提示移除黄色Warning，替换英文提示，明确仅复制粘贴参考，脚本不会自动执行；**
   - 调用独立幂等脚手架脚本`scripts.demo_feedback_payload`生成demo测试数据。
6. `backend/tests/test_feedback_rule.py`：Phase‑I单元+PG集成测试


#### 📝存量文件增量修改
1. `infra/docker/initdb/002_schema.sql`：追加`feedback_item`ALTER语句 + `rule_candidates`建表+索引
2. `backend/app/persistence/models.py`：
   - FeedbackItem ORM新增`suggestion_diff_json`、`rule_candidate_ref`字段；
   - 新增ORM类`RuleCandidate`映射rule_candidates表；
   > ⚠️重要DDL事实：`rule_candidates`表**不存在task_id列**；task业务字符串存储于`review_result.task_id`，查询通过`from_feedback_id → feedback_item → review_result`JOIN获取，禁止往RuleCandidate传入task_id参数。
3. `backend/tests/test_db_schema.py`：追加冒烟测试：校验rule_candidates表、feedback_item新增字段、索引完整性；校验rule_candidates不存在task_id字段。
4. `backend/tests/test_demo.py`：**原test_demo_sprint0_e2e.py重命名为test_demo.py**；Mock‑E2E测试移除真实IR磁盘文件IO依赖，全部内存mock构造ReviewResult、ReviewDefect；修复YAML断言错误：**status从DB RuleCandidate对象读取，禁止从导出yaml草稿读取status字段（yaml草稿只存规则本体，不含DB元字段status/candidate_id）**。
5. `scripts/demo_sprint0.ps1`：扩展Phase‑I反馈提交、rule‑candidate产物导出逻辑；**V1.17加固版本**：
   - 删除powershell here‑string内嵌大段Python代码，彻底规避Markdown复制引入U+2011全角破折号隐形字符；
   - 全部JSON/YAML产物改为**容器内部Python直接落盘**，使用`docker compose cp`二进制拷贝文件回宿主机；彻底规避Windows PowerShell5.1 docker‑exec stdout管道GBK破坏性转码乱码；
   - 修复`docker compose exec -T`换行压扁问题：payload业务日志写入容器`payload_run.log`，ps1执行`cat`读取日志打印控制台，再cp拷贝回宿主机；stdout仅保留单行JSON；
   - PowerShell异常捕获优化：执行docker子进程临时切换`ErrorActionPreference=Continue`，使用`$LASTEXITCODE`判断退出码，规避docker.exe RemoteException；无论payload成功失败均打印完整`payload_run.log`堆栈；
   - 增加脚本退出码校验，子脚本失败直接exit终止，不继续执行后续脏逻辑；
   - **V1.17变更：Step4诊断重构，移除ps1内嵌套bash heredoc/PYEOF；调用`demo_feedback_payload.py --diagnose`做容器环境诊断，不再新增独立diag_env.py文件；Step6手工提示移除黄色Warning，替换英文提示，明确仅复制粘贴参考，脚本不会自动执行；**
   - 调用独立幂等脚手架脚本`scripts.demo_feedback_payload`生成demo测试数据。

#### ❌完全不动文件
Phase‑H RAG全套代码`rag/*.py` + `test_rag_seed.py v1.7`；`app/ir/*`、`app/rules/*`、`app/workflows/*`；上游参考文档`第一阶段工程实施方案_V1.2_增强版_plan.md`只读参考，禁止修改。
### I.3 数据库DDL关键说明
> ⚠️表名严格对齐`test_db_schema.py`冒烟测试：反馈表**`feedback_item`，不是feedbacks**。
1. ALTER `feedback_item`：扩大`feedback_type`字段VARCHAR(64)；新增JSONB `suggestion_diff_json`、`rule_candidate_ref`；新增2条索引。
2. 新建`rule_candidates`完整表，包含`candidate_id`唯一约束、外键`from_feedback_id → feedback_item.id`；status默认`proposed`；配套索引。
> ⚠️重点：`rule_candidates`**无task_id字段**；task业务数据走JOIN链路，不在本表冗余存储。
3. Sprint‑0禁止`Base.metadata.create_all()`；Schema唯一事实来源`002_schema.sql`。

### I.4 Phase‑I验收标准（并入Phase‑K）
- [ ] I‑1 `infra/docker/initdb/002_schema.sql`执行成功；`feedback_item`新增字段生效；`rule_candidates`表+全部索引创建完成；`test_db_schema.py`全部冒烟PASS；校验rule_candidates表不存在task_id字段。
- [ ] I‑2 FeedbackService支持全部6类FeedbackType；DB feedback_item落库字段完整；`suggestion_diff_json`存储正确；**FeedbackCreate入参废弃task_id/defect_id字符串，使用review_result_id、review_defect_id外键；FeedbackOut通过JOIN回填task_id/defect_id业务视图字段**。
- [ ] I‑3 `false_negative` / `new_rule_candidate`反馈自动生成`proposed`状态rule_candidate草稿；`proposed_yaml`草稿YAML语法可解析；**YAML草稿本体不含status/candidate_id等DB元字段，status校验必须读取DB RuleCandidate对象**。
- [ ] I‑4 `knowledge_gap`反馈可以导出`out/knowledge_gap_backlog.json`知识缺口清单；导出通过JOIN ReviewResult/ReviewDefect拿到task_id/defect_id业务字段。
- [ ] I‑5 `tests/test_feedback_rule.py`全部单元+集成pytest PASS（8个用例全部通过）。
- [ ] I‑6 `rule_evolution_service.py` CLI子命令`--list / --export-yaml / --export-knowledge-gap / --output‑json`全部可用；支持容器内写出JSON规避Windows管道乱码。
- [ ] I‑7 `scripts/demo_feedback_payload.py`幂等脚手架脚本：重复运行不会触发`review_defect_defect_id_key`唯一约束冲突；查询复用已存在demo ReviewResult/ReviewDefect记录；**V1.16新增：修复DetachedInstanceError，session上下文内预拷贝主键id至普通int变量；修正RuleResult、RetrievalResult字段映射；日志落盘payload_run.log，stdout仅单行JSON**。
- [ ] I‑8 **V1.17新增**：rule_evolution_service对evidence_refs执行V1.2 §3.2 source/section/reason最小结构化校验；非法结构直接阻断rule_candidate草稿生成；仅校验key是否存在，不校验语义内容。

- [ ] **V1.17‑1** test_demo.py prepare_out_dir执行前shutil.rmtree清空OUT_DIR；防止历史残留文件导致断言误通过
- [ ] **V1.17‑2** rule_evolution_service.py evidence_refs source/section/reason三要素最小assert校验；非法结构阻断草稿生成
- [ ] **V1.17‑3** demo_feedback_payload.py location严格V1.2 §3.1结构sheet/path/coords + ir_refs子字段
- [ ] **V1.17‑4** test_demo.py pg_session scope="module"契约注释完整；强制完整运行模块，禁止单用例执行；细粒度隔离测试下沉test_feedback_rule.py
- [ ] **V1.17‑5** demo_sprint0.ps1 V1.17 Step4诊断调用`demo_feedback_payload.py --diagnose`；彻底消除U+FEFF / PYEOF heredoc语法报错；Step6英文info手工提示

## Phase‑J Sprint0端到端Demo（P0）
### J.1 Phase‑J目标
完整跑通Sprint‑0业务闭环Demo链路：
> 本地完整真实Demo链路（需要.env OPENAI_API_KEY，仅本地运行，CI不执行）
Case001 Buck IR
↓
RuleEngine 确定性规则执行 → rule_results
↓
RAG Seed 知识库（Phase‑H 已闭环，直接调用 retriever.py）
↓
LLM 专家分析生成 ReviewReport，Defect 满足 V1.2 10 字段，携带 evidence 三要素 source/section/reason、review_status 三态
↓
模拟专家提交 2 条反馈：false_negative、knowledge_gap
↓
RuleEvolutionService 生成 rule_candidate 草稿 YAML + knowledge‑gap backlog
↓
全部产物输出到 out/demo_sprint0/

> CI流水线限制：CI不能调用真实OpenAI；因此`tests/test_demo.py`Mock‑LLM冒烟测试，**完全内存mock构造ReviewResult/ReviewDefect，不读取磁盘IR文件、无外网请求；完整真实Demo仅本地开发容器运行（需要.env配置OPENAI_API_KEY）**。

> V1.16重要链路修正说明：
> 1. RuleResult模型真实字段：`category, component, component_refs, evidence_ir_refs, hit_message, net, net_refs, origin, rule_basis, rule_id, rule_name, rule_suggestion, severity`；不存在`defect_id/message/suggestion/risk/location/confidence`；
> 2. RetrievalResult全部为平层一级属性：`source_type、source_title、source_section、snippet`，**无.meta/.metadata嵌套字典**；
> 3. SQLAlchemy禁止with会话外部访问ORM对象id；会话内部拷贝id到普通int变量；规避DetachedInstanceError；
> 4. 业务日志全部写入容器`/app/out/demo_sprint0/payload_run.log`；ps1通过`cat`读取日志打印控制台，再cp拷贝回宿主机产物目录；stdout只输出单行JSON用于ps1解析；
> 5. Trae IDE：`out/`被`.gitignore`忽略，IDE侧边栏会出现部分产物看不见；磁盘文件真实完整；不要修改.gitignore；需要IDE查看产物修改`.trae/.ignore`并重建索引。

> **V1.17链路补充：**
> 1. demo_feedback_payload.py location字段严格遵循V1.2 §3.1结构 `sheet:AUTO‑DEMO`、`path=rule_id`、`coords:{x:0,y:0}`；新增`ir_refs`子key保存原始`evidence_ir_refs`IR引用列表；Sprint‑1待办替换AUTO‑DEMO为IR解析真实sheet/path/coords；
> 2. test_demo.py `prepare_out_dir` fixture执行前`shutil.rmtree`清空OUT_DIR，防止历史产物残留造成pytest断言误通过；
> 3. pg_session保持`scope="module"`；**契约约束：test_demo.py必须完整运行整个模块，禁止单独挑选模块内部单条用例执行；细粒度隔离测试全部下沉test_feedback_rule.py（scope=function）**；
> 4. demo_sprint0.ps1 Step4诊断调用`demo_feedback_payload.py --diagnose`；彻底消除ps1嵌套bash heredoc带来`U+FEFF / wanted PYEOF delimited by end‑of‑file`语法报错；
> 5. Step6手工调试提示改为英文info提示，仅复制粘贴参考，脚本不会自动执行。

6. **V1.17新增待办：替换demo脚本location字段`AUTO‑DEMO` mock占位；IR解析器输出真实sheet/path/coords；永久保留`ir_refs`子字段用于溯源。**

### J.2 Phase‑J 文件清单
#### 🆕新建
1. `backend/tests/test_demo.py`：CI Mock‑E2E冒烟测试，Mock LLM输出，不消耗OpenAI token；移除磁盘IR文件依赖，全部内存mock数据。
2. `backend/scripts/demo_feedback_payload.py`：Demo幂等脚手架脚本，容器内生成ReviewResult/ReviewDefect测试记录，输出feedback json载荷；ps1脚本调用，pytest不依赖。

#### 📝存量修改
1. `scripts/demo_sprint0.ps1`：**V1.16版本**；追加模拟提交反馈、调用rule_evolution导出全部产物逻辑；输出全部产物到`out/demo_sprint0/`目录；修复Windows PowerShell5.1全套编码、docker命令、幂等、退出码校验、exec‑T换行压扁、RemoteException捕获问题。

Demo输出产物清单（全部写入`out/demo_sprint0/`）：
> 区分：脚手架临时输入产物（demo脚本内部使用），业务永久输出产物（rule_candidate_proposed.yaml、knowledge_gap_backlog.json）
1. `fb_false_neg.json`：脚手架生成，false‑negative反馈输入载荷（**仅容器内留存，不会cp拷贝回宿主机**）
2. `fb_knowledge_gap.json`：脚手架生成，knowledge_gap反馈输入载荷（**仅容器内留存，不会cp拷贝回宿主机**）
3. `payload_run.log`：payload脚本完整运行日志（容器生成，docker compose cp拷贝回宿主机；包含IR加载、规则命中、RAG证据数量、DB新建/复用记录）
4. `feedback_false_neg_submit.log`：false‑negative反馈提交日志（ps1宿主机Out‑File直接生成）
5. `feedback_gap_submit.log`：knowledge_gap反馈提交日志（ps1宿主机Out‑File直接生成）
6. `candidate_list.json`：rule_candidate候选列表JSON（容器内生成后cp拷贝回宿主机）
7. `rule_candidate_proposed.yaml`：自动生成proposed状态候选规则草稿（**正式业务输出产物，Sprint‑1继续使用**）
8. `knowledge_gap_backlog.json`：知识缺口导出backlog（**正式业务输出产物，Sprint‑1继续使用**）

> 注意：`report.json`、`feedback_submit_log.json`在**Mock‑E2E test_demo.py内存mock内部生成，用于pytest校验；ps1本地真实demo使用脚手架生成DB记录，不再生成report.json**。

### J.3 Phase‑J验收标准（并入Phase‑K）
- [ ] J‑1：本地容器执行`.\scripts\demo_sprint0.ps1`完整跑完，exit‑code=0；Windows PowerShell5.1环境无编码乱码报错；**payload_run.log完整生成并拷贝至宿主机；stdout仅输出单行JSON；DetachedInstanceError不再复现；RAG证据成功组装写入ReviewDefect.evidence**。
- [ ] J‑2：`out/demo_sprint0/`目录全部业务产物文件`payload_run.log`、`rule_candidate_proposed.yaml`、`knowledge_gap_backlog.json`生成成功；脚手架临时输入文件正常生成。
- [ ] J‑3：DB存在提交的feedback_item记录；存在1条proposed状态rule_candidate记录，proposed_yaml非空；**草稿yaml仅包含规则本体，status从DB读取**。
- [ ] J‑4：`tests/test_demo.py` CI Mock‑E2E冒烟测试全部PASS，无OpenAI外网调用，**不依赖磁盘case IR文件，全部内存mock ReviewResult/ReviewDefect**。
- [ ] J‑5 `scripts/demo_feedback_payload.py`幂等校验：重复执行demo脚本，不会抛出`UniqueViolation duplicate key review_defect_defect_id_key`；**不会抛出DetachedInstanceError**。
- [ ] J‑6 日志校验：`payload_run.log`完整记录IR加载、rule命中、RAG证据数量、DB复用/新建记录；容器exec‑T模式下日志换行完整；宿主机产物目录可以读取完整日志。
- [ ] J‑7 **V1.17新增**：test_demo.py prepare_out_dir fixture执行前shutil.rmtree清空OUT_DIR旧产物；历史残留文件不会造成pytest断言误通过。
- [ ] J‑8 **V1.17新增**：review_defect.location输出严格符合V1.2 §3.1 JSON结构，具备sheet/path/coords必选key；原始evidence_ir_refs保存在子字段ir_refs；Sprint‑1待办替换AUTO‑DEMO mock占位为IR解析真实坐标。

### Phase‑K Sprint‑0关键验收项（更新后的完整12项，追加I‑J子项 + V1.16修复验收）
1. [ ] **Docker Healthy**: `backend` & `postgres` 均为健康状态。
2. [ ] **Healthz 200**: `/api/v1/healthz` 响应正常。
3. [ ] **DB Schema v1.0**: 10张业务表创建完成；Phase‑I追加feedback_item字段、rule_candidates表完整；rule_candidates表确认无task_id列。
4. [ ] **IR Schema v1.0**: 冻结 Component/Pin/Net 规范。
5. [ ] **Case001 Calibration**: 专家校准纪要签字确认。
6. [ ] **3 Power Rules**: 规则 YAML 编写完成且在容器内可运行。
7. [ ] **Case001 Rule Hit**: 命中条数 ≥ 2。
8. [ ] **Benchmark Metrics**: 包含 Precision, Recall, EAR, EVC 四项核心指标。
9. [ ] **Feedback 6 Categories**: 支持全分类反馈录入。
    - [ ] I‑1 DDL执行成功，feedback_item扩展字段 + rule_candidates表完整；rule_candidates无task_id字段
    - [ ] I‑2 6类feedback_type全部落库DB；FeedbackCreate使用review_result_id/review_defect_id外键，废弃task_id/defect_id字符串入参
    - [ ] I‑3 false_negative/new_rule_candidate自动生成proposed rule_candidate草稿；yaml草稿本体不含status元字段，status校验读取DB对象
    - [ ] I‑4 knowledge_gap导出知识缺口backlog JSON，JOIN获取task_id/defect_id业务字段
    - [ ] I‑5 `test_feedback_rule.py`全部pytest PASS（8个集成用例）
    - [ ] I‑6 rule_evolution CLI全部子命令可用，支持`--output‑json`容器内落盘规避Windows管道乱码
    - [ ] I‑7 `demo_feedback_payload.py`幂等脚本，重复运行不触发review_defect唯一约束冲突；**V1.16：修复DetachedInstanceError；RuleResult/RetrievalResult字段映射正确；payload_run.log日志落盘生效**
10. [ ] **5 Business Freezes**: 见第18章节详细清单。
11. [ ] **Environment Freeze**: 见第 17 章节详细清单。
12. [ ] **Sprint 0 Demo PASS**: 全链路演示通过。
    - [ ] J‑1 `demo_sprint0.ps1`容器执行exit‑code=0；Windows PowerShell5.1无编码/容器命令报错；exec‑T日志换行正常；payload_run.log完整输出
    - [ ] J‑2 out/demo_sprint0业务产物`payload_run.log`、`rule_candidate_proposed.yaml`、`knowledge_gap_backlog.json`生成
    - [ ] J‑3 DB feedback_item、proposed rule_candidate记录存在；RAG证据写入ReviewDefect.evidence
    - [ ] J‑4 CI mock‑LLM e2e冒烟测试`tests/test_demo.py`全部PASS，无IR磁盘文件依赖，无OpenAI外网调用
    - [ ] J‑5 demo脚手架幂等，重复执行无UniqueViolation、无DetachedInstanceError报错
    - [ ] J‑6 RAG证据解析正常：RetrievalResult平层字段读取成功，不再报`.meta`/`.metadata`属性异常
### Phase‑L · 风险管理矩阵（追加Phase‑I‑J新增风险 + Windows平台专项风险 + V1.16迭代新增风险）
| 风险项 | Impact | Mitigation | Acceptance |
|---|---|---|---|
| Golden Case 质量 | 基准失效，导致评测误导 | 引入专家二次审核机制 | 专家校准纪要签字 |
| IR Schema 稳定性 | 频繁变更导致解析器重写 | Sprint0冻结V1.0；变更走架构评审 | 变更评审记录 |
| DB Schema变更 | 数据迁移成本高 | Sprint1引入Alembic迁移；Sprint0全部使用initdb脚本重建 | initdb脚本原子化；禁止Base.metadata.create_all |
| Rule 误报/漏报 | 降低工具可信度 | 误报漏报反馈闭环 | EAR指标 ≥ 60% |
| Benchmark 指标稳定性 | 指标波动大，无法客观评估进度 | 固化评测脚本与指标计算公式 | Benchmark Format 冻结文档 |
| 环境一致性风险 | Windows/Linux表现不一 | 强制容器化开发，固定版本；**Demo脚本禁止docker‑exec stdout传递JSON/YAML中文文本；容器内写文件+docker compose cp二进制拷贝** | bootstrap脚本统一验证；ps1脚本Windows PowerShell5.1完整跑通无乱码 |
| RAG种子知识库依赖OpenAI Embedding外网服务 | ingest失败；CI消耗OpenAI token；网络不通导致Sprint0 Phase‑H无法验收 | 1.开发环境配置`.env`密钥；2.CI流水线必须mock OpenAIEmbeddings，禁止真实外网调用；3.密钥可独立配置`OPENAI_EMBEDDING_API_KEY` | 开发容器ingest可正常完成知识库导入；CI单元测试无OpenAI网络请求 |
| **【新增I‑J‑R1】rule_candidate自动生成草稿质量不可控** | 专家误直接把草稿合并进正式规则库，引入语法/业务逻辑错误 | Sprint‑0明确proposed仅草稿模板；强制人工编辑；Sprint‑1才接入benchmark校验候选规则；Demo日志打印醒目警告文字 | 单元测试校验草稿YAML语法可解析；Demo日志输出草稿警告；导出yaml不含status等DB元字段 |
| **【新增I‑J‑R2】Demo脚本依赖OpenAI密钥，CI无法跑完整真实LLM链路** | CI流水线无法执行完整Demo | CI执行mock‑LLM冒烟e2e；完整真实Demo仅本地开发容器运行，文档明确说明边界 | CI：`tests/test_demo.py` mock通过；本地ps1完整Demo可以执行 |
| **【新增I‑J‑R3】Feedback 6类枚举 DB/ORM/Pydantic三者不一致** | 提交反馈抛出枚举值/字段异常 | pytest集成测试全覆盖6类feedback_type；DDL、ORM、Pydantic枚举字面量严格对齐；新增枚举一致性单元用例 | `test_feedback_rule.py`枚举一致性全部用例PASS |
| **【新增I‑J‑R4】Windows PowerShell5.1 docker‑exec stdout UTF‑8中文JSON被GBK破坏性转码** | JSON乱码，ConvertFrom‑Json解析直接失败 | **强制规范：所有JSON/YAML结构化输出禁止走docker‑exec stdout管道；容器内python写出文件，`docker compose cp`二进制拷贝回宿主机；ps1脚本全部遵循该规范** | ps1脚本Windows原生PowerShell5.1完整执行，无乱码解析错误 |
| **【新增I‑J‑R5】Demo脚本重复运行review_defect唯一约束UniqueViolation** | 重复跑demo脚本直接数据库报错，演示中断 | 脚手架`demo_feedback_payload.py`做幂等逻辑：先查询task_id/defect_id，记录存在直接复用主键，不存在才INSERT | 连续多次执行`demo_sprint0.ps1`，无UniqueViolation报错 |
| **【新增I‑J‑R6】Markdown复制粘贴引入U+2011全角破折号隐形特殊字符** | Powershell/Python/YAML语法报错，肉眼无法分辨 | 1. ps1脚本禁止here‑string内嵌超过5行Python源码；抽独立py脚本；2. 复制代码后全局搜索U+2011批量替换为ASCII半角`‑`；3. pytest校验yaml语法 | 脚本、pytest无隐形特殊字符引发语法异常 |
| **【新增I‑J‑R7 V1.16】SQLAlchemy DetachedInstanceError会话游离ORM对象** | 脚本执行崩溃，Demo中断 | session上下文内部预读取主键id拷贝到普通int变量；with会话外部永远不访问ORM模型对象属性 | 多次执行demo脚本不再抛出DetachedInstanceError |
| **【新增I‑J‑R8 V1.16】docker compose exec -T无PTY stdout换行压扁，多条日志挤同一行** | 日志无法阅读，无法区分步骤 | 业务日志全部写入容器`payload_run.log`；stdout仅输出单行JSON；ps1调用cat读取日志再cp拷贝回宿主机产物目录 | 控制台`payload_run.log`输出每条日志独立换行；产物文件内换行完整 |
| **【新增I‑J‑R9 V1.16】RuleResult字段臆测错误（defect_id/message/suggestion/risk/location）** | 属性AttributeError，脚本直接崩溃 | 打印RuleResult全部公开字段；严格使用真实字段`hit_message/rule_suggestion/severity/evidence_ir_refs`；defect_id Sprint‑0本地生成 | 不再抛出RuleResult对象属性不存在异常 |
| **【新增I‑J‑R10 V1.16】RetrievalResult臆测.meta/.metadata嵌套字典** | AttributeError，RAG证据组装失败，evidence置空 | 打印RetrievalResult全部公开字段；使用平层一级属性`source_type/source_title/source_section/snippet`组装evidence字典 | RAG检索成功写入ReviewDefect.evidence，证据数量≥1 |
| **【新增I‑J‑R11 V1.16】Trae IDE索引缓存：out目录部分产物侧边栏看不见** | 开发人员误以为文件丢失 | 1. 不修改`.gitignore`，保持out目录git忽略；2. 调试优先powershell+notepad读取磁盘文件；3. 需要IDE查看产物修改`.trae/.ignore`删除out过滤并**完整重建代码索引** | PowerShell Get‑ChildItem可以完整列出全部产物文件；notepad可打开全部log/json/yaml；磁盘文件完整无丢失 |
| **【新增I‑J‑R12 V1.16】PowerShell docker子进程stderr触发RemoteException，拿不到python traceback** | 脚本直接崩溃，看不到容器内完整报错堆栈 | 执行docker payload前临时保存`ErrorActionPreference`，切为Continue；执行完毕恢复；使用`$LASTEXITCODE`判断退出码；**无论成功失败优先cat打印payload_run.log完整堆栈再exit** | python脚本异常时完整打印payload_run.log内traceback，不会抛出docker.exe RemoteException截断错误信息 |
| **TD‑007技术债务** | Pydantic模型分散在ir/rag/rules各处，新人维护成本高 | Sprint‑0禁止大规模迁移避免import回归；Sprint‑1迭代统一收拢全部Pydantic模型到`app/schemas/*` | Sprint‑1完成迁移后全套pytest PASS |
|**【新增I‑J‑R13 V1.17】test_demo.py pg_session scope=module模块共享session**|单条用例单独执行会产生DB脏数据污染后续case|代码fixture头部增加契约注释；文档明确：test_demo.py必须完整运行整个模块，禁止挑选模块内单条用例执行；细粒度隔离测试使用test_feedback_rule.py(function scope)|执行`pytest tests/test_demo.py`完整模块全部PASS；不要执行单用例|

### §18 Business Freeze清单（Sprint‑0，追加2条Phase‑I）
| 冻结项 | 对应文档/文件 | Sprint 0 完成标准 |
|---|---|---|
| **Schematic IR Schema v1.0** | `docs/ir/schematic_ir_spec_v1.0.md` | 支持 Component 三语义字段解析 |
| **Database Schema v1.0** | `infra/docker/initdb/002_schema.sql` | 10张核心业务表字段固化；Phase‑I追加feedback_item扩展字段、rule_candidates表；rule_candidates确认无task_id列 |
| **Golden Case Format v1.0** | `data/cases/README_CASE_B_DESIGN.md` | 包含 expert_reasoning 与 evidence 结构 |
| **Rule YAML Format v1.0** | `docs/rules/rule_yaml_spec_v1.0.md` | 支持 applicable_condition 与 check_logic |
| **Benchmark Format v1.0** | `docs/benchmark/metrics_definition_v1.0.md` | 明确 Precision/Recall/EAR/EVC 计算公式 |
| **【新增Phase‑I‑Freeze‑6】Feedback Schema v1.0** | `app/schemas/feedback.py` + `infra/docker/initdb/002_schema.sql` | feedback_item字段、6类FeedbackType枚举、rule_candidates表Schema、FeedbackCreate/FeedbackOut Pydantic模型Sprint‑0冻结；Sprint‑1仅允许向后兼容扩展字段，禁止删除字段、修改枚举语义；FeedbackCreate固定使用review_result_id/review_defect_id外键，废弃task_id/defect_id字符串入参。 |
| **【新增Phase‑I‑Freeze‑7】Rule‑Candidate草稿输出格式v1.0** | `app/services/rule_evolution_service.py` | 自动生成proposed_yaml草稿YAML结构冻结；**YAML草稿本体仅存规则本体字段，status/candidate_id等DB元字段不写入yaml文件；status只从DB RuleCandidate对象读取**；Sprint‑1用于benchmark加载候选规则。 |
### Phase‑I‑J完整回归执行命令集合（追加进文档16章节）
```powershell
# 1. 启动容器
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml up -d --build
# 2. 重建数据库（执行002_schema.sql，包含feedback_item Alter、rule_candidates建表）
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m scripts.init_db
# ✅【V1.17重要约束】test_demo.py必须完整执行整个模块；❌禁止挑选单条用例执行：pytest tests/test_demo.py::test_xxx
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest tests/test_demo.py -v
# 3. 全套pytest（单元+集成，包含Phase‑H RAG + Phase‑I Feedback）
# ⚠️test_knowledge_table_has_seed_data需要先ingest_seed_knowledge才PASS（环境校验用例，非业务bug）
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest tests/ -v
# 4. CI模式：只跑单元，跳过@pytest.mark.integration DB集成
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest tests/ -m "not integration" -v
# 5. Rule‑Only benchmark基线
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m tests.test_benchmark
# 6. Phase‑H RAG seed ingest（fake embedding，不触网）
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m app.rag.ingest --config ./data/knowledge/seed_ingest_list.yaml --dry-run
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m app.rag.ingest --config ./data/knowledge/seed_ingest_list.yaml
# 7. Sprint0完整真实Demo（本地开发容器，Windows PowerShell5.1可直接执行；幂等，无需每次init_db）
.\scripts\demo_sprint0.ps1
# ✅【V1.17重要约束】test_demo.py必须完整执行整个模块；❌禁止挑选单条用例执行：pytest tests/test_demo.py::test_xxx
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest tests/test_demo.py -v
# 8. CI Mock‑E2E冒烟测试（无OpenAI外网调用，无磁盘IR文件依赖）
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run pytest tests/test_demo.py -v
# 9. V1.17新增：单独执行容器环境诊断（等价ps1 Step4）
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend /opt/venv/bin/python -m scripts.demo_feedback_payload --diagnose

---

# 第9段：Sprint‑0 红线禁止清单（写入文档末尾）
```markdown
### Sprint‑0 红线禁止清单（写入文档末尾）
1. ❌禁止 Sprint‑0 移动`app/ir`、`app/rag`、`app/rules`内部存量 Pydantic 模型；只新增 feedback/report Pydantic 到`app/schemas/`；存量原地不动；全部收拢 TD‑007 放到 Sprint‑1。
2. ❌禁止`Base.metadata.create_all()`；Schema 唯一事实来源`infra/docker/initdb/002_schema.sql`。
3. ❌Sprint‑0 不做 Alembic 数据库迁移；Alembic 放到 Sprint‑1。
4. ❌Sprint‑0 不接入 LangGraph `knowledge_retrieve`节点；Demo 直接调用 retriever.py；Sprint‑1 再接入。
5. ❌Sprint‑0 不实现 Feedback Web UI；全部通过 Service/CLI/pytest/DB 验证；Web 移交 Sprint‑2。
6. ❌`rule_candidate`草稿 Sprint‑0**禁止自动合并进正式 rules 库，仅输出草稿；人工编辑 + benchmark 校验放到 Sprint‑1**；导出yaml草稿**禁止写入status/candidate_id等数据库元字段**。
7. ❌DDL写表名必须对齐 test_db_schema.py：反馈表叫`feedback_item`，绝对不能写 feedbacks！
8. ❌禁止向`RuleCandidate`ORM构造函数传入task_id参数；rule_candidates表无task_id列；task业务信息通过JOIN `from_feedback_id → feedback_item → review_result`获取。
9. ❌Windows Demo脚本禁止使用`docker‑exec stdout`输出完整JSON/YAML中文结构化文本；**强制容器内写文件 + docker compose cp二进制拷贝回宿主机**。
10. ❌pytest测试禁止从导出rule_candidate_proposed.yaml草稿读取status字段；status校验必须读取DB层RuleCandidate ORM对象。
11. ❌禁止在SQLAlchemy `with get_db_session() as db:`会话结束之后访问ORM对象`.id`等属性；**会话内部预拷贝id到普通int变量，会话外部只使用拷贝后的普通变量，规避DetachedInstanceError**。
12. ❌禁止臆测RuleResult、RetrievalResult字段；开发调试优先打印对象全部公开dir字段确认真实属性；不硬编码臆测的`.meta`、`message`、`defect_id`等不存在字段。
13. ❌不要修改`.gitignore`删除`out/`忽略规则；需要IDE查看产物，只修改`.trae/.ignore`，执行完整重建Trae索引。
14. **❌V1.17新增约束：tests/test_demo.py禁止执行单条用例`pytest tests/test_demo.py::test_xxx`；必须完整运行整个模块`pytest tests/test_demo.py`；需要强隔离用例全部放到test_feedback_rule.py(scope=function)**。
15. **❌V1.17新增约束：evidence_refs入参必须通过source/section/reason三要素最小校验；校验失败直接阻断rule_candidate草稿生成；仅校验key存在，不校验内容语义**。
16. **❌V1.17新增约束：demo_feedback_payload.py location字段禁止直接赋值evidence_ir_refs列表；必须遵循V1.2 §3.1结构sheet/path/coords + ir_refs子字段；Sprint‑1替换AUTO‑DEMO mock占位**。

### 附录：Phase‑I‑J 全部故障复盘（归档，用于后续排错查阅）
> 1. `AttributeError: 'FeedbackItem' object has no attribute 'task_id'`
> 根因：rule_evolution_service直接读取FeedbackItem.task_id；feedback_item表无task_id；修复：JOIN ReviewResult/ReviewDefect拿task_id/defect_id业务字段。
>
> 2. `TypeError: 'task_id' is an invalid keyword argument for RuleCandidate`
> 根因：rule_candidates DDL/ORM不存在task_id字段，代码错误传入task_id；修复：彻底移除传入RuleCandidate的task_id参数；task业务数据走JOIN链路。
>
> 3. `yaml.scanner.ScannerError expected chomping or indentation indicators, but found '‑'`
> 根因：Markdown复制带入U+2011全角破折号`‑`，YAML模板`>‑`非法；修复替换为标准ASCII `>-`。
>
> 4. `FileNotFoundError: schematic_ir.json` test_demo报错
> 根因：Mock‑E2E测试错误读取磁盘IR文件；修复：全部内存mock ReviewResult/ReviewDefect，移除真实IR文件IO依赖。
>
> 5. `KeyError: 'status'`解析导出yaml报错
> 根因：草稿YAML本体不含status元字段；status存在DB rule_candidates表，不在yaml输出；修复：status从DB RuleCandidate对象断言，yaml仅校验语法、version、rule_id。
>
> 6. `UniqueViolation duplicate key value violates unique constraint "review_defect_defect_id_key"`
> 根因：demo脚手架脚本无幂等逻辑，重复运行INSERT固定defect_id；修复`demo_feedback_payload.py`先查询，存在复用主键，不存在再插入。
>
> 7. Windows PowerShell5.1 `ConvertFrom‑Json` JSON乱码解析失败（`VCC鍘昏€︾己澶卞€欓€?`）
> 根因：docker‑exec stdout管道UTF‑8字节被Windows CP936(GBK)破坏性转码；单纯Out‑File‑Encoding utf8无效；修复规范：容器内python写出产物文件，`docker compose cp`二进制拷贝回宿主机，不走stdout传JSON/YAML。
>
> 8. `Error response from daemon: No such container: backend`
> 根因：混用原生`docker cp`，原生docker cp不识别compose服务名backend；修复全部改用`docker compose -f f1 -f f2 cp`。
>
> 9. Markdown复制粘贴U+2011全角破折号，ps1/python/yaml隐形语法报错
> 根因：markdown代码块自动替换半角`-`为U+2011非断连破折号；修复：ps1禁止here‑string内嵌大段python；全局搜索U+2011批量替换ASCII半角`-`。
>
> 10. **V1.16 Bug：`sqlalchemy.orm.exc.DetachedInstanceError: Instance ... is not bound to a Session`**
> 根因：退出`with get_db_session() as db:`会话后，继续访问ORM对象`rr.id / rd.id`；session关闭对象变为detached游离实例，访问id触发过期字段刷新；
> 修复：**会话with块内部读取id赋值普通int变量`out_rr_id/out_rd_id`；with外部只使用普通int变量，不再触碰rr/rd ORM对象**；参考：https://sqlalche.me/e/20/bhk3
>
> 11. **V1.16 Bug：`docker compose exec -T`无PTY模式print换行压扁，多条日志挤同一行**
> 根因：-T关闭伪终端pty，stdout进入块缓冲模式，`\n`换行被docker客户端合并；python print加\n无法修复；
> 修复：业务日志全部写入容器`/app/out/demo_sprint0/payload_run.log`；stdout严格仅输出单行JSON；ps1执行`cat`读取日志打印控制台，再cp拷贝回宿主机产物目录。
>
> 12. **V1.16 Bug：`AttributeError: 'RuleResult' object has no attribute 'defect_id' / message / suggestion / risk / location`**
> 根因：臆测RuleResult字段；RuleResult是规则引擎命中输出，**不会产出defect_id；真实字段hit_message、rule_suggestion、severity、evidence_ir_refs**；
> 修复：打印dir(hit_item)确认全部公开字段；Sprint‑0本地生成`demo_defect_id = f"{hit_item.rule_id}-DEMO‑01"`；使用真实字段映射。
>
> 13. **V1.16 Bug：`AttributeError: 'RetrievalResult' object has no attribute 'meta' / 'metadata'`**
> 根因：臆测元数据是嵌套字典`.meta`；RetrievalResult全部为**平层一级属性：source_type、source_title、source_section、snippet**；
> 修复：打印dir(first_hit)拿到真实字段，直接读取平层属性组装evidence字典。
>
> 14. **V1.16 Bug：docker子进程抛出`docker.exe : Traceback ... RemoteException`，ps1直接终止拿不到python完整traceback**
> 根因：全局`$ErrorActionPreference=Stop`，`& docker @cmd 2>&1`捕获stderr流会转换成terminating异常，截断容器内traceback；
> 修复：执行docker payload前临时保存旧ErrorActionPreference切换Continue；执行完恢复；使用`$LASTEXITCODE`判断docker退出码；**无论payload成功失败优先cat打印payload_run.log完整堆栈，再exit退出**。
>
> 15. **V1.16 IDE现象：Trae IDE侧边栏部分产物看不见（payload_run.log可见，feedback_gap_submit.log看不见），notepad/powershell Get‑ChildItem确认磁盘文件真实存在**
> 根因：`.gitignore`配置`out/`，Trae增量索引过滤被忽略目录；脚本每次删除重建out目录，残留旧索引缓存造成“部分可见部分不可见”假象；**不是文件丢失，磁盘完整**；
> 修复策略：不要修改`.gitignore`；调试优先powershell+notepad读取；需要IDE浏览产物：编辑`.trae/.ignore`删除out过滤规则，**执行完整重建代码索引**。
> 16. **V1.17 Bug：ps1脚本内嵌套bash heredoc引发语法报错**
> ```
> warning: here-document at line 1 delimited by end-of-file (wanted `PYEOF`)
> SyntaxError: invalid non-printable character U+FEFF
> ```
> - 现象：Step4容器Python诊断报错，业务Demo不受影响；
> - 根因：PowerShell here‑string嵌套bash `<<'PYEOF'` heredoc；缩进、引号、UTF‑8‑BOM多重转义污染，导致PYEOF终止标记被破坏；
> - 修复（demo_sprint0.ps1 V1.17）：**彻底移除ps1脚本内部拼接/传递任何bash/python源码字符串；在已有`demo_feedback_payload.py`新增`--diagnose`命令行分支；ps1 Step4直接调用`/opt/venv/bin/python -m scripts.demo_feedback_payload --diagnose`；诊断逻辑放在容器磁盘py文件，完全消除多层字符串转义坑；不再新增独立diag_env.py文件**。
>
> 17. **V1.17 Bug：test_demo.py pg_session scope="module" DeepSeek评审权衡**
> - DeepSeek建议：将`pg_session` fixture `scope="module"`改为`scope="function"`，实现每个用例DB会话完全隔离；
> - ❌拒绝该变更；
> - 根因：`test_demo.py`是**顺序式E2E冒烟模块，测试用例之间共享前置写入的ReviewResult/ReviewDefect测试数据**；若改为`scope="function"`，每条用例执行完毕事务自动回滚，后续用例读取不到前置DB记录，直接测试失败；
> - ✅缓解方案：维持`scope="module"`；fixture头部增加大段契约注释；文档强制约束：**test_demo.py必须完整执行整个模块，禁止单独挑选模块内部单条用例执行；所有需要强隔离的细粒度集成测试全部放在`test_feedback_rule.py`（scope="function"）**；
> - 正确执行命令：`uv run pytest tests/test_demo.py -v`（完整模块）；禁止：`pytest tests/test_demo.py::test_xxx -v`单用例执行。

### 附录：V1.17 Git Commit Message
```git
feat(sprint0): Phase‑I‑J feedback demo V1.17 demo_sprint0.ps1 upgrade

Changes V1.17:
1. demo_sprint0.ps1 V1.17:
   - Step4 diagnostic refactor: remove inline bash heredoc / PYEOF inside powershell script.
   - Reuse existing scripts/demo_feedback_payload.py, add CLI flag --diagnose.
   - --diagnose mode: only run python env & package import check, NO IR/RAG/DB write, zero side‑effect.
   - Delete standalone diag_env.py, reduce extra file maintenance cost.
   - Fix two historical syntax error: U+FEFF BOM & "here‑document wanted PYEOF delimited by end‑of‑file".
2. Step6 manual hint improvement:
   - Remove obsolete yellow warning Write‑Warning.
   - Replace with English info text: clarify these are copy‑paste‑only reference commands, script never execute them automatically.
3. DeepSeek review decision record(no new top‑level chapter, embed into doc header/fault log/risk matrix):
   - ✅ keep prepare_out_dir shutil.rmtree (already landed V1.16)
   - ❌ REJECT change pg_session scope=module → scope=function.
     Reason: test_demo.py is sequential E2E smoke test, test cases share pre‑inserted DB records.
     scope=function will rollback transaction after each case → subsequent test cannot find required data → test failure.
     Mitigation: keep scope=module; add fixture docstring contract; MUST run full module `pytest tests/test_demo.py`; forbid running individual test case; fine‑grained isolated tests in test_feedback_rule.py (scope=function).
4. docs/feedback_demo.md upgrade to V1.17:
   - append version changelog inside document header;
   - update file inventory: demo_feedback_payload.py --diagnose parameter;
   - add two new bug records to fault‑replay appendix;
   - update risk matrix I‑J‑R13;
   - update step4/step6 description;
   - update acceptance items、sprint‑1 todo list、red‑line forbidden list、shell command snippets.

Remain V1.16 fixes (already landed):
- prepare_out_dir shutil.rmtree for test_demo.py
- evidence_refs source/section/reason minimal assert validation rule_evolution_service
- demo_feedback_payload.py location V1.2 §3.1 schema fix(sheet/path/coords + ir_refs sub‑key)
- Fix DetachedInstanceError, docker exec‑T newline squash, RuleResult / RetrievalResult field mapping bugs.

Test:
- .\scripts\demo_sprint0.ps1 full run pass, Step4 diagnostic output complete without PYEOF / U+FEFF error.
- pytest tests/test_demo.py full module run pass.
- test_feedback_rule.py all isolated integration test pass.

Artifact:
out/demo_sprint0/: candidate_list.json, rule_candidate_proposed.yaml, knowledge_gap_backlog.json, payload_run.log, feedback submit logs.

Sprint‑1 TODO unchanged, add new item: replace AUTO‑DEMO mock location with real IR parsed sheet/path/coords，keep ir_refs for traceability.


---

# ✅落地核对清单（全部10段）
- [x] 第1段：头部版本升级V1.17，完整V1.17变更说明
- [x] 第2段：I.2文件清单更新3个脚本说明
- [x] 第3段：I.4追加I‑8 + V1.17‑1~5验收checklist
- [x] 第4段：J.1链路补充 + Sprint‑1第6条待办
- [x] 第5段：J.3追加J‑7、J‑8验收项
- [x] 第6段：风险矩阵追加I‑J‑R13风险行
- [x] 第7段：红线清单追加3条V1.17约束
- [x] 第8段：回归命令增加pytest禁止单用例注释 + `--diagnose`命令
- [x] 第9段：故障复盘追加Bug16、Bug17
- [x] 第10段：文件末尾追加V1.17 git commit附录

> 重要规则：**全程没有创建任何不存在的虚构标题，全部在原有文档已有章节内做追加；原有文字表格完全保留，不做删除重构**。

## 修改完成后校验命令
```powershell
# 进入项目根目录，简单markdown语法校验
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend uv run python -m markdown_it docs/feedback_demo.md --check





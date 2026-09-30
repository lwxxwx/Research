"""
Phase-J CI Mock E2E冒烟测试
⚠️CI流水线：mock全部IR/规则引擎输出，**完全不调用OpenAI外网、不读取真实case文件、不消耗token**
真实完整Demo跑scripts/demo_sprint0.ps1（本地需要.env OPENAI_API_KEY + 完整case数据）
链路（纯内存mock，无外部文件依赖）：
Mock IR数据 → Mock规则引擎输出 → Mock-LLM生成ReviewReport → DB落库ReviewResult+ReviewDefect
→ 模拟提交2条反馈 false_negative + knowledge_gap
→ rule_evolution生成rule_candidate草稿、导出knowledge-gap backlog
只校验链路完整性、产物存在、DB记录；**不校验LLM文本语义正确性、不读取磁盘case文件**

v1.1 变更（对齐 schema v1.2）：
- ★ rule_candidate 从 dict 改为 RuleCandidate 强类型（schema v1.2 改进 4）
- ★ fb_out1/fb_out2 的 model_dump() 改为 model_dump(mode="json")，
     避免 datetime 序列化失败
- ★ review_output_json 加 ensure_ascii=False，与 report.json 风格统一
"""
import json
import os
import shutil  # ✅ NEW: 导入shutil用于rmtree清理OUT_DIR
from pathlib import Path
import pytest
import yaml
from sqlalchemy.orm import Session
from app.schemas.feedback import FeedbackCreate, FeedbackType, RuleCandidate  # ★ v1.1：加 RuleCandidate
from app.services.feedback_service import FeedbackService
from app.services.rule_evolution_service import RuleEvolutionService
from app.persistence.models import ReviewResult, ReviewDefect, RuleCandidate as RuleCandidateORM  # ★ v1.1：ORM 别名
# ✅容器内固定根目录：容器项目根目录永远 /app
#PROJECT_ROOT = Path("/app")
#OUT_DIR = PROJECT_ROOT / "out/demo_sprint0"
OUT_DIR = Path("/out/demo_sprint0")

@pytest.fixture(scope="module")
def pg_session():
    from app.core.config import settings
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine(settings.database_url, echo=False)
    conn = engine.connect()
    trans = conn.begin()
    sess = Session(bind=conn)
    # ✅ NEW: scope=module契约注释 DeepSeek-5.1
    """
    ⚠️【契约说明 scope="module"】
    本fixture为module级别session，模块下**全部测试共享同一个数据库会话事务**。
    👉设计定位：test_demo.py为整体E2E冒烟模块，必须完整执行整个模块；
    ❌不支持单独运行模块内部某一条测试（pytest tests/test_demo.py::xxx），会产生数据污染。
    ✅单元/需要强隔离的测试统一放在 test_feedback_rule.py (scope=function)
    """
    yield sess
    sess.close()
    trans.rollback()
    conn.close()

@pytest.fixture(scope="module")
def prepare_out_dir():
    # ===== 旧代码注释掉 =====
    # os.makedirs(OUT_DIR, exist_ok=True)
    # yield
    # # 测试结束不自动清理，方便人工查看产物

    # ✅ NEW: DeepSeek-5.2 执行前rmtree清空旧产物，避免历史文件造成断言误通过
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # yield之后**不删除产物**，CI跑完保留产物用于人工排查问题


def test_demo_sprint0_mock_e2e(pg_session: Session, prepare_out_dir):
    """CI Mock完整Sprint0 Demo冒烟，无OpenAI外网，无磁盘case文件依赖"""
    # ========= ✅ 全部Mock，删除load_ir / execute_all_rules真实文件读取逻辑 =========
    # Step1 Mock RAG证据（跳过真实retriever检索，Sprint-1接入LangGraph再用真实retriever）
    mock_rag_evidence = [
        {"type":"rag_ref","source_type":"datasheet","source":"STC89C55RC_ds.pdf","section":"晶振电路","reason":"晶振负载电容建议22pF"}
    ]
    # Step2 Mock LLM输出ReviewReport（不调用OpenAI），Defect满足V1.2 10字段 + review_status
    mock_report = {
        "task_id": "R-MOCK-001",
        "report_id": "R-MOCK-001",
        "defects": [
            {
                "defect_id":"DEF-MOCK-001",
                "category":"power",
                "location":{
                    "sheet":"POWER_PAGE1",
                    "path":"U1.VCC",
                    "coords":{"x":1,"y":2}
                },
                "component":"U1",
                "net":"VCC",
                "risk":"critical",
                "evidence": mock_rag_evidence,
                "root_cause":"VCC缺少去耦电容",
                "suggestion":"U1 VCC引脚附近增加100nF X7R去耦电容",
                "confidence":0.92,
                "review_status":"AI_CONFIRMED"
            }
        ],
        "summary": {
            "evidence_coverage":0.85,
            "review_status_distribution":{"AI_CONFIRMED":1,"NEED_EXPERT_REVIEW":0,"LOW_CONFIDENCE":0}
        }
    }
    report_json_path = OUT_DIR / "report.json"
    report_json_path.write_text(json.dumps(mock_report,indent=2,ensure_ascii=False),encoding="utf-8")
    # 【关键】先写入ReviewResult、ReviewDefect拿到真实外键ID，再提交Feedback
    mock_rr = ReviewResult(
        task_id="R-MOCK-001",
        # ★ v1.1：与 report.json 的中文处理保持一致
        # --- 原代码（保留，已弃用） ---
        # review_output_json=json.dumps(mock_report),
        review_output_json=json.dumps(mock_report, ensure_ascii=False),
        is_rule_only=True,
        category="power",
        risk="critical",
        review_status="AI_CONFIRMED"
    )
    pg_session.add(mock_rr)
    pg_session.flush()
    mock_rd = ReviewDefect(
        review_result_id=mock_rr.id,
        defect_id="DEF-MOCK-001",
        category="power",
        location={
            "sheet":"POWER_PAGE1",
            "path":"U1.VCC",
            "coords":{"x":1,"y":2}
        },
        component="U1",
        net="VCC",
        risk="critical",
        evidence=mock_rag_evidence,
        root_cause="VCC缺少去耦电容",
        suggestion="U1 VCC引脚附近增加100nF X7R去耦电容",
        confidence=0.92,
        review_status="AI_CONFIRMED",
        origin="ai_discovered"
    )
    pg_session.add(mock_rd)
    pg_session.flush()
    # Step3 模拟提交2条专家反馈
    fb_svc = FeedbackService()
    # 反馈1 false_negative → 生成rule_candidate草稿
    # ★ v1.1：rule_candidate 从 dict 改为 RuleCandidate 强类型（schema v1.2）
    # --- 原代码（保留，已弃用：dict 缺必填字段，Pydantic 校验失败） ---
    # rule_candidate={"title":"VCC去耦缺失候选规则草稿","severity":"critical"},
    fb1 = FeedbackCreate(
        review_result_id=mock_rr.id,
        review_defect_id=mock_rd.id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="该类场景漏检，需要新增电源去耦规则",
        rule_candidate=RuleCandidate(
            rule_id="POWER_DECOUP_MOCK_001",
            rule_name="VCC去耦缺失候选规则草稿",
            category="power",
            severity="critical",
            rule_basis="Mock E2E：该场景漏检，需新增电源去耦规则",
        ),
        created_by=None
    )
    fb_out1 = fb_svc.submit_feedback(pg_session, fb1)
    assert fb_out1.rule_candidate_ref is not None
    #反馈2 knowledge_gap →知识缺口
    fb2 = FeedbackCreate(
        review_result_id=mock_rr.id,
        review_defect_id=mock_rd.id,
        feedback_type=FeedbackType.KNOWLEDGE_GAP,
        expert_suggestion="需要补充STC89C55RC电源章节datasheet片段",
        created_by=None
    )
    fb_out2 = fb_svc.submit_feedback(pg_session, fb2)
    feedback_log_path = OUT_DIR / "feedback_submit_log.json"
    # ★ v1.1：model_dump(mode="json")，让 datetime 序列化为 ISO 字符串
    # --- 原代码（保留，已弃用：model_dump() 默认 python 模式，datetime 无法 json 序列化） ---
    # feedback_log_path.write_text(json.dumps([fb_out1.model_dump(), fb_out2.model_dump()],indent=2,ensure_ascii=False),encoding="utf-8")
    feedback_log_path.write_text(
        json.dumps(
            [fb_out1.model_dump(mode="json"), fb_out2.model_dump(mode="json")],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Step4 rule_evolution导出产物
    evol_svc = RuleEvolutionService()
    rc_yaml_path = OUT_DIR / "rule_candidate_proposed.yaml"
    evol_svc.export_candidate_yaml(pg_session, fb_out1.rule_candidate_ref, str(rc_yaml_path))
    gap_json_path = OUT_DIR / "knowledge_gap_backlog.json"
    evol_svc.export_knowledge_gap_backlog(pg_session, str(gap_json_path))
    # ===== 全部产物存在性校验 =====
    assert report_json_path.exists()
    assert feedback_log_path.exists()
    assert rc_yaml_path.exists()
    assert gap_json_path.exists()
    # ✅修复：YAML草稿文件是规则本体，**不含DB元字段status**；status校验去查DB RuleCandidate对象，不要读yaml
    # ★ v1.1：ORM 用别名 RuleCandidateORM，避免与 Pydantic RuleCandidate 重名
    # --- 原代码（保留，已弃用：RuleCandidate 现在指 Pydantic，非 ORM） ---
    # rc_db: RuleCandidate = pg_session.query(RuleCandidate).filter(RuleCandidate.candidate_id == fb_out1.rule_candidate_ref).first()
    rc_db: RuleCandidateORM = pg_session.query(RuleCandidateORM).filter(RuleCandidateORM.candidate_id == fb_out1.rule_candidate_ref).first()
    assert rc_db is not None
    assert rc_db.status == "proposed"   # status 在数据库记录校验
    # 仅校验YAML语法合法，解析成功即可，不再读取status key
    rc_text = rc_yaml_path.read_text(encoding="utf-8")
    rc_parsed = yaml.safe_load(rc_text)
    assert rc_parsed is not None
    assert rc_parsed["version"] == "0.1-proposed"
    assert rc_parsed["rule_id"] is not None
    gap_data = json.loads(gap_json_path.read_text(encoding="utf-8"))
    assert len(gap_data)>=1
    print("\n✅ Sprint0 Mock-E2E Demo全部链路冒烟通过，产物输出到out/demo_sprint0")
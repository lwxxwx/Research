# backend/tests/test_feedback_rule.py
"""
Phase-I Feedback & RuleCandidate集成测试
测试范围：
1. FeedbackType 6类枚举校验
2. 全部6类feedback_item落库测试（依赖review_result/review_defect外键）
3. false_negative / new_rule_candidate自动生成proposed rule_candidate草稿
4. knowledge_gap导出backlog JSON
5. rule_candidate草稿YAML语法可解析（仅语法校验，不跑benchmark）
标记@pytest.mark.integration，依赖真实PostgreSQL；CI单元模式-m "not integration"跳过
"""
import json
import pytest
import yaml
from sqlalchemy.orm import Session
from app.schemas.feedback import FeedbackCreate, FeedbackType, SuggestionDiff
from app.services.feedback_service import FeedbackService
from app.services.rule_evolution_service import RuleEvolutionService
from app.persistence.models import FeedbackItem, RuleCandidate, ReviewResult, ReviewDefect


@pytest.fixture(scope="function")
def pg_session():
    """每个测试函数独立事务，测试结束rollback，不污染DB"""
    from app.core.config import settings
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine(settings.database_url, echo=False)
    conn = engine.connect()
    trans = conn.begin()
    sess = Session(bind=conn)
    yield sess
    sess.close()
    trans.rollback()
    conn.close()


@pytest.fixture(scope="function")
def feedback_test_fixture(pg_session: Session):
    """
    测试前置：创建测试用ReviewResult + ReviewDefect
    返回：(review_result_id, review_defect_id)
    """
    rr = ReviewResult(
        task_id="TASK-TEST-001",
        review_output_json={},
        is_rule_only=True,
        category="power",
        risk="medium",
        review_status="AI_CONFIRMED"
    )
    pg_session.add(rr)
    pg_session.flush()

    rd = ReviewDefect(
        review_result_id=rr.id,
        defect_id="DEF-TEST-001",
        category="power",
        location={"sheet":"P1"},
        risk="medium",
        evidence=[],
        root_cause="test",
        suggestion="test",
        confidence=0.8,
        review_status="AI_CONFIRMED",
        origin="rule"
    )
    pg_session.add(rd)
    pg_session.flush()
    return rr.id, rd.id


def test_feedback_type_enum_all_six():
    """单元：校验6类FeedbackType枚举完整无遗漏"""
    vals = set(FeedbackType)
    expect = {
        FeedbackType.CORRECT_DEFECT,
        FeedbackType.FALSE_POSITIVE,
        FeedbackType.FALSE_NEGATIVE,
        FeedbackType.SUGGESTION_UPDATE,
        FeedbackType.NEW_RULE_CANDIDATE,
        FeedbackType.KNOWLEDGE_GAP
    }
    assert vals == expect


@pytest.mark.integration
def test_feedback_correct_defect_no_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-1 correct_defect落库，**不生成rule_candidate**"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.CORRECT_DEFECT,
        expert_suggestion="确认该缺陷正确",
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert out.id > 0
    assert out.feedback_type == FeedbackType.CORRECT_DEFECT
    assert out.rule_candidate_ref is None
    cnt = pg_session.query(RuleCandidate).filter(RuleCandidate.from_feedback_id == out.id).count()
    assert cnt == 0


@pytest.mark.integration
def test_feedback_false_positive_no_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-2 false_positive落库，不生成rule_candidate"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.FALSE_POSITIVE,
        expert_suggestion="误报，实际无风险",
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert out.rule_candidate_ref is None
    assert pg_session.query(RuleCandidate).filter(RuleCandidate.from_feedback_id == out.id).count() == 0


@pytest.mark.integration
def test_feedback_suggestion_update_no_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-3 suggestion_update落库，存储suggestion_diff_json，不生成候选"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.SUGGESTION_UPDATE,
        suggestion_diff=[SuggestionDiff(old="旧建议", new="优化后的建议")],
        expert_suggestion="修改建议文案",
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert len(out.suggestion_diff) == 1
    assert out.suggestion_diff[0].old == "旧建议"
    assert out.rule_candidate_ref is None
    assert pg_session.query(RuleCandidate).filter(RuleCandidate.from_feedback_id == out.id).count() == 0


@pytest.mark.integration
def test_feedback_false_negative_generate_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-4 false_negative → 自动生成proposed rule_candidate草稿"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    hint = {
        "title": "漏检-电源去耦候选",
        "severity": "high",
        "evidence_refs": [
            {
                "source_type": "datasheet",
                "source": "MP2307.pdf",
                "section": "p6",
                "reason": "MP2307 datasheet p6 明确建议 VCC 引脚附近增加 100nF 去耦电容"
            }
        ]
    }
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="该场景漏检，需要新增规则",
        rule_candidate=hint,
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    cand_ref = out.rule_candidate_ref
    assert cand_ref is not None
    rc: RuleCandidate = pg_session.query(RuleCandidate).filter(RuleCandidate.candidate_id == cand_ref).first()
    assert rc is not None
    assert rc.from_feedback_id == out.id
    assert rc.status == "proposed"
    assert rc.severity == "high"
    assert "AUTO-GENERATED DRAFT RULE CANDIDATE" in rc.proposed_yaml
    # 校验草稿YAML语法合法（仅语法，不执行）
    parsed = yaml.safe_load(rc.proposed_yaml)
    assert parsed["rule_id"] is not None
    assert parsed["severity"] == "high"


@pytest.mark.integration
def test_feedback_new_rule_candidate_generate_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-5 new_rule_candidate反馈生成候选草稿"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.NEW_RULE_CANDIDATE,
        expert_suggestion="专家直接提交新规则候选",
        rule_candidate={"title":"专家提交时钟规则草稿","severity":"medium"},
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert out.rule_candidate_ref is not None
    rc = pg_session.query(RuleCandidate).filter(RuleCandidate.candidate_id == out.rule_candidate_ref).first()
    assert rc.status == "proposed"
    assert yaml.safe_load(rc.proposed_yaml) is not None


@pytest.mark.integration
def test_feedback_knowledge_gap_backlog_export(pg_session: Session, feedback_test_fixture, tmp_path):
    """TC-I-6 knowledge_gap落库；导出知识缺口JSON backlog"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    ev1 = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.KNOWLEDGE_GAP,
        expert_suggestion="缺失MP2307热降额datasheet片段，需要入库",
        created_by=None
    )
    # 新建第二组缺陷，避免唯一约束冲突
    rr2 = ReviewResult(task_id="TASK-TEST-002", review_output_json={}, is_rule_only=True, category="power", risk="medium", review_status="AI_CONFIRMED")
    pg_session.add(rr2)
    pg_session.flush()
    rd2 = ReviewDefect(review_result_id=rr2.id, defect_id="DEF-TEST-002", category="power", location={"sheet":"P1"}, risk="medium", evidence=[], root_cause="test", suggestion="test", confidence=0.8, review_status="AI_CONFIRMED", origin="rule")
    pg_session.add(rd2)
    pg_session.flush()

    ev2 = FeedbackCreate(
        review_result_id=rr2.id,
        review_defect_id=rd2.id,
        feedback_type=FeedbackType.KNOWLEDGE_GAP,
        expert_suggestion="缺少CAN终端电阻参考设计",
        created_by=None
    )
    svc.submit_feedback(pg_session, ev1)
    svc.submit_feedback(pg_session, ev2)

    rev = RuleEvolutionService()
    out_file = tmp_path / "knowledge_gap_backlog.json"
    rev.export_knowledge_gap_backlog(pg_session, str(out_file))
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) >=2
    assert any("MP2307热降额" in x["expert_suggestion"] for x in data)


@pytest.mark.integration
def test_rule_evolution_list_candidates(pg_session: Session, feedback_test_fixture):
    """测试list_candidates过滤status"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="漏检生成候选",
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    rev = RuleEvolutionService()
    lst = rev.list_candidates(pg_session, status="proposed")
    ids = [x["candidate_id"] for x in lst]
    assert out.rule_candidate_ref in ids
    empty = rev.list_candidates(pg_session, status="accepted")
    assert len(empty) ==0

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

v1.1 变更（schema v1.1 强类型对齐）：
- ★ rule_candidate 从 dict 改为 RuleCandidate 强类型（schema v1.1 改进 4）
- ★ 新增 test_feedback_out_created_at_is_datetime（schema v1.1 改进 2）
- ★ 删除对 "AUTO-GENERATED DRAFT RULE CANDIDATE" 硬编码字符串的断言
- ★ import 增加 RuleCandidate

v1.2 变更（RuleCandidate 字段集对齐）：
- ★ RuleCandidate 字段集从 5 扩到 11（对齐 rule_format.md + models.py）
  - rule_text → rule_name（改名）
  - rationale → rule_basis（改名）
  - 新增 category 必填、applicable_condition、check_logic、suggestion、
        title、description、evidence_refs
- ★ 更新 test_feedback_false_negative_generate_candidate 的 RuleCandidate 构造
- ★ 更新 test_feedback_new_rule_candidate_generate_candidate 的 RuleCandidate 构造
- ★ 补字段集完整传递断言（验证 rule_name/rule_basis/suggestion/check_logic/
  evidence_refs 都进 proposed_yaml；验证 ORM 冗余字段 title/description/evidence_refs）

v1.3 变更（rule_evolution_service v1.3 覆盖增强）：
- ★ 新增 test_feedback_false_negative_evidence_refs_missing_field_rejected
     覆盖 evidence_refs 缺 section 时抛 ValueError
     （对应 rule_evolution_service v1.3：assert -> raise 的改动）
- ★ 新增 test_feedback_false_negative_evidence_refs_missing_source_rejected
     覆盖 evidence_refs 缺 source 时抛 ValueError
- ★ 新增 test_export_candidate_yaml_refuses_overwrite
     覆盖 export_candidate_yaml 的 FileExistsError（拒绝覆盖）
"""
import json
import pytest
import yaml
from sqlalchemy.orm import Session
from app.schemas.feedback import FeedbackCreate, FeedbackType, SuggestionDiff, RuleCandidate
from app.services.feedback_service import FeedbackService
from app.services.rule_evolution_service import RuleEvolutionService
from app.persistence.models import FeedbackItem, RuleCandidate as RuleCandidateORM, ReviewResult, ReviewDefect


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
    cnt = pg_session.query(RuleCandidateORM).filter(RuleCandidateORM.from_feedback_id == out.id).count()
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
    assert pg_session.query(RuleCandidateORM).filter(RuleCandidateORM.from_feedback_id == out.id).count() == 0


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
    assert pg_session.query(RuleCandidateORM).filter(RuleCandidateORM.from_feedback_id == out.id).count() == 0


@pytest.mark.integration
def test_feedback_false_negative_generate_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-4 false_negative → 自动生成proposed rule_candidate草稿"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()

    # ★ v1.2：RuleCandidate 字段集对齐（11 字段）
    # --- 原代码（保留，已弃用） ---
    # hint = {
    #     "title": "漏检-电源去耦候选",
    #     "severity": "high",
    #     "evidence_refs": [
    #         {
    #             "source_type": "datasheet",
    #             "source": "MP2307.pdf",
    #             "section": "p6",
    #             "reason": "MP2307 datasheet p6 明确建议 VCC 引脚附近增加 100nF 去耦电容"
    #         }
    #     ]
    # }
    # --- v1.1 版本（保留，已弃用） ---
    # hint = RuleCandidate(
    #     rule_id="POWER_DECOUP_001",
    #     rule_text="VCC 引脚附近应增加 100nF 去耦电容（MP2307 datasheet p6 建议）",
    #     severity="high",
    #     category="power",
    #     rationale="漏检-电源去耦候选；来源：MP2307 datasheet p6",
    # )
    hint = RuleCandidate(
        rule_id="POWER_DECOUP_001",
        rule_name="VCC 引脚附近应增加 100nF 去耦电容",             # ★ 改名：原 rule_text
        category="power",
        severity="high",
        rule_basis="漏检-电源去耦候选；来源：MP2307 datasheet p6",  # ★ 改名：原 rationale
        # ★ 新增：验证字段集完整传递
        suggestion="建议在 VCC 引脚 5mm 内增加 0.1μF X7R 陶瓷电容到 GND",
        applicable_condition={"scope": "schematic"},
        check_logic={
            "function": "circuit_checks.check_decoupling",
            "params": {
                "power_net_names": ["VCC"],
                "decouple_cap_value_expected": "0.1μF",
            },
        },
        evidence_refs=[
            {
                "source": "MP2307.pdf",
                "section": "p6",
                "reason": "MP2307 datasheet p6 明确建议 VCC 引脚附近增加 100nF 去耦电容",
            }
        ],
        title="漏检-电源去耦候选",
        description="由 false_negative 反馈自动生成",
    )
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
    rc: RuleCandidateORM = pg_session.query(RuleCandidateORM).filter(RuleCandidateORM.candidate_id == cand_ref).first()
    assert rc is not None
    assert rc.from_feedback_id == out.id
    assert rc.status == "proposed"
    assert rc.severity == "high"

    # ★ 删除硬编码字符串断言，仅保留结构化断言
    # --- 原代码（保留，已弃用） ---
    # assert "AUTO-GENERATED DRAFT RULE CANDIDATE" in rc.proposed_yaml

    # 校验草稿YAML语法合法（仅语法，不执行）
    parsed = yaml.safe_load(rc.proposed_yaml)
    assert parsed["rule_id"] is not None
    assert parsed["severity"] == "high"

    # ★ v1.2：验证字段集完整传递（rule_candidate → proposed_yaml）
    assert parsed["rule_id"] == "POWER_DECOUP_001"
    assert parsed["rule_name"] == "VCC 引脚附近应增加 100nF 去耦电容"
    assert "MP2307" in parsed["rule_basis"]
    assert "0.1μF X7R" in parsed["suggestion"]
    assert parsed["check_logic"]["function"] == "circuit_checks.check_decoupling"

    # ★ v1.2：验证 ORM 冗余字段（title / description / evidence_refs）
    assert rc.title == "漏检-电源去耦候选"
    assert rc.description == "由 false_negative 反馈自动生成"
    assert rc.evidence_refs is not None
    assert len(rc.evidence_refs) == 1
    assert rc.evidence_refs[0]["source"] == "MP2307.pdf"


@pytest.mark.integration
def test_feedback_new_rule_candidate_generate_candidate(pg_session: Session, feedback_test_fixture):
    """TC-I-5 new_rule_candidate反馈生成候选草稿"""
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()

    # ★ v1.2：RuleCandidate 字段集对齐（11 字段）
    # --- 原代码（保留，已弃用） ---
    # rule_candidate={"title":"专家提交时钟规则草稿","severity":"medium"}
    # --- v1.1 版本（保留，已弃用） ---
    # rule_candidate=RuleCandidate(
    #     rule_id="CLOCK_001",
    #     rule_text="时钟走线需满足阻抗与长度匹配要求",
    #     severity="medium",
    #     category="clock",
    # ),
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.NEW_RULE_CANDIDATE,
        expert_suggestion="专家直接提交新规则候选",
        rule_candidate=RuleCandidate(
            rule_id="CLOCK_001",
            rule_name="时钟走线需满足阻抗与长度匹配要求",        # ★ 改名：原 rule_text
            category="clock",
            severity="medium",
            rule_basis="专家经验：高速时钟走线需控制阻抗",        # ★ 新增
        ),
        created_by=None
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert out.rule_candidate_ref is not None
    rc = pg_session.query(RuleCandidateORM).filter(RuleCandidateORM.candidate_id == out.rule_candidate_ref).first()
    assert rc.status == "proposed"
    parsed = yaml.safe_load(rc.proposed_yaml)
    assert parsed is not None

    # ★ v1.2：验证字段集完整传递
    assert parsed["rule_id"] == "CLOCK_001"
    assert parsed["rule_name"] == "时钟走线需满足阻抗与长度匹配要求"


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
    assert len(data) >= 2
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
    assert len(empty) == 0


# ============================================================
# ★ v1.1 新增：schema v1.1 改进 2 回归测试
# FeedbackOut.created_at 类型断言
# ============================================================
@pytest.mark.integration
def test_feedback_out_created_at_is_datetime(pg_session: Session, feedback_test_fixture):
    """schema v1.1：FeedbackOut.created_at 类型为 datetime"""
    from datetime import datetime
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.CORRECT_DEFECT,
        expert_suggestion="created_at 类型检查",
        created_by=None,
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert isinstance(out.created_at, datetime), (
        f"created_at 应为 datetime，实际 {type(out.created_at)}"
    )



# ============================================================
# ★ v1.3 新增：rule_evolution_service 的负例与防御性逻辑测试
# 对应 rule_evolution_service.py v1.3 的：
#   - evidence_refs 三要素校验（assert -> raise 改动）
#   - export_candidate_yaml 的 FileExistsError（拒绝覆盖）
# ============================================================
@pytest.mark.integration
def test_feedback_false_negative_evidence_refs_missing_field_rejected(
    pg_session: Session, feedback_test_fixture
):
    """
    V1.2 §3.2：evidence_refs 每条必须含 source / section / reason 三要素
    缺字段时 rule_evolution_service 应抛 ValueError（v1.3：assert -> raise）

    覆盖点：rule_evolution_service.generate_candidate_from_feedback 的
            evidence_refs 三要素校验
    """
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()

    # ★ 故意缺 "section"
    bad_hint = RuleCandidate(
        rule_id="BAD_EVIDENCE_001",
        rule_name="缺 section 的候选（测试负例）",
        category="power",
        severity="high",
        rule_basis="测试 evidence_refs 三要素校验",
        evidence_refs=[
            {"source": "MP2307.pdf", "reason": "故意缺 section 字段"}
        ],
    )
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="验证 evidence_refs 缺字段时拒绝",
        rule_candidate=bad_hint,
        created_by=None,
    )

    with pytest.raises(ValueError, match="缺失必填字段 section"):
        svc.submit_feedback(pg_session, fb_in)


@pytest.mark.integration
def test_feedback_false_negative_evidence_refs_missing_source_rejected(
    pg_session: Session, feedback_test_fixture
):
    """
    同上：evidence_refs 缺 source 字段时抛 ValueError

    覆盖点：三要素中 source 的校验路径
    """
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()

    bad_hint = RuleCandidate(
        rule_id="BAD_EVIDENCE_002",
        rule_name="缺 source 的候选（测试负例）",
        category="power",
        severity="high",
        rule_basis="测试 evidence_refs 三要素校验",
        evidence_refs=[
            {"section": "p6", "reason": "故意缺 source 字段"}
        ],
    )
    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="验证 evidence_refs 缺 source 时拒绝",
        rule_candidate=bad_hint,
        created_by=None,
    )

    with pytest.raises(ValueError, match="缺失必填字段 source"):
        svc.submit_feedback(pg_session, fb_in)


@pytest.mark.integration
def test_export_candidate_yaml_refuses_overwrite(
    pg_session: Session, feedback_test_fixture, tmp_path
):
    """
    覆盖点：rule_evolution_service.export_candidate_yaml 的
            「拒绝覆盖已有文件」防御性逻辑（FileExistsError）

    场景：
      1) 第一次导出：成功（文件不存在）
      2) 第二次导出：抛 FileExistsError（文件已存在）
    """
    rr_id, rd_id = feedback_test_fixture
    svc = FeedbackService()

    fb_in = FeedbackCreate(
        review_result_id=rr_id,
        review_defect_id=rd_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="测试 export_candidate_yaml 拒绝覆盖",
        created_by=None,
    )
    out = svc.submit_feedback(pg_session, fb_in)
    assert out.rule_candidate_ref is not None

    rev = RuleEvolutionService()
    target = tmp_path / "candidate.yaml"

    # 1) 第一次导出：成功
    rev.export_candidate_yaml(pg_session, out.rule_candidate_ref, str(target))
    assert target.exists(), "首次导出后文件应存在"
    content_first = target.read_text(encoding="utf-8")
    assert "AUTO-GENERATED DRAFT RULE CANDIDATE" in content_first

    # 2) 第二次导出：拒绝覆盖
    with pytest.raises(FileExistsError, match="already exists"):
        rev.export_candidate_yaml(pg_session, out.rule_candidate_ref, str(target))

    # 原文件内容未被修改
    content_second = target.read_text(encoding="utf-8")
    assert content_first == content_second, "第二次导出失败后原文件不应被修改"
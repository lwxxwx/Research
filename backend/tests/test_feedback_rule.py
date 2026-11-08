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

from app.persistence.models import ReviewDefect, ReviewResult
from app.persistence.models import RuleCandidate as RuleCandidateORM
from app.schemas.feedback import FeedbackCreate, FeedbackType, RuleCandidate, SuggestionDiff
from app.services.feedback_service import FeedbackService
from app.services.rule_evolution_service import RuleEvolutionService


@pytest.fixture(scope="function")
def pg_session():
    """每个测试函数独立事务，测试结束rollback，不污染DB"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.config import settings
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
    # 1) proposed 状态应该包含刚创建的候选
    lst = rev.list_candidates(pg_session, status="proposed")
    ids = [x["candidate_id"] for x in lst]
    assert out.rule_candidate_ref in ids

    # 2) accepted 状态：只验证刚创建的候选不在 accepted 里（不假设全库 0 条）
    accepted = rev.list_candidates(pg_session, status="accepted")
    accepted_ids = [x["candidate_id"] for x in accepted]
    assert out.rule_candidate_ref not in accepted_ids, (
        f"刚创建的 proposed 候选不应出现在 accepted 列表，"
        f"实际 accepted 包含 {out.rule_candidate_ref}"
    )


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

def test_update_candidate_status_writes_reviewed_by(client, test_admin_user):
    """G3-C：POST /rule-candidates/{id}/status 写入 reviewed_by = 当前登录用户。

    用纯 ASCII candidate_id，避开 Sprint 2 遗留的 U+2011 特殊连字符问题。
    """
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleCandidate

    cid = f"RC-TEST-{uuid.uuid4().hex[:8].upper()}"
    with get_db_session() as db:
        db.add(RuleCandidate(
            candidate_id=cid, title="test", description="test",
            status="proposed", proposed_yaml="rule_id: TEST\n",
        ))
        db.commit()

    try:
        resp = client.post(
            f"/api/v1/rule-candidates/{cid}/status",
            json={"status": "accepted"},
        )
        assert resp.status_code == 200, resp.text

        with get_db_session() as db:
            rc = db.query(RuleCandidate).filter(RuleCandidate.candidate_id == cid).first()
            assert rc.status == "accepted"
            assert rc.reviewed_by == test_admin_user["id"]
    finally:
        with get_db_session() as db:
            db.query(RuleCandidate).filter(RuleCandidate.candidate_id == cid).delete()
            db.commit()

# ============================================================
# G3-D: rule_candidates 负面用例（R2 要求：无 header / 非法整数 / 角色 403）
# ============================================================

def test_rule_candidates_no_auth_401(client_no_auth):
    """无 X-User-Id 访问 /rule-candidates → 401。"""
    resp = client_no_auth.get("/api/v1/rule-candidates")
    assert resp.status_code == 401


def test_rule_candidates_invalid_uid_401(client_with_invalid_uid):
    """非法整数 X-User-Id 访问 /rule-candidates → 401。"""
    resp = client_with_invalid_uid.get("/api/v1/rule-candidates")
    assert resp.status_code == 401


def test_rule_candidates_engineer_403(client_as_engineer):
    """engineer 访问 /rule-candidates → 403。"""
    resp = client_as_engineer.get("/api/v1/rule-candidates")
    assert resp.status_code == 403


def test_rule_candidates_invalid_status_422(client, test_admin_user):
    """admin 传 rejected 但缺 rejected_reason → 422（Pydantic 或业务校验）。"""
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleCandidate

    cid = f"RC-DTEST-{uuid.uuid4().hex[:8]}"
    with get_db_session() as db:
        db.add(RuleCandidate(
            candidate_id=cid, title="test", description="test",
            status="proposed", proposed_yaml="rule_id: TEST\n",
        ))
        db.commit()

    try:
        # 传 status=rejected 但缺 rejected_reason → 触发 API 层 422
        resp = client.post(
            f"/api/v1/rule-candidates/{cid}/status",
            json={"status": "rejected"},   # 缺 rejected_reason
        )
        assert resp.status_code == 422
    finally:
        with get_db_session() as db:
            db.query(RuleCandidate).filter(RuleCandidate.candidate_id == cid).delete()
            db.commit()

# ============================================================
# [S3-LEGACY-B2] #5：幂等 + 非 gap 类型 ref=None
# ============================================================

def test_create_gap_from_feedback_idempotent():
    """#5：同一 feedback_id 重复调 create_gap_from_feedback → 只生成 1 条 gap（D16 幂等）。

    注意：不能走 API 连点两次（会生成两个不同 feedback_id），
    必须直接用同一 feedback_id 调 service 两次。
    """
    from app.persistence.db import get_db_session
    from app.persistence.models import FeedbackItem, GapFeedbackLink, KnowledgeGap
    from app.services import knowledge_gap_service

    with get_db_session() as db:
        fb = FeedbackItem(
            feedback_type="knowledge_gap",
            expert_suggestion="[B2-IDEMPOTENT] 幂等测试",
            suggestion_diff_json=[],
            attached_refs=[],
        )
        db.add(fb)
        db.commit()
        feedback_id = fb.id

    try:
        with get_db_session() as db:
            gap_id_1 = knowledge_gap_service.create_gap_from_feedback(db, feedback_id)
            db.commit()
        with get_db_session() as db:
            gap_id_2 = knowledge_gap_service.create_gap_from_feedback(db, feedback_id)
            db.commit()

        assert gap_id_1 is not None
        assert gap_id_2 == gap_id_1, f"幂等失败：第一次 {gap_id_1}，第二次 {gap_id_2}"

        with get_db_session() as db:
            link_count = (
                db.query(GapFeedbackLink)
                .filter(GapFeedbackLink.feedback_id == feedback_id)
                .count()
            )
            gap_count = (
                db.query(KnowledgeGap)
                .filter(KnowledgeGap.source_feedback_id == feedback_id)
                .count()
            )
        assert link_count == 1, f"gap_feedback_link 应只有 1 条，实际 {link_count}"
        assert gap_count == 1, f"knowledge_gap 应只有 1 条，实际 {gap_count}"
    finally:
        with get_db_session() as db:
            db.query(GapFeedbackLink).filter(
                GapFeedbackLink.feedback_id == feedback_id
            ).delete(synchronize_session=False)
            db.query(KnowledgeGap).filter(
                KnowledgeGap.source_feedback_id == feedback_id
            ).delete(synchronize_session=False)
            db.query(FeedbackItem).filter(FeedbackItem.id == feedback_id).delete(
                synchronize_session=False
            )
            db.commit()


def test_submit_feedback_non_gap_type_ref_none(client, test_admin_user):
    """#5：非 knowledge_gap 类型（correct_defect）→ 响应 knowledge_gap_ref 为 None。"""
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import ReviewDefect, ReviewResult

    tid = f"TEST-B2-{uuid.uuid4().hex[:8].upper()}"
    did = f"D-{uuid.uuid4().hex[:8].upper()}"     # ← defect_id 全局唯一，避免撞存量
    rr_id = None
    with get_db_session() as db:
        rr = ReviewResult(
            task_id=tid,
            task_status="done",
            created_by=test_admin_user["id"],
            review_output_json={"defects": [], "errors": []},
            evidence_coverage=0.8,
        )
        db.add(rr)
        db.flush()
        rr_id = rr.id
        rd = ReviewDefect(
            review_result_id=rr_id,
            defect_id=did,
            category="other",
            location={},
            risk="medium",
            confidence=0.9,
            root_cause="r",
            suggestion="s",
        )
        db.add(rd)
        db.commit()

    try:
        resp = client.post(
            "/api/v1/feedbacks",
            json={
                "task_id": tid,
                "defect_id": did,
                "feedback_type": "correct_defect",
                "expert_suggestion": "[B2] 非 gap 类型",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["knowledge_gap_ref"] is None
    finally:
        with get_db_session() as db:
            db.query(ReviewDefect).filter(
                ReviewDefect.review_result_id == rr_id
            ).delete(synchronize_session=False)
            db.query(ReviewResult).filter(ReviewResult.task_id == tid).delete(
                synchronize_session=False
            )
            db.commit()

# ============================================================
# [S3-LEGACY-B4] #8 rule rollback
# ============================================================

def _make_rule_with_history(version_current: str, version_target: str):
    """造一条 rule_definition + 一条 history 目标版本，返回 rule_id。"""
    import uuid
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleDefinition, RuleDefinitionHistory

    rule_id = f"RULE-B4-{uuid.uuid4().hex[:8].upper()}"
    with get_db_session() as db:
        db.add(RuleDefinition(
            rule_id=rule_id,
            rule_name="current",
            rule_category="other",
            rule_yaml="rule_id: x\nversion: " + version_current,
            severity="medium",
            enabled=True,
            version=version_current,
        ))
        db.add(RuleDefinitionHistory(
            rule_id=rule_id,
            version=version_target,
            rule_name="target",
            rule_category="other",
            rule_yaml="rule_id: x\nversion: " + version_target,
            severity="medium",
            enabled=True,
            change_source="publish",
        ))
        db.commit()
    return rule_id


def _cleanup_rule(rule_id: str) -> None:
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleDefinition, RuleDefinitionHistory
    with get_db_session() as db:
        db.query(RuleDefinitionHistory).filter(
            RuleDefinitionHistory.rule_id == rule_id
        ).delete(synchronize_session=False)
        db.query(RuleDefinition).filter(
            RuleDefinition.rule_id == rule_id
        ).delete(synchronize_session=False)
        db.commit()


def test_rollback_rule_success():
    """回滚成功：主表内容/版本变目标值，history 新增 rollback 来源一条。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleDefinition, RuleDefinitionHistory
    from app.services.rule_evolution_service import RuleEvolutionService

    rule_id = _make_rule_with_history("v2.0", "v1.0")
    try:
        svc = RuleEvolutionService()
        with get_db_session() as db:
            result = svc.rollback_rule(db, rule_id, "v1.0")
        assert result is not None
        assert result["from_version"] == "v2.0"
        assert result["to_version"] == "v1.0"
        with get_db_session() as db:
            rd = db.query(RuleDefinition).filter(RuleDefinition.rule_id == rule_id).first()
            assert rd.version == "v1.0"
            assert rd.rule_name == "target"
            rollback_rows = (
                db.query(RuleDefinitionHistory)
                .filter(
                    RuleDefinitionHistory.rule_id == rule_id,
                    RuleDefinitionHistory.change_source == "rollback",
                )
                .count()
            )
            assert rollback_rows == 1
    finally:
        _cleanup_rule(rule_id)


def test_rollback_rule_target_not_found():
    """目标版本不存在：返回 None，主表不变，history 无新增。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleDefinition, RuleDefinitionHistory
    from app.services.rule_evolution_service import RuleEvolutionService

    rule_id = _make_rule_with_history("v2.0", "v1.0")
    try:
        svc = RuleEvolutionService()
        with get_db_session() as db:
            before = db.query(RuleDefinitionHistory).filter(
                RuleDefinitionHistory.rule_id == rule_id
            ).count()
            result = svc.rollback_rule(db, rule_id, "v9.9")
        assert result is None
        with get_db_session() as db:
            rd = db.query(RuleDefinition).filter(RuleDefinition.rule_id == rule_id).first()
            assert rd.version == "v2.0"
            after = db.query(RuleDefinitionHistory).filter(
                RuleDefinitionHistory.rule_id == rule_id
            ).count()
            assert after == before
    finally:
        _cleanup_rule(rule_id)


def test_rollback_rule_dry_run_no_write():
    """dry_run=True：只校验，主表与 history 均无变化。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleDefinition, RuleDefinitionHistory
    from app.services.rule_evolution_service import RuleEvolutionService

    rule_id = _make_rule_with_history("v2.0", "v1.0")
    try:
        svc = RuleEvolutionService()
        with get_db_session() as db:
            before = db.query(RuleDefinitionHistory).filter(
                RuleDefinitionHistory.rule_id == rule_id
            ).count()
            result = svc.rollback_rule(db, rule_id, "v1.0", dry_run=True)
        assert result is not None
        assert result["dry_run"] is True
        with get_db_session() as db:
            rd = db.query(RuleDefinition).filter(RuleDefinition.rule_id == rule_id).first()
            assert rd.version == "v2.0"
            after = db.query(RuleDefinitionHistory).filter(
                RuleDefinitionHistory.rule_id == rule_id
            ).count()
            assert after == before
    finally:
        _cleanup_rule(rule_id)


def test_rollback_rule_cli_refuse_exit_1(monkeypatch):
    """CLI：目标版本不存在 → main() 返回 1。"""
    import sys
    from scripts import rollback_rule

    rule_id = _make_rule_with_history("v2.0", "v1.0")
    try:
        monkeypatch.setattr(
            sys, "argv",
            ["rollback_rule.py", "--rule-id", rule_id, "--to-version", "v9.9"],
        )
        assert rollback_rule.main() == 1
    finally:
        _cleanup_rule(rule_id)

def test_rollback_rule_idempotent():
    """连续回滚两次同一目标：第二次返回 noop=True，history 不再新增。"""
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleDefinition, RuleDefinitionHistory
    from app.services.rule_evolution_service import RuleEvolutionService

    rule_id = _make_rule_with_history("v2.0", "v1.0")
    try:
        svc = RuleEvolutionService()

        # 第一次回滚：正常执行
        with get_db_session() as db:
            first = svc.rollback_rule(db, rule_id, "v1.0")
        assert first is not None
        assert first.get("noop") is False
        assert first["from_version"] == "v2.0"
        assert first["to_version"] == "v1.0"

        # 记录第一次回滚后 history 行数
        with get_db_session() as db:
            count_after_first = db.query(RuleDefinitionHistory).filter(
                RuleDefinitionHistory.rule_id == rule_id
            ).count()

        # 第二次回滚同一目标：应命中幂等前置，返回 noop=True
        with get_db_session() as db:
            second = svc.rollback_rule(db, rule_id, "v1.0")
        assert second is not None
        assert second.get("noop") is True

        # 主表版本仍为 v1.0，history 未新增
        with get_db_session() as db:
            rd = db.query(RuleDefinition).filter(
                RuleDefinition.rule_id == rule_id
            ).first()
            assert rd.version == "v1.0"
            count_after_second = db.query(RuleDefinitionHistory).filter(
                RuleDefinitionHistory.rule_id == rule_id
            ).count()
        assert count_after_second == count_after_first, (
            f"幂等命中时 history 不应新增：第一次 {count_after_first}，第二次 {count_after_second}"
        )
    finally:
        _cleanup_rule(rule_id)
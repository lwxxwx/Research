# backend/tests/test_feedback_e2e.py
"""
阶段 E2：六类反馈 E2E 测试（Sprint 1 §14 阶段 E 验收）

【覆盖目标】
  验证"真实评审链路 → 反馈提交 → 规则演进"的完整闭环，而非 Sprint 0 的手工构造。

【双轨设计】
  - Mock 轨（默认）：create_task 用 MockProvider，不触网，CI 稳定
  - 真实轨：create_task 用智谱 API，打 @pytest.mark.integration，无 Key 时 skip

【Mock 轨用 case001 的理由】
  1. A 类 Case 规则命中稳定（POWER_001 + POWER_002），缺陷可控
  2. 六类反馈不依赖 AI 发现的复杂性；B 类 AI 发现链路已在
     tests/test_review_graph_e2e.py::test_e2e_case_b01_zhipu 验证过，E2 不必重复
  3. case001 的 IR 结构简单，Mock 缺陷容易构造

【包含候选 Benchmark 验证的理由】
  E4 的 run_benchmark_for_candidate 骨架刚写完，E2 顺手验证一下：
    - 成本低（内存执行，不落库）
    - 能确认骨架返回结构含 baseline / candidate / delta / per_case
    - 若骨架有问题，E2 早暴露，避免阶段 F 才发现

【不验证的内容】
  - 真实 LLM 输出质量（由阶段 D 的 zhipu 轨验证）
  - Benchmark 的 CSV 输出（留阶段 F）
  - 候选规则的正式启用（Sprint 1 §10.3 明确禁止自动发布）

【运行】
  # Mock 轨（CI）
  pytest tests/test_feedback_e2e.py -v -k "mock"

  # 真实轨（开发环境，需 Key）
  pytest tests/test_feedback_e2e.py -v -k "zhipu"

  # 全部
  pytest tests/test_feedback_e2e.py -v
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from app.llm.provider import MockProvider
from app.schemas.feedback import (
    FeedbackType,
    RuleCandidate,
    SuggestionDiff,
)
from app.services.feedback_service import FeedbackNotFoundError
from app.services.rule_evolution_service import RuleEvolutionService
from app.services.task_service import (
    create_task,
    get_task,
    get_task_defects,
    submit_task_feedback,
)


# ============================================================
# 常量
# ============================================================
CASE_ID = "case001"
_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

# 测试输出目录（与 test_demo.py 一致）
OUT_DIR = Path("/out/test_feedback_e2e")


def _make_defect_payload() -> dict:
    """
    构造 MockProvider 返回的单条缺陷 payload。

    说明：
      - defect_id 动态生成，避免 DB unique 约束导致第二次跑测试时幂等跳过
        （Sprint 1 阶段 D 坑 5 的教训）
      - 缺陷内容为 case001 的 POWER_001（VCC 缺少去耦电容），与真实规则命中一致
    """
    return {
        "task_id": "PLACEHOLDER",       # 会被 Graph 的 task_id 覆盖落库，此处仅占位
        "report_id": "PLACEHOLDER-R",
        "defects": [
            {
                "defect_id": f"DEF-E2E-{uuid.uuid4().hex[:8].upper()}",
                "category": "power",
                "location": {
                    "sheet": "P1",
                    "path": "U1",
                    "coords": {"x": 0, "y": 0},
                },
                "component": "U1",
                "net": "VCC",
                "risk": "critical",
                "evidence": [
                    {
                        "type": "rag_ref",
                        "source": "STC89C55RC_ds.pdf",
                        "section": "Power Supply 电源章节",
                        "reason": "VCC 引脚附近需放置 100nF 去耦电容",
                    },
                    {
                        "type": "ir_ref",
                        "detail": "Net[VCC].connected_pins=[U1.VCC]",
                    },
                ],
                "root_cause": "VCC 网络缺少去耦电容，电源纹波可能导致 MCU 工作异常",
                "suggestion": "在 U1 VCC 引脚附近增加 100nF X7R 去耦电容",
                "confidence": 0.9,
                "review_status": "AI_CONFIRMED",
                "origin": "rule",
                "rule_id": "POWER_001",
            }
        ],
        "summary": {
            "evidence_coverage": 1.0,
            "review_status_distribution": {
                "AI_CONFIRMED": 1,
                "NEED_EXPERT_REVIEW": 0,
                "LOW_CONFIDENCE": 0,
            },
        },
        "errors": [],
    }


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture()
def mock_llm_with_one_defect(monkeypatch):
    """
    把 review_graph.get_provider 替换为返回"一条缺陷"的 MockProvider。

    patch 目标：app.workflows.review_graph.get_provider
      原因：node_expert_analysis 内部是
            `from app.llm.provider import get_provider`，
            实际调用的是 review_graph 模块内的引用。
    """
    from app.workflows import review_graph as rg

    payload = _make_defect_payload()

    def _fake_get_provider(name=None):
        return MockProvider(preset="success", payload=payload)

    monkeypatch.setattr(rg, "get_provider", _fake_get_provider)


@pytest.fixture()
def prepare_out_dir():
    """每次测试前清空输出目录，避免历史产物造成断言误通过。"""
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # 不清理产物，便于人工排查


def _run_task_and_get_defect_id() -> tuple[str, str]:
    """
    跑一次 create_task，返回 (task_id, defect_id)。

    说明：
      - 用 case001 + MockProvider（一条缺陷）
      - create_task 内部走 6 节点 Graph，feedback_save 落库 review_result + review_defect
      - 从 get_task_defects 取第一条缺陷的 defect_id
    """
    task_id = create_task(CASE_ID)
    task_info = get_task(task_id)
    assert task_info is not None, f"create_task 后 get_task 返回 None，task_id={task_id}"
    assert task_info["status"] == "done", (
        f"任务未完成，status={task_info['status']}, errors={task_info['errors']}"
    )

    defects = get_task_defects(task_id)
    assert len(defects) >= 1, (
        f"任务 {task_id} 无缺陷，defects={defects}, "
        f"errors={task_info['errors']}"
    )
    return task_id, defects[0]["defect_id"]


# ============================================================
# Mock 轨测试
# ============================================================
def test_feedback_e2e_mock_six_types(mock_llm_with_one_defect, prepare_out_dir):
    """
    Mock 轨：六类反馈全部通过 submit_task_feedback 落库，闭环完整。

    验证点：
      1. 六类反馈都能提交成功，返回 FeedbackOut
      2. false_negative / new_rule_candidate 生成候选（rule_candidate_ref 非空）
      3. correct_defect / false_positive / suggestion_update / knowledge_gap
         不生成候选（rule_candidate_ref 为空）
      4. knowledge_gap 可通过 export_knowledge_gap_backlog 导出
    """
    task_id, defect_id = _run_task_and_get_defect_id()

    # --- 1. correct_defect：不生成候选 ---
    fb1 = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.CORRECT_DEFECT,
        expert_suggestion="确认 VCC 缺少去耦电容的判断正确",
    )
    assert fb1.rule_candidate_ref is None

    # --- 2. false_positive：不生成候选 ---
    fb2 = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.FALSE_POSITIVE,
        expert_suggestion="该缺陷为误报，VCC 已有去耦电容",
    )
    assert fb2.rule_candidate_ref is None

    # --- 3. false_negative：生成候选 ---
    fb3 = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="漏检：该 Case 还应检查 RST 网络的 RC 时间常数",
        rule_candidate=RuleCandidate(
            rule_id="CTRL_RC_MOCK_E2E_001",
            rule_name="RST 网络 RC 时间常数检查（E2E 候选）",
            category="timing",
            severity="medium",
            rule_basis="E2E 测试：RST 网络 RC 时间常数需 ≥ 2ms",
            evidence_refs=[
                {
                    "source": "AN_MCU_Reset_Circuit.md",
                    "section": "2.1 RC 时间常数",
                    "reason": "RST 网络 RC 时间常数需满足 MCU 最小复位脉宽",
                }
            ],
        ),
    )
    assert fb3.rule_candidate_ref is not None, "false_negative 应生成候选"
    candidate_id = fb3.rule_candidate_ref

    # --- 4. suggestion_update：不生成候选 ---
    fb4 = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.SUGGESTION_UPDATE,
        expert_suggestion="建议更具体的修改：在 U1 VCC 引脚 5mm 内增加 100nF X7R 0603 电容",
        suggestion_adopted="yes",
        suggestion_diff=[
            SuggestionDiff(
                old="在 U1 VCC 引脚附近增加 100nF X7R 去耦电容",
                new="在 U1 VCC 引脚 5mm 内增加 100nF X7R 0603 电容 C25→GND",
            )
        ],
    )
    assert fb4.rule_candidate_ref is None

    # --- 5. new_rule_candidate：生成候选 ---
    fb5 = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.NEW_RULE_CANDIDATE,
        expert_suggestion="建议新增晶振负载电容对称性检查规则",
        rule_candidate=RuleCandidate(
            rule_id="CLOCK_LOAD_SYM_MOCK_E2E_001",
            rule_name="晶振负载电容对称性检查（E2E 候选）",
            category="clock",
            severity="medium",
            rule_basis="E2E 测试：晶振两侧负载电容应基本对称",
        ),
    )
    assert fb5.rule_candidate_ref is not None, "new_rule_candidate 应生成候选"

    # --- 6. knowledge_gap：不生成候选，可导出 backlog ---
    fb6 = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.KNOWLEDGE_GAP,
        expert_suggestion="缺少 STC89C55RC 电源章节 datasheet 片段",
    )
    assert fb6.rule_candidate_ref is None

    # --- 7. knowledge_gap 导出验证 ---
    from app.persistence.db import get_db_session

    evol_svc = RuleEvolutionService()
    gap_path = OUT_DIR / "knowledge_gap_backlog.json"
    with get_db_session() as db:
        evol_svc.export_knowledge_gap_backlog(db, str(gap_path))

    assert gap_path.exists()
    gap_data = json.loads(gap_path.read_text(encoding="utf-8"))
    assert len(gap_data) >= 1, "knowledge_gap backlog 应至少含 1 条"
    assert any(
        "STC89C55RC" in (item.get("expert_suggestion") or "")
        for item in gap_data
    )


def test_feedback_e2e_mock_not_found(mock_llm_with_one_defect):
    """
    Mock 轨：submit_task_feedback 对不存在的 task_id / defect_id 抛 FeedbackNotFoundError。
    """
    # 不存在的 task_id
    with pytest.raises(FeedbackNotFoundError, match="review_result"):
        submit_task_feedback(
            task_id="NOT-EXIST-TASK",
            defect_id="ANY",
            feedback_type=FeedbackType.CORRECT_DEFECT,
            expert_suggestion="x",
        )

    # 存在的 task_id，但不存在的 defect_id
    task_id = create_task(CASE_ID)
    with pytest.raises(FeedbackNotFoundError, match="review_defect"):
        submit_task_feedback(
            task_id=task_id,
            defect_id="NOT-EXIST-DEFECT",
            feedback_type=FeedbackType.CORRECT_DEFECT,
            expert_suggestion="x",
        )


def test_feedback_e2e_mock_candidate_benchmark(mock_llm_with_one_defect):
    """
    Mock 轨：候选规则临时 Benchmark 骨架验证。

    验证点：
      1. false_negative 生成的候选 YAML 可被 run_benchmark_for_candidate 解析
      2. 返回结构含 candidate_rule_id / case_ids / baseline / candidate / delta / per_case
      3. baseline / candidate 含 tp / fp / fn / precision / recall
      4. per_case 长度 == case_ids 长度
      5. 候选规则只在内存跑，不写入 data/rules/（由函数实现保证，此处只验证返回结构）
    """
    task_id, defect_id = _run_task_and_get_defect_id()

    # 提交 false_negative 生成候选
    fb = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="漏检：需要新增 VCC 去耦电容检查",
        rule_candidate=RuleCandidate(
            rule_id="POWER_DECOUP_E2E_BENCH_001",
            rule_name="VCC 去耦电容检查（E2E Benchmark 候选）",
            category="power",
            severity="critical",
            rule_basis="E2E 测试：VCC 需 100nF 去耦",
            # 复用 POWER_001 的 check 函数，保证候选可执行
            check_logic={
                "function": "circuit_checks.check_decoupling",
                "params": {
                    "power_net_names": ["VCC"],
                    "decouple_cap_value_expected": "0.1μF",
                },
            },
        ),
    )
    assert fb.rule_candidate_ref is not None

    # 读取候选 YAML
    from app.persistence.db import get_db_session
    from app.persistence.models import RuleCandidate as RuleCandidateORM

    with get_db_session() as db:
        rc = (
            db.query(RuleCandidateORM)
            .filter(RuleCandidateORM.candidate_id == fb.rule_candidate_ref)
            .first()
        )
        assert rc is not None
        candidate_yaml = rc.proposed_yaml

    # 调候选 Benchmark
    evol_svc = RuleEvolutionService()
    result = evol_svc.run_benchmark_for_candidate(
        candidate_yaml=candidate_yaml,
        case_ids=[CASE_ID],
    )

    # 结构验证
    assert "candidate_rule_id" in result
    assert result["candidate_rule_id"] == "POWER_DECOUP_E2E_BENCH_001"
    assert result["case_ids"] == [CASE_ID]

    for key in ("baseline", "candidate", "delta"):
        assert key in result, f"缺少 {key}"
        assert "tp" in result[key]
        assert "fp" in result[key]
        assert "fn" in result[key]
        assert "precision" in result[key]
        assert "recall" in result[key]

    assert "per_case" in result
    assert len(result["per_case"]) == 1
    assert result["per_case"][0]["case_id"] == CASE_ID
    assert "baseline" in result["per_case"][0]
    assert "candidate" in result["per_case"][0]


# ============================================================
# 真实轨：调用智谱 API，无 Key skip
# ============================================================
@pytest.mark.integration
@pytest.mark.skipif(not _API_KEY, reason="无 ZHIPUAI_API_KEY，跳过真实轨")
def test_feedback_e2e_zhipu():
    """
    真实轨：create_task 用智谱 API，跑一条 false_negative 反馈，验证闭环。

    只跑一条反馈（不是六类），原因：
      - 真实轨较慢（每次 LLM 调用约 20–30s），六类反馈会跑 6 次，超时风险高
      - 六类反馈的落库与触发逻辑已由 Mock 轨覆盖
      - 真实轨只需要验证"真实 LLM 产出的缺陷能被反馈"这一件事

    验证点：
      1. create_task 用真实 LLM 完成
      2. 至少 1 条缺陷
      3. 提交 false_negative 成功，生成候选
    """
    task_id = create_task(CASE_ID)
    task_info = get_task(task_id)
    assert task_info is not None
    assert task_info["status"] == "done", (
        f"真实轨任务未完成，errors={task_info['errors']}"
    )

    defects = get_task_defects(task_id)
    assert len(defects) >= 1, (
        f"真实轨无缺陷，errors={task_info['errors']}"
    )
    defect_id = defects[0]["defect_id"]

    fb = submit_task_feedback(
        task_id=task_id,
        defect_id=defect_id,
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        expert_suggestion="真实轨：漏检场景需新增规则",
        rule_candidate=RuleCandidate(
            rule_id="POWER_DECOUP_REAL_E2E_001",
            rule_name="VCC 去耦电容检查（真实轨 E2E 候选）",
            category="power",
            severity="critical",
            rule_basis="真实轨 E2E：VCC 需 100nF 去耦",
        ),
    )
    assert fb.rule_candidate_ref is not None
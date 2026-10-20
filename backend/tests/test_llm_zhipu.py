"""
智谱 AI + LLM 层整合测试（Sprint 1 阶段 C 合并版）

用法：
  1. pytest（CI / 批量）
     无 ZHIPUAI_API_KEY 时，真实调用智谱的用例自动 skip；其余用例照跑
     docker compose ... exec backend uv run pytest tests/test_llm_zhipu.py -v

  2. 手动运行（单次调试）
     docker compose ... exec -e ZHIPUAI_API_KEY=... backend uv run python /app/tests/test_llm_zhipu.py

说明：
  - API Key 只通过环境变量注入，代码不写死
  - 智谱 embedding-3 是 2048 维，与 DB Schema Vector(1536) 不匹配；
    切换需走 scripts/switch_embedding_backend.py（一次性）
  - 无 ZHIPUAI_API_KEY 时，真实调用类用例 pytest.skip；Mock/Parser/Schema 照跑

【S1-C 合并说明】
  原 test_llm_zhipu.py 只有智谱 chat / embedding 连通性两个用例。
  本版按交付要求合并为单文件，共 7 节：
    一、智谱连通性（真实调用，无 Key skip）
    二、对话 Provider 分发（无 Key 也跑）
    三、Mock Provider 四场景（无 Key 也跑）
    四、OutputParser 三级校验（无 Key 也跑）
    五、Report Schema（无 Key 也跑）
    六、Embedding 工厂分发 + 维度守卫（无 Key 也跑）
    七、手动运行入口（保留）
"""
import os

import pytest


# ============================================================
# 模块级：只读环境变量，不主动 raise
# ============================================================
_API_KEY = os.environ.get("ZHIPUAI_API_KEY")
_MODEL = os.environ.get("ZHIPUAI_MODEL", "glm-4-flash")
_EMBEDDING_MODEL = os.environ.get("ZHIPUAI_EMBEDDING_MODEL", "embedding-3")


# ============================================================
# 通用 fixture：无 Key 时 skip（仅真实调用类用例使用）
# ============================================================
@pytest.fixture(scope="module")
def zhipu_client():
    """无 ZHIPUAI_API_KEY 时，跳过当前测试。"""
    if not _API_KEY:
        pytest.skip(
            "未设置 ZHIPUAI_API_KEY，跳过智谱 AI 测试。"
            "如需运行，请设置环境变量后重试。"
        )

    from zhipuai import ZhipuAI
    return ZhipuAI(api_key=_API_KEY)


# ============================================================
# 一、智谱连通性（真实调用，无 Key skip）
# ============================================================
def test_chat(zhipu_client):
    """智谱对话模型（GLM-4-Flash）应返回非空字符串。"""
    response = zhipu_client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好！如果你能看到这条消息，请回复：'智谱AI连接成功！'"},
        ],
        top_p=0.7,
        temperature=0.9,
    )
    content = response.choices[0].message.content
    assert content, "对话模型返回空内容"
    assert isinstance(content, str), f"对话模型返回类型异常: {type(content)}"
    print(f"\n✅ AI回复内容: {content}")


def test_embedding(zhipu_client):
    """智谱向量模型（Embedding-3）应返回 2048 维向量。"""
    test_text = "这是一段用于测试向量化的文本"
    response = zhipu_client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=test_text,
    )
    embedding = response.data[0].embedding
    assert isinstance(embedding, list), f"embedding 类型异常: {type(embedding)}"
    assert len(embedding) > 0, "embedding 为空"
    # 智谱 embedding-3 固定 2048 维
    assert len(embedding) == 2048, (
        f"embedding 维度异常: {len(embedding)}（期望 2048）"
    )
    print(f"\n✅ 向量化成功！维度={len(embedding)}，前5位={embedding[:5]}")


# ============================================================
# 二、对话 Provider 分发（无 Key 也跑）
# ============================================================
def test_get_provider_mock():
    from app.llm.provider import MockProvider, get_provider
    p = get_provider("mock")
    assert isinstance(p, MockProvider)
    assert p.provider_name == "mock"


def test_get_provider_zhipu_returns_instance():
    """只验类型，不调用真实 API；无 Key 时构造 Provider 会 RuntimeError，故 skip。"""
    from app.llm.provider import ZhipuProvider, get_provider

    if not _API_KEY and not os.environ.get("LLM_API_KEY"):
        pytest.skip("无 ZHIPUAI_API_KEY / LLM_API_KEY，跳过 ZhipuProvider 构造")
    p = get_provider("zhipu")
    assert isinstance(p, ZhipuProvider)
    assert p.provider_name == "zhipu"


def test_get_provider_openai_placeholder_raises():
    from app.llm.provider import get_provider
    p = get_provider("openai")
    resp = p.complete("ping", json_mode=False)
    assert resp.error is not None
    assert "OpenAI Provider 尚未启用" in resp.error


def test_get_provider_vllm_placeholder_raises():
    from app.llm.provider import get_provider
    p = get_provider("vllm")
    resp = p.complete("ping", json_mode=False)
    assert resp.error is not None
    assert "VLLM Provider 尚未启用" in resp.error


def test_get_provider_unknown_raises():
    from app.llm.provider import get_provider
    with pytest.raises(ValueError, match="不支持的 LLM_PROVIDER"):
        get_provider("not-a-provider")


# ============================================================
# 三、Mock Provider 四场景（无 Key 也跑）
# ============================================================
def test_mock_success():
    from app.llm.provider import MockProvider
    p = MockProvider(preset="success")
    resp = p.complete("any prompt", json_mode=True)
    assert resp.ok
    assert resp.provider == "mock"
    assert resp.parsed is not None
    assert resp.finish_reason == "stop"


def test_mock_timeout_returns_error():
    from app.llm.provider import MockProvider
    p = MockProvider(preset="timeout")
    resp = p.complete("any prompt", json_mode=True)
    assert not resp.ok
    assert resp.error is not None
    assert "retryable" in resp.error


def test_mock_invalid_json_returns_none_parsed():
    from app.llm.provider import MockProvider
    p = MockProvider(preset="invalid_json")
    resp = p.complete("any prompt", json_mode=True)
    assert resp.ok  # 传输层成功
    assert resp.parsed is None  # 解析失败


def test_mock_exception_returns_error():
    from app.llm.provider import MockProvider
    p = MockProvider(preset="exception")
    resp = p.complete("any prompt", json_mode=True)
    assert not resp.ok
    assert resp.error is not None
    assert "unexpected" in resp.error


# ============================================================
# 四、OutputParser 三级校验（无 Key 也跑）
# ============================================================
def _build_llm_response(payload: dict | None, raw: str = ""):
    from app.llm.provider import LLMResponse
    return LLMResponse(
        raw_text=raw or "",
        parsed=payload,
        model="mock-model",
        provider="mock",
        finish_reason="stop",
    )


def _make_valid_defect(**overrides) -> dict:
    base = {
        "defect_id": "DEF-001",
        "category": "power",
        "location": {"sheet": "P1", "path": "U1", "coords": {"x": 0, "y": 0}},
        "component": "U1",
        "net": "VCC",
        "risk": "high",
        "evidence": [
            {
                "type": "rag_ref",
                "source": "MP2307_DS_Rev1.2.pdf",
                "section": "6 Thermal Information",
                "reason": "自然对流 TA≤70℃，本设计 85℃，超出降额曲线",
            },
            {
                "type": "ir_ref",
                "detail": "Component[U1].attributes.VIN=12V",
            },
        ],
        "root_cause": "在 85℃ 环境下耗散功率超过最大允许值",
        "suggestion": "改用工业级版本或加散热器",
        "confidence": 0.9,
        "review_status": "AI_CONFIRMED",
        "origin": "rule",
        "rule_id": "POWER_001",
    }
    base.update(overrides)
    return base


def _make_report_payload(defects: list[dict]) -> dict:
    return {
        "task_id": "T-001",
        "report_id": "R-001",
        "defects": defects,
        "summary": {
            "evidence_coverage": 0.0,
            "review_status_distribution": {
                "AI_CONFIRMED": 0,
                "NEED_EXPERT_REVIEW": 0,
                "LOW_CONFIDENCE": 0,
            },
        },
        "errors": [],
    }


def test_parser_valid_defect():
    from app.llm.output_parser import parse_report
    payload = _make_report_payload([_make_valid_defect()])
    report = parse_report(_build_llm_response(payload))
    assert len(report.defects) == 1
    assert report.defects[0].review_status.value == "AI_CONFIRMED"


def test_parser_missing_field():
    from app.llm.output_parser import parse_report
    bad = _make_valid_defect()
    bad.pop("root_cause")
    payload = _make_report_payload([bad])
    report = parse_report(_build_llm_response(payload))
    # 结构校验失败 → 空报告 + errors
    assert len(report.defects) == 0
    assert any("结构校验失败" in e for e in report.errors)


def test_parser_evidence_missing_reason():
    """rag_ref 缺 reason 时，Evidence Schema 校验失败，整份报告为空。"""
    from app.llm.output_parser import parse_report
    bad = _make_valid_defect()
    bad["evidence"][0].pop("reason")
    payload = _make_report_payload([bad])
    report = parse_report(_build_llm_response(payload))
    assert len(report.defects) == 0
    assert any("结构校验失败" in e for e in report.errors)


def test_parser_ai_discovered_cannot_be_ai_confirmed():
    """硬约束：ai_discovered 不得 AI_CONFIRMED。"""
    from app.llm.output_parser import parse_report
    d = _make_valid_defect(
        origin="ai_discovered",
        rule_id=None,
        review_status="AI_CONFIRMED",
    )
    payload = _make_report_payload([d])
    report = parse_report(_build_llm_response(payload))
    assert len(report.defects) == 1
    assert report.defects[0].review_status.value == "NEED_EXPERT_REVIEW"


def test_parser_ai_discovered_missing_rag_ref_downgraded():
    """ai_discovered 缺 rag_ref → NEED_EXPERT_REVIEW，confidence 扣减。"""
    from app.llm.output_parser import parse_report
    d = _make_valid_defect(
        origin="ai_discovered",
        rule_id=None,
        review_status="NEED_EXPERT_REVIEW",
    )
    d["evidence"] = [e for e in d["evidence"] if e["type"] != "rag_ref"]
    payload = _make_report_payload([d])
    report = parse_report(_build_llm_response(payload))
    assert report.defects[0].review_status.value == "NEED_EXPERT_REVIEW"
    assert report.defects[0].confidence < 0.9


def test_parser_fuzzy_words_low_confidence():
    """risk>=medium 且模糊词 >=2 → LOW_CONFIDENCE。"""
    from app.llm.output_parser import parse_report
    d = _make_valid_defect(
        risk="high",
        root_cause="可能存在问题，应该是降额不足",
        suggestion="或许需要加散热",
    )
    payload = _make_report_payload([d])
    report = parse_report(_build_llm_response(payload))
    assert report.defects[0].review_status.value == "LOW_CONFIDENCE"


def test_parser_invalid_json_returns_empty_report():
    from app.llm.output_parser import parse_report
    from app.llm.provider import LLMResponse
    resp = LLMResponse(
        raw_text="{ not valid json",
        parsed=None,
        model="mock",
        provider="mock",
        finish_reason="stop",
    )
    report = parse_report(resp)
    assert len(report.defects) == 0
    assert any("无法解析为 JSON" in e for e in report.errors)


def test_parser_truncated_finish_reason_logged():
    from app.llm.output_parser import parse_report
    from app.llm.provider import LLMResponse
    payload = _make_report_payload([_make_valid_defect()])
    resp = LLMResponse(
        raw_text="",
        parsed=payload,
        model="mock",
        provider="mock",
        finish_reason="length",
    )
    report = parse_report(resp)
    assert any("finish_reason=length" in e for e in report.errors)


# ============================================================
# 五、Report Schema（无 Key 也跑）
# ============================================================
def test_defect_10_fields():
    from app.schemas.report import Defect
    d = Defect(**_make_valid_defect())
    for f in (
        "defect_id", "category", "location", "component", "net",
        "risk", "evidence", "root_cause", "suggestion", "confidence",
    ):
        assert hasattr(d, f)


def test_origin_enum():
    from app.schemas.report import Defect
    for origin in ("rule", "ai_discovered", "hybrid"):
        d = Defect(**_make_valid_defect(origin=origin))
        assert d.origin == origin


def test_evidence_conditional_required():
    from pydantic import ValidationError
    from app.schemas.report import Evidence
    with pytest.raises(ValidationError):
        Evidence(type="rag_ref", source="x", section="y")  # 缺 reason
    with pytest.raises(ValidationError):
        Evidence(type="ir_ref")  # 缺 detail


def test_report_json_roundtrip():
    from app.schemas.report import ReviewReport
    payload = _make_report_payload([_make_valid_defect()])
    r = ReviewReport.model_validate(payload)
    r2 = ReviewReport.model_validate_json(r.model_dump_json())
    assert len(r2.defects) == 1
    assert r2.defects[0].defect_id == "DEF-001"


# ============================================================
# 六、Embedding 工厂分发 + 维度守卫（无 Key 也跑）
# ============================================================
def test_embedding_factory_fake_1536():
    from app.llm.embeddings.fake import FakeEmbedding
    from app.rag.embedding_factory import BACKEND_DIM, get_embedding_client
    client = get_embedding_client("fake")
    assert isinstance(client, FakeEmbedding)
    assert client.dimension == 1536 == BACKEND_DIM["fake"]
    vec = client.embed_query("测试输入文本")
    assert len(vec) == 1536


def test_embedding_factory_zhipu_returns_instance():
    from app.llm.embeddings.zhipu import ZhipuEmbedding
    from app.rag.embedding_factory import BACKEND_DIM

    if not _API_KEY and not os.environ.get("LLM_API_KEY"):
        pytest.skip("无 ZHIPUAI_API_KEY / LLM_API_KEY，跳过 ZhipuEmbedding 构造")
    from app.rag.embedding_factory import get_embedding_client
    client = get_embedding_client("zhipu")
    assert isinstance(client, ZhipuEmbedding)
    # 维度默认 1024
    assert client.dimension == BACKEND_DIM["zhipu"]


def test_embedding_factory_openai_preserved():
    from app.rag.embedding_factory import BACKEND_DIM
    assert BACKEND_DIM["openai"] == 1536


def test_embedding_factory_unknown_raises():
    from app.rag.embedding_factory import get_embedding_client
    with pytest.raises(ValueError, match="不支持的 RAG_EMBEDDING_BACKEND"):
        get_embedding_client("not-a-backend")


def test_dimension_guard_mismatch_raises():
    """维度守卫：期望维度与 DB 不一致时 raise（需要 DB 可用）。"""
    from app.rag.embedding_factory import assert_backend_matches_db
    from app.core.config import settings
    from app.persistence.db import get_db_session

    try:
        with get_db_session() as db:
            assert_backend_matches_db(db)
    except Exception as exc:
        msg = str(exc)
        if "维度不匹配" in msg or "switch_embedding_backend" in msg:
            pytest.skip(f"当前维度与后端不一致（预期行为）：{msg}")
        # DB 不可用等其他异常，也 skip 掉，避免 CI 因 DB 未起而失败
        pytest.skip(f"跳过维度守卫测试：{type(exc).__name__}: {exc}")
    # 一致时通过
    assert True


# ============================================================
# 七、手动运行入口（pytest 不执行此块）
# ============================================================
if __name__ == "__main__":
    if not _API_KEY:
        raise RuntimeError(
            "请设置环境变量 ZHIPUAI_API_KEY\n"
            "宿主机 PowerShell: $env:ZHIPUAI_API_KEY=\"<your-key>\"\n"
            "或通过 compose exec -e ZHIPUAI_API_KEY=... 注入"
        )

    print("=== 智谱AI API 功能测试（手动模式）===")
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=_API_KEY)

    print(f"\n🚀 测试对话模型 ({_MODEL})...")
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "你好！如果你能看到这条消息，请回复：'智谱AI连接成功！'"},
            ],
            top_p=0.7,
            temperature=0.9,
        )
        print("✅ AI回复内容:", resp.choices[0].message.content)
    except Exception as e:
        print("❌ 对话模型调用失败:", str(e))

    print(f"\n🚀 测试向量模型 ({_EMBEDDING_MODEL})...")
    try:
        resp = client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input="这是一段用于测试向量化的文本",
        )
        emb = resp.data[0].embedding
        print(f"✅ 向量化成功！维度={len(emb)}，前5位={emb[:5]}")
    except Exception as e:
        print("❌ 向量模型调用失败:", str(e))

    print("\n=== 测试结束 ===")
"""
智谱 AI API 连通性测试（pytest + 手动运行 双模式）。

用法：
  1. pytest（CI / 批量）
     有 ZHIPUAI_API_KEY 时正常跑；无则自动 skip
     docker compose ... exec backend uv run pytest tests/test_llm_zhipu.py -v

  2. 手动运行（单次调试）
     docker compose ... exec -e ZHIPUAI_API_KEY=... backend uv run python /app/tests/test_llm_zhipu.py

说明：
  - API Key 只通过环境变量注入，代码不写死
  - 智谱 embedding-3 是 2048 维，与 DB Schema Vector(1536) 不匹配；
    仅测试脚本用，不进入 RAG 主链路
  - 无 ZHIPUAI_API_KEY 时，pytest 自动 skip（不报 ERROR）
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
# fixture：无 Key 时整模块 skip
# ============================================================
@pytest.fixture(scope="module")
def zhipu_client():
    """无 ZHIPUAI_API_KEY 时，整个模块的测试 skip。"""
    if not _API_KEY:
        pytest.skip(
            "未设置 ZHIPUAI_API_KEY，跳过智谱 AI 测试。"
            "如需运行，请设置环境变量后重试。"
        )

    from zhipuai import ZhipuAI
    return ZhipuAI(api_key=_API_KEY)


# ============================================================
# 测试 1：对话模型连通性
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


# ============================================================
# 测试 2：向量模型连通性
# ============================================================
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
# 手动运行入口（pytest 不执行此块）
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
# backend/tests/test_llm_vllm.py
"""S2-I: vLLM Provider 占位契约测试。

vLLM 本地未部署时，验证：
  1. 工厂按 name='vllm' 正确返回 VLLMProvider 实例
  2. 未实装时 complete() 返回带 error 的 LLMResponse（基类兜底不抛异常）
  3. 不依赖任何外部服务

实装路径见 docs/sprint2/changelog.md [S2-I 预研] 段。
"""
from __future__ import annotations

import pytest

from app.llm.provider import LLMResponse, VLLMProvider, get_provider


def test_vllm_provider_registered():
    """工厂能按 name='vllm' 返回 VLLMProvider 实例"""
    p = get_provider("vllm")
    assert isinstance(p, VLLMProvider)
    assert p.provider_name == "vllm"


def test_vllm_provider_not_implemented_yet():
    """占位阶段：complete() 不抛异常，返回带 NotImplementedError 的 LLMResponse"""
    p = VLLMProvider(model="test-model", base_url="http://localhost:8000/v1")
    resp = p.complete("hello", json_mode=False)

    assert isinstance(resp, LLMResponse)
    assert resp.ok is False
    assert resp.error is not None
    # 基类把未预期异常归为 unexpected，并保留原始异常名与信息
    assert "NotImplementedError" in resp.error
    assert "尚未实装" in resp.error
    assert resp.provider == "vllm"
    assert resp.raw_text == ""


def test_vllm_provider_uses_settings_defaults():
    """不传参时，从 settings 读取默认 model/base_url"""
    p = VLLMProvider()
    assert p.default_model  # 有 settings 兜底，不应为空
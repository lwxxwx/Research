# backend/app/llm/provider.py
"""
LLM Provider 抽象与实现（Sprint 1 阶段 C 新增）

职责：
  - 统一对话补全接口 LLMProvider.complete()
  - LLMResponse 承载「原始返回 + 解析结果 + 元信息 + 错误」
  - 工厂 get_provider() 按 settings.llm_provider 分发
  - 主用 zhipu；mock 用于 CI/单测；openai / vllm 占位

【边界】
  - 本模块只做「对话补全」，不涉及 embedding（embedding 见 app/llm/embeddings/）
  - 质量判定（review_status / Evidence 三要素 / 模糊词）全部在 output_parser.py，
    LLMResponse 不承担业务质量判定

【重试策略】（Sprint 1 阶段 C 冻结）
  - 单次超时 30s
  - 网络类错误重试 1 次（LLM_MAX_RETRIES=1）
  - 4xx（鉴权/参数/额度）不重试，直接返回 error

【LLMResponse 字段】
  - raw_text      模型原始返回字符串
  - parsed        json_mode 且解析成功时的 dict；失败为 None
  - model         实际调用的模型名
  - provider      后端标识（mock/zhipu/openai/vllm）
  - latency_ms    耗时（毫秒）
  - error         结构化错误描述；成功为 None
  - finish_reason 厂商返回的结束原因；截断（length）时 Parser 直接降级
  - request_id    厂商请求 ID（可选，审计/排障用）

参考：
  - tests/test_llm_zhipu.py（智谱 SDK 用法与无 Key skip 约定）
  - app/core/config.py（llm_provider / zhipuai_api_key / zhipuai_model / llm_api_key）
  - Sprint1_融合版开发落地规划 v1.0 §7
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# ===== [S1-C 新增] 常量 =====
# ============================================================
LLM_TIMEOUT_SECONDS: int = 30
LLM_MAX_RETRIES: int = 1          # 仅网络类错误重试；4xx 不重试


# ============================================================
# ===== [S1-C 新增] LLMResponse =====
# ------------------------------------------------------------
# 字段语义见模块 docstring；本结构不承担质量判定
# ============================================================
@dataclass
class LLMResponse:
    raw_text: str
    parsed: dict[str, Any] | None = None
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    finish_reason: str | None = None
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        """是否成功拿到可用返回（无 error）"""
        return self.error is None


# ============================================================
# ===== [S1-C 新增] LLMProvider 抽象基类 =====
# ============================================================
class LLMProvider(ABC):
    """
    统一对话补全接口。

    实现类必须提供：
      - provider_name: 后端标识（与 settings.llm_provider 对应）
      - default_model: 默认模型名（可被 settings 覆盖）
      - _do_complete(prompt, *, json_mode) -> LLMResponse
    基类负责计时、重试、异常兜底、统一日志。
    """

    provider_name: str = "base"
    default_model: str = ""

    def complete(self, prompt: str, *, json_mode: bool = True) -> LLMResponse:
        """
        统一入口：带计时、重试、异常兜底的对话补全。

        重试策略：
          - 网络类异常（Timeout / ConnectionError 等）：重试 LLM_MAX_RETRIES 次
          - 4xx 类错误（实现类通过 raise _NonRetryableError 表达）：不重试
          - 其他异常：不重试，直接返回 error
        """
        start = time.perf_counter()
        attempts = LLM_MAX_RETRIES + 1
        last_error: str | None = None

        for attempt in range(1, attempts + 1):
            try:
                resp = self._do_complete(prompt, json_mode=json_mode)
                resp.latency_ms = (time.perf_counter() - start) * 1000.0
                if resp.provider == "":
                    resp.provider = self.provider_name
                if resp.model == "":
                    resp.model = self.default_model
                logger.info(
                    "llm.complete ok provider=%s model=%s attempt=%d latency_ms=%.1f",
                    resp.provider, resp.model, attempt, resp.latency_ms,
                )
                return resp

            except _NonRetryableError as exc:
                # 4xx：不重试，立即返回
                last_error = f"non-retryable: {exc}"
                logger.warning("llm.complete non-retryable provider=%s err=%s",
                               self.provider_name, exc)
                break

            except _RetryableError as exc:
                last_error = f"retryable: {exc}"
                logger.warning(
                    "llm.complete retryable provider=%s attempt=%d/%d err=%s",
                    self.provider_name, attempt, attempts, exc,
                )
                if attempt < attempts:
                    continue
                break

            except Exception as exc:  # noqa: BLE001  # 兜底：任何未预期异常
                last_error = f"unexpected: {type(exc).__name__}: {exc}"
                logger.error(
                    "llm.complete unexpected provider=%s err=%s",
                    self.provider_name, last_error, exc_info=True,
                )
                break

        latency_ms = (time.perf_counter() - start) * 1000.0
        return LLMResponse(
            raw_text="",
            parsed=None,
            model=self.default_model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            error=last_error or "unknown error",
            finish_reason=None,
            request_id=None,
        )

    @abstractmethod
    def _do_complete(self, prompt: str, *, json_mode: bool) -> LLMResponse:
        """子类实现：实际调用厂商 API。

        失败时抛：
          - _RetryableError    网络/超时类错误（基类会重试）
          - _NonRetryableError 4xx 类错误（基类不重试）
          - 其他异常           基类不重试，直接 error
        """
        raise NotImplementedError


# ============================================================
# ===== [S1-C 新增] 内部异常：区分可重试 / 不可重试 =====
# ============================================================
class _RetryableError(Exception):
    """网络/超时类错误，基类会重试"""


class _NonRetryableError(Exception):
    """4xx 类错误（鉴权/参数/额度），基类不重试"""


# ============================================================
# ===== [S1-C 新增] 工具：json_mode 下尝试解析 =====
# ============================================================
def _try_parse_json(raw_text: str) -> dict[str, Any] | None:
    """
    json_mode 下尝试解析。失败返回 None，由 Parser 决定降级策略；
    provider 不在此处静默失败，只如实透传。
    """
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ============================================================
# ===== [S1-C 新增] MockProvider =====
# ------------------------------------------------------------
# CI / 单测默认。四场景通过 MockProvider 的 preset 控制：
#   - "success"       正常返回 JSON
#   - "timeout"       抛 _RetryableError（模拟超时）
#   - "invalid_json"  返回非法 JSON（模拟格式错误）
#   - "exception"     抛普通 Exception（模拟未预期异常）
# ============================================================
class MockProvider(LLMProvider):
    provider_name = "mock"
    default_model = "mock-model"

    def __init__(
        self,
        preset: str = "success",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        :param preset: success / timeout / invalid_json / exception
        :param payload: success 场景下返回的 dict；None 时返回最小合法报告
        """
        self.preset = preset
        self.payload = payload if payload is not None else _default_mock_payload()

    def _do_complete(self, prompt: str, *, json_mode: bool) -> LLMResponse:
        if self.preset == "timeout":
            raise _RetryableError("mock timeout")
        if self.preset == "exception":
            raise RuntimeError("mock unexpected exception")
        if self.preset == "invalid_json":
            raw = "{ this is not valid json"
            return LLMResponse(
                raw_text=raw,
                parsed=_try_parse_json(raw) if json_mode else None,
                model=self.default_model,
                provider=self.provider_name,
                finish_reason="stop",
            )
        # success
        raw = json.dumps(self.payload, ensure_ascii=False)
        return LLMResponse(
            raw_text=raw,
            parsed=_try_parse_json(raw) if json_mode else None,
            model=self.default_model,
            provider=self.provider_name,
            finish_reason="stop",
        )


def _default_mock_payload() -> dict[str, Any]:
    """
    Mock 成功场景的最小合法报告，结构对齐 ReviewReport。
    仅用于单测/CI，不进入业务数据。
    """
    return {
        "task_id": "MOCK-TASK-001",
        "report_id": "MOCK-REPORT-001",
        "defects": [],
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


# ============================================================
# ===== [S1-C 新增] ZhipuProvider（主用）=====
# ------------------------------------------------------------
# 参考 tests/test_llm_zhipu.py：
#   - SDK: zhipuai.ZhipuAI(api_key=...)
#   - chat.completions.create(model=..., messages=[...], ...)
#   - 模型默认 glm-4-flash（settings.zhipuai_model 可覆盖）
# Key 取值：settings.zhipuai_api_key or settings.llm_api_key
# ============================================================
class ZhipuProvider(LLMProvider):
    provider_name = "zhipu"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.zhipuai_api_key or settings.llm_api_key
        self.default_model = model or settings.zhipuai_model
        if not self.api_key:
            raise RuntimeError(
                "ZhipuProvider 需要配置 ZHIPUAI_API_KEY 或 LLM_API_KEY"
            )

    def _do_complete(self, prompt: str, *, json_mode: bool) -> LLMResponse:
        # 延迟导入：无 Key 的单测不会触发 SDK import
        try:
            from zhipuai import ZhipuAI
        except ImportError as exc:
            raise _NonRetryableError(f"zhipuai SDK 未安装: {exc}") from exc

        client = ZhipuAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": "你是一名严谨的硬件电路评审专家。"},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.default_model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
        }
        if json_mode:
            # 智谱 GLM 支持 response_format={"type":"json_object"}（部分版本）
            # 不支持的版本会在客户端被忽略或报错，下面的 _classify 会判定
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _classify_zhipu_exception(exc) from exc

        raw_text = _extract_zhipu_text(resp)
        finish_reason = _extract_zhipu_finish_reason(resp)
        request_id = getattr(resp, "id", None)

        return LLMResponse(
            raw_text=raw_text,
            parsed=_try_parse_json(raw_text) if json_mode else None,
            model=self.default_model,
            provider=self.provider_name,
            finish_reason=finish_reason,
            request_id=request_id,
        )


def _extract_zhipu_text(resp: Any) -> str:
    """从智谱返回中提取 message.content；结构异常时返回空串。"""
    try:
        choices = getattr(resp, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def _extract_zhipu_finish_reason(resp: Any) -> str | None:
    try:
        choices = getattr(resp, "choices", None)
        if not choices:
            return None
        reason = getattr(choices[0], "finish_reason", None)
        return reason if isinstance(reason, str) else None
    except Exception:  # noqa: BLE001
        return None


def _classify_zhipu_exception(exc: Exception) -> Exception:
    """
    将智谱/httpx 异常归类为可重试 / 不可重试。
    无法判定的归为普通 Exception（基类不重试，直接 error）。
    """
    # 网络/超时类
    try:
        import httpx  # 智谱 SDK 依赖 httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return _RetryableError(f"network: {exc}")
    except ImportError:
        pass

    # 4xx / 鉴权类
    name = type(exc).__name__
    if "Authentication" in name or "Permission" in name:
        return _NonRetryableError(f"auth: {exc}")
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if 400 <= status < 500:
            return _NonRetryableError(f"http {status}: {exc}")
        if 500 <= status < 600:
            return _RetryableError(f"http {status}: {exc}")

    # 默认：不重试
    return exc


# ============================================================
# ===== [S1-C 新增] OpenAIProvider（占位）=====
# ------------------------------------------------------------
# Sprint 1 主用 zhipu；OpenAI 路径保留为后续可选。
# 调用 complete() 时抛 NotImplementedError 并给清晰提示。
# ============================================================
class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.default_model = model or settings.llm_model or "gpt-4o-mini"

    def _do_complete(self, prompt: str, *, json_mode: bool) -> LLMResponse:
        raise NotImplementedError(
            "OpenAI Provider 尚未启用，Sprint 1 主用 zhipu；"
            "如需启用，在 ZhipuProvider 基础上改为 OpenAI 兼容调用"
        )


# ============================================================
# ===== [S1-C 新增] VLLMProvider（占位）=====
# ------------------------------------------------------------
# Sprint 1 不部署 vLLM；只保留占位，配置 base_url 时供后续切换。
# ============================================================
class VLLMProvider(LLMProvider):
    provider_name = "vllm"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.default_model = model or settings.llm_model or "vllm-model"
        self.base_url = base_url or settings.llm_base_url

    def _do_complete(self, prompt: str, *, json_mode: bool) -> LLMResponse:
        raise NotImplementedError(
            "VLLM Provider 尚未启用，Sprint 1 不部署 vLLM；"
            "如需启用，按 OpenAI-compatible base_url 实现"
        )


# ============================================================
# ===== [S1-C 新增] 工厂 get_provider() =====
# ------------------------------------------------------------
# 按 settings.llm_provider 分发：
#   mock   → MockProvider（CI 默认）
#   zhipu  → ZhipuProvider（开发主用）
#   openai → OpenAIProvider（占位）
#   vllm   → VLLMProvider（占位）
# 未知值 → 抛 ValueError，给出可选值
# ============================================================
_PROVIDERS = {
    "mock": MockProvider,
    "zhipu": ZhipuProvider,
    "openai": OpenAIProvider,
    "vllm": VLLMProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """
    根据 settings.llm_provider（或显式传入 name）返回 Provider 实例。
    """
    key = (name or settings.llm_provider or "mock").lower().strip()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(
            f"不支持的 LLM_PROVIDER={key}，可选值: {sorted(_PROVIDERS.keys())}"
        )
    return cls()


# ============================================================
# ===== [S1-C 新增] __all__ =====
# ============================================================
__all__ = [
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLMResponse",
    "LLMProvider",
    "MockProvider",
    "ZhipuProvider",
    "OpenAIProvider",
    "VLLMProvider",
    "get_provider",
]
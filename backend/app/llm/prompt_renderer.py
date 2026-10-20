# backend/app/llm/prompt_renderer.py
"""
Prompt 渲染工具（Sprint 1 阶段 C 新增）

【为什么不用 str.format()】
  expert_analysis.md 里包含大量 JSON 示例的字面花括号（如 {"task_id": ...}）。
  str.format() 会把它们误判为占位符，导致 KeyError/ValueError。
  本模块用 str.replace() 逐个替换已知占位符，天然不碰字面花括号。

【占位符命名】
  与 expert_analysis.md 保持一致：
    {task_id} / {report_id} / {ir_facts} / {semantic_ctx}
    {rule_hits} / {rules_passed} / {rag_evidence}

【HTML 注释处理】
  expert_analysis.md 文件头的 <!-- --> 元信息默认保留（LLM 读到无害）。
  若调用方希望剥离，传 strip_html_comments=True。
"""
from __future__ import annotations

import re

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def render_prompt(
    template: str,
    *,
    strip_html_comments: bool = False,
    **vars: object,
) -> str:
    """
    用 str.replace 逐个替换 {key} 占位符。

    :param template: Prompt 模板原文（expert_analysis.md 内容）
    :param strip_html_comments: True 时先剥离 <!-- --> 块
    :param vars: 占位符键值对，键名不含花括号
    :return: 渲染后的 Prompt 文本
    """
    if strip_html_comments:
        template = _HTML_COMMENT_RE.sub("", template)

    for key, value in vars.items():
        template = template.replace("{" + key + "}", str(value))
    return template


__all__ = ["render_prompt"]
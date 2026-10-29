// frontend/src/api/feedback.ts
import { request } from "./http";
import type { FeedbackCreateAPI, FeedbackResponse } from "@/types/feedback";

/**
 * POST /api/v1/feedbacks
 * 提交专家反馈（6 类）
 */
export function submitFeedback(
  body: FeedbackCreateAPI,
): Promise<FeedbackResponse> {
  // ===== [S2-E 修正] 适配后端 RuleCandidate 必填字段 =====
  // 前端表单只采集 title / description / severity_hint，
  // 但后端 RuleCandidate 要求 rule_id / rule_name / category / severity / rule_basis。
  // 这里做兜底映射，将 severity_hint 映射为 severity，其余填占位符。
  if (body.rule_candidate && typeof body.rule_candidate === "object") {
    const rc: any = body.rule_candidate;
    // 映射 severity_hint -> severity
    if (rc.severity_hint && !rc.severity) {
      rc.severity = rc.severity_hint;
      delete rc.severity_hint; // 删掉多余字段，否则 Pydantic 报 extra_forbidden
    }
    // 填入后端必填但前端表单未采集的字段（默认占位）
    rc.rule_id = rc.rule_id || "AUTO-GEN"; // 后续规则发布时重新生成
    rc.rule_name = rc.rule_name || rc.title || "未命名规则";
    rc.category = rc.category || "other";
    rc.rule_basis = rc.rule_basis || "专家反馈生成，待审核";
  }

  return request<FeedbackResponse>("/feedbacks", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
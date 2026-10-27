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
  return request<FeedbackResponse>("/feedbacks", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
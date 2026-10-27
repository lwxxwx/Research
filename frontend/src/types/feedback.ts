// frontend/src/types/feedback.ts
// 对齐核对文档第 7–8 章（FeedbackCreateAPI / FeedbackResponse）

/** 6 类反馈类型（Pydantic Literal） */
export type FeedbackType =
  | "correct_defect"
  | "false_positive"
  | "false_negative"
  | "suggestion_update"
  | "knowledge_gap"
  | "new_rule_candidate";

/** 修改建议对比条目（对齐 SuggestionDiff） */
export interface SuggestionDiff {
  old: string;
  new: string;
}

/** 规则候选草稿（对齐 RuleCandidate，11 字段） */
export interface RuleCandidate {
  rule_id: string;
  rule_name: string;
  category: string;
  severity: string;
  rule_basis: string;
  applicable_condition?: Record<string, unknown> | null;
  check_logic?: Record<string, unknown> | null;
  suggestion?: string | null;
  title?: string | null;
  description?: string | null;
  evidence_refs?: Record<string, unknown>[] | null;
}

/** POST /api/v1/feedbacks 请求体（对齐 FeedbackCreateAPI） */
export interface FeedbackCreateAPI {
  task_id: string;
  defect_id: string;
  feedback_type: FeedbackType;
  expert_suggestion?: string | null;
  suggestion_adopted?: "yes" | "partial" | "no" | null;
  suggestion_diff?: SuggestionDiff[] | null;
  rule_candidate?: RuleCandidate | null;
  created_by?: number | null;
}

/** POST /api/v1/feedbacks 响应体（对齐 FeedbackResponse） */
export interface FeedbackResponse {
  id: number;
  task_id: string | null;
  defect_id: string | null;
  review_result_id: number | null;
  review_defect_id: number | null;
  feedback_type: FeedbackType;
  expert_suggestion: string | null;
  suggestion_diff: SuggestionDiff[];
  rule_candidate_ref: string | null;
  created_by: number | null;
  /** FastAPI 序列化为 ISO 8601 字符串 */
  created_at: string;
}
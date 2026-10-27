// frontend/src/types/defect.ts
// 对齐核对文档第 5 章（Defect）+ 第 11 章（report.defects vs GET /defects）

import type { Evidence } from "./evidence";

/** 风险等级（DB 存四态） */
export type Risk = "low" | "medium" | "high" | "critical";

/** 三态质量状态（与 review_result.review_status 对齐） */
export type ReviewStatus =
  | "AI_CONFIRMED"
  | "NEED_EXPERT_REVIEW"
  | "LOW_CONFIDENCE";

/** 发现来源（DB 存三态） */
export type DiscoveryOrigin = "rule" | "ai_discovered" | "hybrid";

/** 专家最终决定（DB 存四态） */
export type ExpertDecision =
  | "pending"
  | "confirmed"
  | "rejected"
  | "needs_verification";

/**
 * 来自 GET /review/tasks/{task_id}/defects（review_defect 表当前状态）。
 * 有 expert_decision_updated_at / expert_decision_by（G1 补）。
 * 无 synthesized（G1 未给 get_task_defects 补）。
 */
export interface DefectFromAPI {
  defect_id: string;
  category: string;
  location: Record<string, unknown>;
  component: string | null;
  net: string | null;
  risk: Risk;
  evidence: Evidence[];
  root_cause: string;
  suggestion: string;
  confidence: number;
  review_status: ReviewStatus;
  origin: DiscoveryOrigin;
  rule_id: string | null;
  expert_decision: ExpertDecision;
  expert_decision_updated_at: string | null;
  expert_decision_by: number | null;
}

/**
 * 来自 report.defects（ReviewReport.model_dump 快照）。
 * 注意：无 expert_decision_updated_at / expert_decision_by（G1 只给 GET /defects 补了）。
 * 有 synthesized（S1-B 新增的 provenance 标记）。
 */
export interface DefectFromReport {
  defect_id: string;
  category: string;
  location: Record<string, unknown>;
  component: string | null;
  net: string | null;
  risk: Risk;
  evidence: Evidence[];
  root_cause: string;
  suggestion: string;
  confidence: number;
  review_status: ReviewStatus;
  origin: DiscoveryOrigin;
  rule_id: string | null;
  expert_decision: ExpertDecision;
  synthesized: boolean;
}

/** GET /api/v1/review/tasks/{task_id}/defects 响应体 */
export interface DefectsResponse {
  items: DefectFromAPI[];
}
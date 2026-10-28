// frontend/src/types/task.ts
// 对齐核对文档第 3–4 章（TaskDetailResponse / TaskListItem）

import type { ReviewStatus, DefectFromReport } from "./defect";

/**
 * 任务状态（对齐 TaskDetailResponse.status）。
 * 注意：后端 Pydantic 声明允许 "pending"，但 get_task 实际只返回 "done" / "failed"。
 * "pending" 场景 = task 不存在 → API 层返回 404。
 */

// ===== [S2-C] 加 pending（S2-A 已加 task_status 列，默认 pending） =====
export type TaskStatus = "pending" | "done" | "failed";
// ===== [S2-C END] =====

// ===== [S2-C] summary_metrics 类型（对齐后端 SummaryMetrics / SampleInfo） =====
export interface SampleInfo {
  ear_sample_size: number;
  ear_sample_threshold: number;
  ai_ner_sample_size: number;
  ai_ner_sample_threshold: number;
}

export interface SummaryMetrics {
  evidence_coverage: number | null;
  ai_new_effective_rate: number | null;
  expert_adoption_rate: number | null;
  review_status_distribution: {
    AI_CONFIRMED: number;
    NEED_EXPERT_REVIEW: number;
    LOW_CONFIDENCE: number;
  };
  sample_info: SampleInfo;
}
// ===== [S2-C END] =====

/** 报告摘要（对齐 ReviewReportSummary） */
export interface ReviewReportSummary {
  evidence_coverage: number | null;
  review_status_distribution: {
    AI_CONFIRMED: number;
    NEED_EXPERT_REVIEW: number;
    LOW_CONFIDENCE: number;
  };
}

/** 报告（对齐 ReviewReport） */
export interface ReviewReport {
  task_id: string;
  report_id: string;
  /** 与 GET /defects 的 items 不同（见核对文档第 11 章） */
  defects: DefectFromReport[];
  summary: ReviewReportSummary;
  errors: string[];
}

/**
 * GET /api/v1/review/tasks/{task_id} 响应体（对齐 TaskDetailResponse）。
 * report 实际至少 {}（空 dict），不会 null；判断空用 Object.keys(report).length === 0
 */
export interface TaskDetail {
  task_id: string;
  status: TaskStatus;
  case_id: string | null;
  report: ReviewReport | Record<string, never> | null;
  errors: string[];
  created_at: string | null;
  defects_count: number;
  // ===== [S2-C] summary_metrics =====
  summary_metrics?: SummaryMetrics | null;
  // ===== [S2-C END] =====
}

/** GET /api/v1/review/tasks 列表项（对齐 TaskListItem） */
export interface TaskListItem {
  task_id: string;
  case_id: string | null;
  case_class: "A" | "B" | null;
  defects_count: number;
  review_status: ReviewStatus;
  created_at: string | null;
  // ===== [S2-C] task_status 透出 =====
  status?: TaskStatus | null;
  // ===== [S2-C END] =====
}

/** GET /api/v1/review/tasks 响应体（对齐 TaskListResponse） */
export interface TaskListResponse {
  total: number;
  items: TaskListItem[];
  errors: string[];
}

/** POST /api/v1/review/tasks 响应体（对齐 CreateTaskResponse） */
export interface CreateTaskResponse {
  task_id: string;
  status: TaskStatus;
  defects_count: number;
}
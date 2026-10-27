// frontend/src/types/task.ts
// 对齐核对文档第 3–4 章（TaskDetailResponse / TaskListItem）

import type { ReviewStatus, DefectFromReport } from "./defect";

/**
 * 任务状态（对齐 TaskDetailResponse.status）。
 * 注意：后端 Pydantic 声明允许 "pending"，但 get_task 实际只返回 "done" / "failed"。
 * "pending" 场景 = task 不存在 → API 层返回 404。
 */
export type TaskStatus = "done" | "failed";

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
}

/** GET /api/v1/review/tasks 列表项（对齐 TaskListItem） */
export interface TaskListItem {
  task_id: string;
  case_id: string | null;
  case_class: "A" | "B" | null;
  defects_count: number;
  review_status: ReviewStatus;
  created_at: string | null;
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
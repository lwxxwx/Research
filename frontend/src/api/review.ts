// frontend/src/api/review.ts
import { request } from "./http";
import type {
  CreateTaskResponse,
  TaskDetail,
  TaskListResponse,
} from "@/types/task";
import type { DefectsResponse } from "@/types/defect";

/** 任务列表查询参数（对齐 GET /review/tasks） */
export interface ListTasksParams {
  case_class?: "A" | "B";
  review_status?: "AI_CONFIRMED" | "NEED_EXPERT_REVIEW" | "LOW_CONFIDENCE";
  limit?: number;
  offset?: number;
}

/**
 * POST /api/v1/review/tasks
 * 发起评审（同步阻塞约 20 秒，超时抛 TIMEOUT_ERROR）
 */
export function createTask(caseId: string): Promise<CreateTaskResponse> {
  return request<CreateTaskResponse>("/review/tasks", {
    method: "POST",
    body: JSON.stringify({ case_id: caseId }),
  });
}

/** GET /api/v1/review/tasks/{task_id} */
export function getTask(taskId: string): Promise<TaskDetail> {
  return request<TaskDetail>(`/review/tasks/${encodeURIComponent(taskId)}`);
}

/** GET /api/v1/review/tasks（筛选 + 分页） */
export function listTasks(params: ListTasksParams = {}): Promise<TaskListResponse> {
  const query = new URLSearchParams();
  if (params.case_class) query.set("case_class", params.case_class);
  if (params.review_status) query.set("review_status", params.review_status);
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));

  const qs = query.toString();
  return request<TaskListResponse>(`/review/tasks${qs ? `?${qs}` : ""}`);
}

/** GET /api/v1/review/tasks/{task_id}/defects */
export function getDefects(taskId: string): Promise<DefectsResponse> {
  return request<DefectsResponse>(
    `/review/tasks/${encodeURIComponent(taskId)}/defects`,
  );
}
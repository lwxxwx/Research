// frontend/src/types/case.ts
// 对齐核对文档第 2 章（CaseItem / CaseListResponse）

/** GET /api/v1/cases 列表项（对齐 CaseItem） */
export interface CaseItem {
  case_id: string;
  /** service 保证只返回 "A" / "B" */
  case_class: "A" | "B";
  title: string | null;
  description: string | null;
  calibration_status: string | null;
  /** service 已归一化为 list[str] */
  tags: string[];
}

/** GET /api/v1/cases 响应体（对齐 CaseListResponse） */
export interface CaseListResponse {
  total: number;
  items: CaseItem[];
  /** 坏 Case 说明，不是 HTTP 错误 */
  errors: string[];
}

// ===== [F07] Case 管理 =====
export interface CaseDetail {
  case_id: string;
  case_dir: string;
  info_yaml: string | null;
  evaluation_yaml: string | null;
  expert_reasoning_md: string | null;
  expected_review_json: Record<string, unknown> | null;
  schematic_ir_json: Record<string, unknown> | null;
}

export interface CaseBenchmarkItem {
  id: number;
  case_id: string;
  mode: string;
  status: string;
  total_gt_defects: number;
  total_pred_defects: number;
  tp: number;
  fp: number;
  fn: number;
  rule_precision: number | null;
  rule_recall: number | null;
  evc: number | null;
  ear: number | null;
  ai_ner: number | null;
  metric_source: string | null;
  task_id: string | null;
  error: string | null;
  created_at: string | null;
}

export interface CaseBenchmarkListResponse {
  total: number;
  items: CaseBenchmarkItem[];
  limit: number;
  offset: number;
}

export interface RunCaseBenchmarkRequest {
  mode: "rule_only" | "full_review";
}
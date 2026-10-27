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
// frontend/src/api/cases.ts
import { request } from "./http";
import type { CaseListResponse } from "@/types/case";

/** GET /api/v1/cases */
export function listCases(): Promise<CaseListResponse> {
  return request<CaseListResponse>("/cases");
}
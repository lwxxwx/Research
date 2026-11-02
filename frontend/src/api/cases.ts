// frontend/src/api/cases.ts
import { request } from "./http";
import type {
  CaseListResponse,
  CaseDetail,
  CaseBenchmarkItem,
  CaseBenchmarkListResponse,
  RunCaseBenchmarkRequest,
} from "@/types/case";

/** GET /api/v1/cases */
export function listCases(): Promise<CaseListResponse> {
  return request<CaseListResponse>("/cases");
}

// ===== [F07] Case 管理 =====
export function getCaseDetail(caseId: string): Promise<CaseDetail> {
  return request<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`);
}

export function runCaseBenchmark(
  caseId: string,
  body: RunCaseBenchmarkRequest,
): Promise<CaseBenchmarkItem> {
  return request<CaseBenchmarkItem>(
    `/cases/${encodeURIComponent(caseId)}/benchmark`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

export function listCaseBenchmarks(
  caseId: string,
  limit = 20,
  offset = 0,
): Promise<CaseBenchmarkListResponse> {
  return request<CaseBenchmarkListResponse>(
    `/cases/${encodeURIComponent(caseId)}/benchmarks?limit=${limit}&offset=${offset}`,
  );
}
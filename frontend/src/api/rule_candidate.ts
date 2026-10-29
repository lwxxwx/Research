// frontend/src/api/rule_candidate.ts
import { request } from "./http";
import type {
  RuleCandidateListResponse,
  RuleCandidateDetail,
  UpdateCandidateStatusRequest,
  UpdateCandidateStatusResponse,
  RuleCandidateStatus
} from "@/types/rule_candidate";

export function listCandidates(status?: RuleCandidateStatus): Promise<RuleCandidateListResponse> {
  const query = status ? `?status=${status}` : "";
  return request<RuleCandidateListResponse>(`/rule-candidates${query}`);
}

export function getCandidate(id: string): Promise<RuleCandidateDetail> {
  return request<RuleCandidateDetail>(`/rule-candidates/${encodeURIComponent(id)}`);
}

export function updateCandidateStatus(
  id: string,
  body: UpdateCandidateStatusRequest
): Promise<UpdateCandidateStatusResponse> {
  return request<UpdateCandidateStatusResponse>(
    `/rule-candidates/${encodeURIComponent(id)}/status`,
    {
      method: "POST",
      body: JSON.stringify(body),
    }
  );
}
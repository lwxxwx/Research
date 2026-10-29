// frontend/src/types/rule_candidate.ts
export type RuleCandidateStatus = "proposed" | "designing" | "benchmarked" | "accepted" | "rejected";

export interface RuleCandidateListItem {
  candidate_id: string;
  from_feedback_id: number | null;
  case_id: string | null;
  title: string;
  description: string;
  severity: string | null;
  status: RuleCandidateStatus;
  created_at: string | null;
}

export interface RuleCandidateListResponse {
  total: number;
  items: RuleCandidateListItem[];
}

export interface RuleCandidateDetail extends RuleCandidateListItem {
  proposed_yaml: string | null;
  evidence_refs: Record<string, any>[];
  benchmark_result: Record<string, any> | null;
  benchmarked_at: string | null;
  rejected_reason: string | null;
  updated_at: string | null;
}

export interface UpdateCandidateStatusRequest {
  status: "accepted" | "rejected";
  rejected_reason?: string | null;
}

export interface UpdateCandidateStatusResponse {
  candidate_id: string;
  status: RuleCandidateStatus;
  rejected_reason: string | null;
  updated_at: string | null;
}
// frontend/src/types/dashboard.ts
// S2-H（四）指标看板类型（对齐后端 DashboardSummaryResponse）

import type { SampleInfo } from "@/types/task";

export interface DashboardScope {
  task_status_filter: "done";
  excluded_task_status: ("pending" | "failed")[];
  total_done_tasks: number;
  has_seed_data: boolean;
}

export interface DashboardMetrics {
  evidence_coverage: number | null;
  ai_new_effective_rate: number | null;
  expert_adoption_rate: number | null;
  review_status_distribution: Record<string, number>;
  expert_decision_distribution: Record<string, number>;
  sample_info: SampleInfo;
}

export interface StatusCount {
  total: number;
  by_status: Record<string, number>;
}

export interface DashboardSummary {
  scope: DashboardScope;
  metrics: DashboardMetrics;
  rule_candidates: StatusCount;
  knowledge_gaps: StatusCount;
}
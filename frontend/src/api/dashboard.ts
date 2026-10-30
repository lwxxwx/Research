// frontend/src/api/dashboard.ts
// S2-H（四）指标看板 API 封装

import { request } from "@/api/http";
import type { DashboardSummary } from "@/types/dashboard";

export function getDashboardSummary(): Promise<DashboardSummary> {
  return request<DashboardSummary>("/dashboard/summary");
}
<!-- frontend/src/components/ExpertDecisionBadge.vue -->
<script setup lang="ts">
import type { ExpertDecision } from "@/types/defect";

const props = defineProps<{
  decision: ExpertDecision | string;
}>();

// ===== 四态徽标（与 expert_decision 对齐，未知值兜底灰色） =====
const DECISION_MAP: Record<
  string,
  { label: string; className: string }
> = {
  pending: {
    label: "待判定",
    className: "bg-gray-100 text-gray-600 border-gray-200",
  },
  confirmed: {
    label: "专家确认",
    className: "bg-emerald-100 text-emerald-700 border-emerald-200",
  },
  rejected: {
    label: "专家驳回",
    className: "bg-red-100 text-red-700 border-red-200",
  },
  needs_verification: {
    label: "需进一步核验",
    className: "bg-amber-100 text-amber-700 border-amber-200",
  },
};

const badge =
  DECISION_MAP[props.decision] ?? {
    label: `未知 (${props.decision})`,
    className: "bg-gray-100 text-gray-600 border-gray-200",
  };
</script>

<template>
  <span
    class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border"
    :class="badge.className"
  >
    {{ badge.label }}
  </span>
</template>
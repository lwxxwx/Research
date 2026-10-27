<!-- frontend/src/components/StatusBadge.vue -->
<script setup lang="ts">
import type { ReviewStatus } from "@/types/defect";

const props = defineProps<{
  status: ReviewStatus | string;
}>();

// ===== 三态徽标（与 review_status 对齐，未知值兜底灰色） =====
const STATUS_MAP: Record<
  string,
  { label: string; className: string }
> = {
  AI_CONFIRMED: {
    label: "AI 已确认",
    className: "bg-emerald-100 text-emerald-700 border-emerald-200",
  },
  NEED_EXPERT_REVIEW: {
    label: "需专家复核",
    className: "bg-amber-100 text-amber-700 border-amber-200",
  },
  LOW_CONFIDENCE: {
    label: "低置信",
    className: "bg-red-100 text-red-700 border-red-200",
  },
};

const badge =
  STATUS_MAP[props.status] ?? {
    label: `未知 (${props.status})`,
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
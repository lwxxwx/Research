<!-- frontend/src/components/ExpertDecisionPanel.vue -->
<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { AlertCircle, Check, HelpCircle, Loader2, X } from "@lucide/vue";

import { updateExpertDecision } from "@/api/review";
import type { ExpertDecision, UpdateExpertDecisionResponse } from "@/types/defect";
import ExpertDecisionBadge from "./ExpertDecisionBadge.vue";

const props = defineProps<{
  taskId: string;
  defectId: string;
  currentDecision: ExpertDecision;
  updatedAt: string | null;
  decisionBy: number | null;
}>();

const emit = defineEmits<{
  (e: "updated", response: UpdateExpertDecisionResponse): void;
}>();

const displayedDecision = ref<ExpertDecision>(props.currentDecision);
const displayedUpdatedAt = ref<string | null>(props.updatedAt);
const displayedDecisionBy = ref<number | null>(props.decisionBy);

watch(() => props.currentDecision, (val) => { displayedDecision.value = val; });
watch(() => props.updatedAt, (val) => { displayedUpdatedAt.value = val; });
watch(() => props.decisionBy, (val) => { displayedDecisionBy.value = val; });

const loading = ref(false);
const errorMsg = ref<string | null>(null);

const isCurrent = (d: ExpertDecision) => displayedDecision.value === d;

const buttons: { value: ExpertDecision; label: string; icon: any; className: string }[] = [
  { value: "confirmed", label: "确认", icon: Check, className: "bg-emerald-600 hover:bg-emerald-700" },
  { value: "rejected", label: "驳回", icon: X, className: "bg-red-600 hover:bg-red-700" },
  { value: "needs_verification", label: "待核验", icon: HelpCircle, className: "bg-amber-600 hover:bg-amber-700" },
];

const auditText = computed(() => {
  if (!displayedUpdatedAt.value) return null;
  const d = new Date(displayedUpdatedAt.value);
  const time = d.toLocaleString();
  const by = displayedDecisionBy.value === null
    ? "操作者 未记录"
    : `操作者 #${displayedDecisionBy.value}`;
  return `判定于 ${time} · ${by}`;
});

async function handleClick(target: ExpertDecision) {
  if (loading.value || isCurrent(target)) return;

  console.log(`[专家判定] 用户点击了「${target}」, 当前缺陷: ${props.defectId}`);

  errorMsg.value = null;
  loading.value = true;

  const prev = {
    decision: displayedDecision.value,
    updatedAt: displayedUpdatedAt.value,
    by: displayedDecisionBy.value,
  };

  displayedDecision.value = target;

  try {
    const resp = await updateExpertDecision(
      props.taskId,
      props.defectId,
      target,
      null,
    );

    displayedDecision.value = resp.expert_decision;
    displayedUpdatedAt.value = resp.expert_decision_updated_at;
    displayedDecisionBy.value = resp.expert_decision_by;

    console.log(`[专家判定] 接口返回成功，审计时间: ${resp.expert_decision_updated_at}`);
    emit("updated", resp);
  } catch (e) {
    displayedDecision.value = prev.decision;
    displayedUpdatedAt.value = prev.updatedAt;
    displayedDecisionBy.value = prev.by;

    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[专家判定] 更新失败，已回滚。原因:`, msg);

    if (msg.includes("404")) {
      errorMsg.value = "缺陷不存在或不属于该任务";
    } else if (msg.includes("422")) {
      errorMsg.value = "非法判定值";
    } else if (msg.includes("TIMEOUT") || msg.includes("timeout")) {
      errorMsg.value = "网络异常，请重试";
    } else {
      errorMsg.value = `更新失败：${msg}`;
    }
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="border-t border-gray-100 pt-3 mt-3">
    <div class="flex items-center justify-between mb-2">
      <h4 class="text-xs font-semibold text-gray-700">专家判定</h4>
      <ExpertDecisionBadge :decision="displayedDecision" />
    </div>

    <div class="flex gap-2">
      <button
        v-for="b in buttons"
        :key="b.value"
        type="button"
        :disabled="loading || isCurrent(b.value)"
        :class="[
          'flex items-center px-3 py-1.5 rounded text-xs font-medium text-white transition-colors',
          b.className,
          (loading || isCurrent(b.value)) ? 'opacity-50 cursor-not-allowed' : '',
        ]"
        @click="handleClick(b.value)"
      >
        <Loader2 v-if="loading && !isCurrent(b.value)" class="w-3 h-3 mr-1 animate-spin" />
        <component v-else :is="b.icon" class="w-3 h-3 mr-1" />
        {{ b.label }}
      </button>
    </div>

    <p v-if="auditText" class="text-xs text-gray-500 mt-2">{{ auditText }}</p>

    <div v-if="errorMsg" class="mt-2 p-2 bg-red-50 border border-red-200 rounded flex items-start">
      <AlertCircle class="w-4 h-4 text-red-600 mt-0.5 mr-1 flex-shrink-0" />
      <p class="text-xs text-red-800">{{ errorMsg }}</p>
    </div>
  </div>
</template>
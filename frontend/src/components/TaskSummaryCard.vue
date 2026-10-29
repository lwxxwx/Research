<!-- frontend/src/components/TaskSummaryCard.vue -->
<script setup lang="ts">
import type { SummaryMetrics } from "@/types/task";

const props = defineProps<{
  metrics: SummaryMetrics | null | undefined;
}>();

function formatPct(val: number | null | undefined): string {
  if (val === null || val === undefined) return "N/A";
  return `${(val * 100).toFixed(2)}%`;
}

function earText(
  val: number | null | undefined,
  sampleSize?: number,
  threshold?: number,
): { text: string; hint: string } {
  if (val !== null && val !== undefined) return { text: formatPct(val), hint: "" };
  if (sampleSize !== undefined && threshold !== undefined && sampleSize < threshold) {
    return { text: "N/A", hint: "样本不足" };
  }
  return { text: "N/A", hint: "待专家反馈" };
}
</script>

<template>
  <!-- 去掉 v-if="metrics"，始终渲染外壳，防止整个 UI 消失 -->
  <div class="card p-4 bg-white border border-gray-200">
    <h3 class="text-sm font-semibold text-gray-700 mb-4">任务摘要指标</h3>
    
    <!-- 如果连 metrics 对象都没有，给个提示 -->
    <div v-if="!metrics" class="text-sm text-gray-500 py-2">
      暂无指标数据（后端未返回 summary_metrics）
    </div>

    <div v-else class="grid grid-cols-2 md:grid-cols-4 gap-4">
      
      <!-- EVC -->
      <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <div class="text-xs text-gray-500 mb-1">EVC (证据完整率)</div>
        <div class="text-xl font-bold text-blue-600">{{ formatPct(metrics.evidence_coverage) }}</div>
      </div>

      <!-- AI_NER -->
      <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <div class="text-xs text-gray-500 mb-1">AI_NER (新增有效发现)</div>
        <div class="flex items-baseline">
          <span class="text-xl font-bold text-indigo-600">
            {{ earText(metrics.ai_new_effective_rate, metrics.sample_info?.ai_ner_sample_size, metrics.sample_info?.ai_ner_sample_threshold).text }}
          </span>
          <span v-if="earText(metrics.ai_new_effective_rate, metrics.sample_info?.ai_ner_sample_size, metrics.sample_info?.ai_ner_sample_threshold).hint" class="text-xs text-gray-400 ml-1">
            · {{ earText(metrics.ai_new_effective_rate, metrics.sample_info?.ai_ner_sample_size, metrics.sample_info?.ai_ner_sample_threshold).hint }}
          </span>
        </div>
      </div>

      <!-- EAR -->
      <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <div class="text-xs text-gray-500 mb-1">EAR (专家采纳率)</div>
        <div class="flex items-baseline">
          <span class="text-xl font-bold text-emerald-600">
            {{ earText(metrics.expert_adoption_rate, metrics.sample_info?.ear_sample_size, metrics.sample_info?.ear_sample_threshold).text }}
          </span>
          <span v-if="earText(metrics.expert_adoption_rate, metrics.sample_info?.ear_sample_size, metrics.sample_info?.ear_sample_threshold).hint" class="text-xs text-gray-400 ml-1">
            · {{ earText(metrics.expert_adoption_rate, metrics.sample_info?.ear_sample_size, metrics.sample_info?.ear_sample_threshold).hint }}
          </span>
        </div>
      </div>

      <!-- 质量分布 -->
      <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <div class="text-xs text-gray-500 mb-1">AI 质量分布</div>
        <div class="flex flex-col space-y-0.5 text-xs mt-1">
          <div class="flex justify-between">
            <span class="text-gray-600">AI 已确认</span>
            <span class="font-bold text-emerald-600">{{ metrics.review_status_distribution?.AI_CONFIRMED || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600">需复核</span>
            <span class="font-bold text-amber-600">{{ metrics.review_status_distribution?.NEED_EXPERT_REVIEW || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600">低置信</span>
            <span class="font-bold text-red-600">{{ metrics.review_status_distribution?.LOW_CONFIDENCE || 0 }}</span>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>
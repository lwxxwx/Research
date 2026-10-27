<!-- frontend/src/components/DefectCard.vue -->
<script setup lang="ts">
import { computed, ref } from "vue";
import { ChevronDown, ChevronRight } from "@lucide/vue";

import type { DefectFromAPI } from "@/types/defect";
import EvidenceChain from "./EvidenceChain.vue";
import StatusBadge from "./StatusBadge.vue";
import ExpertDecisionBadge from "./ExpertDecisionBadge.vue";

const props = defineProps<{
  defect: DefectFromAPI;
  defaultExpanded?: boolean;
}>();

const isExpanded = ref(props.defaultExpanded ?? false);

const riskColor = computed(() => {
  switch (props.defect.risk) {
    case "critical": return "text-red-700 bg-red-100 border-red-200";
    case "high": return "text-orange-700 bg-orange-100 border-orange-200";
    case "medium": return "text-amber-700 bg-amber-100 border-amber-200";
    case "low": return "text-emerald-700 bg-emerald-100 border-emerald-200";
    default: return "text-gray-700 bg-gray-100 border-gray-200";
  }
});

const originText = computed(() => {
  switch (props.defect.origin) {
    case "rule": return "规则发现";
    case "ai_discovered": return "AI 发现";
    case "hybrid": return "混合";
    default: return props.defect.origin;
  }
});
</script>

<template>
  <div class="border border-gray-200 rounded-lg bg-white overflow-hidden">
    <!-- 卡片头部（点击展开/折叠） -->
    <button
      class="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors text-left"
      @click="isExpanded = !isExpanded"
    >
      <div class="flex items-center space-x-3 min-w-0">
        <component :is="isExpanded ? ChevronDown : ChevronRight" class="w-4 h-4 text-gray-400 flex-shrink-0" />
        <span class="font-medium text-gray-900 truncate">{{ defect.defect_id }}</span>
        <span class="text-xs px-2 py-0.5 rounded border" :class="riskColor">{{ defect.risk.toUpperCase() }}</span>
        <span class="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">{{ originText }}</span>
      </div>
      <div class="flex items-center space-x-2 flex-shrink-0 ml-4">
        <StatusBadge :status="defect.review_status" />
        <ExpertDecisionBadge :decision="defect.expert_decision" />
      </div>
    </button>

    <!-- 卡片内容 -->
    <div v-show="isExpanded" class="px-4 py-4 border-t border-gray-100 space-y-4 text-sm">
      <!-- 基本信息网格 -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <span class="text-gray-500">分类：</span>
          <span class="font-medium">{{ defect.category }}</span>
        </div>
        <div>
          <span class="text-gray-500">元件：</span>
          <span class="font-mono">{{ defect.component ?? '-' }}</span>
        </div>
        <div>
          <span class="text-gray-500">网络：</span>
          <span class="font-mono">{{ defect.net ?? '-' }}</span>
        </div>
        <div>
          <span class="text-gray-500">置信度：</span>
          <span>{{ (defect.confidence * 100).toFixed(0) }}%</span>
        </div>
        <div class="col-span-2" v-if="defect.rule_id">
          <span class="text-gray-500">规则 ID：</span>
          <code class="text-xs bg-gray-100 px-1 py-0.5 rounded">{{ defect.rule_id }}</code>
        </div>
      </div>

      <!-- 根因与建议 -->
      <div>
        <h4 class="text-xs font-semibold text-gray-700 mb-1">根因分析</h4>
        <p class="text-gray-600 bg-gray-50 p-2 rounded">{{ defect.root_cause }}</p>
      </div>
      <div>
        <h4 class="text-xs font-semibold text-gray-700 mb-1">修改建议</h4>
        <p class="text-gray-600 bg-gray-50 p-2 rounded">{{ defect.suggestion }}</p>
      </div>

      <!-- 证据链 -->
      <div>
        <h4 class="text-xs font-semibold text-gray-700 mb-2">证据链（Evidence Chain）</h4>
        <EvidenceChain :evidence="defect.evidence" />
      </div>
    </div>
  </div>
</template>
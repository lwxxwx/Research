<!-- frontend/src/components/EvidenceChain.vue -->
<script setup lang="ts">
import type { Evidence } from "@/types/evidence";

defineProps<{
  evidence: Evidence[];
}>();
</script>

<template>
  <div v-if="evidence.length === 0" class="text-xs text-gray-400 italic">
    无证据记录
  </div>
  <div v-else class="space-y-2">
    <div
      v-for="(ev, idx) in evidence"
      :key="idx"
      class="text-xs bg-gray-50 rounded p-2 border border-gray-100"
    >
      <template v-if="ev.type === 'rag_ref'">
        <div class="flex items-start">
          <span class="inline-flex items-center px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 font-medium mr-2 flex-shrink-0">
            RAG
          </span>
          <div>
            <p><span class="font-medium text-gray-700">来源：</span>{{ ev.source || '—' }}</p>
            <p><span class="font-medium text-gray-700">章节：</span>{{ ev.section || '—' }}</p>
            <p class="text-gray-600 mt-1">{{ ev.reason || '—' }}</p>
          </div>
        </div>
      </template>
      <template v-else-if="ev.type === 'ir_ref'">
        <div class="flex items-start">
          <span class="inline-flex items-center px-1.5 py-0.5 rounded bg-purple-100 text-purple-700 font-medium mr-2 flex-shrink-0">
            IR
          </span>
          <code class="text-gray-700 break-all">{{ ev.detail || '—' }}</code>
        </div>
      </template>
      <template v-else>
        <pre class="text-xs overflow-x-auto">{{ JSON.stringify(ev, null, 2) }}</pre>
      </template>
    </div>
  </div>
</template>
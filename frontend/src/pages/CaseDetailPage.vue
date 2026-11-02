<!-- frontend/src/pages/CaseDetailPage.vue -->
<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import {
  getCaseDetail,
  runCaseBenchmark,
  listCaseBenchmarks,
} from "@/api/cases";
import type {
  CaseDetail,
  CaseBenchmarkItem,
} from "@/types/case";

const route = useRoute();
const router = useRouter();

const caseId = route.params.caseId as string;
const detail = ref<CaseDetail | null>(null);
const benchmarks = ref<CaseBenchmarkItem[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const running = ref(false);
const mode = ref<"rule_only" | "full_review">("rule_only");

async function loadDetail() {
  loading.value = true;
  error.value = null;
  try {
    detail.value = await getCaseDetail(caseId);
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

async function loadBenchmarks() {
  try {
    const resp = await listCaseBenchmarks(caseId);
    benchmarks.value = resp.items;
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function handleRunBenchmark() {
  running.value = true;
  error.value = null;
  try {
    const result = await runCaseBenchmark(caseId, { mode: mode.value });
    alert(`Benchmark 完成：status=${result.status}, tp=${result.tp}, fp=${result.fp}, fn=${result.fn}`);
    await loadBenchmarks();
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    running.value = false;
  }
}

function goBack() {
  router.push("/cases");
}

onMounted(async () => {
  await loadDetail();
  await loadBenchmarks();
});
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <button
          class="text-sm text-blue-600 hover:underline"
          @click="goBack"
        >
          ← 返回 Case 列表
        </button>
        <h1 class="text-2xl font-bold text-gray-900">{{ caseId }}</h1>
      </div>
    </div>

    <div v-if="loading" class="text-sm text-gray-500">加载中…</div>
    <div v-else-if="error" class="text-sm text-red-600">错误：{{ error }}</div>

    <div v-else-if="detail" class="space-y-6">
      <!-- Benchmark 操作区 -->
      <div class="bg-white border border-gray-200 rounded-lg p-4">
        <div class="flex items-center space-x-3">
          <label class="text-sm text-gray-600">Benchmark 模式：</label>
          <select
            v-model="mode"
            class="px-2 py-1.5 text-sm border border-gray-300 rounded-md"
          >
            <option value="rule_only">rule_only（纯规则）</option>
            <option value="full_review">full_review（走 LLM，耗时 20–30s）</option>
          </select>
          <button
            class="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            :disabled="running"
            @click="handleRunBenchmark"
          >
            {{ running ? "运行中…" : "运行 Benchmark" }}
          </button>
        </div>
      </div>

      <!-- 资产预览 -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="bg-white border border-gray-200 rounded-lg p-4">
          <h3 class="text-sm font-semibold text-gray-700 mb-2">info.yaml</h3>
          <pre class="text-xs text-gray-700 whitespace-pre-wrap max-h-64 overflow-y-auto">{{ detail.info_yaml || "（无）" }}</pre>
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-4">
          <h3 class="text-sm font-semibold text-gray-700 mb-2">evaluation.yaml</h3>
          <pre class="text-xs text-gray-700 whitespace-pre-wrap max-h-64 overflow-y-auto">{{ detail.evaluation_yaml || "（无）" }}</pre>
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-4">
          <h3 class="text-sm font-semibold text-gray-700 mb-2">expert_reasoning.md</h3>
          <pre class="text-xs text-gray-700 whitespace-pre-wrap max-h-64 overflow-y-auto">{{ detail.expert_reasoning_md || "（无）" }}</pre>
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-4">
          <h3 class="text-sm font-semibold text-gray-700 mb-2">expected_review.json</h3>
          <pre class="text-xs text-gray-700 whitespace-pre-wrap max-h-64 overflow-y-auto">{{ JSON.stringify(detail.expected_review_json, null, 2) }}</pre>
        </div>
      </div>

      <!-- 历史结果 -->
      <div class="bg-white border border-gray-200 rounded-lg p-4">
        <h3 class="text-sm font-semibold text-gray-700 mb-2">历史 Benchmark 结果</h3>
        <div v-if="benchmarks.length === 0" class="text-sm text-gray-500">暂无历史结果</div>
        <table v-else class="w-full text-sm">
          <thead class="text-xs text-gray-500 border-b">
            <tr>
              <th class="text-left py-2">时间</th>
              <th class="text-left">模式</th>
              <th class="text-left">状态</th>
              <th class="text-right">TP</th>
              <th class="text-right">FP</th>
              <th class="text-right">FN</th>
              <th class="text-right">Precision</th>
              <th class="text-right">Recall</th>
              <th class="text-right">EVC</th>
              <th class="text-right">EAR</th>
              <th class="text-right">AI_NER</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="b in benchmarks" :key="b.id" class="border-b hover:bg-gray-50">
              <td class="py-2">{{ b.created_at }}</td>
              <td>{{ b.mode }}</td>
              <td>
                <span
                  class="text-xs px-1.5 py-0.5 rounded"
                  :class="b.status === 'done' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'"
                >{{ b.status }}</span>
              </td>
              <td class="text-right">{{ b.tp }}</td>
              <td class="text-right">{{ b.fp }}</td>
              <td class="text-right">{{ b.fn }}</td>
              <td class="text-right">{{ b.rule_precision?.toFixed(4) ?? "N/A" }}</td>
              <td class="text-right">{{ b.rule_recall?.toFixed(4) ?? "N/A" }}</td>
              <td class="text-right">{{ b.evc?.toFixed(4) ?? "N/A" }}</td>
              <td class="text-right">{{ b.ear?.toFixed(4) ?? "N/A" }}</td>
              <td class="text-right">{{ b.ai_ner?.toFixed(4) ?? "N/A" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
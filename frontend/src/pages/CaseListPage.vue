<!-- frontend/src/pages/CaseListPage.vue -->
<script setup lang="ts">
import { onMounted, ref, computed } from "vue";
import { useRouter } from "vue-router";
import { AlertCircle, FileText, Loader2, RefreshCw, CheckSquare, Square, PlayCircle } from "@lucide/vue";

import { listCases } from "@/api/cases";
import { createTask } from "@/api/review";
import { TIMEOUT_ERROR } from "@/api/http";
import type { CaseItem } from "@/types/case";

const router = useRouter();

// ===== 列表状态 =====
const cases = ref<CaseItem[]>([]);
const errors = ref<string[]>([]);
const loading = ref(true);
const loadError = ref<string | null>(null);

// ===== 单选发起状态 =====
const creatingCaseId = ref<string | null>(null);

// ===== 批量发起状态 =====
const selectedCaseIds = ref<string[]>([]);
const isBatchRunning = ref(false);
const batchProgress = ref({ current: 0, total: 0, currentCaseId: "" });

// ===== 错误提示 =====
const createError = ref<string | null>(null);
const createTimedOut = ref(false);

const MAX_BATCH = 5;

const toggleSelect = (caseId: string) => {
  if (isBatchRunning.value) return;
  const idx = selectedCaseIds.value.indexOf(caseId);
  if (idx >= 0) {
    selectedCaseIds.value.splice(idx, 1);
  } else {
    if (selectedCaseIds.value.length >= MAX_BATCH) {
      alert(`为保证稳定，单次最多选择 ${MAX_BATCH} 个 Case`);
      return;
    }
    selectedCaseIds.value.push(caseId);
  }
};

const isSelected = (caseId: string) => selectedCaseIds.value.includes(caseId);

// ===== [F07] 卡片跳详情 =====
function goToCaseDetail(caseId: string) {
  if (isBatchRunning.value) return;
  router.push(`/cases/${caseId}`);
}

// ===== [F07] 状态颜色映射 =====
const getStatusColor = (status: string | null) => {
  if (!status) return "bg-gray-100 text-gray-500 border-gray-200";
  const s = status.toUpperCase();
  if (s.includes("PENDING")) return "bg-amber-100 text-amber-700 border-amber-200";
  if (s.includes("APPROVED") || s.includes("CONFIRMED")) return "bg-emerald-100 text-emerald-700 border-emerald-200";
  if (s.includes("REJECTED")) return "bg-red-100 text-red-700 border-red-200";
  if (s.includes("CALIBRATED")) return "bg-blue-100 text-blue-700 border-blue-200";
  if (s.includes("PROVISIONAL")) return "bg-purple-100 text-purple-700 border-purple-200";
  return "bg-gray-100 text-gray-500 border-gray-200";
};

// ===== 加载列表 =====
async function loadCases() {
  loading.value = true;
  loadError.value = null;
  try {
    const resp = await listCases();
    cases.value = resp.items;
    errors.value = resp.errors;
    selectedCaseIds.value = [];
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

// ===== 单个发起评审 =====
async function handleCreate(c: CaseItem) {
  if (creatingCaseId.value !== null || isBatchRunning.value) return;

  creatingCaseId.value = c.case_id;
  createError.value = null;
  createTimedOut.value = false;

  try {
    const resp = await createTask(c.case_id);
    await router.push({ name: "task-detail", params: { taskId: resp.task_id } });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg === TIMEOUT_ERROR) {
      createTimedOut.value = true;
      createError.value = "请求超时（30 秒），任务可能已在后台创建。请到任务列表确认是否已生成。";
    } else {
      createError.value = msg;
    }
  } finally {
    creatingCaseId.value = null;
  }
}

// ===== 批量发起评审 =====
async function handleBatchCreate() {
  if (selectedCaseIds.value.length === 0 || isBatchRunning.value) return;
  
  isBatchRunning.value = true;
  createError.value = null;
  createTimedOut.value = false;
  batchProgress.value = { current: 0, total: selectedCaseIds.value.length, currentCaseId: "" };

  const results: { id: string; success: boolean; error?: string }[] = [];
  const toRun = [...selectedCaseIds.value];

  for (let i = 0; i < toRun.length; i++) {
    const cid = toRun[i];
    batchProgress.value.current = i + 1;
    batchProgress.value.currentCaseId = cid;

    try {
      await createTask(cid);
      results.push({ id: cid, success: true });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      results.push({ id: cid, success: false, error: msg });
    }
  }

  isBatchRunning.value = false;
  
  const successCount = results.filter(r => r.success).length;
  const failCount = results.length - successCount;
  
  if (failCount === 0) {
    alert(`批量评审全部成功（${successCount} 个），即将跳转到任务列表。`);
    router.push({ name: "tasks" });
  } else {
    const failDetails = results.filter(r => !r.success).map(r => `${r.id}: ${r.error}`).join("\n");
    alert(`批量评审完成：成功 ${successCount} 个，失败 ${failCount} 个。\n\n失败详情：\n${failDetails}`);
    router.push({ name: "tasks" });
  }
}

// ===== 辅助函数 =====
function goToTasks() { router.push({ name: "tasks" }); }
function dismissError() { createError.value = null; createTimedOut.value = false; }

const canBatch = computed(() => selectedCaseIds.value.length > 0 && !isBatchRunning.value && creatingCaseId.value === null);

onMounted(loadCases);
</script>

<template>
  <div class="space-y-6 pb-20 relative">
    <!-- ===== 页头 ===== -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-semibold text-gray-900">Case 列表</h2>
        <p class="mt-1 text-sm text-gray-500">
          选择 Case 发起评审（约 20 秒）。支持勾选多个 Case 进行批量评审（上限 {{ MAX_BATCH }} 个）。
        </p>
      </div>
      <button class="btn-secondary" :disabled="loading || isBatchRunning" @click="loadCases">
        <RefreshCw class="w-4 h-4 mr-1.5" :class="{ 'animate-spin': loading }" />
        刷新
      </button>
    </div>

    <!-- ===== 错误提示 ===== -->
    <div v-if="loadError" class="card p-4 border-red-200 bg-red-50 flex items-start">
      <AlertCircle class="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
      <div class="flex-1">
        <p class="text-sm font-medium text-red-800">加载 Case 列表失败</p>
        <p class="mt-1 text-sm text-red-700">{{ loadError }}</p>
      </div>
      <button class="btn-secondary ml-3" @click="loadCases">重试</button>
    </div>

    <div v-if="createError" class="card p-4 flex items-start"
      :class="createTimedOut ? 'border-amber-200 bg-amber-50' : 'border-red-200 bg-red-50'">
      <AlertCircle class="w-5 h-5 mt-0.5 mr-3 flex-shrink-0" :class="createTimedOut ? 'text-amber-600' : 'text-red-600'" />
      <div class="flex-1">
        <p class="text-sm font-medium" :class="createTimedOut ? 'text-amber-800' : 'text-red-800'">
          {{ createTimedOut ? "请求超时" : "发起评审失败" }}
        </p>
        <p class="mt-1 text-sm whitespace-pre-wrap" :class="createTimedOut ? 'text-amber-700' : 'text-red-700'">{{ createError }}</p>
        <div class="mt-3 flex items-center space-x-2">
          <button v-if="createTimedOut" class="btn-primary" @click="goToTasks">去任务列表确认</button>
          <button class="btn-secondary" @click="dismissError">关闭</button>
        </div>
      </div>
    </div>

    <!-- ===== 加载中 ===== -->
    <div v-if="loading" class="flex items-center justify-center py-16">
      <Loader2 class="w-6 h-6 text-blue-600 animate-spin" />
      <span class="ml-2 text-gray-600">加载中...</span>
    </div>

    <!-- ===== 空列表 ===== -->
    <div v-else-if="cases.length === 0" class="card p-12 text-center text-gray-500">
      没有可用的 Case。
    </div>

    <!-- ===== Case 列表 ===== -->
    <div v-else class="space-y-8">
      <div v-for="group in ['A', 'B'] as const" :key="group" v-show="cases.some((c) => c.case_class === group)">
        <h3 class="text-sm font-semibold text-gray-700 mb-3 flex items-center">
          <span class="inline-flex items-center justify-center w-6 h-6 rounded-full mr-2 text-xs font-bold"
            :class="group === 'A' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'">
            {{ group }}
          </span>
          {{ group === "A" ? "A 类 · 规则基线" : "B 类 · AI 辅助发现" }}
        </h3>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <!-- 卡片主体：绝对不使用 absolute 定位 -->
          <div
            v-for="c in cases.filter((x) => x.case_class === group)"
            :key="c.case_id"
            class="card hover:shadow-md transition-shadow flex flex-col"
            :class="isSelected(c.case_id) ? 'border-blue-500 ring-1 ring-blue-500' : ''"
          >
            <!-- 卡片顶部：左侧是跳转区域，右侧固定放状态胶囊 -->
            <div class="flex items-start justify-between p-4 border-b border-gray-50">
              
              <!-- 左侧标题区域：点击跳详情 -->
              <div class="flex-1 cursor-pointer min-w-0 pr-3" @click="goToCaseDetail(c.case_id)">
                <div class="flex items-center mb-1">
                  <FileText class="w-4 h-4 text-gray-400 mr-2 flex-shrink-0" />
                  <code class="text-xs font-mono text-gray-600 truncate">{{ c.case_id }}</code>
                </div>
                <h4 class="text-base font-medium text-gray-900 truncate">
                  {{ c.title ?? c.case_id }}
                </h4>
              </div>

              <!-- 右侧：状态胶囊 + 多选（整体点击可选中） -->
              <div 
                class="flex-shrink-0 flex items-center h-full pt-1 cursor-pointer"
                @click.stop="toggleSelect(c.case_id)"
              >
                <div 
                  class="flex items-center px-2.5 py-1.5 rounded-md border transition-all text-xs font-medium"
                  :class="isSelected(c.case_id) 
                    ? 'bg-blue-600 text-white border-blue-600 shadow-sm' 
                    : getStatusColor(c.calibration_status) + ' hover:opacity-80'"
                >
                  <CheckSquare v-if="isSelected(c.case_id)" class="w-3.5 h-3.5 mr-1.5" />
                  <Square v-else class="w-3.5 h-3.5 mr-1.5 opacity-60" />
                  <span v-if="isSelected(c.case_id)">已选择</span>
                  <span v-else class="truncate max-w-[80px]">{{ c.calibration_status || "未知" }}</span>
                </div>
              </div>
            </div>

            <!-- 中间：描述内容，点击跳详情 -->
            <div class="p-4 flex-1 flex flex-col cursor-pointer" @click="goToCaseDetail(c.case_id)">
              <p v-if="c.description" class="text-sm text-gray-500 line-clamp-2 flex-1">{{ c.description }}</p>
              <div v-if="c.tags.length > 0" class="mt-2 flex flex-wrap gap-1">
                <span v-for="tag in c.tags" :key="tag" class="text-xs px-2 py-0.5 rounded bg-gray-50 text-gray-600 border border-gray-200">
                  {{ tag }}
                </span>
              </div>
            </div>

            <!-- 底部：操作按钮 -->
            <div class="p-4 pt-0">
              <button
                class="btn-primary w-full"
                :disabled="creatingCaseId !== null || isBatchRunning"
                @click.stop="handleCreate(c)"
              >
                <Loader2 v-if="creatingCaseId === c.case_id" class="w-4 h-4 mr-1.5 animate-spin" />
                单次评审
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== 底部悬浮批量操作栏 ===== -->
    <div v-if="selectedCaseIds.length > 0" class="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] p-4 z-20">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div class="text-sm text-gray-700">
          已选择 <strong class="text-blue-600 font-bold text-lg">{{ selectedCaseIds.length }}</strong> 个 Case
          <span class="text-gray-400 ml-2">(最多 {{ MAX_BATCH }} 个)</span>
        </div>
        <div class="flex items-center space-x-3">
          <button class="text-sm text-gray-500 hover:text-gray-700" :disabled="isBatchRunning" @click="selectedCaseIds = []">
            取消选择
          </button>
          <button class="btn-primary flex items-center" :disabled="!canBatch" @click="handleBatchCreate">
            <Loader2 v-if="isBatchRunning" class="w-4 h-4 mr-2 animate-spin" />
            <PlayCircle v-else class="w-4 h-4 mr-2" />
            {{ isBatchRunning ? '批量评审进行中...' : '开始批量评审' }}
          </button>
        </div>
      </div>
    </div>

    <!-- ===== 批量进度遮罩弹窗 ===== -->
    <div v-if="isBatchRunning" class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-sm">
      <div class="bg-white rounded-xl shadow-xl p-8 max-w-md w-full text-center space-y-4">
        <Loader2 class="w-10 h-10 text-blue-600 animate-spin mx-auto" />
        <h3 class="text-xl font-bold text-gray-900">批量评审中</h3>
        <p class="text-gray-600">
          正在处理第 <span class="font-bold text-blue-600">{{ batchProgress.current }}</span> / {{ batchProgress.total }} 个 Case
        </p>
        <div class="bg-gray-100 rounded-full h-2 w-full overflow-hidden">
          <div class="bg-blue-600 h-2 transition-all duration-300" :style="{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }"></div>
        </div>
        <code class="text-sm text-gray-500 block bg-gray-50 p-2 rounded">{{ batchProgress.currentCaseId }}</code>
        <p class="text-xs text-gray-400 mt-4 leading-relaxed">
          系统正在串行执行，每个 Case 预计需要 15~25 秒。<br/>
          当前批次总耗时约 {{ batchProgress.total * 20 }} 秒，请耐心等待，<strong>请勿关闭页面</strong>...
        </p>
      </div>
    </div>
  </div>
</template>
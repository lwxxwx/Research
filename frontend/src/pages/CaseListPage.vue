<!-- frontend/src/pages/CaseListPage.vue -->
<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { AlertCircle, FileText, Loader2, RefreshCw } from "@lucide/vue";

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

// ===== 当前正在发起评审的 case_id（用于按钮 loading）=====
const creatingCaseId = ref<string | null>(null);

// ===== 发起评审错误提示（区别于列表加载错误）=====
const createError = ref<string | null>(null);
// ===== 超时后的引导（禁止自动重试，引导去任务列表）=====
const createTimedOut = ref(false);

/** 加载 Case 列表 */
async function loadCases() {
  loading.value = true;
  loadError.value = null;
  try {
    const resp = await listCases();
    cases.value = resp.items;
    errors.value = resp.errors;
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

/** 点击 Case → 发起评审 */
async function handleCreate(c: CaseItem) {
  // 防重复提交（loading 期间禁止点击）
  if (creatingCaseId.value !== null) return;

  creatingCaseId.value = c.case_id;
  createError.value = null;
  createTimedOut.value = false;

  try {
    const resp = await createTask(c.case_id);
    // 成功后跳转任务详情
    await router.push({ name: "task-detail", params: { taskId: resp.task_id } });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);

    if (msg === TIMEOUT_ERROR) {
      // ===== 超时：禁止自动重试，引导去任务列表（可能已创建）=====
      createTimedOut.value = true;
      createError.value =
        "请求超时（20 秒），任务可能已在后台创建。请到任务列表确认是否已生成。";
    } else {
      // ===== 其他错误：可重试 =====
      createError.value = msg;
    }
  } finally {
    creatingCaseId.value = null;
  }
}

/** 手动去任务列表 */
function goToTasks() {
  router.push({ name: "tasks" });
}

/** 关闭错误提示 */
function dismissError() {
  createError.value = null;
  createTimedOut.value = false;
}

onMounted(loadCases);
</script>

<template>
  <div class="space-y-6">
    <!-- ===== 页头 ===== -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-semibold text-gray-900">Case 列表</h2>
        <p class="mt-1 text-sm text-gray-500">
          选择 Case 发起评审（约 20 秒）。A 类为规则基线，B 类为 AI 辅助发现。
        </p>
      </div>
      <button
        class="btn-secondary"
        :disabled="loading"
        @click="loadCases"
      >
        <RefreshCw class="w-4 h-4 mr-1.5" :class="{ 'animate-spin': loading }" />
        刷新
      </button>
    </div>

    <!-- ===== 列表加载错误 ===== -->
    <div
      v-if="loadError"
      class="card p-4 border-red-200 bg-red-50 flex items-start"
    >
      <AlertCircle class="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
      <div class="flex-1">
        <p class="text-sm font-medium text-red-800">加载 Case 列表失败</p>
        <p class="mt-1 text-sm text-red-700">{{ loadError }}</p>
      </div>
      <button class="btn-secondary ml-3" @click="loadCases">重试</button>
    </div>

    <!-- ===== 发起评审错误（含超时）===== -->
    <div
      v-if="createError"
      class="card p-4 flex items-start"
      :class="createTimedOut ? 'border-amber-200 bg-amber-50' : 'border-red-200 bg-red-50'"
    >
      <AlertCircle
        class="w-5 h-5 mt-0.5 mr-3 flex-shrink-0"
        :class="createTimedOut ? 'text-amber-600' : 'text-red-600'"
      />
      <div class="flex-1">
        <p
          class="text-sm font-medium"
          :class="createTimedOut ? 'text-amber-800' : 'text-red-800'"
        >
          {{ createTimedOut ? "请求超时" : "发起评审失败" }}
        </p>
        <p
          class="mt-1 text-sm"
          :class="createTimedOut ? 'text-amber-700' : 'text-red-700'"
        >
          {{ createError }}
        </p>
        <div class="mt-3 flex items-center space-x-2">
          <button
            v-if="createTimedOut"
            class="btn-primary"
            @click="goToTasks"
          >
            去任务列表确认
          </button>
          <button class="btn-secondary" @click="dismissError">关闭</button>
        </div>
      </div>
    </div>

    <!-- ===== 坏 Case 提示（errors 非空）===== -->
    <div
      v-if="errors.length > 0"
      class="card p-3 border-amber-200 bg-amber-50 text-sm text-amber-800"
    >
      有 {{ errors.length }} 个 Case 资产异常（已跳过）：
      <ul class="mt-1 list-disc list-inside text-xs">
        <li v-for="(e, idx) in errors" :key="idx">{{ e }}</li>
      </ul>
    </div>

    <!-- ===== 加载中 ===== -->
    <div v-if="loading" class="flex items-center justify-center py-16">
      <Loader2 class="w-6 h-6 text-blue-600 animate-spin" />
      <span class="ml-2 text-gray-600">加载中...</span>
    </div>

    <!-- ===== 空列表 ===== -->
    <div
      v-else-if="cases.length === 0"
      class="card p-12 text-center text-gray-500"
    >
      没有可用的 Case。
    </div>

    <!-- ===== Case 列表（按 A / B 分组）===== -->
    <div v-else class="space-y-8">
      <div
        v-for="group in ['A', 'B'] as const"
        :key="group"
        v-show="cases.some((c) => c.case_class === group)"
      >
        <!-- 分组标题 -->
        <h3 class="text-sm font-semibold text-gray-700 mb-3 flex items-center">
          <span
            class="inline-flex items-center justify-center w-6 h-6 rounded-full mr-2 text-xs font-bold"
            :class="
              group === 'A'
                ? 'bg-blue-100 text-blue-700'
                : 'bg-purple-100 text-purple-700'
            "
          >
            {{ group }}
          </span>
          {{ group === "A" ? "A 类 · 规则基线" : "B 类 · AI 辅助发现" }}
        </h3>

        <!-- Case 卡片网格 -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div
            v-for="c in cases.filter((x) => x.case_class === group)"
            :key="c.case_id"
            class="card p-4 hover:shadow-md transition-shadow flex flex-col"
          >
            <!-- 头部 -->
            <div class="flex items-start justify-between">
              <div class="flex items-center min-w-0">
                <FileText class="w-4 h-4 text-gray-400 mr-2 flex-shrink-0" />
                <code class="text-xs font-mono text-gray-600 truncate">
                  {{ c.case_id }}
                </code>
              </div>
              <span
                v-if="c.calibration_status"
                class="ml-2 text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-600 flex-shrink-0"
              >
                {{ c.calibration_status }}
              </span>
            </div>

            <!-- 标题 -->
            <h4 class="mt-2 text-base font-medium text-gray-900">
              {{ c.title ?? c.case_id }}
            </h4>

            <!-- 描述 -->
            <p
              v-if="c.description"
              class="mt-1 text-sm text-gray-500 line-clamp-2 flex-1"
            >
              {{ c.description }}
            </p>

            <!-- tags -->
            <div v-if="c.tags.length > 0" class="mt-2 flex flex-wrap gap-1">
              <span
                v-for="tag in c.tags"
                :key="tag"
                class="text-xs px-2 py-0.5 rounded bg-gray-50 text-gray-600 border border-gray-200"
              >
                {{ tag }}
              </span>
            </div>

            <!-- 发起按钮 -->
            <button
              class="btn-primary mt-4 w-full"
              :disabled="creatingCaseId !== null"
              @click="handleCreate(c)"
            >
              <Loader2
                v-if="creatingCaseId === c.case_id"
                class="w-4 h-4 mr-1.5 animate-spin"
              />
              发起评审
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
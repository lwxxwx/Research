<!-- frontend/src/pages/TaskListPage.vue -->
<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import { AlertCircle, ChevronLeft, ChevronRight, RefreshCw } from "@lucide/vue";

import { listTasks } from "@/api/review";
import { useTaskStore } from "@/stores/task";
import StatusBadge from "@/components/StatusBadge.vue";
import LoadingSpinner from "@/components/LoadingSpinner.vue";
import type { TaskListItem } from "@/types/task";

const taskStore = useTaskStore();

// ===== 分页 / 筛选 =====
const PAGE_SIZE = 20;
const currentPage = ref(0); // 0-based
const caseClassFilter = ref<"" | "A" | "B">("");
const reviewStatusFilter = ref<
  "" | "AI_CONFIRMED" | "NEED_EXPERT_REVIEW" | "LOW_CONFIDENCE"
>("");

// ===== 列表数据（直接读 store，回退时刷新视图）=====
const items = computed(() => taskStore.taskList);
const total = computed(() => taskStore.taskListTotal);
const loading = computed(() => taskStore.taskListLoading);
const error = computed(() => taskStore.taskListError);

// ===== 分页计算 =====
const totalPages = computed(() =>
  Math.max(1, Math.ceil(total.value / PAGE_SIZE)),
);
const canPrev = computed(() => currentPage.value > 0);
const canNext = computed(() => currentPage.value < totalPages.value - 1);

/** 加载当前页 */
async function fetchPage() {
  taskStore.setTaskListLoading(true);
  taskStore.setTaskListError(null);
  try {
    const resp = await listTasks({
      case_class: caseClassFilter.value || undefined,
      review_status: reviewStatusFilter.value || undefined,
      limit: PAGE_SIZE,
      offset: currentPage.value * PAGE_SIZE,
    });
    taskStore.setTaskList(resp.items, resp.total);
  } catch (e) {
    taskStore.setTaskListError(e instanceof Error ? e.message : String(e));
  } finally {
    taskStore.setTaskListLoading(false);
  }
}

/** 筛选变化 → 重置到第 1 页 */
watch([caseClassFilter, reviewStatusFilter], () => {
  currentPage.value = 0;
  fetchPage();
});

/** 翻页 */
function prevPage() {
  if (canPrev.value) {
    currentPage.value -= 1;
    fetchPage();
  }
}
function nextPage() {
  if (canNext.value) {
    currentPage.value += 1;
    fetchPage();
  }
}

/** 清空筛选 */
function clearFilters() {
  caseClassFilter.value = "";
  reviewStatusFilter.value = "";
}

/** 时间格式化：ISO → 本地时间（YYYY-MM-DD HH:mm） */
function formatTime(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return iso;
  }
}

/** case_class 显示（None → 分类未知） */
function caseClassText(c: "A" | "B" | null): string {
  return c ?? "未知";
}

onMounted(fetchPage);
</script>

<template>
  <div class="space-y-6">
    <!-- ===== 页头 ===== -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-semibold text-gray-900">任务列表</h2>
        <p class="mt-1 text-sm text-gray-500">
          共 {{ total }} 个任务（当前页 {{ items.length }} 个）
        </p>
      </div>
      <button class="btn-secondary" :disabled="loading" @click="fetchPage">
        <RefreshCw class="w-4 h-4 mr-1.5" :class="{ 'animate-spin': loading }" />
        刷新
      </button>
    </div>

    <!-- ===== 筛选栏 ===== -->
    <div class="card p-4 flex flex-wrap items-end gap-4">
      <div>
        <label class="block text-xs font-medium text-gray-600 mb-1">
          Case 类别
        </label>
        <select
          v-model="caseClassFilter"
          class="block w-32 rounded-md border-gray-300 shadow-sm text-sm focus:border-blue-500 focus:ring-blue-500"
        >
          <option value="">全部</option>
          <option value="A">A 类</option>
          <option value="B">B 类</option>
        </select>
      </div>

      <div>
        <label class="block text-xs font-medium text-gray-600 mb-1">
          质量状态
        </label>
        <select
          v-model="reviewStatusFilter"
          class="block w-40 rounded-md border-gray-300 shadow-sm text-sm focus:border-blue-500 focus:ring-blue-500"
        >
          <option value="">全部</option>
          <option value="AI_CONFIRMED">AI 已确认</option>
          <option value="NEED_EXPERT_REVIEW">需专家复核</option>
          <option value="LOW_CONFIDENCE">低置信</option>
        </select>
      </div>

      <button
        v-if="caseClassFilter || reviewStatusFilter"
        class="btn-secondary text-xs"
        @click="clearFilters"
      >
        清空筛选
      </button>
    </div>

    <!-- ===== 错误 ===== -->
    <div
      v-if="error"
      class="card p-4 border-red-200 bg-red-50 flex items-start"
    >
      <AlertCircle class="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
      <div class="flex-1">
        <p class="text-sm font-medium text-red-800">加载任务列表失败</p>
        <p class="mt-1 text-sm text-red-700">{{ error }}</p>
      </div>
      <button class="btn-secondary ml-3" @click="fetchPage">重试</button>
    </div>

    <!-- ===== 加载中 ===== -->
    <LoadingSpinner v-if="loading" text="加载任务中..." />

    <!-- ===== 空列表 ===== -->
    <div
      v-else-if="items.length === 0"
      class="card p-12 text-center text-gray-500"
    >
      暂无任务。去
      <RouterLink to="/cases" class="text-blue-600 hover:underline">
        Case 列表
      </RouterLink>
      发起评审。
    </div>

    <!-- ===== 表格 ===== -->
    <div v-else class="card overflow-hidden">
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              task_id
            </th>
            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              case_id
            </th>
            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              类别
            </th>
            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
              缺陷数
            </th>
            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              质量状态
            </th>
            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              创建时间
            </th>
            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
              操作
            </th>
          </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
          <tr
            v-for="item in items"
            :key="item.task_id"
            class="hover:bg-gray-50 transition-colors"
          >
            <td class="px-4 py-3 whitespace-nowrap">
              <code class="text-xs font-mono text-gray-700">{{ item.task_id }}</code>
            </td>
            <td class="px-4 py-3 whitespace-nowrap">
              <code class="text-xs font-mono text-gray-600">
                {{ item.case_id ?? "—" }}
              </code>
            </td>
            <td class="px-4 py-3 whitespace-nowrap">
              <span
                class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                :class="
                  item.case_class === 'A'
                    ? 'bg-blue-100 text-blue-700'
                    : item.case_class === 'B'
                      ? 'bg-purple-100 text-purple-700'
                      : 'bg-gray-100 text-gray-600'
                "
              >
                {{ caseClassText(item.case_class) }}
              </span>
            </td>
            <td class="px-4 py-3 whitespace-nowrap text-right text-sm text-gray-900 font-medium">
              {{ item.defects_count }}
            </td>
            <td class="px-4 py-3 whitespace-nowrap">
              <StatusBadge :status="item.review_status" />
            </td>
            <td class="px-4 py-3 whitespace-nowrap text-xs text-gray-500">
              {{ formatTime(item.created_at) }}
            </td>
            <td class="px-4 py-3 whitespace-nowrap text-right">
              <RouterLink
                :to="{ name: 'task-detail', params: { taskId: item.task_id } }"
                class="text-sm text-blue-600 hover:text-blue-700 hover:underline"
              >
                查看详情 →
              </RouterLink>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ===== 分页 ===== -->
    <div v-if="!loading && total > 0" class="flex items-center justify-between">
      <span class="text-sm text-gray-600">
        第 {{ currentPage + 1 }} / {{ totalPages }} 页
      </span>
      <div class="flex items-center space-x-2">
        <button
          class="btn-secondary"
          :disabled="!canPrev || loading"
          @click="prevPage"
        >
          <ChevronLeft class="w-4 h-4 mr-1" />
          上一页
        </button>
        <button
          class="btn-secondary"
          :disabled="!canNext || loading"
          @click="nextPage"
        >
          下一页
          <ChevronRight class="w-4 h-4 ml-1" />
        </button>
      </div>
    </div>
  </div>
</template>
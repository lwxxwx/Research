<!-- frontend/src/pages/TaskDetailPage.vue -->
<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { AlertCircle, ArrowLeft, Loader2 } from "@lucide/vue";

import { getTask, getDefects } from "@/api/review";
import type { TaskDetail } from "@/types/task";
import type { DefectFromAPI } from "@/types/defect";
import DefectCard from "@/components/DefectCard.vue";
import FeedbackPanel from "@/components/FeedbackPanel.vue";
import StatusBadge from "@/components/StatusBadge.vue";

const route = useRoute();
const router = useRouter();
const taskId = route.params.taskId as string;

const task = ref<TaskDetail | null>(null);
const defects = ref<DefectFromAPI[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);

// 当前展开反馈面板的 defectId
const feedbackDefectId = ref<string | null>(null);

async function loadData() {
  loading.value = true;
  error.value = null;
  try {
    // 1. 先查 task (如果 404 则中断)
    const t = await getTask(taskId);
    task.value = t;

    // 2. 只有 done 且 report 存在才查 defects
    if (t.status === "done" && t.report) {
      const d = await getDefects(taskId);
      defects.value = d.items;
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

function toggleFeedback(id: string) {
  feedbackDefectId.value = feedbackDefectId.value === id ? null : id;
}

// ===== [S2-D] 专家判定更新后，用响应体更新本地 defects 数组（不重新拉全量） =====
function onExpertDecisionUpdated(defectId: string, response: {
  expert_decision: string;
  expert_decision_updated_at: string;
  expert_decision_by: number | null;
}) {
  const idx = defects.value.findIndex((d) => d.defect_id === defectId);
  if (idx >= 0) {
    defects.value[idx] = {
      ...defects.value[idx],
      expert_decision: response.expert_decision as any,
      expert_decision_updated_at: response.expert_decision_updated_at,
      expert_decision_by: response.expert_decision_by,
    };
  }
}
// ===== [S2-D END] =====

onMounted(loadData);
watch(() => route.params.taskId, (newId) => {
  if (newId) {
    taskId = newId as string;
    loadData();
  }
});
</script>

<template>
  <div class="space-y-6">
    <!-- 返回按钮 -->
    <div>
      <button class="flex items-center text-sm text-gray-500 hover:text-gray-900" @click="router.push('/tasks')">
        <ArrowLeft class="w-4 h-4 mr-1" /> 返回任务列表
      </button>
    </div>

    <!-- 加载中 -->
    <div v-if="loading" class="flex items-center justify-center py-20 text-gray-500">
      <Loader2 class="w-6 h-6 animate-spin mr-2 text-blue-600" />
      加载任务详情中...
    </div>

    <!-- 错误 -->
    <div v-else-if="error" class="card p-6 border-red-200 bg-red-50 text-center">
      <AlertCircle class="w-12 h-12 text-red-500 mx-auto mb-3" />
      <h3 class="text-lg font-medium text-red-900">加载失败</h3>
      <p class="text-sm text-red-700 mt-1">{{ error }}</p>
      <button class="btn-secondary mt-4" @click="loadData">重试</button>
    </div>

    <!-- 详情主体 -->
    <template v-else-if="task">
      <!-- 头部摘要 -->
      <div class="card p-6 border-l-4 border-blue-500">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div class="flex items-center space-x-3 mb-1">
              <h1 class="text-2xl font-bold text-gray-900">任务详情</h1>
              <StatusBadge :status="task.status === 'done' ? 'AI_CONFIRMED' : 'LOW_CONFIDENCE'" />
            </div>
            <code class="text-sm font-mono text-gray-600 block">{{ task.task_id }}</code>
          </div>
          <div class="flex items-center space-x-6 text-sm text-gray-600 bg-gray-50 px-4 py-2 rounded-lg">
            <div>
              <span class="block text-xs text-gray-500">Case ID</span>
              <span class="font-medium">{{ task.case_id ?? '未知' }}</span>
            </div>
            <div>
              <span class="block text-xs text-gray-500">缺陷数</span>
              <span class="font-medium text-blue-600">{{ task.defects_count }}</span>
            </div>
            <div>
              <span class="block text-xs text-gray-500">创建时间</span>
              <span>{{ task.created_at ? new Date(task.created_at).toLocaleString() : '—' }}</span>
            </div>
          </div>
        </div>

        <!-- 三态分布 (如果有) -->
        <div v-if="task.report?.summary?.review_status_distribution" class="mt-4 pt-4 border-t border-gray-100 flex gap-6">
          <div class="text-sm"><span class="text-gray-500">AI 已确认：</span><span class="font-bold text-emerald-600">{{ task.report.summary.review_status_distribution.AI_CONFIRMED }}</span></div>
          <div class="text-sm"><span class="text-gray-500">需专家复核：</span><span class="font-bold text-amber-600">{{ task.report.summary.review_status_distribution.NEED_EXPERT_REVIEW }}</span></div>
          <div class="text-sm"><span class="text-gray-500">低置信：</span><span class="font-bold text-red-600">{{ task.report.summary.review_status_distribution.LOW_CONFIDENCE }}</span></div>
        </div>
      </div>

      <!-- 失败提示 (如果 status=failed) -->
      <div v-if="task.status === 'failed'" class="card p-6 border-red-200 bg-red-50">
        <h3 class="text-lg font-medium text-red-900 flex items-center">
          <AlertCircle class="w-5 h-5 mr-2" /> 评审执行失败
        </h3>
        <ul class="list-disc list-inside mt-2 text-sm text-red-700 space-y-1">
          <li v-for="(e, i) in task.errors" :key="i">{{ e }}</li>
        </ul>
      </div>

      <!-- 报告为空 -->
      <div v-else-if="!task.report || Object.keys(task.report).length === 0" class="card p-12 text-center text-gray-500">
        报告未生成或为空。
      </div>

      <!-- 缺陷列表 -->
      <div v-else class="space-y-4">
        <h3 class="text-lg font-medium text-gray-900 border-b pb-2">缺陷明细 ({{ defects.length }})</h3>
        <div v-if="defects.length === 0" class="text-gray-500 text-center py-8">无缺陷。</div>
        
        <div v-else class="space-y-4">
          <div v-for="d in defects" :key="d.defect_id" class="relative">
            <DefectCard
              :defect="d"
              :task-id="task.task_id"
              @expert-decision-updated="onExpertDecisionUpdated"
            />
            <!-- 快速反馈入口 -->
            <div class="mt-2 text-right">
              <button @click="toggleFeedback(d.defect_id)" class="text-sm text-blue-600 hover:text-blue-800 underline">
                {{ feedbackDefectId === d.defect_id ? '收起反馈' : '快速反馈' }}
              </button>
            </div>
            <!-- 反馈面板 -->
            <FeedbackPanel 
              v-if="feedbackDefectId === d.defect_id" 
              :task-id="task.task_id" 
              :defect-id="d.defect_id" 
              @success="toggleFeedback(d.defect_id)"
            />
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
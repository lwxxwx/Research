<!-- frontend/src/components/FeedbackPanel.vue -->
<script setup lang="ts">
import { ref } from "vue";
import { AlertCircle, CheckCircle, Loader2 } from "@lucide/vue";

import { submitFeedback } from "@/api/feedback";
import type { FeedbackType } from "@/types/feedback";

const props = defineProps<{
  taskId: string;
  defectId: string;
}>();

const emit = defineEmits<{
  (e: "success", feedbackId: number): void;
}>();

const feedbackType = ref<FeedbackType>("correct_defect");
const expertSuggestion = ref("");
const suggestionAdopted = ref<"yes" | "partial" | "no" | "">("");
const suggestionDiffText = ref("");
const ruleCandidateText = ref("");
const submitting = ref(false);
const error = ref<string | null>(null);
const successId = ref<number | null>(null);

const FEEDBACK_TYPES: { value: FeedbackType; label: string; desc: string }[] = [
  { value: "correct_defect", label: "确认缺陷", desc: "缺陷正确，建议可用" },
  { value: "false_positive", label: "误报", desc: "系统报错了" },
  { value: "false_negative", label: "漏检", desc: "系统没报，专家发现了" },
  { value: "suggestion_update", label: "修改建议", desc: "缺陷对，但建议不准" },
  { value: "new_rule_candidate", label: "新规则候选", desc: "建议加规则" },
  { value: "knowledge_gap", label: "知识缺失", desc: "缺 Datasheet 等" },
];

async function handleSubmit() {
  error.value = null;
  successId.value = null;
  submitting.value = true;

  try {
    // 处理 JSON 字段
    let suggestion_diff = null;
    if (feedbackType.value === "suggestion_update" && suggestionDiffText.value.trim()) {
      try { suggestion_diff = JSON.parse(suggestionDiffText.value); } 
      catch { throw new Error("建议对比的 JSON 格式错误"); }
    }

    let rule_candidate = null;
    if ((feedbackType.value === "new_rule_candidate" || feedbackType.value === "false_negative") && ruleCandidateText.value.trim()) {
      try { rule_candidate = JSON.parse(ruleCandidateText.value); } 
      catch { throw new Error("规则候选的 JSON 格式错误"); }
    }

    const resp = await submitFeedback({
      task_id: props.taskId,
      defect_id: props.defectId,
      feedback_type: feedbackType.value,
      expert_suggestion: expertSuggestion.value || null,
      suggestion_adopted: feedbackType.value === "suggestion_update" && suggestionAdopted.value ? suggestionAdopted.value : null,
      suggestion_diff,
      rule_candidate,
    });

    successId.value = resp.id;
    emit("success", resp.id);
    
    // 成功后清空部分输入
    expertSuggestion.value = "";
    suggestionDiffText.value = "";
    ruleCandidateText.value = "";
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <div class="card p-4 bg-gray-50 border-gray-200 mt-2">
    <h4 class="text-sm font-semibold text-gray-800 mb-3">提交专家反馈</h4>

    <!-- 成功提示 -->
    <div v-if="successId" class="mb-4 p-3 bg-emerald-50 border border-emerald-200 rounded flex items-start">
      <CheckCircle class="w-5 h-5 text-emerald-600 mt-0.5 mr-2 flex-shrink-0" />
      <p class="text-sm text-emerald-800">反馈已提交（ID: {{ successId }}）</p>
    </div>

    <!-- 错误提示 -->
    <div v-if="error" class="mb-4 p-3 bg-red-50 border border-red-200 rounded flex items-start">
      <AlertCircle class="w-5 h-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />
      <p class="text-sm text-red-800">{{ error }}</p>
    </div>

    <div class="space-y-4">
      <!-- 类型选择 -->
      <div>
        <label class="block text-xs font-medium text-gray-600 mb-1">反馈类型</label>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-2">
          <label
            v-for="t in FEEDBACK_TYPES" :key="t.value"
            class="flex items-start p-2 border rounded cursor-pointer hover:bg-white"
            :class="feedbackType === t.value ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500' : 'border-gray-200 bg-white'"
          >
            <input type="radio" :value="t.value" v-model="feedbackType" class="mt-1 mr-2" />
            <div>
              <span class="text-sm font-medium text-gray-900">{{ t.label }}</span>
              <p class="text-xs text-gray-500">{{ t.desc }}</p>
            </div>
          </label>
        </div>
      </div>

      <!-- 专家建议 -->
      <div>
        <label class="block text-xs font-medium text-gray-600 mb-1">专家建议（可选）</label>
        <textarea v-model="expertSuggestion" rows="2" class="w-full rounded border-gray-300 text-sm focus:border-blue-500 focus:ring-blue-500" placeholder="填写修改建议或说明..."></textarea>
      </div>

      <!-- 建议采纳（仅 suggestion_update） -->
      <div v-if="feedbackType === 'suggestion_update'">
        <label class="block text-xs font-medium text-gray-600 mb-1">是否采纳原建议</label>
        <select v-model="suggestionAdopted" class="w-full rounded border-gray-300 text-sm">
          <option value="">请选择</option>
          <option value="yes">全部采纳</option>
          <option value="partial">部分采纳</option>
          <option value="no">不采纳</option>
        </select>
        <label class="block text-xs font-medium text-gray-600 mb-1 mt-2">建议对比 (JSON)</label>
        <textarea v-model="suggestionDiffText" rows="2" class="w-full rounded border-gray-300 text-xs font-mono" placeholder='[{"old": "补加电容", "new": "补 100nF"}]'></textarea>
      </div>

      <!-- 规则候选（仅 new_rule_candidate / false_negative） -->
      <div v-if="feedbackType === 'new_rule_candidate' || feedbackType === 'false_negative'">
        <label class="block text-xs font-medium text-gray-600 mb-1">规则候选 (JSON)</label>
        <textarea v-model="ruleCandidateText" rows="3" class="w-full rounded border-gray-300 text-xs font-mono" placeholder='{"rule_id": "POWER_006", "rule_name": "热降额", ...}'></textarea>
      </div>

      <button class="btn-primary w-full" :disabled="submitting" @click="handleSubmit">
        <Loader2 v-if="submitting" class="w-4 h-4 mr-2 animate-spin" />
        提交反馈
      </button>
    </div>
  </div>
</template>
<!-- frontend/src/components/FeedbackPanel.vue -->
<script setup lang="ts">
import { ref } from "vue";
import { AlertCircle, CheckCircle, Loader2 } from "@lucide/vue";

import { submitFeedback } from "@/api/feedback";
import type { FeedbackType, SuggestionDiffItem, RuleCandidateInput } from "@/types/feedback";
// ===== [S2-E] 引入新表单组件 =====
import SuggestionDiffEditor from "./SuggestionDiffEditor.vue";
import RuleCandidateForm from "./RuleCandidateForm.vue";
// ===== [S2-E END] =====

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
// ===== [S2-E] 结构化状态替换 JSON 文本域 =====
const suggestionDiff = ref<SuggestionDiffItem[]>([{ old: "", new: "" }]);
const ruleCandidate = ref<RuleCandidateInput | null>(null);
// ===== [S2-E END] =====
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
    // ===== [S2-E] 表单校验 =====
    if (feedbackType.value === "new_rule_candidate") {
      if (!ruleCandidate.value?.title?.trim() || !ruleCandidate.value?.description?.trim()) {
        throw new Error("规则标题和描述不能为空");
      }
    }
    if (feedbackType.value === "suggestion_update") {
      const hasValidDiff = suggestionDiff.value.some(d => d.old.trim() || d.new.trim());
      if (!hasValidDiff) {
        throw new Error("请至少填写一组建议对比");
      }
    }
    // ===== [S2-E END] =====

    // 处理 diff 和 rule_candidate 负载（过滤掉全空的行）
    const finalDiff = feedbackType.value === "suggestion_update"
      ? suggestionDiff.value.filter(d => d.old.trim() || d.new.trim())
      : null;

    const finalRuleCandidate = (feedbackType.value === "new_rule_candidate" || feedbackType.value === "false_negative")
      ? ruleCandidate.value
      : null;

    const resp = await submitFeedback({
      task_id: props.taskId,
      defect_id: props.defectId,
      feedback_type: feedbackType.value,
      expert_suggestion: expertSuggestion.value || null,
      suggestion_adopted: feedbackType.value === "suggestion_update" && suggestionAdopted.value ? suggestionAdopted.value : null,
      suggestion_diff: finalDiff as any,
      rule_candidate: finalRuleCandidate as any,
    });

    successId.value = resp.id;
    emit("success", resp.id);
    
    // 成功后清空输入
    expertSuggestion.value = "";
    suggestionDiff.value = [{ old: "", new: "" }];
    ruleCandidate.value = null;
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

    <div v-if="successId" class="mb-4 p-3 bg-emerald-50 border border-emerald-200 rounded flex items-start">
      <CheckCircle class="w-5 h-5 text-emerald-600 mt-0.5 mr-2 flex-shrink-0" />
      <p class="text-sm text-emerald-800">反馈已提交（ID: {{ successId }}）</p>
    </div>

    <div v-if="error" class="mb-4 p-3 bg-red-50 border border-red-200 rounded flex items-start">
      <AlertCircle class="w-5 h-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />
      <p class="text-sm text-red-800">{{ error }}</p>
    </div>

    <div class="space-y-4">
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

      <div>
        <label class="block text-xs font-medium text-gray-600 mb-1">专家建议（可选）</label>
        <textarea v-model="expertSuggestion" rows="2" class="w-full rounded border-gray-300 text-sm focus:border-blue-500 focus:ring-blue-500" placeholder="填写修改建议或说明..."></textarea>
      </div>

      <!-- ===== [S2-E] 建议修改对比（结构化 UI） ===== -->
      <div v-if="feedbackType === 'suggestion_update'">
        <label class="block text-xs font-medium text-gray-600 mb-1">是否采纳原建议</label>
        <select v-model="suggestionAdopted" class="w-full rounded border-gray-300 text-sm mb-3">
          <option value="">请选择</option>
          <option value="yes">全部采纳</option>
          <option value="partial">部分采纳</option>
          <option value="no">不采纳</option>
        </select>
        <label class="block text-xs font-medium text-gray-600 mb-1">建议对比</label>
        <SuggestionDiffEditor v-model="suggestionDiff" />
      </div>

      <!-- ===== [S2-E] 规则候选表单（结构化 UI） ===== -->
      <div v-if="feedbackType === 'new_rule_candidate' || feedbackType === 'false_negative'">
        <label class="block text-xs font-medium text-gray-600 mb-1">规则候选草稿</label>
        <RuleCandidateForm v-model="ruleCandidate" />
      </div>

      <button class="btn-primary w-full" :disabled="submitting" @click="handleSubmit">
        <Loader2 v-if="submitting" class="w-4 h-4 mr-2 animate-spin" />
        提交反馈
      </button>
    </div>
  </div>
</template>
<!-- frontend/src/components/RuleCandidateForm.vue -->
<script setup lang="ts">
import type { RuleCandidateInput } from "@/types/feedback";

const props = defineProps<{
  modelValue: RuleCandidateInput | null;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", val: RuleCandidateInput | null): void;
}>();

function updateField(field: keyof RuleCandidateInput, val: string) {
  emit("update:modelValue", {
    title: props.modelValue?.title ?? "",
    description: props.modelValue?.description ?? "",
    severity_hint: props.modelValue?.severity_hint ?? "",
    [field]: val,
  });
}
</script>

<template>
  <div class="space-y-3 bg-gray-50 p-3 rounded border border-gray-200">
    <div>
      <label class="block text-xs font-medium text-gray-700 mb-1">规则标题 <span class="text-red-500">*</span></label>
      <input
        type="text"
        :value="modelValue?.title ?? ''"
        class="w-full rounded border-gray-300 text-sm"
        placeholder="例如：Buck 热降额阈值规则 POWER_006"
        @input="updateField('title', ($event.target as HTMLInputElement).value)"
      />
    </div>
    
    <div>
      <label class="block text-xs font-medium text-gray-700 mb-1">规则描述 <span class="text-red-500">*</span></label>
      <textarea
        :value="modelValue?.description ?? ''"
        rows="3"
        class="w-full rounded border-gray-300 text-sm"
        placeholder="描述规则的适用条件和检查逻辑..."
        @input="updateField('description', ($event.target as HTMLTextAreaElement).value)"
      ></textarea>
    </div>

    <div>
      <label class="block text-xs font-medium text-gray-700 mb-1">严重度提示（可选）</label>
      <select
        :value="modelValue?.severity_hint ?? ''"
        class="w-full rounded border-gray-300 text-sm"
        @change="updateField('severity_hint', ($event.target as HTMLSelectElement).value)"
      >
        <option value="">请选择</option>
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
        <option value="critical">Critical</option>
      </select>
    </div>
  </div>
</template>
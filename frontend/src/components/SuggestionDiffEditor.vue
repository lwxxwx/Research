<!-- frontend/src/components/SuggestionDiffEditor.vue -->
<script setup lang="ts">
import { Plus, Trash2 } from "@lucide/vue";
import type { SuggestionDiffItem } from "@/types/feedback";

const props = defineProps<{
  modelValue: SuggestionDiffItem[];
}>();

const emit = defineEmits<{
  (e: "update:modelValue", val: SuggestionDiffItem[]): void;
}>();

function addItem() {
  emit("update:modelValue", [...props.modelValue, { old: "", new: "" }]);
}

function removeItem(index: number) {
  const copy = [...props.modelValue];
  copy.splice(index, 1);
  emit("update:modelValue", copy);
}

function updateItem(index: number, field: "old" | "new", val: string) {
  const copy = [...props.modelValue];
  copy[index] = { ...copy[index], [field]: val };
  emit("update:modelValue", copy);
}
</script>

<template>
  <div class="space-y-3">
    <div
      v-for="(item, index) in modelValue"
      :key="index"
      class="flex gap-3 items-start bg-gray-50 p-3 rounded border border-gray-200"
    >
      <div class="flex-1 space-y-2">
        <div>
          <label class="block text-xs text-gray-500 mb-1">旧建议</label>
          <textarea
            :value="item.old"
            rows="2"
            class="w-full rounded border-gray-300 text-xs bg-white"
            placeholder="修改前的建议..."
            @input="updateItem(index, 'old', ($event.target as HTMLTextAreaElement).value)"
          ></textarea>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1">新建议</label>
          <textarea
            :value="item.new"
            rows="2"
            class="w-full rounded border-gray-300 text-xs bg-white"
            placeholder="修改后的建议..."
            @input="updateItem(index, 'new', ($event.target as HTMLTextAreaElement).value)"
          ></textarea>
        </div>
      </div>
      <button
        type="button"
        class="mt-6 p-1.5 text-red-500 hover:bg-red-50 rounded disabled:opacity-30 disabled:cursor-not-allowed"
        :disabled="modelValue.length <= 1"
        title="删除这组对比"
        @click="removeItem(index)"
      >
        <Trash2 class="w-4 h-4" />
      </button>
    </div>

    <button type="button" class="text-sm text-blue-600 hover:text-blue-800 flex items-center" @click="addItem">
      <Plus class="w-4 h-4 mr-1" /> 添加一组对比
    </button>
  </div>
</template>
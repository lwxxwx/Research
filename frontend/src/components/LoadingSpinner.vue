<!-- frontend/src/components/LoadingSpinner.vue -->
<script setup lang="ts">
import { Loader2 } from "@lucide/vue";

withDefaults(
  defineProps<{
    /** 提示文本，传空字符串则不显示文本 */
    text?: string;
    /** 尺寸 */
    size?: "sm" | "md" | "lg";
  }>(),
  { text: "加载中...", size: "md" },
);

const sizeClass = {
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-8 h-8",
} as const;
</script>

<template>
  <div class="flex items-center justify-center py-8">
    <Loader2 class="animate-spin text-blue-600" :class="sizeClass[size]" />
    <span v-if="text" class="ml-2 text-gray-600">{{ text }}</span>
  </div>
</template>
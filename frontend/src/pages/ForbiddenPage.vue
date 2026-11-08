<!-- frontend/src/pages/ForbiddenPage.vue -->
<script setup lang="ts">
import { useRouter } from "vue-router";
const router = useRouter();
</script>

<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center px-4">
    <div class="text-center">
      <h1 class="text-6xl font-bold text-gray-300 mb-4">403</h1>
      <p class="text-lg text-gray-700 mb-6">权限不足，无法访问此页面</p>
      <button
        class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        @click="router.push('/cases')"
      >
        返回 Case 列表
      </button>
    </div>
  </div>
</template>
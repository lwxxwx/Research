<!-- frontend/src/pages/NotFoundPage.vue -->
<script setup lang="ts">
import { RouterLink } from "vue-router";
</script>

<template>
  <div class="text-center py-12">
    <h2 class="text-2xl font-semibold text-gray-900">404 页面不存在</h2>
    <p class="mt-2 text-gray-600">请检查 URL，或返回 Case 列表。</p>
    <RouterLink to="/cases" class="btn-primary mt-4 inline-flex">
      返回 Case 列表
    </RouterLink>
  </div>
</template>
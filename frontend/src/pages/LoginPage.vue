<!-- frontend/src/pages/LoginPage.vue -->
<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { useAuthStore } from "@/stores/auth";
import { UnauthorizedError } from "@/api/http";

const router = useRouter();
const authStore = useAuthStore();

const username = ref("");
const loading = ref(false);
const errorMsg = ref<string | null>(null);

async function handleLogin() {
  if (!username.value.trim() || loading.value) return;
  loading.value = true;
  errorMsg.value = null;
  try {
    await authStore.login(username.value.trim());
    const redirect = (router.currentRoute.value.query.redirect as string) || "/cases";
    router.push(redirect);
  } catch (e) {
    if (e instanceof UnauthorizedError) {
      errorMsg.value = "用户不存在或已禁用";
    } else {
      errorMsg.value = e instanceof Error ? e.message : String(e);
    }
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center px-4">
    <div class="w-full max-w-sm bg-white rounded-xl shadow-md p-8">
      <h1 class="text-2xl font-semibold text-center text-gray-900 mb-1">原理图评审系统</h1>
      <p class="text-sm text-gray-500 text-center mb-6">Sprint 3 · 身份模拟登录</p>

      <form @submit.prevent="handleLogin" class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">用户名</label>
          <input
            v-model="username"
            type="text"
            placeholder="请输入用户名（如 admin / eng1）"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            :disabled="loading"
            autofocus
          />
        </div>
        <button
          type="submit"
          class="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 disabled:bg-blue-300"
          :disabled="loading || !username.trim()"
        >
          {{ loading ? "登录中..." : "登录" }}
        </button>
      </form>

      <p v-if="errorMsg" class="mt-4 text-sm text-red-600 text-center">{{ errorMsg }}</p>
      <p class="mt-6 text-xs text-gray-400 text-center">
        无密码登录（开发 / 内网测试专用）
      </p>
    </div>
  </div>
</template>
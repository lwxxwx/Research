<!-- frontend/src/App.vue -->
<script setup lang="ts">
import { RouterLink, RouterView, useRoute, useRouter } from "vue-router";
import { computed } from "vue";
import { FileText, ListChecks, Sparkles, Database, BarChart3, LogOut, User } from "@lucide/vue";
import { useAuthStore } from "@/stores/auth";

const route = useRoute();
const router = useRouter();
const authStore = useAuthStore();

const isCasesActive = computed(() => route.path.startsWith("/cases"));
const isTasksActive = computed(() => route.path.startsWith("/tasks"));
const isRuleCandidatesActive = computed(() => route.path.startsWith("/rule-candidates"));
const isKnowledgeBaseActive = computed(() => route.path.startsWith("/knowledge-base"));
const isDashboardActive = computed(() => route.path.startsWith("/dashboard"));

// 登录页 / 403 页不显示顶部导航
const showHeader = computed(() => !["/login", "/403"].includes(route.path));

// 角色 → 菜单可见性
const canSeeTasks = computed(() =>
  ["engineer", "reviewer", "admin"].includes(authStore.role || "")
);
const canSeeRuleCandidates = computed(() =>
  ["rule_admin", "admin"].includes(authStore.role || "")
);
const canSeeKnowledgeBase = computed(() =>
  ["rule_admin", "admin"].includes(authStore.role || "")
);
const canSeeDashboard = computed(() =>
  ["admin", "rule_admin", "reviewer"].includes(authStore.role || "")
);

function handleLogout() {
  authStore.logout();
  router.push("/login");
}
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <!-- ===== 顶部导航栏（仅登录后显示）===== -->
    <header v-if="showHeader" class="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <!-- Logo + 标题 -->
          <div class="flex items-center space-x-3">
            <FileText class="w-6 h-6 text-blue-600" />
            <h1 class="text-lg font-semibold text-gray-900">原理图评审系统</h1>
            <span class="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">Sprint 3</span>
          </div>

          <!-- 导航链接（按角色显示）-->
          <nav class="flex items-center space-x-1">
            <RouterLink
              to="/cases"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="isCasesActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'"
            >
              <FileText class="w-4 h-4 mr-1.5" />Case 列表
            </RouterLink>

            <RouterLink
              v-if="canSeeTasks"
              to="/tasks"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="isTasksActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'"
            >
              <ListChecks class="w-4 h-4 mr-1.5" />任务列表
            </RouterLink>

            <RouterLink
              v-if="canSeeRuleCandidates"
              to="/rule-candidates"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="isRuleCandidatesActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'"
            >
              <Sparkles class="w-4 h-4 mr-1.5" />规则候选
            </RouterLink>

            <RouterLink
              v-if="canSeeKnowledgeBase"
              to="/knowledge-base"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="isKnowledgeBaseActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'"
            >
              <Database class="w-4 h-4 mr-1.5" />知识库
            </RouterLink>

            <RouterLink
              v-if="canSeeDashboard"
              to="/dashboard"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="isDashboardActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'"
            >
              <BarChart3 class="w-4 h-4 mr-1.5" />看板
            </RouterLink>
          </nav>

          <!-- 用户信息 + 退出 -->
          <div v-if="authStore.isAuthenticated" class="flex items-center space-x-3 ml-4">
            <div class="flex items-center text-sm text-gray-700">
              <User class="w-4 h-4 mr-1" />
              <span class="font-medium">{{ authStore.user?.username }}</span>
              <span class="ml-2 text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded">
                {{ authStore.user?.role }}
              </span>
            </div>
            <button
              class="text-sm text-gray-600 hover:text-gray-900 flex items-center"
              @click="handleLogout"
            >
              <LogOut class="w-4 h-4 mr-1" />退出
            </button>
          </div>
        </div>
      </div>
    </header>

    <!-- ===== 主内容区 ===== -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <RouterView />
    </main>
  </div>
</template>
<!-- frontend/src/App.vue -->
<script setup lang="ts">
import { RouterLink, RouterView, useRoute } from "vue-router";
import { computed } from "vue";
// ===== [S2-G / S2-H] 引入图标 =====
import { FileText, ListChecks, Sparkles, Database } from "@lucide/vue";

const route = useRoute();

// 导航高亮：当前路由是否在某前缀下
const isCasesActive = computed(() => route.path.startsWith("/cases"));
const isTasksActive = computed(() => route.path.startsWith("/tasks"));
// ===== [S2-G] 规则候选路由高亮 =====
const isRuleCandidatesActive = computed(() => route.path.startsWith("/rule-candidates"));
// ===== [S2-H] 知识库路由高亮 =====
const isKnowledgeBaseActive = computed(() => route.path.startsWith("/knowledge-base"));
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <!-- ===== 顶部导航栏 ===== -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <!-- Logo + 标题 -->
          <div class="flex items-center space-x-3">
            <FileText class="w-6 h-6 text-blue-600" />
            <h1 class="text-lg font-semibold text-gray-900">
              原理图评审系统
            </h1>
            <span class="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
              Sprint 2
            </span>
          </div>

          <!-- 导航链接 -->
          <nav class="flex items-center space-x-1">
            <RouterLink
              to="/cases"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="
                isCasesActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              "
            >
              <FileText class="w-4 h-4 mr-1.5" />
              Case 列表
            </RouterLink>
            <RouterLink
              to="/tasks"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="
                isTasksActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              "
            >
              <ListChecks class="w-4 h-4 mr-1.5" />
              任务列表
            </RouterLink>
            <!-- ===== [S2-G] 规则候选导航 ===== -->
            <RouterLink
              to="/rule-candidates"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="
                isRuleCandidatesActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              "
            >
              <Sparkles class="w-4 h-4 mr-1.5" />
              规则候选
            </RouterLink>
            <!-- ===== [S2-G END] ===== -->
            <!-- ===== [S2-H] 知识库导航 ===== -->
            <RouterLink
              to="/knowledge-base"
              class="flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors"
              :class="
                isKnowledgeBaseActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              "
            >
              <Database class="w-4 h-4 mr-1.5" />
              知识库
            </RouterLink>
            <!-- ===== [S2-H END] ===== -->
          </nav>
        </div>
      </div>
    </header>

    <!-- ===== 主内容区 ===== -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <RouterView />
    </main>
  </div>
</template>
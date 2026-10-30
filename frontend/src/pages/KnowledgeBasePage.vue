<!-- frontend/src/pages/KnowledgeBasePage.vue -->
<script setup lang="ts">
import { onMounted, ref } from "vue";
import { AlertCircle, CheckCircle, Database, FileUp, Loader2, RefreshCw } from "@lucide/vue";

import { listKnowledgeDocs, uploadKnowledgeDoc } from "@/api/knowledge";
import type { KnowledgeDocItem } from "@/types/knowledge";

const docs = ref<KnowledgeDocItem[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);

// 上传状态
const isUploading = ref(false);
const uploadError = ref<string | null>(null);
const uploadSuccess = ref<string | null>(null);

async function loadDocs() {
  loading.value = true;
  error.value = null;
  try {
    const resp = await listKnowledgeDocs();
    docs.value = resp.items;
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

async function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  if (!file.name.endsWith(".md")) {
    uploadError.value = "仅支持 .md 格式的 Markdown 文件";
    target.value = "";
    return;
  }

  isUploading.value = true;
  uploadError.value = null;
  uploadSuccess.value = null;

  try {
    const resp = await uploadKnowledgeDoc(file);
    uploadSuccess.value = `上传成功！文档「${resp.title}」已切分为 ${resp.chunk_count} 个分片并入库。`;
    await loadDocs(); // 刷新列表
  } catch (e) {
    uploadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    isUploading.value = false;
    target.value = ""; // 清空 input
  }
}

onMounted(loadDocs);
</script>

<template>
  <div class="space-y-6 max-w-5xl mx-auto">
    <!-- 页头 -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-semibold text-gray-900 flex items-center">
          <Database class="w-6 h-6 mr-2 text-blue-600" /> RAG 知识库管理
        </h2>
        <p class="mt-1 text-sm text-gray-500">
          上传 Datasheet 或设计规范 Markdown 文件，系统会自动切片并向量化，供 AI 评审时检索。
        </p>
      </div>
      <div class="flex items-center space-x-3">
        <button class="btn-secondary" :disabled="loading || isUploading" @click="loadDocs">
          <RefreshCw class="w-4 h-4 mr-1.5" :class="{ 'animate-spin': loading }" /> 刷新
        </button>
        
        <!-- 隐藏的 input，通过 label 触发 -->
        <label class="btn-primary flex items-center cursor-pointer" :class="isUploading ? 'opacity-70 cursor-not-allowed' : ''">
          <Loader2 v-if="isUploading" class="w-4 h-4 mr-2 animate-spin" />
          <FileUp v-else class="w-4 h-4 mr-2" />
          {{ isUploading ? '解析入库中...' : '上传 Markdown' }}
          <input type="file" accept=".md" class="hidden" @change="handleFileUpload" :disabled="isUploading" />
        </label>
      </div>
    </div>

    <!-- 错误与成功提示 -->
    <div v-if="uploadError" class="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
      <AlertCircle class="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
      <p class="text-sm text-red-800">{{ uploadError }}</p>
    </div>
    <div v-if="uploadSuccess" class="p-4 bg-emerald-50 border border-emerald-200 rounded-lg flex items-start">
      <CheckCircle class="w-5 h-5 text-emerald-600 mt-0.5 mr-3 flex-shrink-0" />
      <p class="text-sm text-emerald-800">{{ uploadSuccess }}</p>
    </div>

    <!-- 列表加载中 -->
    <div v-if="loading" class="flex items-center justify-center py-16 text-gray-500">
      <Loader2 class="w-6 h-6 animate-spin mr-2 text-blue-600" /> 加载知识库列表中...
    </div>
    
    <!-- 列表错误 -->
    <div v-else-if="error" class="card p-6 border-red-200 bg-red-50 text-center">
      <p class="text-sm text-red-700">{{ error }}</p>
      <button class="btn-secondary mt-4" @click="loadDocs">重试</button>
    </div>

    <!-- 数据表格 -->
    <div v-else class="card border border-gray-200 overflow-hidden">
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">文档标题</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">来源类型</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">入库时间</th>
          </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200 text-sm">
          <tr v-if="docs.length === 0">
            <td colspan="4" class="px-6 py-8 text-center text-gray-500">知识库为空</td>
          </tr>
          <tr v-for="d in docs" :key="d.id" class="hover:bg-gray-50">
            <td class="px-6 py-4 whitespace-nowrap text-gray-500 font-mono">{{ d.id }}</td>
            <td class="px-6 py-4 text-gray-900 font-medium">{{ d.title }}</td>
            <td class="px-6 py-4 whitespace-nowrap">
              <span class="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs border border-blue-100">
                {{ d.source_type ?? '未知' }}
              </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-gray-500">
              {{ d.created_at ? new Date(d.created_at).toLocaleString() : '-' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
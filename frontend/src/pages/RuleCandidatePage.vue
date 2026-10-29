<!-- frontend/src/pages/RuleCandidatePage.vue -->
<script setup lang="ts">
import { onMounted, ref, computed } from "vue";
import { AlertCircle, CheckCircle, Loader2, XCircle, FileCode2 } from "@lucide/vue";
import { listCandidates, getCandidate, updateCandidateStatus } from "@/api/rule_candidate";
import type { RuleCandidateListItem, RuleCandidateDetail, RuleCandidateStatus } from "@/types/rule_candidate";

const loading = ref(true);
const error = ref<string | null>(null);

const candidates = ref<RuleCandidateListItem[]>([]);
const currentStatus = ref<RuleCandidateStatus | "">("proposed");
const selectedId = ref<string | null>(null);
const detail = ref<RuleCandidateDetail | null>(null);
const detailLoading = ref(false);

// 驳回理由弹窗
const showRejectModal = ref(false);
const rejectReason = ref("");
const submitting = ref(false);

const statusTabs: { label: string; value: RuleCandidateStatus | "" }[] = [
  { label: "待审核 (proposed)", value: "proposed" },
  { label: "已批准 (accepted)", value: "accepted" },
  { label: "已驳回 (rejected)", value: "rejected" },
];

async function loadList() {
  loading.value = true;
  error.value = null;
  try {
    const res = await listCandidates(currentStatus.value || undefined);
    candidates.value = res.items;
    if (candidates.value.length > 0 && !selectedId.value) {
      selectCandidate(candidates.value[0].candidate_id);
    } else if (candidates.value.length === 0) {
      detail.value = null;
      selectedId.value = null;
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

async function selectCandidate(id: string) {
  selectedId.value = id;
  detailLoading.value = true;
  try {
    detail.value = await getCandidate(id);
  } catch (e) {
    alert(`加载详情失败: ${e}`);
  } finally {
    detailLoading.value = false;
  }
}

async function handleApprove() {
  if (!detail.value) return;
  if (!confirm("确定批准该候选规则？(批准后不会自动写入正式规则库，需人工合入)")) return;
  
  submitting.value = true;
  try {
    await updateCandidateStatus(detail.value.candidate_id, { status: "accepted" });
    alert("已批准");
    await loadList();
  } catch (e) {
    alert(`操作失败: ${e}`);
  } finally {
    submitting.value = false;
  }
}

async function handleReject() {
  if (!rejectReason.value.trim()) {
    alert("驳回必须填写理由");
    return;
  }
  if (!detail.value) return;

  submitting.value = true;
  try {
    await updateCandidateStatus(detail.value.candidate_id, {
      status: "rejected",
      rejected_reason: rejectReason.value
    });
    showRejectModal.value = false;
    rejectReason.value = "";
    await loadList();
  } catch (e) {
    alert(`操作失败: ${e}`);
  } finally {
    submitting.value = false;
  }
}

onMounted(loadList);
</script>

<template>
  <div class="space-y-6">
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">规则候选审核</h1>
        <p class="text-sm text-gray-500 mt-1">审核由专家反馈生成的规则草稿，批准后需人工合入正式 YAML 目录。</p>
      </div>
    </div>

    <div class="flex space-x-2 border-b border-gray-200 pb-2">
      <button
        v-for="tab in statusTabs" :key="tab.value"
        class="px-4 py-2 text-sm font-medium rounded-t-lg"
        :class="currentStatus === tab.value ? 'bg-white border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'"
        @click="currentStatus = tab.value; loadList()"
      >
        {{ tab.label }}
      </button>
    </div>

    <div class="flex flex-col md:flex-row gap-6">
      <!-- 左侧列表 -->
      <div class="w-full md:w-1/3 space-y-3">
        <div v-if="loading" class="flex items-center text-gray-500 py-4"><Loader2 class="w-4 h-4 animate-spin mr-2"/> 加载中...</div>
        <div v-else-if="candidates.length === 0" class="text-gray-500 text-sm py-4 text-center border rounded-lg bg-gray-50">暂无数据</div>
        
        <div v-else class="space-y-2 max-h-[70vh] overflow-y-auto pr-2">
          <div
            v-for="c in candidates" :key="c.candidate_id"
            class="p-3 border rounded-lg cursor-pointer transition-colors"
            :class="selectedId === c.candidate_id ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500' : 'border-gray-200 bg-white hover:bg-gray-50'"
            @click="selectCandidate(c.candidate_id)"
          >
            <div class="flex justify-between items-start mb-1">
              <span class="text-xs font-mono text-gray-500">{{ c.candidate_id }}</span>
              <span class="text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">{{ c.severity }}</span>
            </div>
            <div class="font-medium text-gray-900 text-sm line-clamp-1">{{ c.title }}</div>
            <div class="text-xs text-gray-500 mt-1 line-clamp-1">Case: {{ c.case_id ?? 'Unknown' }}</div>
          </div>
        </div>
      </div>

      <!-- 右侧详情 -->
      <div class="w-full md:w-2/3">
        <div v-if="detailLoading" class="card p-12 flex justify-center"><Loader2 class="w-8 h-8 animate-spin text-blue-500"/></div>
        
        <div v-else-if="detail" class="card p-6 space-y-6 bg-white border border-gray-200">
          <!-- 头部信息 -->
          <div class="flex justify-between items-start">
            <div>
              <h2 class="text-xl font-bold text-gray-900">{{ detail.title }}</h2>
              <p class="text-sm text-gray-500 mt-1">{{ detail.description }}</p>
              <div class="flex items-center space-x-3 mt-3 text-xs text-gray-600">
                <span>ID: <code class="bg-gray-100 px-1 rounded">{{ detail.candidate_id }}</code></span>
                <span v-if="detail.case_id">Case: {{ detail.case_id }}</span>
                <span v-if="detail.created_at">创建于: {{ new Date(detail.created_at).toLocaleString() }}</span>
              </div>
            </div>
            
            <div class="flex space-x-2" v-if="detail.status === 'proposed'">
              <button class="btn-primary flex items-center bg-emerald-600 hover:bg-emerald-700" :disabled="submitting" @click="handleApprove">
                <CheckCircle class="w-4 h-4 mr-1"/> 批准
              </button>
              <button class="btn-secondary flex items-center text-red-600 border-red-200 hover:bg-red-50" :disabled="submitting" @click="showRejectModal = true">
                <XCircle class="w-4 h-4 mr-1"/> 驳回
              </button>
            </div>
            <div v-else class="text-sm font-bold" :class="detail.status === 'accepted' ? 'text-emerald-600' : 'text-red-600'">
              {{ detail.status === 'accepted' ? '已批准' : '已驳回' }}
            </div>
          </div>

          <!-- YAML 预览 -->
          <div>
            <h3 class="text-sm font-semibold text-gray-700 mb-2 flex items-center">
              <FileCode2 class="w-4 h-4 mr-1"/> YAML 草稿预览
            </h3>
            <div class="bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre class="text-xs text-green-400 font-mono"><code>{{ detail.proposed_yaml || '# 无 YAML 内容' }}</code></pre>
            </div>
          </div>

          <!-- 元数据 -->
          <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span class="text-gray-500">来源反馈 ID：</span>
              <span class="font-mono">{{ detail.from_feedback_id ?? '—' }}</span>
            </div>
            <div>
              <span class="text-gray-500">Benchmark 状态：</span>
              <span v-if="detail.benchmarked_at" class="text-blue-600">已跑 ({{ new Date(detail.benchmarked_at).toLocaleString() }})</span>
              <span v-else class="text-gray-400">未跑</span>
            </div>
          </div>
          
          <div v-if="detail.rejected_reason" class="p-3 bg-red-50 text-red-800 text-sm rounded border border-red-100">
            <strong>驳回理由：</strong> {{ detail.rejected_reason }}
          </div>
        </div>
      </div>
    </div>

    <!-- 驳回弹窗 -->
    <div v-if="showRejectModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="bg-white rounded-lg p-6 w-96 max-w-full">
        <h3 class="text-lg font-bold mb-4">驳回候选规则</h3>
        <textarea v-model="rejectReason" rows="3" class="w-full rounded border-gray-300 text-sm mb-4" placeholder="请填写驳回理由..."></textarea>
        <div class="flex justify-end space-x-2">
          <button class="btn-secondary" @click="showRejectModal = false">取消</button>
          <button class="btn-primary bg-red-600 hover:bg-red-700" :disabled="submitting" @click="handleReject">确认驳回</button>
        </div>
      </div>
    </div>
  </div>
</template>
<template>
  <div class="dashboard-page">
    <div class="page-header">
      <div class="header-left">
        <h1>指标看板</h1>
        <p class="scope-hint">
          统计范围：仅 task_status = done 任务（排除 pending / failed）；规则候选与知识缺口独立统计。
        </p>
      </div>
      <button class="refresh-btn" :disabled="loading" @click="loadData">
        {{ loading ? "加载中…" : "刷新" }}
      </button>
    </div>

    <div v-if="data?.scope.has_seed_data" class="seed-warning">
      检测到被排除的 Seed 数据，真实指标已自动剔除 Seed 反馈。
    </div>

    <div v-if="loading && !data" class="state-block">加载中…</div>
    <div v-else-if="error" class="state-block error">{{ error }}</div>

    <template v-else-if="data">
      <!-- 4 格指标卡 -->
      <div class="metric-cards">
        <div class="metric-card">
          <div class="label">证据覆盖率 EVC</div>
          <div class="value" :class="{ na: data.metrics.evidence_coverage === null }">
            {{ formatPct(data.metrics.evidence_coverage) }}
          </div>
        </div>
        <div class="metric-card">
          <div class="label">AI 新缺陷有效率 AI_NER</div>
          <div class="value" :class="{ na: data.metrics.ai_new_effective_rate === null }">
            {{ formatPct(data.metrics.ai_new_effective_rate) }}
          </div>
          <div class="sub">
            样本 {{ data.metrics.sample_info.ai_ner_sample_size }} / 门槛
            {{ data.metrics.sample_info.ai_ner_sample_threshold }}
          </div>
        </div>
        <div class="metric-card">
          <div class="label">专家采纳率 EAR</div>
          <div class="value" :class="{ na: data.metrics.expert_adoption_rate === null }">
            {{ formatPct(data.metrics.expert_adoption_rate) }}
          </div>
          <div class="sub">
            样本 {{ data.metrics.sample_info.ear_sample_size }} / 门槛
            {{ data.metrics.sample_info.ear_sample_threshold }}
          </div>
        </div>
        <div class="metric-card">
          <div class="label">已统计任务数</div>
          <div class="value">{{ data.scope.total_done_tasks }}</div>
          <div class="sub">仅 done</div>
        </div>
      </div>

      <!-- 分布区 -->
      <div class="dist-grid">
        <section class="dist-card">
          <h2>AI 质量分布</h2>
          <table>
            <tbody>
              <tr>
                <td>AI 确认</td>
                <td>{{ data.metrics.review_status_distribution.AI_CONFIRMED ?? 0 }}</td>
              </tr>
              <tr>
                <td>需专家复核</td>
                <td>{{ data.metrics.review_status_distribution.NEED_EXPERT_REVIEW ?? 0 }}</td>
              </tr>
              <tr>
                <td>低置信</td>
                <td>{{ data.metrics.review_status_distribution.LOW_CONFIDENCE ?? 0 }}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section class="dist-card">
          <h2>专家判定分布</h2>
          <table>
            <tbody>
              <tr>
                <td>待判定</td>
                <td>{{ data.metrics.expert_decision_distribution.pending ?? 0 }}</td>
              </tr>
              <tr>
                <td>已确认</td>
                <td>{{ data.metrics.expert_decision_distribution.confirmed ?? 0 }}</td>
              </tr>
              <tr>
                <td>已驳回</td>
                <td>{{ data.metrics.expert_decision_distribution.rejected ?? 0 }}</td>
              </tr>
              <tr>
                <td>需进一步核验</td>
                <td>{{ data.metrics.expert_decision_distribution.needs_verification ?? 0 }}</td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>

      <!-- 规则候选 / 知识缺口 -->
      <div class="dist-grid">
        <section class="dist-card">
          <h2>规则候选（共 {{ data.rule_candidates.total }}）</h2>
          <table>
            <tbody>
              <tr v-for="(label, key) in ruleCandidateLabels" :key="key">
                <td>{{ label }}</td>
                <td>{{ data.rule_candidates.by_status[key] ?? 0 }}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section class="dist-card">
          <h2>知识缺口（共 {{ data.knowledge_gaps.total }}）</h2>
          <table>
            <tbody>
              <tr v-for="(label, key) in knowledgeGapLabels" :key="key">
                <td>{{ label }}</td>
                <td>{{ data.knowledge_gaps.by_status[key] ?? 0 }}</td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { getDashboardSummary } from "@/api/dashboard";
import type { DashboardSummary } from "@/types/dashboard";

const data = ref<DashboardSummary | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const ruleCandidateLabels: Record<string, string> = {
  proposed: "待提议",
  designing: "设计中",
  benchmarked: "已基准测试",
  accepted: "已批准",
  rejected: "已驳回",
};

const knowledgeGapLabels: Record<string, string> = {
  open: "待处理",
  in_progress: "处理中",
  resolved: "已解决",
  wontfix: "不修复",
};

function formatPct(v: number | null): string {
  if (v === null || v === undefined) return "N/A · 样本不足";
  return (v * 100).toFixed(2) + "%";
}

async function loadData() {
  loading.value = true;
  error.value = null;
  try {
    data.value = await getDashboardSummary();
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : "加载失败";
  } finally {
    loading.value = false;
  }
}

onMounted(loadData);
</script>

<style scoped>
.dashboard-page {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}
.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.page-header h1 {
  margin: 0 0 6px;
  font-size: 22px;
}
.scope-hint {
  color: #6b7280;
  font-size: 13px;
  margin: 0;
}
.refresh-btn {
  flex-shrink: 0;
  padding: 6px 14px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  font-size: 13px;
}
.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.seed-warning {
  background: #fef3c7;
  border: 1px solid #f59e0b;
  color: #92400e;
  padding: 8px 12px;
  border-radius: 6px;
  margin-bottom: 16px;
  font-size: 13px;
}
.metric-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}
.metric-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  background: #fff;
}
.metric-card .label {
  color: #6b7280;
  font-size: 13px;
}
.metric-card .value {
  font-size: 26px;
  font-weight: 600;
  margin: 6px 0;
  color: #111827;
}
.metric-card .value.na {
  font-size: 15px;
  font-weight: 500;
  color: #9ca3af;
}
.metric-card .sub {
  color: #9ca3af;
  font-size: 12px;
}
.dist-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}
.dist-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  background: #fff;
}
.dist-card h2 {
  font-size: 15px;
  margin: 0 0 10px;
  color: #111827;
}
.dist-card table {
  width: 100%;
  border-collapse: collapse;
}
.dist-card td {
  padding: 6px 0;
  font-size: 14px;
  color: #374151;
  border-bottom: 1px solid #f3f4f6;
}
.dist-card tr:last-child td {
  border-bottom: none;
}
.dist-card td:last-child {
  text-align: right;
  font-weight: 600;
  color: #111827;
}
.state-block {
  padding: 40px;
  text-align: center;
  color: #6b7280;
}
.state-block.error {
  color: #b91c1c;
}
@media (max-width: 900px) {
  .metric-cards {
    grid-template-columns: repeat(2, 1fr);
  }
  .dist-grid {
    grid-template-columns: 1fr;
  }
}
</style>
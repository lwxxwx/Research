// frontend/src/router/index.ts
import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      redirect: "/cases",
    },
    {
      path: "/cases",
      name: "cases",
      component: () => import("@/pages/CaseListPage.vue"),
      meta: { title: "Case 列表" },
    },
    {
      path: "/tasks",
      name: "tasks",
      component: () => import("@/pages/TaskListPage.vue"),
      meta: { title: "任务列表" },
    },
    {
      path: "/tasks/:taskId",
      name: "task-detail",
      component: () => import("@/pages/TaskDetailPage.vue"),
      meta: { title: "任务详情" },
      props: true,
    },
    // ===== [S2-G] 规则候选审核页 =====
    {
      path: "/rule-candidates",
      name: "rule-candidates",
      component: () => import("@/pages/RuleCandidatePage.vue"),
      meta: { title: "规则候选审核" },
    },
    // ===== [S2-G END] =====
    // ===== [S2-H] RAG 知识库管理 =====
    {
      path: "/knowledge-base",
      name: "knowledge-base",
      component: () => import("@/pages/KnowledgeBasePage.vue"),
      meta: { title: "知识库管理" },
    },
    // ===== [S2-H（四）] 指标看板 =====
    {
      path: "/dashboard",
      name: "dashboard",
      component: () => import("@/pages/DashboardPage.vue"),
      meta: { title: "指标看板" },
    },
    // ===== [S2-H（四） END] =====
    // ===== [S2-H END] =====
    {
      path: "/:pathMatch(.*)*",
      name: "not-found",
      component: () => import("@/pages/NotFoundPage.vue"),
      meta: { title: "404" },
    },
  ],
});

// 路由切换时更新 document.title
router.afterEach((to) => {
  const base = "原理图评审系统";
  const pageTitle = to.meta?.title as string | undefined;
  document.title = pageTitle ? `${pageTitle} · ${base}` : base;
});

export default router;
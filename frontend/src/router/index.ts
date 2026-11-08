// frontend/src/router/index.ts
import { createRouter, createWebHistory } from "vue-router";
import { useAuthStore } from "@/stores/auth";

// 白名单：不需要鉴权、不触发 bootstrap
const PUBLIC_ROUTES = new Set(["/login", "/403"]);

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/login",
      name: "login",
      component: () => import("@/pages/LoginPage.vue"),
      meta: { title: "登录" },
    },
    {
      path: "/403",
      name: "forbidden",
      component: () => import("@/pages/ForbiddenPage.vue"),
      meta: { title: "权限不足" },
    },
    {
      path: "/",
      redirect: "/cases",
    },
    {
      path: "/cases",
      name: "cases",
      component: () => import("@/pages/CaseListPage.vue"),
      meta: { title: "Case 列表", requiresAuth: true },
    },
    {
      path: "/cases/:caseId",
      name: "case-detail",
      component: () => import("@/pages/CaseDetailPage.vue"),
      meta: { title: "Case 详情", requiresAuth: true },
    },
    {
      path: "/tasks",
      name: "tasks",
      component: () => import("@/pages/TaskListPage.vue"),
      meta: { title: "任务列表", requiresAuth: true, roles: ["engineer", "reviewer", "admin"] },
    },
    {
      path: "/tasks/:taskId",
      name: "task-detail",
      component: () => import("@/pages/TaskDetailPage.vue"),
      meta: { title: "任务详情", requiresAuth: true, roles: ["engineer", "reviewer", "admin"] },
      props: true,
    },
    {
      path: "/rule-candidates",
      name: "rule-candidates",
      component: () => import("@/pages/RuleCandidatePage.vue"),
      meta: { title: "规则候选审核", requiresAuth: true, roles: ["rule_admin", "admin"] },
    },
    {
      path: "/knowledge-base",
      name: "knowledge-base",
      component: () => import("@/pages/KnowledgeBasePage.vue"),
      meta: { title: "知识库管理", requiresAuth: true, roles: ["rule_admin", "admin"] },
    },
    {
      path: "/dashboard",
      name: "dashboard",
      component: () => import("@/pages/DashboardPage.vue"),
      meta: { title: "指标看板", requiresAuth: true, roles: ["admin", "rule_admin", "reviewer"] },
    },
    {
      path: "/:pathMatch(.*)*",
      name: "not-found",
      component: () => import("@/pages/NotFoundPage.vue"),
      meta: { title: "404" },
    },
  ],
});

router.beforeEach(async (to) => {
  // 白名单：不触发 bootstrap
  if (PUBLIC_ROUTES.has(to.path)) {
    return true;
  }

  const authStore = useAuthStore();

  // 懒加载：首次导航时 bootstrap（内部有请求锁）
  if (!authStore.bootstrapped) {
    await authStore.bootstrap();
  }

  // 未登录：跳登录页（带 redirect 参数）
  if (!authStore.isAuthenticated) {
    return { path: "/login", query: { redirect: to.fullPath } };
  }

  // 角色校验
  const allowedRoles = to.meta?.roles as string[] | undefined;
  if (allowedRoles && authStore.role && !allowedRoles.includes(authStore.role)) {
    return { path: "/403" };
  }

  return true;
});

router.afterEach((to) => {
  const base = "原理图评审系统";
  const pageTitle = to.meta?.title as string | undefined;
  document.title = pageTitle ? `${pageTitle} · ${base}` : base;
});

export default router;
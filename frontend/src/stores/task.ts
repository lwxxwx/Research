// frontend/src/stores/task.ts
import { defineStore } from "pinia";
import { ref } from "vue";

import type { TaskDetail, TaskListItem } from "@/types/task";

/**
 * 任务 store（H 阶段用）。
 *
 * 职责：
 *   - 缓存当前任务详情（TaskDetail）
 *   - 缓存任务列表（TaskListItem[]）
 *   - 管理 loading / error 状态
 *
 * 注意：store 只缓存 UI 需要的数据，不做业务逻辑（业务逻辑在 api/*.ts）。
 */
export const useTaskStore = defineStore("task", () => {
  // ===== 当前任务详情 =====
  const currentTask = ref<TaskDetail | null>(null);
  const currentTaskLoading = ref(false);
  const currentTaskError = ref<string | null>(null);

  // ===== 任务列表 =====
  const taskList = ref<TaskListItem[]>([]);
  const taskListTotal = ref(0);
  const taskListLoading = ref(false);
  const taskListError = ref<string | null>(null);

  // ===== 当前任务：setter / clear =====
  function setCurrentTask(task: TaskDetail | null) {
    currentTask.value = task;
    currentTaskError.value = null;
  }
  function setCurrentTaskLoading(loading: boolean) {
    currentTaskLoading.value = loading;
  }
  function setCurrentTaskError(error: string | null) {
    currentTaskError.value = error;
  }

  // ===== 任务列表：setter / clear =====
  function setTaskList(items: TaskListItem[], total: number) {
    taskList.value = items;
    taskListTotal.value = total;
    taskListError.value = null;
  }
  function setTaskListLoading(loading: boolean) {
    taskListLoading.value = loading;
  }
  function setTaskListError(error: string | null) {
    taskListError.value = error;
  }

  return {
    // 当前任务
    currentTask,
    currentTaskLoading,
    currentTaskError,
    setCurrentTask,
    setCurrentTaskLoading,
    setCurrentTaskError,
    // 任务列表
    taskList,
    taskListTotal,
    taskListLoading,
    taskListError,
    setTaskList,
    setTaskListLoading,
    setTaskListError,
  };
});
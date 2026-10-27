// frontend/src/api/http.ts
// fetch 封装：baseURL + 20 秒超时 + TIMEOUT 错误类型

/** API baseURL（开发 /api/v1 由 Vite proxy 转发；生产 /api/v1 由 Nginx 反代） */
const BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

/** 默认超时（毫秒）：POST /review/tasks 约 20 秒，超时后不自动重试 */
const DEFAULT_TIMEOUT_MS = 20_000;

/** 超时错误类型（UI 层用于区分"超时"和"其他错误"） */
export const TIMEOUT_ERROR = "TIMEOUT";

/**
 * HTTP 请求封装。
 *
 * @param path 相对路径（如 "/review/tasks"），不带 /api/v1
 * @param options fetch 配置
 * @param timeoutMs 超时（毫秒），默认 20 秒
 * @returns 解析后的 JSON（泛型 T）
 * @throws Error 网络错误 / HTTP 非 2xx / 超时（message === TIMEOUT_ERROR）
 */
export async function request<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const url = `${BASE_URL}${path}`;

  // ===== AbortController 实现超时 =====
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    clearTimeout(timer);

    // ===== HTTP 非 2xx → 抛错（含后端 detail） =====
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body?.detail) {
          detail = typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
        }
      } catch {
        // body 不是 JSON，忽略
      }
      throw new Error(detail);
    }

    // ===== 204 No Content → 返回 undefined =====
    if (response.status === 204) {
      return undefined as T;
    }

    // ===== 解析 JSON =====
    return (await response.json()) as T;
  } catch (e) {
    clearTimeout(timer);

    // ===== AbortError（超时）→ 统一为 TIMEOUT_ERROR =====
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error(TIMEOUT_ERROR);
    }
    throw e;
  }
}
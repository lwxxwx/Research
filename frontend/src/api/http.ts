// frontend/src/api/http.ts
// ============================================================
// HTTP 请求封装（带超时、错误处理、开发环境日志）
// ============================================================

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const DEFAULT_TIMEOUT_MS = 20_000;

export const TIMEOUT_ERROR = "TIMEOUT";

export async function request<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  let base = BASE_URL.endsWith("/") ? BASE_URL.slice(0, -1) : BASE_URL;
  let p = path.startsWith("/") ? path : `/${path}`;
  const url = `${base}${p}`;

  const method = options.method || "GET";
  
  // 开发者提示：打印请求开始
  if (import.meta.env.DEV) {
    let logBody: any = options.body;
    if (options.body instanceof FormData) {
      logBody = "[FormData]";
    } else if (typeof options.body === "string" && options.body.startsWith("{")) {
      try { logBody = JSON.parse(options.body); } catch { logBody = options.body; }
    }
    console.log(`[API 请求开始] ${method} ${url}`, logBody || "");
  }
  
  const startTime = performance.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // ===== 动态处理 Content-Type =====
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  };
  // 如果 body 不是 FormData，且没有手动指定 Content-Type，默认设为 application/json
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers,
    });

    clearTimeout(timer);
    const duration = (performance.now() - startTime).toFixed(0);

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body?.detail) {
          detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
      } catch { /* ignore */ }
      
      // 开发者提示：打印请求失败
      console.error(`[API 失败] ${method} ${url} - 耗时: ${duration}ms - 错误:`, detail);
      throw new Error(detail);
    }

    if (response.status === 204) return undefined as T;
    const data = await response.json();
    
    // 开发者提示：打印请求成功
    if (import.meta.env.DEV) {
      console.log(`[API 成功] ${method} ${url} - 耗时: ${duration}ms - 状态: ${response.status}`);
    }
    return data as T;

  } catch (e) {
    clearTimeout(timer);
    if (e instanceof DOMException && e.name === "AbortError") {
      // 开发者提示：打印超时
      console.warn(`[API 超时] ${method} ${url} - 超过 ${timeoutMs}ms 未响应`);
      throw new Error(TIMEOUT_ERROR);
    }
    throw e;
  }
}
// frontend/src/api/http.ts
// ============================================================
// HTTP 请求封装（带超时、错误处理、开发环境日志）
// Sprint 3 G4：注入 X-User-Id；401 清身份跳登录；403 抛专用异常
// Sprint 3 legacy B3：Bearer JWT 优先，回退 X-User-Id
// ============================================================

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const DEFAULT_TIMEOUT_MS = 20_000;

export const TIMEOUT_ERROR = "TIMEOUT";
export const USER_ID_STORAGE_KEY = "user_id";
export const ACCESS_TOKEN_STORAGE_KEY = "access_token";

/** 403：已认证但权限不足（页面级提示，不跳登录） */
export class PermissionDeniedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PermissionDeniedError";
  }
}

/** 401：身份失效（清身份跳登录） */
export class UnauthorizedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnauthorizedError";
  }
}

function getStoredUserId(): string | null {
  try {
    return localStorage.getItem(USER_ID_STORAGE_KEY);
  } catch {
    return null;
  }
}

function getStoredToken(): string | null {
  try {
    return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function handleUnauthorized(): void {
  // 清身份：user_id + access_token 同时清
  try {
    localStorage.removeItem(USER_ID_STORAGE_KEY);
    localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    /* ignore */
  }
  // 跳登录：用 window.location 避免 Vue Router 在 http.ts 里循环依赖
  if (typeof window !== "undefined" && window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
}

export async function request<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  let base = BASE_URL.endsWith("/") ? BASE_URL.slice(0, -1) : BASE_URL;
  let p = path.startsWith("/") ? path : `/${path}`;
  const url = `${base}${p}`;
  const method = options.method || "GET";

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

  const headers: Record<string, string> = {
    ...((options.headers as Record<string, string>) || {}),
  };
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  // ===== [S3-LEGACY-B3] 鉴权头注入：Bearer 优先，回退 X-User-Id =====
  const token = getStoredToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    const storedUserId = getStoredUserId();
    if (storedUserId && !headers["X-User-Id"]) {
      headers["X-User-Id"] = storedUserId;
    }
  }
  // ===== [S3-LEGACY-B3 END] =====

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
      } catch {
        /* ignore */
      }

      console.error(`[API 失败] ${method} ${url} - 耗时: ${duration}ms - 错误:`, detail);

      if (response.status === 401) {
        handleUnauthorized();
        throw new UnauthorizedError(detail);
      }
      if (response.status === 403) {
        throw new PermissionDeniedError(detail);
      }
      throw new Error(detail);
    }

    if (response.status === 204) return undefined as T;
    const data = await response.json();

    if (import.meta.env.DEV) {
      console.log(`[API 成功] ${method} ${url} - 耗时: ${duration}ms - 状态: ${response.status}`);
    }
    return data as T;
  } catch (e) {
    clearTimeout(timer);
    if (e instanceof DOMException && e.name === "AbortError") {
      console.warn(`[API 超时] ${method} ${url} - 超过 ${timeoutMs}ms 未响应`);
      throw new Error(TIMEOUT_ERROR);
    }
    throw e;
  }
}
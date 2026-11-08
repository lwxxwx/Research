import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { login as apiLogin, me as apiMe } from "@/api/auth";
import { USER_ID_STORAGE_KEY, UnauthorizedError } from "@/api/http";

const ACCESS_TOKEN_STORAGE_KEY = "access_token";   // ← 新增

export interface AuthUser {
  user_id: number;
  username: string;
  role: string;
  is_active: boolean;
}

export const useAuthStore = defineStore("auth", () => {
  const user = ref<AuthUser | null>(null);
  const bootstrapped = ref(false);
  let bootstrapPromise: Promise<AuthUser | null> | null = null;

  const isAuthenticated = computed(() => user.value !== null);
  const role = computed(() => user.value?.role ?? null);

  function setUser(u: AuthUser | null): void {
    user.value = u;
    if (u) {
      try { localStorage.setItem(USER_ID_STORAGE_KEY, String(u.user_id)); } catch { /* ignore */ }
    } else {
      try {
        localStorage.removeItem(USER_ID_STORAGE_KEY);
        localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);   // ← 清 token
      } catch { /* ignore */ }
    }
  }

  function setToken(token: string | null): void {            // ← 新增
    try {
      if (token) localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
      else localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
    } catch { /* ignore */ }
  }

  function clear(): void {
    setUser(null);
    setToken(null);
    bootstrapped.value = false;
    bootstrapPromise = null;
  }

  async function bootstrap(): Promise<AuthUser | null> {
    if (bootstrapped.value) return user.value;
    if (bootstrapPromise) return bootstrapPromise;

    let storedId: string | null = null;
    try { storedId = localStorage.getItem(USER_ID_STORAGE_KEY); } catch { /* ignore */ }
    if (!storedId) { bootstrapped.value = true; return null; }

    bootstrapPromise = (async () => {
      try {
        const me = await apiMe();
        const u: AuthUser = {
          user_id: me.user_id, username: me.username, role: me.role, is_active: me.is_active,
        };
        setUser(u);
        bootstrapped.value = true;
        return u;
      } catch (e) {
        if (e instanceof UnauthorizedError) clear();
        bootstrapped.value = true;
        return null;
      } finally {
        bootstrapPromise = null;
      }
    })();

    return bootstrapPromise;
  }

  async function login(username: string): Promise<AuthUser> {
    const resp = await apiLogin(username);
    setToken(resp.access_token ?? null);                    // ← 存 token
    setUser({
      user_id: resp.user_id, username: resp.username, role: resp.role, is_active: true,
    });
    const me = await bootstrap();
    return me ?? (user.value as AuthUser);
  }

  function logout(): void { clear(); }

  return {
    user, isAuthenticated, role, bootstrapped,
    bootstrap, login, logout, setUser, setToken, clear,
  };
});
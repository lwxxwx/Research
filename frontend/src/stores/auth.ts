// frontend/src/stores/auth.ts
// Sprint 3 G4：认证状态 store
// 约束：
//   1. role 唯一可信源 = GET /auth/me 返回；不写 localStorage
//   2. localStorage 仅存 user_id（只用于 X-User-Id 注入）
//   3. bootstrap() 加请求锁，防并发
import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { login as apiLogin, me as apiMe } from "@/api/auth";
import { USER_ID_STORAGE_KEY, UnauthorizedError } from "@/api/http";

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
      try {
        localStorage.setItem(USER_ID_STORAGE_KEY, String(u.user_id));
      } catch {
        /* ignore */
      }
    } else {
      try {
        localStorage.removeItem(USER_ID_STORAGE_KEY);
      } catch {
        /* ignore */
      }
    }
  }

  function clear(): void {
    setUser(null);
    bootstrapped.value = false;
    bootstrapPromise = null;
  }

  /**
   * 启动时拉取 /auth/me。
   * - 有请求锁：并发调复用同一个 Promise
   * - 无 user_id：直接返回 null，不调接口
   * - 401：清身份返回 null
   */
  async function bootstrap(): Promise<AuthUser | null> {
    if (bootstrapped.value) return user.value;
    if (bootstrapPromise) return bootstrapPromise;

    let storedId: string | null = null;
    try {
      storedId = localStorage.getItem(USER_ID_STORAGE_KEY);
    } catch {
      /* ignore */
    }
    if (!storedId) {
      bootstrapped.value = true;
      return null;
    }

    bootstrapPromise = (async () => {
      try {
        const me = await apiMe();
        const u: AuthUser = {
          user_id: me.user_id,
          username: me.username,
          role: me.role,
          is_active: me.is_active,
        };
        setUser(u);
        bootstrapped.value = true;
        return u;
      } catch (e) {
        if (e instanceof UnauthorizedError) {
          clear();
        }
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
    // 登录后调 me 拿完整信息（含 is_active）
    setUser({
      user_id: resp.user_id,
      username: resp.username,
      role: resp.role,
      is_active: true,
    });
    const me = await bootstrap();
    return me ?? (user.value as AuthUser);
  }

  function logout(): void {
    clear();
  }

  return {
    user,
    isAuthenticated,
    role,
    bootstrapped,
    bootstrap,
    login,
    logout,
    setUser,
    clear,
  };
});
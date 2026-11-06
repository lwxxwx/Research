// frontend/src/api/auth.ts
// Sprint 3 G4：登录 / 当前用户 / 登出
import { request } from "./http";

export interface LoginResponse {
  user_id: number;
  username: string;
  role: string;
}

export interface AuthMeResponse {
  user_id: number;
  username: string;
  role: string;
  is_active: boolean;
}

export async function login(username: string): Promise<LoginResponse> {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

export async function me(): Promise<AuthMeResponse> {
  return request<AuthMeResponse>("/auth/me", { method: "GET" });
}
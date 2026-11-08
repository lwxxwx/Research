// frontend/src/api/auth.ts
import { request } from "./http";

export interface LoginResponse {
  user_id: number;
  username: string;
  role: string;
  access_token: string | null;      // ← 新增
  token_type: string;               // ← 新增
  expires_in: number;               // ← 新增
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
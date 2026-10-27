// frontend/vite.config.ts
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // ===== [H 阶段] /api → 后端 FastAPI（localhost:8000） =====
      // 开发时前端 fetch /api/v1/cases → Vite proxy → http://localhost:8000/api/v1/cases
      // 同源请求（浏览器看是 localhost:5173/api/v1/...），无 CORS
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // 不 rewrite：/api/v1/review/tasks → http://localhost:8000/api/v1/review/tasks
      },
    },
  },
});
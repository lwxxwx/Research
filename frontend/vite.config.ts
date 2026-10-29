import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // ===== [修复] 代理 /api 请求到后端 =====
      '/api': {
        target: 'http://localhost:8000', // 指向本地 Docker 的 backend
        changeOrigin: true,
        // 不需要 rewrite，因为后端本身就有 /api/v1 前缀
      }
    }
  }
})
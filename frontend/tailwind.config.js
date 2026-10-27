/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ===== 三态质量状态（与 review_status 对齐）=====
        "status-ai-confirmed": "#10b981",       // 绿
        "status-need-expert": "#f59e0b",        // 黄
        "status-low-confidence": "#ef4444",     // 红
        "status-unknown": "#6b7280",            // 灰（兜底）
        // ===== 四态专家决定（与 expert_decision 对齐）=====
        "decision-pending": "#6b7280",          // 灰
        "decision-confirmed": "#10b981",        // 绿
        "decision-rejected": "#ef4444",         // 红
        "decision-needs-verification": "#f59e0b", // 黄
      },
    },
  },
  plugins: [],
};
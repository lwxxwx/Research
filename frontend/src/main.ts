// frontend/src/main.ts
import { createApp } from "vue";
import { createPinia } from "pinia";

import App from "./App.vue";
import router from "./router";

// ===== Tailwind 全局样式 =====
import "./style.css";

const app = createApp(App);

app.use(createPinia());
app.use(router);

app.mount("#app");
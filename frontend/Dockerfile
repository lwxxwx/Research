# frontend/Dockerfile
# Sprint 3 legacy 批次④（D23）：多阶段构建，把 Vite 产物打包进 Nginx 镜像
# 本地开发继续用 5173（vite dev server），3000 用于验证生产打包制品

# ---------- 构建阶段 ----------
FROM node:20-alpine AS build
WORKDIR /app

# 先拷依赖清单，利用层缓存
COPY package.json package-lock.json ./
RUN npm ci

# 再拷源码并构建
COPY . .
RUN npm run build

# ---------- 运行阶段 ----------
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# 单页应用：任意路径回退到 index.html（刷新 /cases 等深链不 404）
RUN printf 'server {\n\
  listen 80;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
# Sprint 0 · A1 Git / Branch Workflow & 根文件手动实现计划

## 1. 任务摘要
本计划详细描述了 Sprint 0 阶段 Phase A1 的手动执行步骤。目标是确立项目的 Git 分支管理规范，并完成根目录下基础工程文件的初始化。

## 2. 当前状态分析
- **Git**: 仅存在 `main` 分支。
- **根文件**: 缺失 `.gitignore`, `README.md`, `CONTRIBUTING.md`, `.dockerignore`, `.env.example`。

## 3. 详细执行步骤

### 步骤 1：Git 分支架构搭建
在 PowerShell 中执行以下命令，建立三级分支体系：

```powershell
# 1. 确保在 main 分支
git checkout main

# 2. 创建并切换到 develop (开发主分支)

# 3. 从 develop 创建并切换到 sprint0/base (Sprint 0 基准分支)
git checkout -b sprint0/base

# 4. 推送到远程 (如已配置远程仓库)
# git push -u origin develop
# git push -u origin sprint0/base
```

### 步骤 2：创建根目录基础文件
手动在项目根目录 `e:\projects\Schematic_Design_Review\` 创建以下文件并填入内容：

#### 2.1 创建 `.gitignore`
**目的**：防止环境敏感数据和容器运行时产生的大量数据进入版本库。
**文件内容**：
```text
# --- Python / UV ---
__pycache__/
*.py[cod]
*$py.class
.venv/
.uv/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# --- Environment ---
.env
.env.local

# --- Docker & Persistence ---
pgdata/
out/

# --- IDE ---
.vscode/
.idea/
*.swp
*.swo

# --- OS ---
Thumbs.db
.DS_Store
```

#### 2.2 创建 `.dockerignore`
**目的**：优化 Docker 构建上下文，提升构建速度。
**文件内容**：
```text
.git/
.venv/
__pycache__/
pgdata/
out/
.trae/
*.md
.gitignore
.dockerignore
```

#### 2.3 创建 `.env.example`
**目的**：提供环境变量模板，方便开发者快速配置。
**文件内容**：
```text
# --- 数据库配置 ---
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=schematic_review
# 容器内部通信 URL
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/schematic_review

# --- AI 服务 ---
LLM_PROVIDER=mock
LLM_API_KEY=your_api_key_here
DATA_DIR=/app/data

# --- 应用配置 ---
APP_ENV=dev
```

#### 2.4 创建 `README.md`
**目的**：项目入门指南。
**文件内容**：
```markdown
# AI 原理图评审系统 (MVP)

本项目基于 Docker + uv 构建，旨在通过 AI 技术实现原理图评审的自动化。

## 快速开始

### 前提条件
- Git
- Docker Desktop
- IDE (推荐 Trae / VS Code)

### 启动开发环境
1. **初始化环境**
   ```powershell
   .\scripts\bootstrap.ps1
   ```
2. **启动服务**
   ```powershell
   .\scripts\dev.ps1
   ```

## 开发规范
请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。
```

#### 2.5 创建 `CONTRIBUTING.md`
**目的**：规范团队协作。
**文件内容**：
```markdown
# 贡献指南

## 分支管理
- `main`: 生产/发布分支。
- `develop`: 开发集成分支。
- `feature/*`: 功能开发分支。
- `sprint*/base`: 各阶段 Sprint 的基准分支。

## Commit 规范
使用 Angular 规范：`<type>(<scope>): <subject>`
- `feat`: 新功能
- `fix`: 修复问题
- `docs`: 文档变更
- `style`: 格式变更
- `refactor`: 重构
```

### 步骤 3：提交基础架构
```powershell
git add .
git commit -m "feat(infra): initialize project root files and branch workflow"
```

## 4. 验证步骤
1. 执行 `git branch` 确认存在 `main`, `develop`, `sprint0/base` 分支。
2. 检查根目录下 5 个新文件是否存在且内容完整。
3. 尝试修改 `.env.example` 为 `.env` 并确认其被 `.gitignore` 成功过滤（不出现在 `git status` 中）。

## 5. 决策与假设
- **假设**：开发者具备基本的 Git 操作能力。
- **决策**：初期不强制要求推送远程，仅完成本地架构搭建。

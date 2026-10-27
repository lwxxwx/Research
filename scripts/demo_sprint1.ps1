# scripts/demo_sprint1.ps1
# ============================================================================
# Sprint 1 终极全栈 Demo 脚本（阶段 I 交付）
# ============================================================================
# 【流程】
#   Step 1: 环境预检（Docker / 容器状态）
#   Step 2: 配置预检（LLM Key）
#   Step 3: 后端核心链路测试（真实轨 A 类 Case + Mock 轨 E2E）
#   Step 4: API 契约联调测试（验证 7 个端点通）
#   Step 5: 展示产物与候选规则
# ============================================================================

param(
    [switch]$ForceMock
)

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PROJECT_ROOT = $projectRoot

$f1 = Join-Path $projectRoot "infra/docker/docker-compose.yml"
$f2 = Join-Path $projectRoot "infra/docker/docker-compose.dev.yml"

$outDir = Join-Path $projectRoot "out/demo_sprint1"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Sprint 1 终极 Demo（后端 Graph + API + 前端 MVP）" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ============================================================================
# Step 1: 预检
# ============================================================================
Write-Host "`n[Step 1] 预检环境..." -ForegroundColor Yellow

$dockerCheck = cmd /c "docker info --format '{{.ServerVersion}}' 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Desktop 未运行" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Docker: $dockerCheck"

$psCheck = cmd /c "docker compose -f `"$f1`" -f `"$f2`" ps --status running 2>&1"
if ($psCheck -notmatch "backend") {
    Write-Host "⚠️ backend 容器未运行，请先执行 docker compose up -d" -ForegroundColor Yellow
} else {
    Write-Host "  ✅ backend 运行中"
}

# ============================================================================
# Step 2: 配置预检
# ============================================================================
Write-Host "`n[Step 2] 预检 LLM 配置..." -ForegroundColor Yellow

$keyCheck = cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec -T backend printenv ZHIPUAI_API_KEY 2>&1"
$hasKey = ($LASTEXITCODE -eq 0) -and ($keyCheck -match '\w{20,}')

if ($hasKey) {
    Write-Host "  ✅ ZHIPUAI_API_KEY 已配置" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  ZHIPUAI_API_KEY 未配置，强制降级为 Mock 轨" -ForegroundColor Yellow
    $ForceMock = $true
}

# ============================================================================
# Step 3: 后端核心链路测试
# ============================================================================
Write-Host "`n[Step 3] 运行后端 Graph 业务测试..." -ForegroundColor Yellow

if ($ForceMock) {
    Write-Host "  模式：Mock 轨（不触网）" -ForegroundColor Cyan
    $demoFlag = "-e DEMO_REAL_LLM=0"
} else {
    Write-Host "  模式：真实轨（调用智谱 API）" -ForegroundColor Cyan
    $demoFlag = "-e DEMO_REAL_LLM=1"
}

$pytestCmd = "docker compose -f `"$f1`" -f `"$f2`" exec $demoFlag -T backend uv run pytest tests/test_demo.py -v -s"
cmd /c $pytestCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ 后端 Graph 测试通过" -ForegroundColor Green
} else {
    Write-Host "  ⚠️ 后端 Graph 测试降级（前端可继续）" -ForegroundColor Yellow
}

# ============================================================================
# Step 4: API 契约联调测试（H 阶段新增）
# ============================================================================
Write-Host "`n[Step 4] 运行 API 契约联调..." -ForegroundColor Yellow
Write-Host "  执行 tests/test_api.py 的 Mock 测试（不触网）" -ForegroundColor Cyan

$apiPytestCmd = "docker compose -f `"$f1`" -f `"$f2`" exec -T backend uv run pytest tests/test_api.py -v -m `"not integration`""
cmd /c $apiPytestCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ API 联调测试通过" -ForegroundColor Green
} else {
    Write-Host "  ❌ API 联调失败，请检查路由和 Schema" -ForegroundColor Red
}

# ============================================================================
# Step 5: 展示产物
# ============================================================================
Write-Host "`n[Step 5] 展示 Demo 产物..." -ForegroundColor Yellow

if (Test-Path $outDir) {
    $files = Get-ChildItem $outDir -File
    if ($files.Count -gt 0) {
        Write-Host "  产物目录：$outDir" -ForegroundColor Green
        foreach ($f in $files) {
            $sizeKB = [math]::Round( ($f.Length / 1KB), 1 )
            Write-Host "    - $($f.Name)  ($sizeKB KB)"
        }
    } else {
        Write-Host "  ⚠️  产物目录为空" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  产物目录不存在：$outDir" -ForegroundColor Yellow
}

Write-Host "`n  🔗 前端服务地址：http://localhost:8080（或 npm run dev 的 5173）" -ForegroundColor Cyan

# ============================================================================
# Step 6: 候选规则列表
# ============================================================================
Write-Host "`n[Step 6] 展示候选规则..." -ForegroundColor Yellow
$listCmd = "docker compose -f `"$f1`" -f `"$f2`" exec -T backend uv run python -m app.services.rule_evolution_service --list"
cmd /c $listCmd

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Sprint 1 Demo 执行完毕！" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
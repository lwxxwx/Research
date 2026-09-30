# scripts/init_db.ps1
<#
.SYNOPSIS
    一键初始化数据库 schema（幂等）

.DESCRIPTION
    包装 docker compose exec backend uv run python -m scripts.init_db
    核心逻辑在 scripts/init_db.py，本脚本只做：
      1. 定位项目根目录
      2. 组装 docker compose -f 命令
      3. 检查 backend 容器是否运行
      4. cmd /c 包装执行

.PARAMETER Help
    显示帮助

.EXAMPLE
    .\scripts\init_db.ps1
    正常初始化数据库

.EXAMPLE
    .\scripts\init_db.ps1 -Help
    显示帮助

.NOTES
    Sprint 0 Phase C · 数据库初始化
    与 scripts/init_db.py 配合使用
    本脚本不修改 init_db.py；如需 dry-run，请直接运行：
        docker compose ... exec backend uv run python -m scripts.init_db
#>

param(
    [switch]$Help = $false
)

$ErrorActionPreference = "Stop"

# ============================================================
# Help 分支
# ============================================================
if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\scripts\init_db.ps1 [-Help]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  一键初始化数据库 schema（幂等，已初始化时跳过 DDL）"
    Write-Host ""
    Write-Host "  -Help     显示本帮助"
    Write-Host ""
    exit 0
}

# ============================================================
# 定位项目根目录
# ------------------------------------------------------------
# $PSScriptRoot 是 PowerShell 内置变量，指向本脚本所在目录（scripts/）
# 项目根目录 = scripts/ 的上一级
# ============================================================
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    # 兼容 PowerShell 5.1 在 dot-source 时的行为
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$projectRoot = Split-Path -Parent $scriptDir

Write-Host "📁 项目根目录: $projectRoot" -ForegroundColor DarkGray

# ============================================================
# 定位 compose 文件（绝对路径）
# ============================================================
$f1 = Join-Path $projectRoot "infra/docker/docker-compose.yml"
$f2 = Join-Path $projectRoot "infra/docker/docker-compose.dev.yml"

if (-not (Test-Path $f1)) {
    Write-Host "❌ 未找到 $f1" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $f2)) {
    Write-Host "❌ 未找到 $f2" -ForegroundColor Red
    exit 1
}

# ============================================================
# 检查 backend 容器是否在运行
# ============================================================
Write-Host "🔍 检查 backend 容器状态..." -ForegroundColor Cyan
$composePrefix = "docker compose -f `"$f1`" -f `"$f2`""

# 用 cmd /c 包装，规避 PowerShell 5.1 stderr 污染
$psOutput = cmd /c "$composePrefix ps --format json 2>nul"
$psExitCode = $LASTEXITCODE

if ($psExitCode -ne 0) {
    Write-Host "❌ docker compose ps 执行失败 (exit=$psExitCode)" -ForegroundColor Red
    Write-Host "   请确认 Docker Desktop 已启动，或手动运行：" -ForegroundColor Yellow
    Write-Host "   $composePrefix ps" -ForegroundColor Yellow
    exit 1
}

# ps 输出为 JSON 数组（可能为空）
if ([string]::IsNullOrWhiteSpace($psOutput)) {
    Write-Host "⚠️  backend 容器未运行" -ForegroundColor Yellow
    Write-Host "   请先启动：" -ForegroundColor Yellow
    Write-Host "   .\scripts\dev.ps1 start" -ForegroundColor Yellow
    exit 1
}

# 检查 backend 是否在列表里
$backendRunning = $false
try {
    $services = $psOutput | ConvertFrom-Json
    foreach ($svc in $services) {
        if ($svc.Service -eq "backend" -or $svc.Name -like "*backend*") {
            $backendRunning = $true
            break
        }
    }
} catch {
    # JSON 解析失败，退化为字符串匹配
    if ($psOutput -match "backend") {
        $backendRunning = $true
    }
}

if (-not $backendRunning) {
    Write-Host "⚠️  backend 容器未运行" -ForegroundColor Yellow
    Write-Host "   请先启动：" -ForegroundColor Yellow
    Write-Host "   .\scripts\dev.ps1 start" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ backend 容器运行中" -ForegroundColor Green

# ============================================================
# 组装 init_db.py 命令
# ============================================================
$pythonCmd = "python -m scripts.init_db"
$fullCmd = "$composePrefix exec -T backend uv run $pythonCmd"

Write-Host ""
Write-Host "🚀 执行数据库初始化..." -ForegroundColor Cyan
Write-Host "   $fullCmd" -ForegroundColor DarkGray
Write-Host ""

# ============================================================
# 执行（cmd /c 包装）
# ============================================================
cmd /c $fullCmd
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "✅ 数据库初始化完成" -ForegroundColor Green
} else {
    Write-Host "❌ 数据库初始化失败 (exit=$exitCode)" -ForegroundColor Red
}
exit $exitCode
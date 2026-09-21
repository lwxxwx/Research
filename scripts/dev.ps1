<#
Sprint0 Dev Environment Manager V2.2 (方案A)
Usage:
  .\scripts\dev.ps1 [start|stop|restart|logs|status|help] [-Build]
Examples:
  .\scripts\dev.ps1 start
  .\scripts\dev.ps1 start -Build
  .\scripts\dev.ps1 stop
  .\scripts\dev.ps1 restart -Build
  .\scripts\dev.ps1 logs
  .\scripts\dev.ps1 status

【方案A-新增】本版本变更（相对 V1.3-fix）：
1. 注入 PROJECT_ROOT，供 docker-compose.dev.yml 使用
2. $f1/$f2 改为基于项目根的绝对路径
3. 所有 compose 调用统一使用 cmd /c 包装
4. 移除 --project-directory（PowerShell 5.1 下与 -f 组合会导致丢 -f）
5. 启动前预创建 bind mount 源目录
#>
param(
    [string]$Action = "start",
    [switch]$Build
)
$ErrorActionPreference = "Continue"

function Get-ProjectRoot {
    param([string]$StartDir = $PWD.Path)
    $current = Resolve-Path $StartDir
    while ($current) {
        if (Test-Path (Join-Path $current ".git")) { return $current.Path }
        if (Test-Path (Join-Path $current ".env.example")) { return $current.Path }
        $parent = Split-Path $current -Parent
        if ($parent -eq $current) { break }
        $current = $parent
    }
    return $StartDir
}
function Write-Success { Write-Host "✓ $($args[0])" -ForegroundColor Green }
function Write-Error { Write-Host "✗ $($args[0])" -ForegroundColor Red }
function Write-Info { Write-Host "ℹ $($args[0])" -ForegroundColor Cyan }
function Write-Warning { Write-Host "⚠ $($args[0])" -ForegroundColor Yellow }

$projectRoot = Get-ProjectRoot
Set-Location $projectRoot

# ============================================================
# [原逻辑 - 保留注释，便于对照回滚]
# （原版本此处无 PROJECT_ROOT 注入）
# ============================================================

# 【方案A-新增】注入 PROJECT_ROOT
$env:PROJECT_ROOT = $projectRoot
Write-Info "PROJECT_ROOT = $env:PROJECT_ROOT"

# ============================================================
# [原逻辑 - 保留注释，便于对照回滚]
# （原版本此处无源目录预创建）
# ============================================================

# 【方案A-新增】预创建所有 bind mount 源目录
$mountSources = @("backend", "data", "scripts", "infra", "out")
foreach ($d in $mountSources) {
    $p = Join-Path $projectRoot $d
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Force -Path $p | Out-Null
        Write-Info "Created mount source: $p"
    }
}
$gitkeep = Join-Path $projectRoot "out\.gitkeep"
if (-not (Test-Path $gitkeep)) {
    New-Item -ItemType File -Force -Path $gitkeep | Out-Null
}

# ------------------------------------------------------------
# [原逻辑 - 保留注释，便于对照回滚]
# $f1 = "infra/docker/docker-compose.yml"
# $f2 = "infra/docker/docker-compose.dev.yml"
# ------------------------------------------------------------

# 【方案A-新增】使用基于项目根的绝对路径
$f1 = Join-Path $projectRoot "infra/docker/docker-compose.yml"
$f2 = Join-Path $projectRoot "infra/docker/docker-compose.dev.yml"

if (-not (Test-Path $f1) -or -not (Test-Path $f2)) {
    Write-Error "Compose config file missing: $f1 / $f2"
    exit 1
}

# 【方案A-新增】统一 compose 参数串
$composeArgStr = "-f `"$f1`" -f `"$f2`""

function Show-Help {
    Write-Host "`nSprint0 Dev Environment Manager" -ForegroundColor Cyan
    Write-Host "Usage: .\scripts\dev.ps1 [action] [-Build]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  start       Start dev stack (default)"
    Write-Host "  stop        Stop dev stack (down)"
    Write-Host "  restart     Restart stack"
    Write-Host "  logs        Follow container logs"
    Write-Host "  status      Show container ps status"
    Write-Host "  help        Show this help message"
    Write-Host ""
    Write-Host "  -Build      Append --build when up/restart"
}

switch ($Action.ToLower()) {
    "start" {
        Write-Info "Starting dev environment..."

        # ------------------------------------------------------------
        # [原逻辑 - 保留注释，便于对照回滚]
        # $cmd = @("compose","-f",$f1,"-f",$f2,"up","-d")
        # if ($Build) { $cmd += "--build" }
        # & docker @cmd
        # ------------------------------------------------------------

        # 【方案A-新增】cmd /c 包装
        $buildFlag = if ($Build) { "--build" } else { "" }
        cmd /c "docker compose $composeArgStr up -d $buildFlag"

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dev environment started"
            Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Cyan
            Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
        }
        else {
            Write-Error "start failed"
            exit 1
        }
    }
    "stop" {
        Write-Info "Stopping dev environment..."

        # ------------------------------------------------------------
        # [原逻辑 - 保留注释，便于对照回滚]
        # $cmd = @("compose","-f",$f1,"-f",$f2,"down")
        # & docker @cmd
        # ------------------------------------------------------------

        # 【方案A-新增】cmd /c 包装
        cmd /c "docker compose $composeArgStr down"
        Write-Success "Dev environment stopped"
    }
    "restart" {
        Write-Info "Restarting dev environment..."

        # ------------------------------------------------------------
        # [原逻辑 - 保留注释，便于对照回滚]
        # $downCmd = @("compose","-f",$f1,"-f",$f2,"down")
        # & docker @downCmd
        # $upCmd = @("compose","-f",$f1,"-f",$f2,"up","-d")
        # if ($Build) { $upCmd += "--build" }
        # & docker @upCmd
        # ------------------------------------------------------------

        # 【方案A-新增】cmd /c 包装
        cmd /c "docker compose $composeArgStr down"
        $buildFlag = if ($Build) { "--build" } else { "" }
        cmd /c "docker compose $composeArgStr up -d $buildFlag"

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dev environment restarted"
            Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Cyan
            Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
        }
        else {
            Write-Error "restart failed"
            exit 1
        }
    }
    "logs" {
        # ------------------------------------------------------------
        # [原逻辑 - 保留注释，便于对照回滚]
        # $cmd = @("compose","-f",$f1,"-f",$f2,"logs","-f")
        # & docker @cmd
        # ------------------------------------------------------------

        # 【方案A-新增】cmd /c 包装
        cmd /c "docker compose $composeArgStr logs -f"
    }
    "status" {
        # ------------------------------------------------------------
        # [原逻辑 - 保留注释，便于对照回滚]
        # $cmd = @("compose","-f",$f1,"-f",$f2,"ps")
        # & docker @cmd
        # ------------------------------------------------------------

        # 【方案A-新增】cmd /c 包装
        cmd /c "docker compose $composeArgStr ps"
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown action: $Action"
        Show-Help
        exit 1
    }
}
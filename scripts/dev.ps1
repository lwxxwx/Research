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

S2-H 优化：
1. start / restart 成功后自动执行 uv sync，防止 pyproject.toml 变更导致容器内依赖漂移
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

$env:PROJECT_ROOT = $projectRoot
Write-Info "PROJECT_ROOT = $env:PROJECT_ROOT"

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

$f1 = Join-Path $projectRoot "infra/docker/docker-compose.yml"
$f2 = Join-Path $projectRoot "infra/docker/docker-compose.dev.yml"

if (-not (Test-Path $f1) -or -not (Test-Path $f2)) {
    Write-Error "Compose config file missing: $f1 / $f2"
    exit 1
}

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
        $buildFlag = if ($Build) { "--build" } else { "" }
        cmd /c "docker compose $composeArgStr up -d $buildFlag"

        if ($LASTEXITCODE -eq 0) {
            Write-Info "Syncing python dependencies (uv sync)..."
            cmd /c "docker compose $composeArgStr exec -T backend uv sync"
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "uv sync failed, continue"
            } else {
                Write-Success "Dependencies synced"
            }
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
        cmd /c "docker compose $composeArgStr down"
        Write-Success "Dev environment stopped"
    }
    "restart" {
        Write-Info "Restarting dev environment..."
        cmd /c "docker compose $composeArgStr down"
        $buildFlag = if ($Build) { "--build" } else { "" }
        cmd /c "docker compose $composeArgStr up -d $buildFlag"

        if ($LASTEXITCODE -eq 0) {
            Write-Info "Syncing python dependencies (uv sync)..."
            cmd /c "docker compose $composeArgStr exec -T backend uv sync"
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "uv sync failed, continue"
            } else {
                Write-Success "Dependencies synced"
            }
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
        cmd /c "docker compose $composeArgStr logs -f"
    }
    "status" {
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
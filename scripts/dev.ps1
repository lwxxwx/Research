<#
Sprint0 Dev Environment Manager V1.3‑fix
Usage:
  .\scripts\dev.ps1 [start|stop|restart|logs|status|help] [-Build]
Examples:
  .\scripts\dev.ps1 start
  .\scripts\dev.ps1 start -Build
  .\scripts\dev.ps1 stop
  .\scripts\dev.ps1 restart -Build
  .\scripts\dev.ps1 logs
  .\scripts\dev.ps1 status
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
$f1 = "infra/docker/docker-compose.yml"
$f2 = "infra/docker/docker-compose.dev.yml"

if (-not (Test-Path $f1) -or -not (Test-Path $f2)) {
    Write-Error "Compose config file missing: $f1 / $f2"
    exit 1
}

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
        $cmd = @("compose","-f",$f1,"-f",$f2,"up","-d")
        if ($Build) { $cmd += "--build" }
        & docker @cmd
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
        $cmd = @("compose","-f",$f1,"-f",$f2,"down")
        & docker @cmd
        Write-Success "Dev environment stopped"
    }
    "restart" {
        Write-Info "Restarting dev environment..."
        $downCmd = @("compose","-f",$f1,"-f",$f2,"down")
        & docker @downCmd
        $upCmd = @("compose","-f",$f1,"-f",$f2,"up","-d")
        if ($Build) { $upCmd += "--build" }
        & docker @upCmd
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
        $cmd = @("compose","-f",$f1,"-f",$f2,"logs","-f")
        & docker @cmd
    }
    "status" {
        $cmd = @("compose","-f",$f1,"-f",$f2,"ps")
        & docker @cmd
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

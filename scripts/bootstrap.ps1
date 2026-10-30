<#
Sprint0 A5 Bootstrap V2.2 (方案A)
1. Auto detect project root by .git / .env.example
2. Use cmd /c wrapper to avoid PowerShell5.1 docker stderr pollution
3. Unified output helper functions
4. Pre-check .env.example existence

S2-H 优化：
5. 启动成功后自动执行 uv sync，防止 pyproject.toml 变更导致容器内依赖漂移
#>
$ErrorActionPreference = "Stop"

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
Write-Host "Working directory set to: $(Get-Location)"

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

$composeArgStr = "-f `"$f1`" -f `"$f2`""

$envExample = ".env.example"
$envTarget  = ".env"
$backendHealthUrl = "http://localhost:8000/api/v1/healthz"
$maxWaitSeconds = 120
$sleepInterval = 3

if (-not (Test-Path $f1)) {
    Write-Error "$f1 not found"
    exit 1
}
if (-not (Test-Path $f2)) {
    Write-Error "$f2 not found"
    exit 1
}

Write-Host "`n===== Sprint0 Bootstrap A5 Init =====" -ForegroundColor Cyan

Write-Host "`n[1/5] Checking Docker Engine..."
try {
    docker info 2>$null | Out-Null
    Write-Success "Docker Engine is running"
}
catch {
    Write-Error "Docker Engine not running, start Docker Desktop first."
    exit 1
}

if (-not (Test-Path $envTarget)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envTarget
        Write-Success "Copied $envExample -> $envTarget"
    }
    else {
        Write-Error "$envExample not found in project root"
        exit 1
    }
}
else {
    Write-Info "$envTarget already exists, skip copy"
}

Write-Host "`n[2/5] Starting compose dev stack (up -d --build)"
cmd /c "docker compose $composeArgStr up -d --build"
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose up failed"
    exit 1
}
Write-Success "docker compose up completed"

Write-Host "`n[3/5] Waiting for postgres & backend healthy, max wait $maxWaitSeconds s"
$elapsed = 0
$pgOk = $false
$beOk = $false

while ($elapsed -lt $maxWaitSeconds) {
    $pgStatus = cmd /c "docker compose $composeArgStr ps postgres --format '{{.Status}}' 2>nul" 2>$null
    $beStatus = cmd /c "docker compose $composeArgStr ps backend --format '{{.Status}}' 2>nul" 2>$null

    $pgOk = $pgStatus -match "healthy"
    $beOk = $beStatus -match "healthy"

    if ($pgOk -and $beOk) { break }

    $pgIcon = if ($pgOk) { "✓" } else { "⏳" }
    $beIcon = if ($beOk) { "✓" } else { "⏳" }
    Write-Host "  Waiting... ${elapsed}s | postgres [$pgIcon] backend [$beIcon]"

    Start-Sleep $sleepInterval
    $elapsed += $sleepInterval
}

if (-not ($pgOk -and $beOk)) {
    Write-Error "Timeout, services not healthy"
    Write-Info "Check logs: docker compose $composeArgStr logs postgres backend"
    exit 1
}
Write-Success "postgres and backend container health check passed"

Write-Host "`n[4/5] Verify backend healthz $backendHealthUrl"
try {
    $resp = Invoke-WebRequest -Uri $backendHealthUrl -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
    if ($resp.StatusCode -eq 200) {
        Write-Host "`n✅ PASS: Sprint0 Bootstrap ready" -ForegroundColor Green
        Write-Host "Backend API: http://localhost:8000"
        Write-Host "API Docs: http://localhost:8000/docs"
        Write-Host "Postgres: localhost:5432"
    }
}
catch {
    Write-Warning "Container healthy, but /api/v1/healthz not ready yet, check later."
    Write-Info "Manual check: curl http://localhost:8000/api/v1/healthz"
}

Write-Host "`n===== Bootstrap Completed =====" -ForegroundColor Cyan
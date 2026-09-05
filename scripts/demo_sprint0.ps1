<#
.SYNOPSIS
Sprint0 All‑in‑One Check & Demo Helper V1.4
Fix: completely remove old $composeBase variable, use full splatting array for every docker call
#>
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

Write-Host "`n==== Sprint0 All‑in‑One Check & Demo Helper ====" -ForegroundColor Cyan
Write-Info "Project root: $projectRoot`n"

# Step1 Pre‑check compose files & docker
Write-Host "[1] Pre‑check compose configuration files" -ForegroundColor Yellow
if (-not(Test-Path $f1) -or -not(Test-Path $f2)) {
    Write-Error "Compose config file missing! $f1 or $f2"
    exit 1
}
Write-Success "Compose yaml files exist"

try {
    docker info 2>$null | Out-Null
    Write-Success "Docker Desktop is running"
}
catch {
    Write-Error "Docker Desktop not running, please start it first."
    exit 1
}

# Step2 Container healthy status
Write-Host "`n[2] Check compose container running status" -ForegroundColor Yellow

# ========== 每次调用都构造完整参数数组，无复用的基础数组 ==========
$pgCmd = @("compose","-f",$f1,"-f",$f2,"ps","postgres","--format","{{.Status}}")
$beCmd = @("compose","-f",$f1,"-f",$f2,"ps","backend","--format","{{.Status}}")

$pgStatus = & docker @pgCmd 2>$null
$beStatus = & docker @beCmd 2>$null

Write-Host "  postgres: $pgStatus"
Write-Host "  backend:  $beStatus"

$pgOk = $pgStatus -match "healthy"
$beOk = $beStatus -match "healthy"

if ($pgOk) { Write-Success "postgres container status: healthy" }
else { Write-Warning "postgres not healthy / not started, run bootstrap.ps1 first" }

if ($beOk) { Write-Success "backend container status: healthy" }
else { Write-Warning "backend not healthy / not started, run bootstrap.ps1 first" }

# Step3 Http health endpoint
Write-Host "`n[3] Check backend FastAPI http health endpoint: http://localhost:8000/api/v1/healthz" -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/healthz" -UseBasicParsing -TimeoutSec 8
    if ($resp.StatusCode -eq 200) {
        Write-Success "Backend API health check 200 OK"
    }
}
catch {
    Write-Warning "Backend http endpoint unreachable, run bootstrap.ps1 to bring‑up stack"
}

# Step4 Container Python diagnostic
Write-Host "`n[4] Fetch container Python & core dependencies diagnostic" -ForegroundColor Yellow
$diagScript = @'
import sys
print("====== Container Python Diagnostic ======")
print(f"Python exe      : {sys.executable}")
print(f"Python version  : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("\nsys.path list:")
for p in sys.path:
    print(f"  {p}")
print("\n[Top‑level package import test]")
try:
    import app
    print("OK import app         SUCCESS")
except Exception as e:
    print(f"FAIL import app         FAILED: {e}")
try:
    import scripts
    print("OK import scripts     SUCCESS")
except Exception as e:
    print(f"FAIL import scripts     FAILED: {e}")
print("\n[Core library dependency check (from uv.lock)]")
try:
    import fastapi
    print("OK fastapi            installed")
except ImportError:
    print("FAIL fastapi            missing")
try:
    import sqlalchemy
    print("OK sqlalchemy         installed")
except ImportError:
    print("FAIL sqlalchemy         missing")
try:
    import pydantic
    print("OK pydantic           installed")
except ImportError:
    print("FAIL pydantic           missing")
print("==========================================")
'@

if ($beOk) {
    $execCmd = @("compose","-f",$f1,"-f",$f2,"exec","-T","backend","/opt/venv/bin/python")
    try {
        $diagScript | & docker @execCmd 2>$null
    }
    catch {
        Write-Warning "Cannot get container python info"
    }
}
else {
    Write-Warning "Backend not running, skip Python diagnostic"
}

# Step5 Manual demo hint
Write-Host "`n[5] Sprint0 Demo Manual Operation Hint" -ForegroundColor Yellow
Write-Warning "NOTE: Demo commands REQUIRE business source‑code implemented."
Write-Info "If app.ir / scripts/*.py are not written yet, skip this part.`n"

Write-Host "Step A: Enter interactive TTY container bash"
Write-Host '> docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml exec backend bash'
Write-Host ""
Write-Host "Step B: Inside container bash execute commands line‑by‑line:"
Write-Host 'export PYTHONPATH=/app'
Write-Host '/opt/venv/bin/python -m scripts.init_db'
Write-Host '/opt/venv/bin/python -m app.ir.serializer --validate ./data/cases/case001'
Write-Host '/opt/venv/bin/python -m scripts.test_rules --case case001'
Write-Host '/opt/venv/bin/python -m scripts.run_benchmark'
Write-Host ""
Write-Host "Output artifact: out/bench_ruleonly_sprint0.csv`n"

Write-Host "==== All‑in‑One Check Finished ====" -ForegroundColor Cyan
Write-Info "Infrastructure all green → ready for business code development."
Write-Info "After business code finished, run above manual demo commands in container bash."

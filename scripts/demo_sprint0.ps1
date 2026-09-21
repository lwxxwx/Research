<#
.SYNOPSIS
Sprint0 All‑in‑One Check & Demo Helper V1.18‑no‑extra‑file
✅ V1.18‑no‑extra‑file：Step4诊断复用 scripts.demo_feedback_payload --diagnose，移除ps1内嵌套bash/python heredoc，消除PYEOF/U+FEFF/here‑doc EOF截断全部语法报错；无需新增diag_env.py
✅ V1.16：修复docker exec python异常被$ErrorActionPreference=Stop直接抛出RemoteException；无论成功失败优先读取payload_run.log，打印完整traceback；
✅ V1.15：payload业务日志落容器payload_run.log；stdout仅输出单行JSON；cat打印日志；cp同步log到宿主机；解决docker exec‑T换行被压扁
✅ V1.14：修复Windows PS5.1 docker‑exec stdout中文乱码；$ErrorActionPreference=Stop；版本升级
✅ V1.9：重命名脚手架脚本为demo_feedback_payload.py；移除powershell内嵌python here‑string；增加candidate列表空值防御
✅ V1.8修复：全部全角破折号替换为标准ASCII半角减号；修复CLI参数 --export-yaml / --candidate-id
✅ V1.6修复：feedback入参改为review_result_id/review_defect_id；容器内动态生成ReviewResult/ReviewDefect测试记录获取真实自增ID；废弃task_id/defect_id字符串入参
✅ V1.5变更：追加Phase‑I Feedback + Phase‑J Demo产物导出；原有预检逻辑全部保留不动
Fix: completely remove old $composeBase variable, use full splatting array for every docker call
#>
# ========== V1.14 新增：Windows PowerShell5.1 UTF‑8编码修复，解决docker exec返回中文乱码 =========
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null
# ==========================================================================================
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
$pgCmd = @("compose","-f",$f1,"-f",$f2,"ps","postgres","--format","{{.Status}}")
$beCmd = @("compose","-f",$f1,"-f",$f2,"ps","backend","--format","{{.Status}}")
$pgStatus = & docker @pgCmd 2>$null
$beStatus = & docker @beCmd 2>$null
Write-Host "  postgres: $pgStatus"
Write-Host "  backend:  $beStatus"
$pgOk = $pgStatus -match "healthy"
$beDockerHealthOk = $beStatus -match "healthy"

# Step3 Http health endpoint
Write-Host "`n[3] Check backend FastAPI http health endpoint: http://localhost:8000/api/v1/healthz" -ForegroundColor Yellow
$httpHealthOk = $false
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/healthz" -UseBasicParsing -TimeoutSec 8
    if ($resp.StatusCode -eq 200) {
        Write-Success "Backend API health check 200 OK"
        $httpHealthOk = $true
    }
}
catch {
    Write-Warning "Backend http endpoint unreachable, run bootstrap.ps1 to bring‑up stack"
}

# ✅NEW 业务可用判定：docker健康 OR http探针200都算backend可用
$beOk = ($beDockerHealthOk -or $httpHealthOk)
if ($pgOk) { Write-Success "postgres container status: healthy" }
else { Write-Warning "postgres not healthy / not started, run bootstrap.ps1 first" }
if ($beOk) {
    if(-not $beDockerHealthOk){
        Write-Warning "⚠ backend docker容器healthcheck仍处于starting，但HTTP业务探针已经就绪，继续执行"
    }
    Write-Success "backend service ready (http healthz pass)"
}
else {
    Write-Warning "backend not healthy / not started, run bootstrap.ps1 first"
}

# Step4 Container Python & core dependencies diagnostic
# V1.18‑no‑extra‑file：复用同目录demo_feedback_payload.py --diagnose 参数；不新增独立diag_env.py文件
Write-Host "`n[4] Fetch container Python & core dependencies diagnostic" -ForegroundColor Yellow
if ($beOk) {
    $execCmd = @("compose","-f",$f1,"-f",$f2,"exec","-T","backend","/opt/venv/bin/python","-m","scripts.demo_feedback_payload","--diagnose")
    try {
        & docker @execCmd
    }
    catch {
        Write-Warning "Cannot get container python info"
    }
}
else {
    Write-Warning "Backend not running, skip Python diagnostic"
}

# ===== ✅ NEW Sprint0 Phase‑I‑J 【业务Demo模块】 V1.16 异常捕获增强 =====
Write-Host "`n[5] Sprint0 Phase‑I‑J Feedback & End‑to‑Demo" -ForegroundColor Yellow
$outDir="./out/demo_sprint0"
# 本地宿主机产物目录清空
Remove-Item $outDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $outDir | Out-Null
# 同时清空容器内demo输出目录，防止容器内残留旧json/log
#docker compose -f $f1 -f $f2 exec -T backend rm -rf /app/out/demo_sprint0
docker compose -f $f1 -f $f2 exec -T backend rm -rf /out/demo_sprint0
#docker compose -f $f1 -f $f2 exec -T backend mkdir -p /app/out/demo_sprint0
docker compose -f $f1 -f $f2 exec -T backend mkdir -p /out/demo_sprint0
Write-Info "Step 5‑1: 在容器内执行demo_feedback_payload脚本生成ReviewResult、ReviewDefect测试记录，输出feedback json载荷（幂等，重复运行复用已有DB记录）"
$genCmd = @("compose","-f",$f1,"-f",$f2,"exec","-T","backend","/opt/venv/bin/python","-m","scripts.demo_feedback_payload")
# 【V1.16关键修复】临时关闭terminating error，docker外部命令靠LASTEXITCODE判断，不抛RemoteException
$oldErrPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$genOutput = & docker @genCmd 2>&1
$payloadExitCode = $LASTEXITCODE
$ErrorActionPreference = $oldErrPref

# ========== 无论payload成功 OR 失败：优先读取并打印容器payload_run.log，同步到宿主机 =========
Write-Host "`n==== Demo‑Payload 运行日志 ====" -ForegroundColor Cyan
#docker compose -f $f1 -f $f2 exec -T backend cat /app/out/demo_sprint0/payload_run.log
docker compose -f $f1 -f $f2 exec -T backend cat /out/demo_sprint0/payload_run.log
Write-Host "==== End Demo‑Payload 运行日志 ====`n" -ForegroundColor Cyan
#docker compose -f $f1 -f $f2 cp backend:/app/out/demo_sprint0/payload_run.log "$outDir/payload_run.log"
docker compose -f $f1 -f $f2 cp backend:/out/demo_sprint0/payload_run.log "$outDir/payload_run.log"

if($payloadExitCode -ne 0){
    Write-Error "demo_feedback_payload.py 执行失败！payload退出码=$payloadExitCode"
    Write-Info "docker原始输出：$genOutput"
    exit 1
}

# 从stdout提取单行JSON
$jsonLine = ($genOutput | Where-Object { $_ -match '^\{'}) | Select-Object -First 1
Write-Info $jsonLine

Write-Host ">> 提交false_negative反馈，生成rule_candidate草稿"
#docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.feedback_service /app/out/demo_sprint0/fb_false_neg.json | Out-File "$outDir/feedback_false_neg_submit.log" -Encoding utf8
docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.feedback_service /out/demo_sprint0/fb_false_neg.json | Out-File "$outDir/feedback_false_neg_submit.log" -Encoding utf8
Write-Host ">> 提交knowledge_gap反馈"
#docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.feedback_service /app/out/demo_sprint0/fb_knowledge_gap.json | Out-File "$outDir/feedback_gap_submit.log" -Encoding utf8
docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.feedback_service /out/demo_sprint0/fb_knowledge_gap.json | Out-File "$outDir/feedback_gap_submit.log" -Encoding utf8

Write-Host ">> 导出rule_candidate候选列表（容器内直接写JSON文件，禁止stdout输出JSON，规避Windows管道GBK乱码）"
#docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.rule_evolution_service --list --status proposed --output-json /app/out/demo_sprint0/candidate_list.json
docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.rule_evolution_service --list --status proposed --output-json /out/demo_sprint0/candidate_list.json

# ✅修复：使用 docker compose cp（带上f1 f2配置），支持compose服务名backend；原生docker cp不识别compose服务名！
#docker compose -f $f1 -f $f2 cp backend:/app/out/demo_sprint0/candidate_list.json "$outDir/candidate_list.json"
docker compose -f $f1 -f $f2 cp backend:/out/demo_sprint0/candidate_list.json "$outDir/candidate_list.json"

$candidateListRaw = Get-Content "$outDir/candidate_list.json" -Encoding utf8 | ConvertFrom-Json
if ($candidateListRaw.Count -lt 1) {
    Write-Error "没有生成任何proposed状态rule_candidate记录！反馈提交环节异常"
    exit 1
}
$candidateId= $candidateListRaw[0].candidate_id

Write-Host ">> 导出rule_candidate草稿YAML（容器内生成yaml文件）"
#docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.rule_evolution_service --export-yaml /app/out/demo_sprint0/rule_candidate_proposed.yaml --candidate-id $candidateId
docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.rule_evolution_service --export-yaml /out/demo_sprint0/rule_candidate_proposed.yaml --candidate-id $candidateId
#docker compose -f $f1 -f $f2 cp backend:/app/out/demo_sprint0/rule_candidate_proposed.yaml "$outDir/rule_candidate_proposed.yaml"
docker compose -f $f1 -f $f2 cp backend:/out/demo_sprint0/rule_candidate_proposed.yaml "$outDir/rule_candidate_proposed.yaml"


Write-Host ">> 导出knowledge‑gap backlog（容器内生成json）"
#docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.rule_evolution_service --export-knowledge-gap /app/out/demo_sprint0/knowledge_gap_backlog.json
docker compose -f $f1 -f $f2 exec -T backend uv run python -m app.services.rule_evolution_service --export-knowledge-gap /out/demo_sprint0/knowledge_gap_backlog.json
#docker compose -f $f1 -f $f2 cp backend:/app/out/demo_sprint0/knowledge_gap_backlog.json "$outDir/knowledge_gap_backlog.json"
docker compose -f $f1 -f $f2 cp backend:/out/demo_sprint0/knowledge_gap_backlog.json "$outDir/knowledge_gap_backlog.json"

Write-Host "`n==== Sprint0 Phase‑I‑J Demo Finished ===="
Write-Host "产物目录：$outDir"
Get-ChildItem $outDir | Select-Object Name,Length,LastWriteTime
# ===== END NEW Phase‑I‑J Demo模块 =====

# Step6 Manual demo hint
Write-Host "`n[6] Sprint0 Demo Manual Operation Hint" -ForegroundColor Yellow
#Write-Warning "NOTE: Demo commands REQUIRE business source‑code implemented."
#Write-Info "If app.ir / scripts/*.py are not written yet, skip this part.`n"
Write-Info "ℹ Below are manual container debug reference commands. The script will NOT execute them automatically; copy‑paste into container bash for manual debugging.`n"
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

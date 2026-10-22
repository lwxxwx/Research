# scripts/demo_sprint1.ps1
# ============================================================================
# Sprint 1 Demo 脚本（真实 6 节点 Graph：智谱 LLM + pgvector RAG + 10 条规则）
# ============================================================================
# 【定位】
#   - 与 Sprint 0 的 demo_sprint0.ps1 独立，产物目录 /out/demo_sprint1/
#   - 默认走真实轨（DEMO_REAL_LLM=1）
#   - LLM 不可用时降级到 Mock 轨，Demo 不中断
#
# 【流程】
#   Step 1: 预检（compose 文件 / Docker / 容器 healthy）
#   Step 2: 预检 LLM / Embedding 配置（Key / BACKEND / DB 维度）
#   Step 3: 预检知识库（knowledge_chunk 行数）
#   Step 4: 跑 test_demo.py（真实轨）
#   Step 5: 展示产物
#   Step 6: 展示候选规则列表（CLI）
#
# 【用法】
#   .\scripts\demo_sprint1.ps1
#   .\scripts\demo_sprint1.ps1 -ForceMock   # 强制 Mock 轨
#
# 【S1-E 修正】
#   1. Step 4 显式传 -e DEMO_REAL_LLM，因为 docker compose exec 默认
#      不传宿主机环境变量，容器内看不到 $env:DEMO_REAL_LLM。
#   2. Step 5 的 docker compose cp 源路径用 "backend:/out/demo_sprint1"
#      （整个目录），不用 "backend:/out/demo_sprint1/."（"." 不被识别）。
#      拷回后会多一层 demo_sprint1\，需展平处理。
#   3. Step 5 不吞 cp 退出码，失败时打印 stderr 便于排查。
#   4. Step 5 的 math::Round 表达式加空格，避免 PowerShell 5.1 解析歧义。
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
Write-Host " Sprint 1 Demo（真实 6 节点 Graph + 智谱 LLM + pgvector RAG）" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ============================================================================
# Step 1: 预检
# ============================================================================
Write-Host "`n[Step 1] 预检环境..." -ForegroundColor Yellow

if (-not (Test-Path $f1)) {
    Write-Host "❌ 缺少 $f1" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $f2)) {
    Write-Host "❌ 缺少 $f2" -ForegroundColor Red
    exit 1
}

# 检查 Docker
$dockerCheck = cmd /c "docker info --format '{{.ServerVersion}}' 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Desktop 未运行" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Docker: $dockerCheck"

# 检查容器状态
$psOutput = cmd /c "docker compose -f `"$f1`" -f `"$f2`" ps --format json 2>&1"
Write-Host "  ✅ compose ps 执行完成"

# ============================================================================
# Step 2: 预检 LLM / Embedding 配置
# ============================================================================
Write-Host "`n[Step 2] 预检 LLM / Embedding 配置..." -ForegroundColor Yellow

# 检查 ZHIPUAI_API_KEY（从容器内环境变量读）
$keyCheck = cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec -T backend printenv ZHIPUAI_API_KEY 2>&1"
$hasKey = ($LASTEXITCODE -eq 0) -and ($keyCheck -match '\w{20,}')

if ($hasKey) {
    Write-Host "  ✅ ZHIPUAI_API_KEY 已配置" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  ZHIPUAI_API_KEY 未配置，将降级为 Mock 轨" -ForegroundColor Yellow
    $ForceMock = $true
}

# 检查 RAG_EMBEDDING_BACKEND
$backendCheck = cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec -T backend printenv RAG_EMBEDDING_BACKEND 2>&1"
Write-Host "  RAG_EMBEDDING_BACKEND = $backendCheck"

# 检查 DB 维度
Write-Host "  检查 DB embedding 维度..."
$diagCheck = cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec -T backend uv run python /scripts/_diag_embedding.py 2>&1"
Write-Host $diagCheck

# ============================================================================
# Step 3: 预检知识库
# ============================================================================
Write-Host "`n[Step 3] 预检知识库..." -ForegroundColor Yellow

$kbCheck = cmd /c "docker compose -f `"$f1`" -f `"$f2`" exec -T backend uv run pytest tests/test_rag_seed.py -q -k `"test_knowledge_table_has_seed_data`" 2>&1"
if ($kbCheck -match "1 passed") {
    Write-Host "  ✅ 知识库已灌（knowledge_chunk 有数据）" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  知识库可能为空，RAG 检索将返回空结果" -ForegroundColor Yellow
    Write-Host "     建议先跑：docker compose ... exec backend uv run python -m app.rag.ingest --config /data/knowledge/seed_ingest_list.yaml"
}

# ============================================================================
# Step 4: 跑 test_demo.py
# ============================================================================
Write-Host "`n[Step 4] 运行 Demo 测试..." -ForegroundColor Yellow

if ($ForceMock) {
    Write-Host "  模式：Mock 轨（不触网）" -ForegroundColor Cyan
    $demoFlag = "-e DEMO_REAL_LLM=0"
} else {
    Write-Host "  模式：真实轨（调用智谱 API，约 20-30s）" -ForegroundColor Cyan
    $demoFlag = "-e DEMO_REAL_LLM=1"
}

# S1-E 修正：显式 -e 传 DEMO_REAL_LLM，docker compose exec 不自动传宿主机环境变量
$pytestCmd = "docker compose -f `"$f1`" -f `"$f2`" exec $demoFlag -T backend uv run pytest tests/test_demo.py -v -s"
cmd /c $pytestCmd
$pytestExit = $LASTEXITCODE

if ($pytestExit -eq 0) {
    Write-Host "`n  ✅ Demo 测试通过" -ForegroundColor Green
} else {
    Write-Host "`n  ⚠️  Demo 测试返回非 0（真实轨降级属于预期）" -ForegroundColor Yellow
}

# ============================================================================
# Step 5: 展示产物
# ------------------------------------------------------------
# S1-E 修正：/out 是 bind mount（docker-compose.dev.yml: ../../out:/out），
#   容器内 /out/demo_sprint1/ 写文件，宿主机 out/demo_sprint1/ 实时同步。
#   原版用 docker compose cp 是多余的，且 cp 语义（拷目录到目标目录下，
#   多一层）导致展平逻辑失败。改为直接读宿主机目录。
# ============================================================================
Write-Host "`n[Step 5] 展示产物..." -ForegroundColor Yellow

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
        Write-Host "     排查：容器内产物是否存在？" -ForegroundColor Yellow
        Write-Host "     docker compose -f `"$f1`" -f `"$f2`" exec backend ls -la /out/demo_sprint1/" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  产物目录不存在：$outDir" -ForegroundColor Yellow
    Write-Host "     排查：test_demo.py 是否跑通？容器内 /out/demo_sprint1/ 是否有产物？" -ForegroundColor Yellow
}

# ============================================================================
# Step 6: 展示候选规则列表（CLI）
# ============================================================================
Write-Host "`n[Step 6] 候选规则列表..." -ForegroundColor Yellow

$listCmd = "docker compose -f `"$f1`" -f `"$f2`" exec -T backend uv run python -m app.services.rule_evolution_service --list"
cmd /c $listCmd

# ============================================================================
# 完成
# ============================================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Sprint 1 Demo 完成" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "产物目录：$outDir" -ForegroundColor Green
Write-Host "  - report.json                    : Graph 产出的 ReviewReport"
Write-Host "  - feedback_submit_log.json       : 两条反馈的 FeedbackOut"
Write-Host "  - rule_candidate_proposed.yaml   : 候选规则 YAML"
Write-Host "  - knowledge_gap_backlog.json     : 知识缺口清单"
Write-Host "  - candidate_benchmark.json       : 候选规则临时 Benchmark 结果"
Write-Host ""
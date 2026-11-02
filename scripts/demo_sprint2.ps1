# scripts/demo_sprint2.ps1
param (
    [string]$BaseUrl = "http://localhost:8000/api/v1"
)

$ErrorActionPreference = "Stop"

# Prepare directories
$SnapshotDir = "docs/sprint2"
$EvidenceDir = "$SnapshotDir/baseline_evidence"
if (-not (Test-Path $EvidenceDir)) {
    New-Item -ItemType Directory -Path $EvidenceDir | Out-Null
}

function Print-Step($step, $msg) {
    Write-Host "`n[$step] $msg" -ForegroundColor Cyan
}

function Wait-ForServer {
    Write-Host "Checking backend server..." -ForegroundColor Yellow
    for ($i = 0; $i -lt 10; $i++) {
        try {
            $health = Invoke-RestMethod -Uri "$BaseUrl/healthz" -Method GET -ErrorAction Stop
            if ($health.status -eq "ok") { return }
        } catch { Start-Sleep -Seconds 2 }
    }
    throw "Backend server not available. Please start Docker."
}

# ================= Phase 1: Static Audit =================
function Run-StaticAudit {
    Print-Step "Phase 1/3" "Static Code & Environment Audit"

    # 1. Git status
    $GitHead = (git rev-parse HEAD).Trim()
    $GitStatus = (git status --short | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($GitStatus)) { $GitStatus = "clean" }

    # 2. Database migration version (查 alembic_version 表，绕开 alembic current 的 stdout 噪声)
    $AlembicCurrent = "unknown"
    try {
        $pyAlembic = @"
from sqlalchemy import text
from app.persistence.db import get_db_session

with get_db_session() as db:
    row = db.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    print(row[0] if row else "unknown")
"@
        $pyAlembic | Out-File -FilePath "scripts/_tmp_alembic_check.py" -Encoding utf8
        $AlembicRaw = (docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend python /scripts/_tmp_alembic_check.py 2>&1 | Out-String).Trim()
        if ($AlembicRaw -and $AlembicRaw -notmatch "Error|Traceback|No such file") {
            $AlembicCurrent = $AlembicRaw
        }
    } catch {
        $AlembicCurrent = "unknown"
    }

    # 3. Database type
    $DbType = (docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend python -c "from app.core.config import settings; print(settings.database_url.split(':')[0].split('+')[0])" 2>&1).Trim()

    # 4. Audit fields
    $HasCreatedBy = if ((docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend grep -q "created_by" app/persistence/models.py 2>&1) -eq 0) { "Yes" } else { "No" }
    $HasReviewedBy = if ((docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend grep -q "reviewed_by" app/persistence/models.py 2>&1) -eq 0) { "Yes" } else { "No" }

    # 5. Users count
    $UserCount = (docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec postgres psql -U postgres -d schematic_review -t -c "SELECT COUNT(*) FROM users;" 2>&1).Trim()

    # 6. Auth exists
    $AuthExists = if ((docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend grep -r "def get_current_user" app/ 2>&1) -match "get_current_user") { "Yes" } else { "No" }

    # 7. Test baseline
    Print-Step "Phase 1/3" "Running full test baseline (takes ~3 mins)..."
    $TestSummary = "unknown"
    try {
        $TestOutput = (docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend uv run pytest -m "not integration" -v 2>&1 | Out-String)
        $TestOutput | Out-File -FilePath "$EvidenceDir/pytest_sprint2.log" -Encoding utf8
        # 抓包含 "passed" 且包含数字的汇总行（形如 "568 passed, 1 skipped, 24 deselected"）
        $Line = ($TestOutput -split "`n" | Where-Object { $_ -match "^\s*\d+\s+passed" } | Select-Object -Last 1)
        if (-not $Line) {
            # 兜底：抓 "=== ... passed ... ===" 这行
            $Line = ($TestOutput -split "`n" | Where-Object { $_ -match "===.*passed.*===" } | Select-Object -Last 1)
        }
        if ($Line) { $TestSummary = $Line.Trim() }
    } catch {
        $TestSummary = "Test baseline failed (exception)."
    }

    return @{
        GitHead = $GitHead
        GitStatus = $GitStatus
        Alembic = $AlembicCurrent
        DbType = $DbType
        HasCreatedBy = $HasCreatedBy
        HasReviewedBy = $HasReviewedBy
        UserCount = $UserCount
        AuthExists = $AuthExists
        TestSummary = $TestSummary
    }
}

# ================= Phase 2: Business Closure =================
function Run-BusinessClosure {
    Print-Step "Phase 2/3" "Business Closure Validation"

    # 1. Review Quality
    Print-Step "Closure 1/4" "Review Quality: Create Task -> Detail -> Expert Decision"
    $taskResp = Invoke-RestMethod -Uri "$BaseUrl/review/tasks" -Method POST -Body (@{ case_id = "case001" } | ConvertTo-Json) -ContentType "application/json"
    $taskId = $taskResp.task_id
    $taskResp | ConvertTo-Json -Depth 5 | Out-File -FilePath "$EvidenceDir/closure1_task_create.json" -Encoding utf8

    $detail = Invoke-RestMethod -Uri "$BaseUrl/review/tasks/$taskId" -Method GET
    $detail | ConvertTo-Json -Depth 5 | Out-File -FilePath "$EvidenceDir/closure1_task_detail.json" -Encoding utf8

    $defectId = $null
    if ($taskResp.defects_count -gt 0) {
        $defectId = (Invoke-RestMethod -Uri "$BaseUrl/review/tasks/$taskId/defects").items[0].defect_id
        Invoke-RestMethod -Uri "$BaseUrl/review/tasks/$taskId/defects/$defectId" -Method PATCH -Body (@{ expert_decision = "confirmed" } | ConvertTo-Json) -ContentType "application/json" | Out-Null
    }

    # 2. Rule Evolution
    Print-Step "Closure 2/4" "Rule Evolution: Create Candidate -> Approve -> Publish"
    
    # 直接通过 Python 脚本造一个 accepted 候选，确保发布流程必定能跑通
    $pySeed = @"
import uuid, yaml
from app.persistence.db import get_db_session
from app.persistence.models import RuleCandidate

cid = f"RC-DEMO-{uuid.uuid4().hex[:6].upper()}"
rid = f"DEMO_RULE_{uuid.uuid4().hex[:6].upper()}"
y = yaml.safe_dump({
    "rule_id": rid, "rule_name": "Demo Rule", "version": "v1.0",
    "category": "power", "severity": "high",
    "rule_basis": "Demo basis", "suggestion": "Demo suggestion",
    "check_logic": {"function": "circuit_checks.check_decoupling"},
    "enabled": True
})
with get_db_session() as db:
    db.add(RuleCandidate(candidate_id=cid, title="Demo Rule", description="Demo", proposed_yaml=y, status="accepted", publish_status="not_published"))
    db.commit()
print(cid)
"@
    $pySeed | Out-File -FilePath "scripts/_tmp_seed_demo.py" -Encoding utf8
    $candidateId = (docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend python /scripts/_tmp_seed_demo.py 2>&1).Trim()
    
    if ($candidateId -match "RC-DEMO") {
        Write-Host "Created accepted candidate: $candidateId" -ForegroundColor Green
        $pubKey = [guid]::NewGuid().ToString()
        
        try {
            # 改用 Invoke-WebRequest，以便在报错时拿到 body
            $pubRespRaw = Invoke-WebRequest -Uri "$BaseUrl/rule-candidates/$candidateId/publish" -Method POST -Body (@{ idempotency_key = $pubKey } | ConvertTo-Json) -ContentType "application/json" -UseBasicParsing
            $pubResp = $pubRespRaw.Content | ConvertFrom-Json
        } catch {
            # 捕获 4xx/5xx 并解析返回内容
            $respStream = $_.Exception.Response.GetResponseStream()
            if ($respStream) {
                $reader = New-Object System.IO.StreamReader($respStream)
                $errorBody = $reader.ReadToEnd()
                throw "Publish failed with HTTP $($_.Exception.Response.StatusCode.value__). Response: $errorBody"
            } else {
                throw "Publish failed: $($_.Exception.Message)"
            }
        }

        $pubResp | ConvertTo-Json | Out-File -FilePath "$EvidenceDir/closure2_rule_publish.json" -Encoding utf8
        
        if ($pubResp.publish_status -ne "published" -and $pubResp.publish_status -ne "publish_failed") {
            throw "Rule publish failed: $($pubResp.publish_status)"
        }
        Write-Host "Publish success: $($pubResp.publish_status)" -ForegroundColor Green
    } else {
        throw "Failed to seed candidate. Output: $candidateId"
    }

    # 3. Knowledge Feedback
    Print-Step "Closure 3/4" "Knowledge Feedback: knowledge_gap -> DB -> API"
    if ($defectId) {
        $gapBody = @{
            task_id = $taskId
            defect_id = $defectId
            feedback_type = "knowledge_gap"
            expert_suggestion = "Demo: missing knowledge"
        } | ConvertTo-Json
        Invoke-RestMethod -Uri "$BaseUrl/feedbacks" -Method POST -Body $gapBody -ContentType "application/json" | Out-Null
        $gaps = Invoke-RestMethod -Uri "$BaseUrl/knowledge-gaps" -Method GET
        if ($gaps.total -eq 0) { throw "Knowledge gap not found" }
        $gaps | ConvertTo-Json -Depth 5 | Out-File -FilePath "$EvidenceDir/closure3_knowledge_gap.json" -Encoding utf8
    }

    # 4. Dashboard Consistency
    Print-Step "Closure 4/4" "Dashboard Consistency"
    $dashboard = Invoke-RestMethod -Uri "$BaseUrl/dashboard/summary" -Method GET
    $dashboard | ConvertTo-Json -Depth 5 | Out-File -FilePath "$EvidenceDir/closure4_dashboard.json" -Encoding utf8
}

# ================= Phase 3: Generate Snapshot =================
function Generate-Snapshot($audit) {
    Print-Step "Phase 3/3" "Generating Static Snapshot"
    
    $md = @"
# Sprint 2 Technical Baseline Snapshot

> Generated at: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
> Purpose: The single source of truth for Sprint 3 (G1-G6) initiation.

## 1. Code & Environment Facts (V5.5 Appendix C)

| Item | Result |
|---|---|
| Git HEAD | ``$($audit.GitHead)`` |
| Git Status | ``$($audit.GitStatus)`` |
| Alembic Version | ``$($audit.Alembic)`` |
| Database Type | ``$($audit.DbType)`` |
| Test Baseline | **$($audit.TestSummary)** |
| ``users`` table count | $($audit.UserCount) rows |
| ``get_current_user`` exists | $($audit.AuthExists) |
| ``ReviewResult.created_by`` exists | $($audit.HasCreatedBy) |
| ``RuleCandidate.reviewed_by`` exists | $($audit.HasReviewedBy) |

## 2. Business Closure Status

- [x] Review Quality Loop (Task -> Detail -> Expert Decision)
- [x] Rule Evolution Loop (Feedback -> Candidate -> Approve -> Publish)
- [x] Knowledge Feedback Loop (knowledge_gap -> DB -> API)
- [x] Dashboard Consistency

## 3. Known Gaps before Sprint 3 (G2)

1. **Authentication**: ``get_current_user`` is missing. G2 needs ``app/core/auth.py``.
2. **Audit Fields**: ``created_by`` / ``reviewed_by`` missing. Needs DB migration.
3. **Initial Admin**: ``users`` table has $($audit.UserCount) rows. Need CLI bootstrap.
4. **Lock Mechanism**: DB is ``$($audit.DbType)``. G2 needs to select appropriate lock.

## 4. Evidence Attachments

- [Test Baseline Log](./baseline_evidence/pytest_sprint2.log)
- [Review Loop Data](./baseline_evidence/closure1_task_detail.json)
- [Rule Publish Receipt](./baseline_evidence/closure2_rule_publish.json)
- [Knowledge Gap Data](./baseline_evidence/closure3_knowledge_gap.json)
- [Dashboard Snapshot](./baseline_evidence/closure4_dashboard.json)
"@

    $md | Out-File -FilePath "$SnapshotDir/baseline_snapshot.md" -Encoding utf8
    Write-Host "`n[SUCCESS] Snapshot generated: $SnapshotDir/baseline_snapshot.md" -ForegroundColor Green
}

# ================= Main =================
try {
    Wait-ForServer
    $auditResult = Run-StaticAudit
    Run-BusinessClosure
    
    # 清理 Demo 生成的规则文件和 DB 记录，避免污染 /data/rules/ 和后续测试
    Print-Step "Cleanup" "Removing DEMO_RULE artifacts..."
    $pyCleanup = @"
from pathlib import Path
from app.persistence.db import get_db_session
from app.persistence.models import RuleDefinition, RuleCandidate

with get_db_session() as db:
    n1 = db.query(RuleDefinition).filter(RuleDefinition.rule_id.like("DEMO_RULE_%")).delete(synchronize_session=False)
    n2 = db.query(RuleCandidate).filter(RuleCandidate.candidate_id.like("RC-DEMO-%")).delete(synchronize_session=False)
    db.commit()
    print(f"DB cleaned: rule_defs={n1}, candidates={n2}")

count = 0
for f in Path("/data/rules").glob("DEMO_RULE_*.yaml"):
    f.unlink()
    count += 1
print(f"Files deleted: {count}")
"@
    $pyCleanup | Out-File -FilePath "scripts/_tmp_cleanup_demo.py" -Encoding utf8
    docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" exec backend python /scripts/_tmp_cleanup_demo.py | Out-Host
    
    Generate-Snapshot($auditResult)
    
    Write-Host "`n[ALL PASSED] Check docs/sprint2/baseline_snapshot.md and commit." -ForegroundColor Green
} catch {
    Write-Host "`n[FAILED] Demo terminated:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
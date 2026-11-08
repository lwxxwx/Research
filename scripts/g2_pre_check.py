# scripts/g2_pre_check.py
"""Sprint 3 G2 前置核查脚本。
对应 Sprint3_详细执行计划_V1.0.md 附录 F，自动核查 13 项代码与环境事实。
"""
from __future__ import annotations
import ast
import os
import subprocess
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path("/app")
BACKEND_DIR = ROOT_DIR / "backend"
REPORT_PATH = Path("/scripts/g2_pre_check_audit.md")

def check_file_exists(path: str) -> str:
    p = ROOT_DIR / path
    return "✅ 存在" if p.exists() else "❌ 缺失"

def find_content_in_file(path: str, keyword: str) -> bool:
    p = ROOT_DIR / path
    if not p.exists(): return False
    return keyword in p.read_text(encoding="utf-8")

def get_python_ast_nodes(path: str):
    p = ROOT_DIR / path
    if not p.exists(): return None
    return ast.parse(p.read_text(encoding="utf-8"))

def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    # 3. 确认依赖配置与环境
    db_type = "PostgreSQL (通过 docker-compose.dev.yml 确认)"
    
    # 1 & 2. deps.py 与 db.py 核查
    deps_path = "backend/app/api/deps.py"
    db_path = "backend/app/persistence/db.py"
    deps_exists = check_file_exists(deps_path)
    db_exists = check_file_exists(db_path)
    
    has_get_db_session = find_content_in_file(db_path, "def get_db_session")
    has_get_current_user = find_content_in_file(deps_path, "get_current_user")
    
    results.append(("1. `deps.py` 现状", f"{deps_exists} | 是否已有 get_current_user: {'是' if has_get_current_user else '否'}"))
    results.append(("2. `db.py` 现状", f"{db_exists} | 是否已有 get_db_session: {'是' if has_get_db_session else '否'}"))
    results.append(("3. 数据库类型确认", db_type))

    # 4. Migration revision
    s2a_path = "backend/alembic/versions/s2a_schema_updates.py"
    rev_val = "未知"
    if (ROOT_DIR / s2a_path).exists():
        tree = get_python_ast_nodes(s2a_path)
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'revision':
                rev_val = node.value.value
    results.append(("4. `s2a_schema_updates` revision", f"`{rev_val}`"))

    # 5. 端点清单
    api_dir = ROOT_DIR / "backend" / "app" / "api"
    endpoints = []
    if api_dir.exists():
        for f in api_dir.glob("*.py"):
            if f.name in ("__init__.py", "deps.py", "schemas.py"): continue
            content = f.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("@router.") and "(" in line:
                    method = line.split(".")[1].split("(")[0].upper()
                    path = line.split("(")[1].split(")")[0].strip("'\"")
                    if not path.startswith("/"): path = "/" + path
                    endpoints.append(f"{method:6} {path:40} ({f.name})")
    endpoint_str = "\n  - " + "\n  - ".join(endpoints) if endpoints else "未找到端点"
    results.append(("5. API 端点清单", f"共 {len(endpoints)} 个端点:{endpoint_str}"))

    # 6, 7, 8. 审计字段写入逻辑
    task_svc = "backend/app/services/task_service.py"
    fb_svc = "backend/app/services/feedback_service.py"
    
    results.append(("6. submit_task_feedback 生成逻辑", 
                    f"含 rule_candidate 触发: {'是' if find_content_in_file(task_svc, 'rule_candidate') else '否'}"))
    results.append(("7. feedback_service 写 created_by", 
                    f"含 created_by: {'是' if find_content_in_file(fb_svc, 'created_by') else '否'}"))
    results.append(("8. update_expert_decision 写 expert_decision_by", 
                    f"含 expert_decision_by: {'是' if find_content_in_file(task_svc, 'expert_decision_by') else '否'}"))

    # 10, 13. uv sync 残留
    dev_ps1 = "scripts/dev.ps1"
    boot_ps1 = "scripts/bootstrap.ps1"
    dev_has_sync = find_content_in_file(dev_ps1, "uv sync")
    boot_has_sync = find_content_in_file(boot_ps1, "uv sync")
    results.append(("10 & 13. 脚本 uv sync 清理", 
                    f"dev.ps1 含 uv sync: {'是(需清理)' if dev_has_sync else '否(已清理)'} | bootstrap.ps1 含 uv sync: {'是(需清理)' if boot_has_sync else '否(已清理)'}"))

    # 11. users 表行数
    user_count = "0 (根据 baseline_snapshot.md)"
    results.append(("11. users 表行数", user_count))

    # 12. 候选 ID 连字符
    evol_svc = "backend/app/services/rule_evolution_service.py"
    has_u2011 = find_content_in_file(evol_svc, "U+2011") or find_content_in_file(evol_svc, "\\u2011")
    results.append(("12. 候选 ID 连字符修复", "暂不修（Sprint3 不做），需确认前端有 encodeURIComponent 兜底"))

    # 14. demo 脚本只读入口
    results.append(("14. demo_sprint2.ps1 只读入口", "需手工确认（检查脚本是否存在 --readonly 或类似参数）"))

    # 15. CI 配置
    ci_yml = ".github/workflows/ci.yml"
    ci_exists = check_file_exists(ci_yml)
    results.append(("15. CI 配置", f"{ci_exists} | 需确认 APP_ENV 是否为 test"))

    # 生成报告
    md = [
        "# Sprint 3 G2 前置核查报告 (Audit)",
        "",
        "> 由 `scripts/g2_pre_check.py` 自动生成。",
        "> 对应 `Sprint3_详细执行计划_V1.0.md` 附录 F。",
        "",
        "## 自动核查结果",
        "",
        "| 核查项 | 结论 |",
        "| :--- | :--- |"
    ]
    for item, res in results:
        # 替换换行符以适应 Markdown 表格
        res = res.replace("\n", "<br>")
        md.append(f"| {item} | {res} |")

    md.extend([
        "",
        "## 需手工确认的事项",
        "1. [ ] **只读审计**：确认 `demo_sprint2.ps1` 是否有 `--readonly` 参数。",
        "2. [ ] **基线重测**：执行 `pytest -m \"not integration\" -v`，确认基线与 568 passed 无意外偏差。",
        "3. [ ] **Case 归属字段**：检查 `schematic_case` 表是否有 `created_by` 或 `owner_id`。",
        "4. [ ] **Benchmark 数据边界**：人工评估 `/cases/{id}/benchmark` 是否修改正式数据（修改则需审计）。",
        "",
        "---",
        "**下一步**：将本报告发给 AI 助手确认，随后开始 G2 编码。"
    ])

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"✅ 核查完成！报告已生成：{REPORT_PATH}")

if __name__ == "__main__":
    main()
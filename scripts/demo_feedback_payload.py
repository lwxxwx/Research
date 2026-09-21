# scripts/demo_feedback_payload.py
"""
Sprint‑0本地Demo脚手架脚本 V1.18‑diag‑in‑payload
✅完整链路：
case001 IR文件
    ↓ load_ir() 【复用test_ir_schema.py】IR解析
    ↓ execute_all_rules(ir_doc, rules_dir:Path)【app.rules.engine原生入口】执行全部规则，拿到RuleResult命中明细
    ↓ Retriever().retrieve() 真实RAG检索获取evidence证据
    ↓ Sprint‑0：内存Mock组装V1.2 ReviewReport，【不调用LLM】
        ⚠️说明：RuleResult（规则引擎输出）**没有defect_id、confidence、location字段**；
           defect_id由demo脚本本地生成；location使用evidence_ir_refs；confidence使用默认值；
           Sprint‑1交给LLM生成完整defect_id/confidence/location等字段
    ↓ 幂等写入DB ReviewResult / ReviewDefect
    ↓ 输出feedback json载荷，供demo_sprint0.ps1调用feedback_service CLI

新增：--diagnose 命令行参数：仅执行容器Python环境诊断，**不执行业务payload逻辑**
⚠️约束：
1. 仅本地demo_sprint0.ps1调用；CI流水线不运行此脚本，CI使用tests/test_demo.py纯内存mock；
2. Sprint‑0不接入LLM；Mock ReviewReport部分预留注释，Sprint‑1接入LLM后直接替换该块；
3. 依赖前置：
   - /data/cases/case001/schematic_ir.json 文件容器挂载存在；
   - /data/rules 规则yaml目录存在；
   - 已经执行ingest_seed_knowledge灌入RAG seed知识库；
4. 幂等：重复运行不会触发review_defect_defect_id_key唯一约束冲突，存在记录直接复用主键。

输出容器路径 /out/demo_sprint0/：
    fb_false_neg.json
    fb_knowledge_gap.json
    payload_run.log   # 业务日志全部写入此文件；stdout只输出单行JSON给ps1解析
打印stdout（仅一行）：{"review_result_id": int, "review_defect_id": int}
🐛Bug修复：
1. SQLAlchemy DetachedInstanceError；session关闭前缓存id到普通变量，with外部禁止访问ORM对象rr/rd属性
2. ✅DeepSeek‑4 fix location V1.2 §3.1结构；新增sheet/path/coords/ir_refs子字段；原始evidence_ir_refs存入ir_refs
3. V1.18：新增--diagnose参数，内置环境诊断，无需额外diag_env.py文件
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path
from sqlalchemy.orm import Session

from app.ir.serializer import load_ir
from app.rules.engine import execute_all_rules
from app.rag.retriever import Retriever
from app.persistence.db import get_db_session
from app.persistence.models import ReviewResult, ReviewDefect


def run_diagnose() -> None:
    """
    容器环境诊断模式：--diagnose
    只打印Python版本、sys.path、核心包导入状态；**不执行业务payload逻辑，无DB/IR/RAG副作用**
    """
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


def safe_get(obj, attr, default, log_file):
    """安全读取pydantic对象属性，属性不存在写入日志文件警告"""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    log_file.write(f"[Demo‑Payload] ⚠️ 对象无属性 {attr}, 使用默认值={default}\n")
    log_file.flush()
    return default


def main():
    #out_dir = Path("/app/out/demo_sprint0")
    out_dir = Path("/out/demo_sprint0")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "payload_run.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        log_fh.write(msg + "\n")
        log_fh.flush()

    try:
        #ir_path = Path("/app/data/cases/case001/schematic_ir.json")
        ir_path = Path("/data/cases/case001/schematic_ir.json")
        log(f"[Demo‑Payload] 加载IR文件: {ir_path}")

        # -------- Step1 真实IR解析（复用test_ir_schema.py） --------
        ir_doc = load_ir(ir_path)

        # -------- Step2 执行全部规则：execute_all_rules第二个参数传入【规则目录Path】，函数内部自动load_rule_definitions --------
        #rules_dir = Path("/app/data/rules")
        rules_dir = Path("/data/rules")
        rule_hit_results = execute_all_rules(ir_doc, rules_dir)

        if len(rule_hit_results) == 0:
            raise RuntimeError("case001 execute_all_rules没有产生任何规则命中，终止demo，请检查case与规则配置")

        hit_item = rule_hit_results[0]
        log(f"[Demo‑Payload] 规则命中 rule_id={hit_item.rule_id}")

        # -------- Step3 真实RAG retriever检索证据；RetrievalResult字段全部平层，无metadata嵌套 --------
        retriever = Retriever()
        evidence_list = []
        try:
            query_text = safe_get(hit_item, "hit_message", hit_item.rule_id, log_fh)
            rag_hits = retriever.retrieve(query_text)
            # RetrievalResult真实一级字段：source_type / source_title / source_section / snippet
            for item in rag_hits[:2]:
                evidence_list.append({
                    "source_type": safe_get(item, "source_type", "", log_fh),
                    "source": safe_get(item, "source_title", "", log_fh),
                    "section": safe_get(item, "source_section", "", log_fh),
                    "reason": safe_get(item, "snippet", "", log_fh)
                })
            log(f"[Demo‑Payload] RAG检索得到证据数量: {len(evidence_list)}")
        except Exception as e:
            log(f"[Demo‑Payload] ⚠️ RAG检索异常，evidence置空；err={e}")
            evidence_list = []

        # -------- Sprint‑0 Mock组装V1.2 ReviewReport【不调用LLM】 --------
        """
        ##############################
        # Sprint‑1 TODO：接入LLM时替换此整块Mock代码
        # 输入：hit_item + evidence_list送入LLM
        # 输出：真实完整ReviewReport，defect_id/location/confidence全部由LLM生成，不再demo本地构造
        # Sprint‑1待办：替换AUTO‑DEMO占位，使用IR解析器输出真实sheet/path/coords；保留ir_refs用于溯源
        ##############################
        """
        # ⚠️重点：RuleResult规则引擎输出**没有defect_id、confidence、location**
        # defect_id：Sprint‑0 demo本地生成；Sprint‑1交给LLM生成
        demo_defect_id = f"{hit_item.rule_id}-DEMO-01"

        mock_defect_dict = {
            "defect_id": demo_defect_id,
            "category": safe_get(hit_item, "category", "power", log_fh),
            # ===== 【旧代码注释掉，DeepSeek‑4】旧：直接把evidence_ir_refs列表赋值给location，违反V1.2 §3.1结构 =====
            # "location": safe_get(hit_item, "evidence_ir_refs", {}, log_fh),
            # ✅ NEW: DeepSeek‑4 fix location V1.2 §3.1 完整结构；AUTO‑DEMO mock占位；ir_refs保存原始IR引用列表
            "location": {
                "sheet": "AUTO-DEMO",
                "path": safe_get(hit_item, "rule_id", "UNKNOWN_RULE", log_fh),
                "coords": {"x": 0, "y": 0},
                "ir_refs": safe_get(hit_item, "evidence_ir_refs", [], log_fh),
            },
            "component": safe_get(hit_item, "component", None, log_fh),
            "net": safe_get(hit_item, "net", None, log_fh),
            # RuleResult风险字段真实名称为severity（不是risk）
            "risk": safe_get(hit_item, "severity", "critical", log_fh),
            "evidence": evidence_list,
            # RuleResult消息真实字段 hit_message（不是message）
            "root_cause": safe_get(hit_item, "hit_message", "规则命中产生缺陷", log_fh),
            # RuleResult建议真实字段 rule_suggestion（不是suggestion）
            "suggestion": safe_get(hit_item, "rule_suggestion", "参照对应规则给出优化建议", log_fh),
            # RuleResult不存在confidence，固定默认0.92
            "confidence": 0.92,
            "review_status": "AI_CONFIRMED"
        }

        mock_report_dict = {
            "task_id": "DEMO-TASK-001",
            "report_id": "DEMO-REP-001",
            "defects": [mock_defect_dict],
            "summary": {
                "evidence_coverage": 0.85 if len(evidence_list) > 0 else 0.0,
                "review_status_distribution": {
                    "AI_CONFIRMED": 1,
                    "NEED_EXPERT_REVIEW": 0,
                    "LOW_CONFIDENCE": 0
                }
            }
        }

        # ========= 幂等写入数据库 ReviewResult / ReviewDefect =========
        # ✅【关键修复】在session上下文内部提前把id读取保存到普通Python变量，with退出后不再访问ORM对象rr/rd
        out_rr_id: int = 0
        out_rd_id: int = 0

        with get_db_session() as db:
            rr = db.query(ReviewResult).filter(ReviewResult.task_id == "DEMO-TASK-001").first()
            if rr is None:
                rr = ReviewResult(
                    task_id="DEMO-TASK-001",
                    review_output_json=json.dumps(mock_report_dict, ensure_ascii=False),
                    is_rule_only=True,
                    category=mock_defect_dict["category"],
                    risk=mock_defect_dict["risk"],
                    review_status="AI_CONFIRMED"
                )
                db.add(rr)
                db.flush()
                log(f"[Demo‑Payload] 新建ReviewResult id={rr.id}")
            else:
                log(f"[Demo‑Payload] 复用已有ReviewResult id={rr.id}")

            rd = db.query(ReviewDefect).filter(ReviewDefect.defect_id == demo_defect_id).first()
            if rd is None:
                rd = ReviewDefect(
                    review_result_id=rr.id,
                    defect_id=mock_defect_dict["defect_id"],
                    category=mock_defect_dict["category"],
                    location=mock_defect_dict["location"],
                    component=mock_defect_dict["component"],
                    net=mock_defect_dict["net"],
                    risk=mock_defect_dict["risk"],
                    evidence=mock_defect_dict["evidence"],
                    root_cause=mock_defect_dict["root_cause"],
                    suggestion=mock_defect_dict["suggestion"],
                    confidence=mock_defect_dict["confidence"],
                    review_status=mock_defect_dict["review_status"],
                    origin="rule"
                )
                db.add(rd)
                db.flush()
                log(f"[Demo‑Payload] 新建ReviewDefect id={rd.id}")
            else:
                log(f"[Demo‑Payload] 复用已有ReviewDefect id={rd.id}")

            # ========= 输出feedback提交载荷json =========
            fb_false_neg = {
                "review_result_id": rr.id,
                "review_defect_id": rd.id,
                "feedback_type": "false_negative",
                "expert_suggestion": "该场景漏检，需要生成候选规则草稿",
                "rule_candidate": {"title": "VCC去耦缺失候选", "severity": "critical"},
                "created_by": None
            }
            fb_gap = {
                "review_result_id": rr.id,
                "review_defect_id": rd.id,
                "feedback_type": "knowledge_gap",
                "expert_suggestion": "缺失STC89C55RC电源章节datasheet片段，需要补充seed知识库",
                "created_by": None
            }

            with open(out_dir / "fb_false_neg.json", "w", encoding="utf-8") as f:
                json.dump(fb_false_neg, f, indent=2, ensure_ascii=False)
            with open(out_dir / "fb_knowledge_gap.json", "w", encoding="utf-8") as f:
                json.dump(fb_gap, f, indent=2, ensure_ascii=False)

            # 🚨【最重要】session还没关闭！立刻把id拷贝进普通int变量，离开with之后绝对禁止访问rr/rd对象
            out_rr_id = int(rr.id)
            out_rd_id = int(rd.id)

        # 👉 已经退出with，session已经close；**只使用普通变量out_rr_id/out_rd_id，不要再碰rr、rd ORM实例**
        print(json.dumps({"review_result_id": out_rr_id, "review_defect_id": out_rd_id}), flush=True)

    except Exception as exc:
        log(f"[Demo‑Payload] ❌脚本异常：{exc}")
        log(traceback.format_exc())
        raise
    finally:
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo_feedback_payload Sprint0 demo脚本")
    parser.add_argument("--diagnose", action="store_true", help="仅执行容器Python环境诊断，不执行业务payload链路")
    args = parser.parse_args()

    if args.diagnose:
        run_diagnose()
        sys.exit(0)
    main()

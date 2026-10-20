# scripts/switch_embedding_backend.py
"""
Embedding 后端切换脚本（Sprint 1 阶段 C 新增）

【用途】
  一条命令完成"切 Embedding 后端"的全部动作：
  迁移 → 清库 → 改 models.py → 重灌 → 冒烟。

【只管 Embedding，不管对话模型】
  - 对话模型：改 LLM_PROVIDER 环境变量 + 重启 backend 容器，不需要本脚本
  - Embedding：本脚本负责迁移 + 清库 + 重灌

【用法（容器内执行）】
  docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
    exec backend uv run python /scripts/switch_embedding_backend.py --to zhipu

【安全】
  - 仅开发环境可用：APP_ENV != dev 时拒绝执行
  - 必须在容器内运行：检测 /data/knowledge 挂载
  - 清库前打印警告，需 --yes 或交互确认

【S1-C 修正】
  - BACKEND_DIM 不再本地硬编码，从 app.rag.embedding_factory 读，单一事实来源
  - 维度读取改用 format_type 解析，避免 atttypmod 偏差（1024 算成 1020）
  - 修改 models.py 后，用正则只替换 KnowledgeChunk.embedding 所在行的 Vector(N)
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


# --- 原代码（保留，已注释） ---
# BACKEND_DIM = {"fake": 1536, "openai": 1536, "zhipu": 2048}
# --- 原代码（保留，已注释） ---
# ===== [S1-C 修正] 从工厂 import，保证单一事实来源 =====
# 说明：延迟到 main() 内 import，避免脚本被 pytest 收集时触发 DB 连接。
# ===== [S1-C 修正 END] =====

MODELS_PATH = pathlib.Path("/app/app/persistence/models.py")

SEED_CONFIG = "/data/knowledge/seed_ingest_list.yaml"


def _check_in_container() -> None:
    if not pathlib.Path("/data/knowledge").is_dir():
        raise SystemExit(
            "拒绝执行：未检测到 /data/knowledge 挂载。\n"
            "本脚本必须在 backend 容器内运行：\n"
            "  docker compose ... exec backend uv run python /scripts/switch_embedding_backend.py --to <backend>"
        )


def _check_env() -> None:
    from app.core.config import settings
    if settings.app_env.lower() != "dev":
        raise SystemExit(
            f"拒绝执行：APP_ENV={settings.app_env}，本脚本仅开发环境可用"
        )


def _confirm(backend: str, dim: int, force: bool) -> None:
    if force:
        return
    print(
        f"\n⚠️  即将切换到 {backend}（{dim} 维）。\n"
        f"   动作：Alembic 迁移 → 清空 knowledge_chunk → 改 models.py → 重灌\n"
        f"   开发库数据将被清空。\n"
    )
    answer = input("确认继续？输入 yes：").strip().lower()
    if answer != "yes":
        raise SystemExit("已取消")


def _current_db_dim() -> int | None:
    from sqlalchemy import text
    from app.persistence.db import engine
    with engine.connect() as conn:
        # --- 原代码（保留，已注释） ---
        # return conn.execute(
        #     text(
        #         "SELECT atttypmod - 4 FROM pg_attribute "
        #         "WHERE attrelid = 'knowledge_chunk'::regclass AND attname = 'embedding'"
        #     )
        # ).scalar()
        # --- 原代码（保留，已注释） ---
        # ===== [S1-C 修正] 用 format_type 解析，避免 atttypmod 偏差 =====
        actual_type = conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'knowledge_chunk'::regclass AND attname = 'embedding'"
            )
        ).scalar()
        if actual_type and "(" in actual_type and ")" in actual_type:
            try:
                return int(actual_type.split("(")[1].rstrip(")"))
            except (ValueError, IndexError):
                return None
        return None
        # ===== [S1-C 修正 END] =====


def _run_alembic(action: str) -> None:
    if action == "upgrade":
        cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
    else:
        cmd = [sys.executable, "-m", "alembic", "downgrade", "42104f6cb89d"]
    print(f"[1/5] {' '.join(cmd[1:])} ...")
    subprocess.run(cmd, check=True)


def _truncate_chunks() -> None:
    from sqlalchemy import text
    from app.persistence.db import engine
    print("[2/5] 清空 knowledge_chunk（旧向量维度不兼容）...")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE knowledge_chunk RESTART IDENTITY CASCADE"))


def _sync_models_py(dim: int) -> None:
    print(f"[3/5] 同步 models.py 的 KnowledgeChunk.embedding -> Vector({dim}) ...")
    if not MODELS_PATH.exists():
        print(f"  [WARN] 未找到 {MODELS_PATH}，跳过")
        return
    content = MODELS_PATH.read_text(encoding="utf-8")
    # --- 原代码（保留，已注释） ---
    # new_content = re.sub(r"Vector\((\d+)\)", f"Vector({dim})", content)
    # --- 原代码（保留，已注释） ---
    # ===== [S1-C 修正] 只替换 KnowledgeChunk.embedding 所在行的 Vector(N)，
    # 避免误伤注释或其他 Vector(N) 字面量 =====
    pattern = re.compile(
        r'(embedding:\s*Mapped\[Optional\[Vector\]\]\s*=\s*mapped_column\()'
        r'Vector\(\d+\)(\))'
    )
    new_content, n = pattern.subn(rf'\g<1>Vector({dim})\g<2>', content)
    if n == 0:
        print("  [WARN] 未匹配到 KnowledgeChunk.embedding 的 Vector(N)，未修改")
        return
    # ===== [S1-C 修正 END] =====
    if new_content != content:
        MODELS_PATH.write_text(new_content, encoding="utf-8")
        print(f"  models.py 已更新（替换 {n} 处）")
    else:
        print("  models.py 无变化")


def _reingest() -> None:
    print("[4/5] 重灌知识库（python -m app.rag.ingest）...")
    subprocess.run(
        [sys.executable, "-m", "app.rag.ingest", "--config", SEED_CONFIG],
        check=True,
    )


def _smoke_test() -> None:
    print("[5/5] 检索冒烟 ...")
    from app.rag.retriever import Retriever
    ret = Retriever()
    results = ret.retrieve(query="晶振负载电容", top_k=1)
    print(f"  检索到 {len(results)} 条结果")
    if results:
        print(f"  首条 score={results[0].score:.4f} source={results[0].source_title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="切换 RAG Embedding 后端（开发环境）")
    parser.add_argument("--to", required=True, choices=["fake", "openai", "zhipu"])
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    backend = args.to
    # ===== [S1-C 修正] 从工厂读维度，单一事实来源 =====
    from app.rag.embedding_factory import BACKEND_DIM
    target_dim = BACKEND_DIM[backend]
    # ===== [S1-C 修正 END] =====

    _check_in_container()
    _check_env()
    _confirm(backend, target_dim, args.yes)

    current = _current_db_dim()
    print(f"当前 DB 维度：{current}，目标：{target_dim}")
    if current == target_dim:
        print("维度已一致，仅需确认 RAG_EMBEDDING_BACKEND 已切到目标后端。")
        print(f"  请确认 .env 中 RAG_EMBEDDING_BACKEND={backend}")
        return

    if target_dim > (current or 0):
        _run_alembic("upgrade")
    else:
        _run_alembic("downgrade")
    _truncate_chunks()
    _sync_models_py(target_dim)
    _reingest()
    _smoke_test()

    print(
        f"\n✅ 切换完成。请确认环境变量：\n"
        f"   RAG_EMBEDDING_BACKEND={backend}（.env）\n"
        f"   并重启 backend 容器：docker compose ... restart backend"
    )


if __name__ == "__main__":
    main()
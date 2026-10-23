# tests/test_benchmark_zhipu.py
"""
Sprint 1 阶段 F · 真实智谱 Full Review Benchmark（Integration）

【定位】
  真实 LLM 轨的 10-Case Full Review Benchmark。
  - Rule Only 由 tests/test_benchmark.py 覆盖（不依赖 LLM）
  - Mock 轨由 tests/test_benchmark.py 覆盖（不触网）
  - 本文件只跑真实智谱，验证"真实 LLM 在 10 Case 上的表现"

【运行方式】
  # 显式启用 integration 测试
  pytest tests/test_benchmark_zhipu.py -v -m integration

  # 无 Key 时自动 skip
  pytest tests/test_benchmark_zhipu.py -v

【容错策略（S1-F 冻结）】
  - 至少 8 个 Case 成功（status=done）→ 测试通过
  - 失败 ≤ 2 个：容忍偶发 429 / 网络抖动，但逐条打印失败 Case
  - 失败 > 2 个：测试失败（可能是系统性故障）
  - 硬失败（任何一条都让测试失败）：
      * 某条 status=done 但 task_id is None
      * 某条 metric_source != golden_label_simulated
      * 某条 status=done 但 ear / ai_ner 非 None（无真实反馈却有值）

【重要边界】
  - 「至少 8 成功」是容错阈值，不证明 10 Case 基准完整通过
  - 模拟指标允许合法 None（分母 0 时）
  - CSV 先写后断言，失败不阻止落盘
  - 不默认清理历史数据；清理按批次标识或时间窗口

【真实落库副作用】
  本测试会真实落库 10 条 review_result + 若干 review_defect。
  task_id 格式 RUN-YYYYMMDD-HHMMSS-XXXX，按批次可识别。
  测试不默认清理，阶段 J 的 cleanup 脚本按批次标识精确清理。

【skipif 的 Key 来源】
  用 settings.zhipuai_api_key or settings.llm_api_key，
  与 app/llm/provider.py::ZhipuProvider 实际读取的配置源一致，
  避免"测试认为有 Key、provider 却没有"的误判。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from app.core.config import settings
from app.services.benchmark_service import (
    BENCH_OUT_DIR,
    MODE_FULL_REVIEW,
    STATUS_DONE,
    STATUS_FAILED,
    run_all_benchmark,
    write_csv,
)


# ============================================================
# 常量
# ============================================================
CASE_IDS = [
    "case001", "case002", "case003", "case004", "case005", "case006",
    "case_b01_buck_thermal_derating_B",
    "case_b02_reset_rc_B",
    "case_b03_io_direction_B",
    "case_b04_power_sequence_B",
]

# 容错阈值：至少 8 个 Case 成功
MIN_SUCCESS = 8

# 批次标识：供人工排查 + 后续清理定位
BATCH_TAG = f"BENCH-S1F-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# CSV 输出路径（/out 是 bind mount，容器内写即宿主机可见）
CSV_PATH = os.path.join(BENCH_OUT_DIR, "bench_full_review_sprint1.csv")


# ============================================================
# skipif：与 provider 实际读取的 Key 来源一致
# ------------------------------------------------------------
# 用 settings 而不是 os.environ，避免环境变量名不一致导致的误判：
#   - ZhipuProvider 读 settings.zhipuai_api_key or settings.llm_api_key
#   - 测试也读同一来源
# ============================================================
def _has_zhipu_key() -> bool:
    return bool(settings.zhipuai_api_key or settings.llm_api_key)


_HAS_KEY = _has_zhipu_key()


# ============================================================
# 主测试：10 Case 真实 Full Review
# ============================================================
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_KEY,
    reason="无 ZHIPUAI_API_KEY / LLM_API_KEY，跳过真实智谱 Integration 测试",
)
def test_full_review_10cases_zhipu():
    """
    10 Case 真实 Full Review Benchmark。

    流程：
      1. run_all_benchmark 跑 10 Case（失败保留，不中断）
      2. 先写 CSV（无论成功失败），失败不阻止落盘
      3. 逐条打印每个 Case 的 status / task_id / error
      4. 分三层断言：
         a) 硬失败：数量、task_id、metric_source、无真实反馈却有真实指标值
         b) 模拟指标：允许 None，非 None 时在 [0,1]
         c) 容错：至少 MIN_SUCCESS 个成功
    """
    print(f"\n[BENCH] batch_tag = {BATCH_TAG}")
    print(f"[BENCH] running {len(CASE_IDS)} cases in FULL_REVIEW mode (real Zhipu)")
    print(f"[BENCH] expected duration: ~7 minutes\n")

    # ========== 1. 跑批次 ==========
    results = run_all_benchmark(CASE_IDS, mode=MODE_FULL_REVIEW)

    # ========== 2. 先写 CSV（失败不阻止落盘） ==========
    csv_write_error: Exception | None = None
    try:
        write_csv(results, CSV_PATH)
        print(f"[BENCH] CSV written: {CSV_PATH}")
    except Exception as exc:
        csv_write_error = exc
        print(f"[BENCH] CSV write failed: {type(exc).__name__}: {exc}")

    # ========== 3. 逐条打印 ==========
    n_done = 0
    n_failed = 0
    failed_cases: list[str] = []

    print("\n[BENCH] per-case results:")
    print("-" * 100)
    for r in results:
        status_mark = "✅" if r.status == STATUS_DONE else "❌"
        print(
            f"  {status_mark} {r.case_id:45s} | "
            f"status={r.status:8s} | "
            f"task_id={r.task_id or '-':30s} | "
            f"tp={r.tp} fp={r.fp} fn={r.fn} | "
            f"error={r.error or '-'}"
        )
        if r.status == STATUS_DONE:
            n_done += 1
        else:
            n_failed += 1
            failed_cases.append(r.case_id)
    print("-" * 100)
    print(f"[BENCH] summary: done={n_done}, failed={n_failed}")
    if failed_cases:
        print(f"[BENCH] failed cases: {failed_cases}")
    print()

    # ========== 4. CSV 写失败：先报错 ==========
    assert csv_write_error is None, (
        f"CSV 写出失败: {type(csv_write_error).__name__}: {csv_write_error}"
    )

    # ========== 5. 硬失败断言 ==========
    assert len(results) == len(CASE_IDS), (
        f"结果数量不匹配：期望 {len(CASE_IDS)}，实际 {len(results)}"
    )

    for r in results:
        if r.status == STATUS_DONE:
            # 成功 Case 的硬约束
            assert r.task_id is not None, (
                f"{r.case_id}: status=done 但 task_id 为 None（程序缺陷）"
            )
            assert r.metric_source == "golden_label_simulated", (
                f"{r.case_id}: metric_source={r.metric_source!r}，"
                f"应为 'golden_label_simulated'"
            )
            assert r.ear is None, (
                f"{r.case_id}: 无真实反馈时 ear 应为 None，实际 {r.ear!r}"
            )
            assert r.ai_ner is None, (
                f"{r.case_id}: 无真实反馈时 ai_ner 应为 None，实际 {r.ai_ner!r}"
            )

            # 模拟指标：允许 None，非 None 时在 [0,1]
            if r.ai_ner_simulated is not None:
                assert 0.0 <= r.ai_ner_simulated <= 1.0, (
                    f"{r.case_id}: ai_ner_simulated={r.ai_ner_simulated} 越界"
                )
            if r.ear_simulated is not None:
                assert 0.0 <= r.ear_simulated <= 1.0, (
                    f"{r.case_id}: ear_simulated={r.ear_simulated} 越界"
                )

        elif r.status == STATUS_FAILED:
            # 失败 Case 必须有 error 说明
            assert r.error, (
                f"{r.case_id}: status=failed 但 error 为空（信息缺失）"
            )
        else:
            pytest.fail(
                f"{r.case_id}: 非预期的 status={r.status!r}"
            )

    # ========== 6. 容错断言 ==========
    assert n_done >= MIN_SUCCESS, (
        f"成功数不足：{n_done}/{len(CASE_IDS)} < {MIN_SUCCESS}。"
        f"失败 Case: {failed_cases}。"
        f"请排查失败原因（API 限流 / 网络 / 程序缺陷）。"
    )

    # 全部通过时的提示
    if n_failed == 0:
        print(f"[BENCH] ✅ 全部 {len(CASE_IDS)} Case 通过")
    else:
        print(
            f"[BENCH] ⚠️  {n_done}/{len(CASE_IDS)} 通过，"
            f"{n_failed} 个失败在容错阈值内，但需人工排查：{failed_cases}"
        )


# ============================================================
# 辅助：手动跑批次（不走 pytest，供 Demo 用）
# ------------------------------------------------------------
# 用法：
#   python -m tests.test_benchmark_zhipu
# 与 pytest 的区别：不触发 skipif，直接跑 10 Case + 写 CSV
# ============================================================
def _manual_run() -> int:
    """
    手动跑 10 Case 真实 Full Review + 写 CSV。

    :return: 退出码（0=至少 MIN_SUCCESS 成功；1=失败过多或 CSV 写失败）
    """
    if not _HAS_KEY:
        print("❌ 无 ZHIPUAI_API_KEY / LLM_API_KEY，无法跑真实轨")
        return 1

    print(f"[BENCH] batch_tag = {BATCH_TAG}")
    results = run_all_benchmark(CASE_IDS, mode=MODE_FULL_REVIEW)

    try:
        write_csv(results, CSV_PATH)
        print(f"[BENCH] CSV written: {CSV_PATH}")
    except Exception as exc:
        print(f"[BENCH] CSV write failed: {type(exc).__name__}: {exc}")
        return 1

    n_done = sum(1 for r in results if r.status == STATUS_DONE)
    n_failed = len(results) - n_done
    print(f"[BENCH] done={n_done}, failed={n_failed}")
    return 0 if n_done >= MIN_SUCCESS else 1


if __name__ == "__main__":
    import sys
    sys.exit(_manual_run())
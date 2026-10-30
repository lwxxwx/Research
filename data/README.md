## [S2-I] 2026-10-30 - 资产治理与 CI/CD

### 交付内容
- **vLLM 占位契约测试**：新增 `backend/tests/test_llm_vllm.py`，3 条用例验证工厂分发、占位语义、settings 兜底。
- **Benchmark CSV 导出**：新增 `scripts/export_benchmark_csv.py`，支持 rule_only / full_review 模式导出，产物格式符合 S1-F 冻结契约。
- **data 资产治理**：新增 `data/README.md`，约定 Case / 知识库 / 规则文件的命名与新增流程。
- **诊断脚本隔离**：`.gitignore` 已忽略 `scripts/_diag_*.py`、`scripts/_cleanup_*.py`、`scripts/_tmp_*.py`，避免误提交。
- **CI/CD**：新增 `.github/workflows/ci.yml`，包含 backend（uv sync + pytest 非 integration）与 frontend（npm ci + type-check + build）两个并行 job。

### CI 行为
- 触发条件：push 到 `main` / `develop` / `feature/**`，以及向 `main` / `develop` 提 PR。
- 后端：Python 3.11 + uv 0.12.6，`uv sync --extra dev --frozen`，跑 `pytest -m "not integration"`。
- 前端：Node 20 + npm ci，跑 `type-check` + `build`。
- 不依赖真实 LLM 或外部 DB，CI 稳定可跑。

### 已知限制
- 集成测试（`@pytest.mark.integration`）不在 CI 跑，留给本地或有 GPU 的 runner。
- Benchmark CSV 导出暂不作为 CI artifact，留待后续按需补充。
- vLLM 实装（Sprint 3 计划）后，需在 CI 里加一个可选 job，或仅在本地 GPU 机器跑集成测试。
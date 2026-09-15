# Golden Case Format v1.0
> Business Freeze · Sprint 0 业务冻结项（版本锁定 v1.0，变更需架构评审）
> 文档定位：`data/cases/README_CASE_B_DESIGN.md`

## 1. 案例分类定义

- **A 类用例（6 个）**：标准正确 / 少量可检测电气问题的原理图，适合规则引擎基线测试（Rule-Only）。IR 三语义字段覆盖率 100%；缺陷均可被确定性规则稳定检出（`origin=rule`）；`strict` 模式校验必须 PASS。
- **B 类用例（4 个）**：存在复杂隐患、边界场景、部分元件语义缺失，用于测试告警、漏报、误报。**必须包含至少 1 个当前规则无法稳定检出、需 AI/RAG 发现的问题（`origin=ai_discovered`）**；`lib_name` 全部必填。

> case001 定位为 **A 类**（Sprint0 完整落地用例，Rule-Only 基准）。

## 2. 每个 case 目录必须包含的文件清单

```
caseXXX/
├─ info.yaml                 # 案例元信息（case_class / ai_expected_count / known_design_defects）
├─ schematic_ir.json        # SchematicIRDocument v1.0 IR 中间表示
├─ expert_reasoning.md      # 专家推理过程（V1.2 §2.3 的 4 步模板）
├─ expected_review.json     # Ground-Truth 预期评审结果（Defect 10 字段规范）
├─ evaluation.yaml           # Benchmark 评测配置（判定条件 + 权重）
└─ evidence/                 # 证据文件夹（RAG 注入 / 对照）
   ├─ datasheet/*.md         # Datasheet 关键摘录
   ├─ application_note/*.md  # 应用笔记摘录
   └─ reference_design/*.md  # 参考设计摘录
```

> A 类至少补齐 `evaluation.yaml`；为统一验收，case001 按完整结构落地。

## 3. IR 校验规则

Golden Case 必须使用 strict 模式：

```powershell
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
  exec backend uv run python -m app.ir.serializer --validate ./data/cases/case001/schematic_ir.json --strict
```

强制约束（strict 模式报错项）：
- 所有元件 `lib_name` 不为空字符串；
- 所有元件 `intent` / `context` 非空（三语义字段）；
- `connected_pins` 严格 `{ref}.{pin_id}` 格式，禁止物理引脚 `number`；
- `pin_id` 全部存在于对应元件引脚列表；
- 允许：`constraint` 可为空字符串。

## 4. expected_review.json Defect 10 字段规范（V1.2）

每条 defect 必须含：`defect_id / category / location / component / net / risk / evidence / root_cause / suggestion / confidence` 共 10 个字段。

- `category` 枚举：power / clock / interface / memory / thermal / esd / timing / layout / parametric / other
- `risk` 语义化：low / medium / high / critical（与 severity 同义映射）
- 每条 `evidence` 必须满足三要素 `source / section / reason`（三字段非空且长度 ≥ 10），另有 `type`（rag_ref / ir_ref）
- `ai_discovered` 类缺陷必须至少 1 条 `rag_ref` + 1 条 `ir_ref`
- 顶层字段：`case_id / review_version / overall_risk_level / summary / defects / observations / good_practices / known_ignores / benchmark_meta / expert_sign_off`

## 5. evaluation.yaml 字段说明

- `pass_criteria`：对每条缺陷断言（origin / rule_id / component 交集 / net 交集 / evidence 数量 / suggestion 关键词 / root_cause 关键词）
- `benchmark_weights`：A 类 `rule_metrics_weight:0.8 / ai_metrics_weight:0.2`；B 类 `0.4 / 0.6`
- `benchmark_tags`、`skip_categories`、`strict_ir_validation`

## 6. 测试执行入口

```powershell
# 单用例规则测试
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
  exec backend uv run python -m scripts.test_rules --case case001

# 全量 Golden Case benchmark（Rule-Only）
docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
  exec backend uv run python -m scripts.run_benchmark
```

## 7. Sprint0 验收锚点

- K7：**Case001 Rule Hit ≥ 2**（case001 以 POWER_001 + POWER_002 命中达成）
- Phase E 验收：案例目录结构完整；`expected_review.json` 满足 10 字段规范
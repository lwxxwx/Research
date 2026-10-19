# Sprint 1 规则设计阻塞报告

> 日期：2026-10-19
> 状态：🟢 阶段 B 完成；B 类 4 个 Case 待硬件专家校准
> 责任：Tech Lead B → hardware-expert

## 1. 已解决的阻塞

| 原阻塞 | 解决方案 | 状态 |
|---|---|---|
| `CTRL_001/002` 的 `category: control` 与 `rule_format.md` 枚举冲突 | 改用 `interface` / `timing` | ✅ 已解决 |
| `POWER_003` 降额依据缺失 | 用通用降额准则（GJB/Z 35）占位；后续替换 | ✅ 已解决（占位） |
| `POWER_002` 8051 专向 | 加 `reset_polarity` 参数；用 `pin_type` 排除 MCU | ✅ 已解决 |
| 5 条新规则的设计依据缺失 | 按"IR 可判定拓扑类缺陷"原则设计 | ✅ 已解决 |
| B 类 4 个 Case 的构造 | Design Agent 设计 + Review Agent 审查 | ✅ 已解决（PROVISIONAL） |
| 复用 case001 的 evidence 带缺陷引用 | 用"去缺陷引用"版替换 | ✅ 已解决 |

## 2. 剩余待办（B 类专家校准）

**B 类 4 个 Case 目前是 PROVISIONAL**，需硬件专家审阅并签字：

| Case | 需要专家审阅 | 状态 |
|---|---|---|
| case_b01_buck_thermal_derating_B | Buck 热降额计算、GJB/Z 35 条款、证据链 | ⏳ PROVISIONAL |
| case_b02_reset_rc_B | RC 容差叠加分析、最坏工况推导、Intel 手册引用 | ⏳ PROVISIONAL |
| case_b03_io_direction_B | I2C 开漏特性、上拉电阻取值、规则边界 | ⏳ PROVISIONAL |
| case_b04_power_sequence_B | 多路时序推导、灌电流路径、规则边界 | ⏳ PROVISIONAL |

**专家校准流程**：

1. hardware-expert 审阅 `expert_reasoning.md` + `expected_review.json` + `evidence/`
2. 修正电路 / AI-only 候选 / 证据
3. 签字 `CALIBRATED`
4. 更新 `info.yaml::calibration_status` 从 `PROVISIONAL` 到 `CALIBRATED`
5. 更新 `expected_review.json::expert_sign_off::reviewer` 从 `PROVISIONAL_AI_ASSISTANT` 到 `hardware-expert`

## 3. 不受阻塞的任务

以下任务**不依赖专家校准**，已全部完成：

- ✅ 10 条规则实现 + 单测
- ✅ 10 个 Case 资产（6A + 4B）
- ✅ 覆盖矩阵
- ✅ 规则命中测试
- ✅ B 类资产完整性测试
- ✅ A/B 类命名结构测试

## 4. 阶段 B 的 P0 目标完成情况

| # | 交付内容 | 状态 |
|---|---|---|
| A | Sprint 0 回归 + Alembic 唯一 Schema | ✅ |
| B | 10 个 Case 资产 + 覆盖矩阵 + 校验脚本 | ✅ 10/10 |
| B' | 10 条规则（含 POWER_003）+ 单测 | ✅ 10/10 |
| expert_decision 持久化 | ✅ |
| AI_NER / EAR 公式 + 单测 | ⏳ 阶段 F |

**阶段 B 的 P0 目标全部完成**（除阶段 F 的 AI_NER / EAR 公式）。

## 5. 建议

**进入阶段 C（LLM 层）**：

- LLM Provider 抽象（Mock / OpenAI / vLLM）
- Prompt（expert_analysis）
- OutputParser
- 单测

**B 类专家校准**可并行进行，不影响阶段 C 的推进。
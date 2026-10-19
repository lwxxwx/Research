# 专家推理过程（Case B02）

> 案例：8051 复位 RC 裕量不足
> 类型：B 类（纯 AI-only，无 rule 缺陷）
> 本文件按 V1.2 §2.3 的 4 步模板撰写
> ⚠️ 状态：PROVISIONAL（Tech Lead B 设计 + hardware-expert 审查占位，待真实硬件专家校准）

## Step 1 · 从 Schematic IR 提取事实

| 事实 | IR 路径 |
|---|---|
| 主控 STC89C55RC，11.0592MHz 晶振 | `Component[U1]` / `Component[Y1]` |
| 复位 RC：R1=22kΩ ±10%，C3=100nF ±10% | `Component[R1]` / `Component[C3]` 的 `attributes.tolerance_pct` |
| 复位拓扑：R2 上拉 RST-VCC（正确） | `Net[RST]` 含 `R2.pin1`；`Net[VCC]` 含 `R2.pin2` |
| 标称 τ = 22kΩ × 100nF = 2.2ms | 计算 |
| 工程阈值 min_reset_high_ms = 2ms | `CTRL_002` 规则参数 |

## Step 2 · 识别规则能覆盖的边界

| 规则 | 能覆盖（命中） | 边界说明 |
|---|---|---|
| POWER_001 | ❌ 不命中：C4=0.1μF | - |
| POWER_002 | ❌ 不命中：R2 上拉正确 | - |
| POWER_003 | ❌ 不命中：无功耗属性 | - |
| IO_001 | ✅ 命中（low） | 不计入 defect |
| CLOCK_001/002 | ❌ 不命中：C1/C2=22pF 正确 | - |
| CTRL_001 | ❌ 不命中：无使能脚 | - |
| **CTRL_002** | **❌ 不命中：标称 τ=2.2ms ≥ 2ms** | **规则只看标称值，不看容差裕量** |

**规则覆盖不到的**：

- **容差裕量分析**：需读取 R/C 的 `tolerance_pct`，做 ±10% 叠加推导
- **最坏工况计算**：τ_min = R_min × C_min = 19.8kΩ × 90nF = 1.782ms

## Step 3 · 匹配依据（evidence/）

| 依据文件 | 章节 | 内容 |
|---|---|---|
| `evidence/datasheet/Intel_MCS51_UM_reset.md` | Reset Timing | 复位需 ≥ 2 机器周期；@11.0592MHz ≈ 2.17μs |
| `evidence/application_note/AN-8051-002_reset_timing.md` | RC 裕量准则 | 工程要求 τ ≥ 1.5 × 最小要求（3ms） |
| `evidence/application_note/AN-8051-002_reset_timing.md` | 最坏工况推导 | τ_min = 19.8kΩ × 90nF = 1.782ms |
| `evidence/reference_design/8051_MinSystem_RefDes.md` | 复位设计 | 标准 RC 复位电路（复用 case001） |

### 关键计算

1. **标称 τ**：22kΩ × 100nF = 2.2ms
2. **工程要求**：τ ≥ 2ms（min_reset_high_ms）+ 50% 裕量 = 3ms（可靠复位）
3. **最坏工况**：R1=19.8kΩ（-10%），C3=90nF（-10%）→ τ_min = 1.782ms
4. **对比**：1.782ms < 2ms → **最坏工况下复位失败**

## Step 4 · 给出结论

| Defect | 规则 | 结论 | 对应 expected_review.json |
|---|---|---|---|
| DEF-B02-1 | **（AI）** | 复位 RC 最坏工况 τ=1.782ms < 2ms，high | `defects[0]` |

**评审结论**：**FAIL（需修复）**

## 边界说明

若 C3 容差 0%、R1 容差 0%（理想器件），则 τ=2.2ms > 2ms，**不触发本缺陷**。本 Case 的缺陷依赖 ±10% 容差叠加的最坏工况（τ_min=1.782ms < 2ms）。

## 专家补充观察

- 本 Case 为纯 AI-only（无 rule 缺陷），验证 AI 的"容差裕量分析"能力
- 与 A 类 `case006`（CTRL_002 命中 τ=10μs）形成互补：A 类验证规则，B 类验证 AI 发现规则外的问题

## 签字

- reviewer：**PROVISIONAL_AI_ASSISTANT**（Tech Lead B 设计 + hardware-expert 审查占位）
- calibration_status：**PROVISIONAL**
- calibration_date：2026-10-19
- pending_review_by：**hardware-expert**
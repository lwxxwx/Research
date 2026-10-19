# 专家推理过程（Case B01）

> 案例：8051 + LM2596 Buck 电源系统
> 类型：B 类（AI 辅助发现）
> 本文件按 V1.2 §2.3 的 4 步模板撰写
> ⚠️ 状态：PROVISIONAL（Tech Lead B 设计 + hardware-expert 审查占位，待真实硬件专家校准）

## Step 1 · 从 Schematic IR 提取事实

| 事实 | IR 路径 |
|---|---|
| 主控芯片 STC89C55RC，5V 供电 | `Component[U1]` |
| Buck 芯片 LM2596S-5.0（TO-263） | `Component[U2]` |
| 输入 12V，输出 5V | `Net[VIN_12V].voltage = 12.0`，`Net[VCC_5V].voltage = 5.0` |
| 负载电流 1A | `extra_meta.load_current_a = 1.0` |
| 环境温度 85℃ | `extra_meta.ambient_temperature_c = 85` |
| VCC_5V 网络只有 220μF 电解，无 0.1μF 陶瓷 | `Net[VCC_5V].connected_pins` 中无陶瓷电容 |
| C_in 旁无 100nF 高频陶瓷 | `Net[VIN_12V].connected_pins = [U2.VIN, C_in.pin1]` |
| U2 的 `attributes` 无功耗/热参数 | `Component[U2].attributes` 只有 `rated_current_a` / `output_voltage_v` / `package` |
| 复位 RC、晶振负载电容正确 | `Net[RST]`、`Net[XTAL_OSC]` 拓扑正确 |

## Step 2 · 识别规则能覆盖的边界

| 规则 | 能覆盖（命中） | 边界说明 |
|---|---|---|
| POWER_001（去耦电容检查） | ✅ 命中：VCC_5V 无 0.1μF 陶瓷 | 不判断高频去耦（100nF）位置，但按 §7.1 可判"无陶瓷去耦" |
| POWER_002（复位 RC 拓扑） | ❌ 不命中：R2 上拉正确 | - |
| POWER_003（器件功耗降额） | ❌ 不命中：U2.attributes 无 power_dissipation_actual | **规则需要 IR 携带功耗属性；本 Case 的 IR 故意不带，确保 AI-only** |
| IO_001（IO 悬空） | ✅ 命中：大量 IO 悬空（low） | 不计入 defect |
| IO_002 / IO_003 | ❌ 不命中 | - |
| CLOCK_001 / CLOCK_002 | ❌ 不命中：C1/C2=22pF 正确，晶振完整 | - |
| CTRL_001 / CTRL_002 | ❌ 不命中：U2 无 EN/OE；复位 τ=100ms 正确 | - |

**规则覆盖不到的**：

- **热降额裕量**：需要读取 Datasheet 热阻参数 + 效率曲线 + GJB/Z 35 降额准则，进行多步推理
- **EMC 布局风险**：依赖 PCB 布局，IR 无法表达（作为观察项，非 defect）

## Step 3 · 匹配依据（evidence/）

| 依据文件 | 章节 | 内容 |
|---|---|---|
| `evidence/datasheet/STC89C55RC_DS_p6_power.md` | 7.1 Power Supply Decoupling | 0.1μF 陶瓷去耦要求（复用 case001） |
| `evidence/datasheet/LM2596_DS_thermal.md` | Thermal Considerations | θJA ≈ 50℃/W，T_J_max = 125℃ |
| `evidence/datasheet/LM2596_DS_thermal.md` | Efficiency | 12V→5V/1A 时 η ≈ 80% |
| `evidence/application_note/AN-8051-001_p3_decoupling.md` | 3. Power Supply Decoupling | 0.1μF 陶瓷 < 5mm（复用 case001） |
| `evidence/application_note/GJBZ35-93_semiconductor_derating.md` | 半导体器件 Ⅱ 级降额 | T_J ≤ 95℃ |
| `evidence/reference_design/Buck_Converter_Thermal_Design.md` | 热设计原则 | T_J = T_A + P_loss × θJA |

### 热降额计算完整推导

1. **效率**：η ≈ 80%（LM2596 Datasheet Efficiency 图，12V→5V/1A）
2. **输入功率**：P_in = P_out / η = 5W / 0.8 = 6.25W
3. **功耗**：P_loss = P_in - P_out = 6.25W - 5W = 1.25W
4. **温升**：ΔT = P_loss × θJA = 1.25W × 50℃/W = 62.5℃
5. **结温**：T_J = T_A + ΔT = 85 + 62.5 = 147.5℃
6. **对比**：T_J = 147.5℃ > T_J_max = 125℃ → 超限；且 > 95℃（GJB/Z 35 Ⅱ级）→ 降额违规

## Step 4 · 给出结论

| Defect | 规则 | 结论 | 对应 expected_review.json |
|---|---|---|---|
| DEF-B01-1 | POWER_001 | VCC_5V 无 0.1μF 陶瓷去耦，critical | `defects[0]` |
| DEF-B01-2 | **（AI）** | LM2596 结温 147.5℃，超 T_J_max 和 GJB/Z 35 降额线，high | `defects[1]` |

**评审结论**：**FAIL（需修复）**

## 边界说明（Review Agent 建议）

本缺陷（DEF-B01-2）的触发依赖以下条件，任一不满足则**不应触发**：

| 条件 | 本 Case 值 | 不触发条件 |
|---|---|---|
| 环境温度 | 85℃ | < 40℃ 且负载 < 0.5A 时，T_J < 116.25℃（仍可能超 95℃ 降额线，但不超 125℃ 绝对最大值）|
| 负载电流 | 1A | < 0.5A 时 P_loss < 0.625W，温升 < 31.25℃，T_J < 116.25℃ |
| θJA | 50℃/W（TO-263 无散热片） | 加散热片后 θJA < 25℃/W，T_J < 116.25℃ |

**注意**：

- 本 Case 的 85℃ + 1A 组合下，T_J = 147.5℃ **同时超过** T_J_max=125℃（绝对最大值）和 GJB/Z 35 Ⅱ级降额线 95℃
- 如果测试用例改为 25℃ + 0.5A，热降额缺陷**不应触发**（供负例使用）
- 本 Case 的 `expected_review.json` 只在 85℃ + 1A 场景下有效

## 专家补充观察（非 defect）

- **EMC 风险**：Buck SW 节点（`SW_NODE`）高频开关噪声可能辐射干扰 11.0592MHz 晶振。建议布局时 SW 节点面积最小化、远离晶振。该问题依赖 PCB 布局，IR 无法表达，未列入正式 defect，留待后续 Sprint 有布局数据后加入。

## 签字

- reviewer：**PROVISIONAL_AI_ASSISTANT**（Tech Lead B 设计 + hardware-expert 审查占位）
- calibration_status：**PROVISIONAL**
- calibration_date：2026-10-19
- pending_review_by：**hardware-expert**
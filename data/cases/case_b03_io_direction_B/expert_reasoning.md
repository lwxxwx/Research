# 专家推理过程（Case B03）

> 案例：8051 + I2C EEPROM，I2C 总线缺上拉
> 类型：B 类（1 rule + 1 ai_discovered）
> 本文件按 V1.2 §2.3 的 4 步模板撰写
> ⚠️ 状态：PROVISIONAL（Tech Lead B 设计 + hardware-expert 审查占位，待真实硬件专家校准）

## Step 1 · 从 Schematic IR 提取事实

| 事实 | IR 路径 |
|---|---|
| 主控 STC89C55RC，11.0592MHz | `Component[U1]` / `Component[Y1]` |
| I2C EEPROM AT24C02 | `Component[U2]` |
| SDA 网络：`U1.P3_6`（bidirectional）+ `U2.SDA`（bidirectional），**无上拉** | `Net[SDA].connected_pins = ["U1.P3_6", "U2.SDA"]` |
| SCL 网络：`U1.P3_7`（bidirectional）+ `U2.SCL`（input），**无上拉** | `Net[SCL].connected_pins = ["U1.P3_7", "U2.SCL"]` |
| BUS_CONFLICT 网络：`U1.P1_0`（output）+ `U1.P1_1`（output） | `Net[BUS_CONFLICT]` |

## Step 2 · 识别规则能覆盖的边界

| 规则 | 能覆盖（命中） | 边界说明 |
|---|---|---|
| POWER_001 | ❌ 不命中：C4=0.1μF | - |
| POWER_002 | ❌ 不命中：R2 上拉正确 | - |
| POWER_003 | ❌ 不命中：无功耗属性 | - |
| IO_001 | ✅ 命中（low） | 不计入 defect |
| **IO_002** | **✅ 命中：BUS_CONFLICT（2 OUTPUT）** | **P1_0 + P1_1 都是 output** |
| **IO_003** | **❌ 不命中：SDA/SCL 是 bidirectional** | **external_outputs 要求 direction==output** |
| CLOCK_001/002 | ❌ 不命中 | C1/C2=22pF 正确 |
| CTRL_001/002 | ❌ 不命中 | 无使能脚 / τ=100ms 正确 |

**规则覆盖不到的**：

- **I2C 开漏总线缺上拉**：需要 I2C 规范知识（开漏特性 + 上拉要求），不是简单的网络拓扑检查
- IO_002 只看"多个 OUTPUT 同网"，IO_003 看"MCU vs 外设 output"，**都不检查"上拉电阻是否存在"**

## Step 3 · 匹配依据（evidence/）

| 依据文件 | 章节 | 内容 |
|---|---|---|
| `evidence/datasheet/STC89C55RC_DS_p6_power.md` | 7.1 Power Supply Decoupling | 0.1μF 去耦（复用 case001） |
| `evidence/datasheet/AT24C02_DS_i2c.md` | I2C Bus Timing | SDA/SCL 上升时间 ≤ 1000ns |
| `evidence/datasheet/NXP_I2C_UM10204.md` | 7.1 Pull-up Resistor Sizing | 标准模式 100kHz 推荐 4.7kΩ，1.7kΩ~5.7kΩ |
| `evidence/application_note/I2C_Bus_Design.md` | 2. Open-Drain Bus Requirement | 开漏总线只能拉低，高电平靠上拉 |
| `evidence/reference_design/8051_MinSystem_RefDes.md` | 电源/复位设计 | 复用 case001 |

> **复用文件说明**：
> - `STC89C55RC_DS_p6_power.md` 和 `8051_MinSystem_RefDes.md` 从 case001 复用，
>   仅引用其 Datasheet / 参考设计内容。
> - 本 Case 的 `POWER_001` 不命中（C4=0.1μF 已正确），`POWER_002` 不命中（R2 上拉已正确）。
> - 这两个文件已改为不包含 case001 的缺陷引用。

### 关键分析

1. **I2C 是开漏总线**：器件只能拉低，不能输出高电平
2. **高电平来源**：外部上拉电阻
3. **无上拉**：总线浮空，主机发送逻辑 1 时总线无法拉高
4. **后果**：ACK 位（从机拉低 SDA）无法被主机识别，通信完全失效

## Step 4 · 给出结论

| Defect | 规则 | 结论 | 对应 expected_review.json |
|---|---|---|---|
| DEF-B03-1 | IO_002 | P1_0 + P1_1 同网，critical | `defects[0]` |
| DEF-B03-2 | **（AI）** | I2C SDA/SCL 缺上拉，critical | `defects[1]` |

**评审结论**：**FAIL（需修复）**

## 边界说明

若 I2C 总线速率 < 10kHz 且总线长度 < 10cm，某些 CMOS 输入引脚可凭寄生电容维持短时间高电平，通信可**勉强工作**（但极不可靠）。本 Case 的缺陷在标准模式 100kHz 下**必然触发**。

## 专家补充观察

- 本 Case 验证 AI 的"电气特性分析"能力（开漏总线 + 上拉要求）
- 与 IO_002（规则）形成互补：规则发现"多个 output 同网"，AI 发现"开漏总线缺上拉"

## 签字

- reviewer：**PROVISIONAL_AI_ASSISTANT**（Tech Lead B 设计 + hardware-expert 审查占位）
- calibration_status：**PROVISIONAL**
- calibration_date：2026-10-19
- pending_review_by：**hardware-expert**（待真实专家签字）
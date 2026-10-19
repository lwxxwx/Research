# 专家推理过程（Case B04）

> 案例：8051 + 多路电源 - 上电时序异常
> 类型：B 类（1 rule + 1 ai_discovered）
> 本文件按 V1.2 §2.3 的 4 步模板撰写
> ⚠️ 状态：PROVISIONAL（Tech Lead B 设计 + hardware-expert 审查占位，待真实硬件专家校准）

## Step 1 · 从 Schematic IR 提取事实

| 事实 | IR 路径 |
|---|---|
| 主控 STC89C55RC（5V），11.0592MHz | `Component[U1]` / `Component[Y1]` |
| SPI Flash W25Q16（3.3V） | `Component[U2]` |
| 5V LDO：AMS1117-5.0，输出电容 `C1=220μF` | `Component[U4]` 的 `attributes.output_capacitance_uf = 220` |
| 3.3V LDO：AMS1117-3.3，输入电容 `C2=10μF` | `Component[U3]` 的 `attributes.input_capacitance_uf = 10` |
| VCC_5V 有 `C3=0.1μF` 去耦 | `Net[VCC_5V]` 含 `C3.pin1` |
| **VCC_3V3 无 0.1μF 去耦** | `Net[VCC_3V3].connected_pins = ["U3.VOUT", "U2.VCC"]` |
| SPI 总线拓扑：CS/CLK/MOSI/MISO | `Net[SPI_CS]` / `Net[SPI_CLK]` / `Net[SPI_MOSI]` / `Net[SPI_MISO]` |

## Step 2 · 识别规则能覆盖的边界

| 规则 | 能覆盖（命中） | 边界说明 |
|---|---|---|
| **POWER_001** | **✅ 命中：VCC_3V3 缺去耦** | **VCC_5V 有 C3，VCC_3V3 无** |
| POWER_002 | ❌ 不命中：R2 上拉正确 | - |
| POWER_003 | ❌ 不命中：无功耗属性 | - |
| IO_001 | ✅ 命中（low） | `U1` + `U2` 各一条，不计入 defect |
| IO_002 | ❌ 不命中 | - |
| **IO_003** | **❌ 不命中：SPI_MISO 的 U1.P1_5 是 input** | **不是 output，不计入 mcu_outputs** |
| CLOCK_001/002 | ❌ 不命中：C4/C5=22pF 正确 | - |
| CTRL_001 | ❌ 不命中 | - |
| CTRL_002 | ❌ 不命中：τ=100ms 正确 | - |

**规则覆盖不到的**：

- **上电时序**：10 条规则中**无任何时序检查项**
- 多路电源上电顺序、器件启动时间、IO 灌电流路径，**均需 AI/RAG 推理**

## Step 3 · 匹配依据（evidence/）

| 依据文件 | 章节 | 内容 |
|---|---|---|
| `evidence/datasheet/STC89C55RC_DS_p6_power.md` | 7.1 Power Supply Decoupling | 0.1μF 去耦（复用 case001） |
| `evidence/datasheet/STC89C55RC_DS_abs_max.md` | Absolute Maximum Ratings | IO 引脚电压 ≤ VCC+0.5V |
| `evidence/datasheet/AMS1117_DS_startup.md` | Startup Time | 启动时间 ≈ 1~10ms（与输出电容相关） |
| `evidence/datasheet/W25Q16_DS_power_seq.md` | Power-Up Timing | SPI Flash 上电时序要求 |
| `evidence/application_note/Power_Sequence_Design.md` | 3. Multi-Rail Power Sequencing | 核心电压必须先于 IO 电压 |
| `evidence/reference_design/8051_MinSystem_RefDes.md` | 电源/复位设计 | 复用 case001 |

### 关键分析

**上电时序推导**：

1. **U4(5V LDO) 输出电容 C1=220μF** → 充电时间常数 ≈ C1 × R_load ≈ 220μF × 5Ω = 1.1ms（典型负载 1A）
2. **U3(3.3V LDO) 输入电容 C2=10μF** → 充电时间常数 ≈ 10μF × 5Ω = 50μs
3. **3.3V 比 5V 早建立约 1~10ms**
4. 上电瞬间：3.3V 先建立 → SPI Flash(W25Q16) IO 输出 3.3V → 8051 VCC 仍未建立（0V）
5. **灌电流路径**：SPI Flash 的 IO 引脚 → 内部 ESD 保护二极管 → 8051 的 IO 引脚 → 8051 的 VCC 网络
6. **风险**：8051 的 IO 电压 3.3V > VCC(0V) + 0.5V → 触发闩锁（latch-up）

## Step 4 · 给出结论

| Defect | 规则 | 结论 | 对应 expected_review.json |
|---|---|---|---|
| DEF-B04-1 | POWER_001 | VCC_3V3 缺 0.1μF 去耦，critical | `defects[0]` |
| DEF-B04-2 | **（AI）** | 上电时序异常，high | `defects[1]` |

**评审结论**：**FAIL（需修复）**

## 边界说明

- 若 5V 先于 3.3V 建立（如通过 PMIC 时序控制芯片），本缺陷不触发
- 若 U4 输入/输出电容与 U3 相同，启动速度相近，本缺陷不触发
- 若 SPI 线上有电平转换/隔离，本缺陷不触发
- 本 Case 的 220μF vs 10μF 组合下，时序差约 1~10ms，**必然触发**

## 专家补充观察

- 本 Case 验证 AI 的"多路时序分析"能力（10 条规则无时序检查）
- 与 B01（热降额）/ B02（容差裕量）/ B03（开漏特性）形成 4 个独立维度的 AI 发现能力验证

## 签字

- reviewer：**PROVISIONAL_AI_ASSISTANT**（Tech Lead B 设计 + hardware-expert 审查占位）
- calibration_status：**PROVISIONAL**
- calibration_date：2026-10-19
- pending_review_by：**hardware-expert**（待真实专家签字）
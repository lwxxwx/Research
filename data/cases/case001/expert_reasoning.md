# 专家推理过程（Case001）

> 案例：8051 最小系统（STC89C55RC）
> 类型：A 类（Rule-Only 基线）
> 本文件按 V1.2 §2.3 的 4 步模板撰写；缺陷全部由确定性规则检出（origin=rule）

## Step 1 · 从 Schematic IR 提取事实

| 事实 | IR 路径 |
| :--- | :--- |
| 主控芯片为 STC89C55RC（DIP-40） | `Component[U1].lib_name` |
| VCC 网络仅连接 U1.VCC | `Net[VCC].connected_pins = ["U1.VCC"]` |
| 晶振 11.0592MHz，负载电容 22pF | `Component[Y1].value`, `Component[C1/C2].value` |
| 晶振网络为单一合并网络（XTAL1 与 XTAL2 共网） | `Net[XTAL_OSC].connected_pins = [U1.XTAL1, U1.XTAL2, Y1.pin1, Y1.pin2, C1.pin1, C2.pin1]` |
| 复位 RC：R1=10kΩ, C3=10μF | `Component[R1/C3].value` |
| 复位网络 RST：R1、C3、SW1 并联于 RST-GND | `Net[RST].connected_pins`、`Net[GND].connected_pins` |
| EA 接地（独立网络 EA_GND） | `Net[EA_GND].connected_pins = [U1.EA]` |
| 工作电压 5V | `Net[VCC].voltage = 5.0` |
| 网络清单（共 5 个） | VCC / GND / XTAL_OSC / RST / EA_GND |

## Step 2 · 识别规则能覆盖的边界（A 类缺陷均可规则检出）

| 规则 | 能覆盖（命中） | 边界说明 |
| :--- | :--- | :--- |
| POWER_001（去耦电容检查） | ✅ 判断 VCC 网络是否挂载去耦电容 → **命中**（VCC 无电容） | 不判断容值匹配、布局距离 < 5mm 等参数级问题 |
| POWER_002（复位 RC 检查） | ✅ 判断复位 RC 拓扑与 RST 网络是否有到 VCC 的上拉路径 → **命中**（C3/SW1 错接 RST-GND） | 不判断电解电容容差对 τ 的具体影响 |

> 说明：Phase F 规则库中 POWER_002 名义定义为「反馈分压」（DC-DC 场景），本 case 为 8051 最小系统、无反馈分压电路；复位 RC 检查为本 case 语境下的规则命中项，规则清单以 `data/rules` 最终合入为准，详见冻结纪要。

## Step 3 · 匹配依据（指向 evidence/ 下具体文件与章节）

| 依据文件 | 章节 | 内容 |
| :--- | :--- | :--- |
| `evidence/datasheet/STC89C55RC_DS_p6_power.md` | 7.1 Power Supply Decoupling | 每个 VCC 引脚应配 0.1μF 陶瓷去耦电容（支撑 DEF-CASE001-1） |
| `evidence/datasheet/STC89C55RC_DS_p6_power.md` | 6.3 Reset Circuit | RST 需高电平 ≥2 机器周期；R1 下拉 + C3 上电延时（支撑 DEF-CASE001-2） |
| `evidence/application_note/AN-8051-001_p3_decoupling.md` | 3. Power Supply Decoupling | 去耦电容缺失导致 MCU 随机复位（支撑 DEF-CASE001-1） |
| `evidence/application_note/AN-8051-001_p3_decoupling.md` | 7. Reset Circuit Design | C3=VCC 侧上电延时电容；手动复位应拉高 RST（支撑 DEF-CASE001-2） |
| `evidence/reference_design/8051_MinSystem_RefDes.md` | 2. 电源设计 / 4. 复位设计 | 标准最小系统含去耦电容；RC 复位 + 手动按键（支撑两条缺陷） |

## Step 4 · 给出结论（对应 expected_review.json 中 rule origin 的 defect）

| Defect | 规则 | 结论 | 对应 expected_review.json |
| :--- | :--- | :--- | :--- |
| DEF-CASE001-1 | POWER_001 | U1.VCC 缺 0.1μF 去耦电容，critical | `defects[0]` |
| DEF-CASE001-2 | POWER_002 | 复位 RC 连接异常：C3/SW1 错接 RST-GND，RST 无上拉路径，无法高电平复位，medium | `defects[1]` |

**评审结论**：**CONDITIONAL_PASS**

理由：晶振（单一 XTAL_OSC 网络）、EA 接地拓扑正确；但存在 2 条规则可检出缺陷——去耦电容缺失（critical）与复位电路连接异常（medium），必须修复。缺陷数量满足 Sprint0 K7（Rule Hit ≥ 2）。

## 专家签字

- 校准人：**hardware-expert**
- 校准日期：2026-09-11
- 校准状态：**CALIBRATED**
- 正式校准纪要：`docs/freeze/case001_expert_calibration.md`

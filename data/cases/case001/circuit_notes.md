# Case001 电路设计说明

> 案例元信息与设计背景，供规则引擎/RAG/评审参考
> Golden Case Format v1.0 · Sprint 0 Business Freeze

## 1. 案例来源

本用例来自 **8051 教学最小系统**，没有外设，仅保证芯片启动运行。

## 2. 电路组成

| 元件 | 型号/参数 | 作用 |
| :--- | :--- | :--- |
| U1 | STC89C55RC（DIP-40） | 8051 微控制器 |
| Y1 | 11.0592MHz 晶振 | 系统时钟源 |
| C1/C2 | 22pF 陶瓷电容 | 晶振负载电容 |
| C3 | 10μF 电解电容 | 复位延时电容 |
| R1 | 10kΩ 电阻 | 复位下拉电阻 |
| SW1 | 常开按键 | 手动复位 |

**元件数**：7
**网络数**：5（VCC / GND / XTAL_OSC / RST / EA_GND）
**引脚数**：52

> 网络命名以 `schematic_ir.json` 为准：晶振为**单一合并网络 `XTAL_OSC`**（U1.XTAL1 与 U1.XTAL2 共网），EA 为独立接地网络 `EA_GND`。

## 3. 已知设计缺陷（均可被规则检出，Rule Hit ≥ 2，满足 Sprint0 K7）

### 缺陷 E1：VCC 缺去耦电容（critical）

- **现象**：U1 的 VCC（pin40）与 GND 之间没有去耦电容
- **后果**：电源噪声抑制不足，MCU 可能随机复位
- **规则命中**：POWER_001
- **对应 Ground Truth**：`DEF-CASE001-1`

### 缺陷 E2：复位电路 RC 连接异常（medium）

- **现象**：R1、C3、SW1 全部并联跨接于 RST 与 GND 之间，RST 网络无任何到 VCC 的上拉路径
- **后果**：STC89C55RC 为高电平复位，上电时 RST 恒为低，无法产生高电平复位脉冲；SW1 手动复位（按下接地）同样无效
- **规则命中**：POWER_002（复位 RC 检查）
- **对应 Ground Truth**：`DEF-CASE001-2`

## 4. 已验证的优良实践

1. **复位电路参数合理**：10kΩ + 10μF → τ=100ms，满足 8051 复位保持要求（缺陷仅在连接位置，参数本身正确）
2. **晶振负载电容匹配**：22pF 与 11.0592MHz 晶振匹配，单一 XTAL_OSC 网络拓扑正确
3. **EA 硬件接地**：选择内部 Flash，避免浮空

## 5. 观察项（非缺陷）

1. **IO 端口全部悬空**：P0/P1/P2/P3 无外设，最小系统正常现象
2. **PSEN/ALE 悬空**：无外部程序存储器，正常

## 6. 校验状态

- ✅ IR 严格模式校验 PASS（`--strict`）
- ✅ 三语义字段覆盖率 100%
- ✅ `expected_review.json` 满足 Defect 10 字段规范 + Evidence 三要素
- ✅ Case001 Rule Hit = 2（POWER_001 + POWER_002），满足 Sprint0 K7

## 7. 备注

本用例作为 Sprint 0 Rule-Only Benchmark 的基线数据源（A 类）。

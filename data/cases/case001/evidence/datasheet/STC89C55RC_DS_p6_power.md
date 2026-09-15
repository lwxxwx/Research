# Datasheet 摘录 · STC89C55RC

> 来源：STC89C55RC Datasheet V2.1
> 章节：4.2 / 6.1 / 6.3 / 7.1
> 摘录人：hardware-expert
> 摘录日期：2026-09-11

## 4.2 Recommended Operating Conditions

| Symbol | Parameter | Min | Typ | Max | Unit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| VCC | Operating Voltage | 4.75 | 5.0 | 5.25 | V |
| TA | Operating Temperature | -40 | 25 | 85 | °C |
| Fosc | Oscillator Frequency | 0 | 11.0592 | 40 | MHz |

## 6.1 Crystal Oscillator

8051 内部振荡器需外接晶振。推荐电路：

```
        XTAL1 ──┬── Y1 ──┬── XTAL2
                │        │
               C1        C2
                │        │
               GND      GND
```

- 晶振频率：11.0592MHz
- 负载电容 C1/C2：22pF ±5%
- 布局要求：晶振与 MCU 距离 < 10mm，走线对称

## 6.3 Reset Circuit

8051 复位引脚 RST 需保持高电平至少 2 个机器周期。推荐 RC 复位电路：

- R1：10kΩ（下拉电阻）
- C3：10μF（上电延时电容）
- 时间常数：τ = R1 × C3 = 100ms ≥ 2 × 机器周期

## 7.1 Power Supply Decoupling

**强制要求**：每个 VCC 引脚应配置 **0.1μF 陶瓷去耦电容**，尽量靠近引脚放置。

- **U1 pin40 (VCC)**：需配 0.1μF 去耦电容
- **去耦电容缺失**：可能导致电源噪声，MCU 工作不稳定

**关联缺陷 E1（DEF-CASE001-1）**：当前原理图 VCC 网络仅有 U1.VCC 一个引脚，无任何去耦电容，违反本节强制要求。

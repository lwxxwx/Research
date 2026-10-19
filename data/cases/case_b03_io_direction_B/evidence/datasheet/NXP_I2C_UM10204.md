# NXP I2C Bus Specification UM10204 摘录 · 上拉电阻

> 来源：NXP I2C-bus specification and user manual (UM10204)
> 章节：7.1 Pull-up Resistor Sizing
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## 1. 开漏总线特性

- SDA 和 SCL 都是双向开漏引脚
- 器件只能拉低总线，不能输出高电平
- 高电平由外部上拉电阻提供

## 2. 上拉电阻计算

- **最小上拉电阻**：受最大灌电流限制（R_min = (VDD - VOL_max) / IOL_max）
- **最大上拉电阻**：受总线电容和上升时间限制（R_max = t_r / (0.8473 × C_bus)）
- **标准模式 100kHz，5V 系统，C_bus ≈ 100pF**：
  - R_min ≈ 1.7kΩ
  - R_max ≈ 5.7kΩ
  - **推荐 4.7kΩ**

## 3. 关联缺陷

- **DEF-B03-2**：I2C SDA/SCL 缺上拉，推荐 4.7kΩ
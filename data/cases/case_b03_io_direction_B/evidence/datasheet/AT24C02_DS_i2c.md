# AT24C02 Datasheet 摘录 · I2C Bus

> 来源：AT24C02 Datasheet (Atmel/Microchip)
> 章节：I2C Bus Timing
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## 1. I2C 总线特性

- **接口类型**：I2C（开漏，需外部上拉）
- **地址**：0x50（A0/A1/A2 接地）
- **电压**：1.8V~5.5V

## 2. 时序参数（标准模式 100kHz）

| 参数 | 值 | 单位 |
|---|---|---|
| SCL 频率 | 100 | kHz |
| SDA/SCL 上升时间 t_r | ≤ 1000 | ns |
| SDA/SCL 下降时间 t_f | ≤ 300 | ns |

## 3. 上拉电阻要求

- SDA 和 SCL 必须外接上拉电阻
- 标准模式 100kHz 下典型 4.7kΩ
- 无上拉 → 总线浮空 → ACK 无法传输 → 通信失败

## 关联缺陷

- **DEF-B03-2**：I2C SDA/SCL 缺上拉
---
source_type: datasheet
source_title: I2C 总线电气特性与上拉电阻选型
source_section: Open-Drain Bus Requirement 开漏总线要求
part_numbers: ["AT24C02"]
related_rule_ids: ["IO_003", "IO_002"]
---
## 1. 开漏总线要求

- 开漏特性：器件只能拉低，不能输出高电平
- 高电平来源：外部上拉电阻
- 无上拉后果：总线浮空，通信完全失效
- SDA 和 SCL 都是双向开漏引脚

## 2. 上拉电阻选型

| 模式 | 速率 | 典型上拉 |
|---|---|---|
| 标准模式 | 100kHz | 4.7kΩ |
| 快速模式 | 400kHz | 1.5kΩ~2.2kΩ |
| 快速模式 Plus | 1MHz | 1kΩ 以下 |
| 高速模式 | 3.4MHz | 内置电流源，无需上拉 |

## 3. 上拉电阻计算公式（NXP UM10204）

- 最小上拉电阻：R_min = (VDD - VOL_max) / IOL_max
- 最大上拉电阻：R_max = t_r / (0.8473 × C_bus)
- 标准模式 100kHz，5V 系统，C_bus ≈ 100pF：
  - R_min ≈ 1.7kΩ
  - R_max ≈ 5.7kΩ
  - 推荐 4.7kΩ

## 4. AT24C02 时序参数（标准模式 100kHz）

| 参数 | 值 | 单位 |
|---|---|---|
| SCL 频率 | 100 | kHz |
| SDA/SCL 上升时间 t_r | ≤ 1000 | ns |
| SDA/SCL 下降时间 t_f | ≤ 300 | ns |

## 5. 常见错误

- SDA/SCL 无上拉 → 总线浮空 → ACK 无法传输 → 通信失败
- 上拉电阻过大 → 上升时间超标 → 高速模式失效
- 上拉电阻过小 → 灌电流超标 → 器件损坏

> 注：本摘录为占位，具体条款需硬件专家核对 NXP UM10204 原文。
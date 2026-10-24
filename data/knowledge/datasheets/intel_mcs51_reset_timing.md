---
source_type: datasheet
source_title: Intel MCS-51 User Manual 复位时序
source_section: Reset Timing
part_numbers: ["STC89C55RC", "8051"]
related_rule_ids: ["CTRL_002"]
---
## 1. 复位时序要求

- 手册要求：RST 引脚保持高电平至少 2 个机器周期
- 机器周期：12 / 晶振频率
- @11.0592MHz：机器周期 ≈ 12 / 11.0592MHz ≈ 1.085μs
- 2 机器周期 ≈ 2.17μs

## 2. 工程实践

实际工程要求 RST 高电平持续 ≥ 2ms，以覆盖：
- 电源上电时间
- 晶振起振时间
- 电源波动
- 温度漂移

本 Case 的 min_reset_high_ms = 2 是工程经验值，不是 Intel 手册硬性要求。

## 3. 与 RC 容差的关系

Intel 手册 2.17μs 是理论下限；工程 2ms 是实践要求。
RC 复位电路需按工程 2ms 设计，并留 1.5× 裕量（τ ≥ 3ms）。
# STC89C55RC Datasheet 摘录 · Absolute Maximum Ratings

> 来源：STC89C55RC Datasheet V2.1
> 章节：Absolute Maximum Ratings
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## Absolute Maximum Ratings

| 参数 | 值 | 单位 |
|---|---|---|
| VCC 到 GND 电压 | -0.5 ~ +6.5 | V |
| 任意 IO 引脚到 GND 电压 | -0.5 ~ VCC+0.5 | V |
| 工作温度 | -40 ~ +85 | ℃ |

## IO 引脚电压约束

**关键要求**：任意 IO 引脚电压**不得超过 VCC + 0.5V**，否则：
- ESD 保护二极管导通
- 可能触发闩锁（latch-up）
- 可能损坏芯片

## 关联缺陷

- **DEF-B04-2**：上电时序异常，8051 的 VCC 未建立（0V）时，SPI Flash 的 IO 输出 3.3V，超过 VCC+0.5V = 0.5V，触发 ESD 保护二极管导通
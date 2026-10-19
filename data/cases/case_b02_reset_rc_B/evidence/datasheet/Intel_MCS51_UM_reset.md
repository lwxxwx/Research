# Intel MCS-51 User Manual 摘录 · Reset Timing

> 来源：Intel MCS-51 User Manual (1980)
> 章节：Reset Timing
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## 复位时序要求

- **手册要求**：RST 引脚保持高电平至少 2 个机器周期
- **机器周期**：12 / 晶振频率
- **@11.0592MHz**：机器周期 ≈ 12 / 11.0592MHz ≈ 1.085μs
- **2 机器周期** ≈ 2.17μs

## 工程实践

- 实际工程要求 RST 高电平持续 ≥ 2ms，以覆盖：
  - 电源上电时间
  - 晶振起振时间
  - 电源波动
  - 温度漂移
- 本 Case 的 `min_reset_high_ms = 2` 是工程经验值，不是 Intel 手册硬性要求

## 关联缺陷

- **DEF-B02-1**：复位 RC 最坏工况 τ=1.782ms < 2ms（工程阈值），但 > 2.17μs（Intel 手册要求）
- 缺陷核心是"工程裕量不足"，不是"违反 Intel 手册"
# AMS1117 Datasheet 摘录 · 启动时间

> 来源：AMS1117 Datasheet
> 章节：Startup Time / Application Information
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## 启动时间

- **典型启动时间**：1~10ms
- **影响因素**：
  - 输出电容（越大启动越慢）
  - 负载电流（越小启动越快）
  - 输入电压

## 输出电容对启动的影响

- 输出电容充电时间 ≈ C_out × V_out / I_load
- 例：C_out=220μF, V_out=5V, I_load=1A → t ≈ 1.1ms
- 例：C_out=10μF, V_out=3.3V, I_load=0.5A → t ≈ 66μs

## 关联缺陷

- **DEF-B04-2**：U4(5V LDO) 输出电容 220μF，U3(3.3V LDO) 输入电容 10μF → 5V 建立慢于 3.3V 约 1~10ms
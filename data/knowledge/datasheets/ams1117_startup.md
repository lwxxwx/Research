---
source_type: datasheet
source_title: AMS1117 Datasheet 启动时间
source_section: Startup Time 启动特性
part_numbers: ["AMS1117"]
related_rule_ids: ["POWER_001", "CTRL_002"]
---
## 1. 启动时间

- 典型启动时间：1~10ms
- 影响因素：
  - 输出电容（越大启动越慢）
  - 负载电流（越小启动越快）
  - 输入电压

## 2. 输出电容对启动的影响

- 输出电容充电时间 ≈ C_out × V_out / I_load
- 例：C_out=220μF, V_out=5V, I_load=1A → t ≈ 1.1ms
- 例：C_out=10μF, V_out=3.3V, I_load=0.5A → t ≈ 66μs

## 3. 多路 LDO 时序分析

当两路 LDO 输出电容不同：
- 大电容 LDO 启动慢，小电容 LDO 启动快
- 若小电容 LDO 是 IO 电压、大电容 LDO 是核心电压
- 则 IO 电压先于核心电压建立 → 灌电流风险

本 Case：U4(5V LDO) 输出电容 220μF，U3(3.3V LDO) 输入电容 10μF
→ 5V 建立慢于 3.3V 约 1~10ms → 时序异常
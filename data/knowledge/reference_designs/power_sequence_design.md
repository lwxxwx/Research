---
source_type: design_standard
source_title: 多路电源时序设计准则
source_section: Multi-Rail Power Sequencing 多路电源时序
part_numbers: []
related_rule_ids: ["POWER_001", "CTRL_002"]
---
## 1. 核心原则

- 核心电压（VCC_CORE）必须先于或同步于 IO 电压（VCC_IO）
- 反之，IO 电压通过 ESD 保护二极管灌入核心电压域，可能触发闩锁（latch-up）
- 多路电源上电顺序必须明确设计，不能依赖"自然上电"

## 2. 常见错误

- 双 LDO 未做时序控制
- 大电容 LDO 与小电容 LDO 混用（启动速度不一致）
- 无 PMIC 时序控制
- IO 电压先于核心电压建立

## 3. 修复方案

- 使用带 EN 引脚的 LDO + RC 延时
- 使用 PMIC 时序控制芯片
- 使用电源监控（PGOOD 信号）
- SPI/IO 线上加电平转换/隔离

## 4. 灌电流路径分析

上电时序异常时，IO 电压通过 ESD 保护二极管灌入未上电的 VCC 域：
- 8051 VCC 未建立（0V）时，SPI Flash 的 IO 输出 3.3V
- 超过 VCC+0.5V = 0.5V → ESD 二极管导通 → 闩锁风险
- 电流路径：Flash.DO → SPI_MISO → 8051.P1.5 → ESD 二极管 → VCC

> 注：本摘录为占位，具体条款需硬件专家核对。
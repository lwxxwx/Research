---
source_type: datasheet
source_title: W25Q16 Datasheet 上电时序
source_section: Power-Up Timing 上电时序
part_numbers: ["W25Q16"]
related_rule_ids: ["POWER_001", "CTRL_002"]
---
## 1. Power-Up Timing

- VCC 上升时间：≤ 5ms
- VCC 到稳定时间：≤ 10ms
- 上电后首次访问延迟：≥ 100μs

## 2. IO 引脚方向

- CS/CLK/DI：input（接收 8051 信号）
- DO：output（发送数据到 8051）
- WP/HOLD：input

## 3. 上电时序风险

SPI Flash 上电后 DO 引脚即进入 output 状态。
若此时 8051 VCC 尚未建立（0V），DO 的输出电压通过 SPI_MISO 灌入 8051 的 IO 引脚：
- 8051 IO 电压约束：≤ VCC + 0.5V
- VCC=0V 时，DO 输出 3.3V > 0.5V → ESD 保护二极管导通
- 可能触发闩锁，损坏 8051

修复：确保 8051 VCC 先于或同步于 SPI Flash VCC 建立。
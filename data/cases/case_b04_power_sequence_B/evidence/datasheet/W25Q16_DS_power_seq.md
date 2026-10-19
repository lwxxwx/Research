# W25Q16 Datasheet 摘录 · 上电时序

> 来源：W25Q16 Datasheet (Winbond)
> 章节：Power-Up Timing
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## Power-Up Timing

- **VCC 上升时间**：≤ 5ms
- **VCC 到稳定时间**：≤ 10ms
- **上电后首次访问延迟**：≥ 100μs

## IO 引脚

- CS/CLK/DI：input（接收 8051 信号）
- **DO：output（发送数据到 8051）**
- WP/HOLD：input

## 关联缺陷

- **DEF-B04-2**：SPI Flash 上电后 IO 已输出，若 8051 VCC 未建立，DO 的 output 电压通过 SPI_MISO 灌入 8051
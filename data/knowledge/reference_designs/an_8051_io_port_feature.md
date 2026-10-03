---
source_type: application_note
source_title: 8051 IO端口设计笔记 AN‑MCU‑IO‑02
source_section: P0/P1/P2/P3端口电气特性
part_numbers: ["STC89C55RC"]
related_rule_ids: ["IO_001"]
---
STC89C55RC包含4组8位IO口：P0、P1、P2、P3。
P1、P2、P3端口内置上拉电阻；P0端口内部无上拉。
最小系统调试场景，IO引脚允许悬空；产品硬件设计中P0用作输出必须增加外部上拉。
IO端口不能超过芯片最大灌拉电流，端口电压不允许超过VCC或者低于GND。

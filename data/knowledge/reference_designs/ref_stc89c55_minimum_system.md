---
source_type: reference_design
source_title: STC89C55RC最小系统参考设计
source_section: Minimum Working Circuit 最小工作条件
part_numbers: ["STC89C55RC"]
related_rule_ids: ["POWER_001","POWER_002"]
---
STC89C55RC可以正常启动运行，必须同时具备三套基础电路：
1. +5V可靠电源供电VCC/GND；
2. XTAL1/XTAL2外接晶振与匹配电容的时钟振荡电路；
3. RST引脚正确的上电复位电路。
缺少任意一套电路，单片机无法正常执行固件程序。
P0‑P3为通用IO端口，最小系统下IO口可以悬空，不影响芯片启动。
DIP‑40封装STC89C55RC：P0口无上拉，实际产品需要外部上拉电阻。

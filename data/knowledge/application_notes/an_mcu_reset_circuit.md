---
source_type: application_note
source_title: 8051最小系统应用笔记 AN‑MCU‑RST‑01
source_section: 上电复位电路
part_numbers: ["STC89C55RC"]
related_rule_ids: ["MCU_003"]
---
8051系列高电平复位，RST引脚需要维持至少2个机器周期高电平完成复位。
经典RC上电复位：R1=10kΩ上拉电阻，C3=10μF复位电容。
上电瞬间电容充电，RST保持高电平；电容充满后RST拉低，MCU开始运行程序。
增加手动复位按键SW1并联在电容两端，按下按键C3放电，强制触发系统复位。
RST引脚不可悬空，悬空将导致MCU随机复位、运行异常。

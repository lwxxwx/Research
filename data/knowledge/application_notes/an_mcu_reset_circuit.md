---
source_type: application_note
source_title: 8051最小系统应用笔记 AN‑MCU‑RST‑01
source_section: 上电复位电路
part_numbers: ["STC89C55RC"]
related_rule_ids: ["MCU_003"]
---
8051系列高电平复位，RST引脚需要维持至少2个机器周期高电平完成复位。
经典RC上电复位：R1=10kΩ下拉电阻（接RST-GND），C3=10μF上电延时电容（接VCC-RST）。
上电瞬间C3耦合VCC跳变，RST产生短暂高电平；随后R1将RST拉低，MCU开始运行程序。
增加手动复位按键SW1并联在RST与VCC之间，按下按键RST拉高，强制触发系统复位。
RST引脚不可悬空，悬空将导致MCU随机复位、运行异常。

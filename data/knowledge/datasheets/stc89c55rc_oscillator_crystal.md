---
source_type: datasheet
source_title: STC89C55RC datasheet
source_section: Oscillator Circuit 晶振电路设计
part_numbers: ["STC89C55RC"]
related_rule_ids: []
---
STC89C55RC使用片内振荡器，XTAL1、XTAL2引脚外接石英晶振Y1。
典型晶振规格11.0592MHz，两侧匹配电容C1、C2取值22pF。
电容一端分别接XTAL1、XTAL2引脚，另一端统一连接GND。
晶振与匹配电容必须就近摆放至MCU的XTAL引脚，减少走线长度，降低时钟信号干扰。
晶振电路负载电容不匹配，会造成MCU时钟不稳、起振失败。

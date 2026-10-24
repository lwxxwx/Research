---
source_type: datasheet
source_title: LM2596 Datasheet 热特性与降额
source_section: Thermal Considerations 热设计
part_numbers: ["LM2596"]
related_rule_ids: ["POWER_003", "POWER_001"]
---
## 1. Absolute Maximum Ratings

| 参数 | 值 | 单位 |
|---|---|---|
| 最大结温 T_J_max | 125 | ℃ |
| 存储温度 | -65 ~ 150 | ℃ |

## 2. Thermal Considerations（TO-263 封装）

| 参数 | 值 | 单位 | 备注 |
|---|---|---|---|
| θJA 结到环境热阻 | 50 | ℃/W | TO-263 封装，无散热片 |
| θJA 加散热片 | 25 | ℃/W | 需外加散热片 |

## 3. 效率与功耗计算

12V→5V/1A 典型效率 η ≈ 80%。

功耗计算：
1. P_out = 5V × 1A = 5W
2. P_in = P_out / η = 6.25W
3. P_loss = P_in - P_out = 1.25W
4. ΔT = P_loss × θJA = 1.25W × 50℃/W = 62.5℃
5. T_J = T_A + ΔT = 85 + 62.5 = 147.5℃
6. 对比 T_J_max = 125℃ → 超限

## 4. Buck 转换器热设计准则

开关稳压器功耗：P_loss = P_out × (1/η - 1)
结温估算：T_J = T_A + P_loss × θJA

降额要求：T_J ≤ T_J_max × 降额系数（GJB/Z 35 Ⅱ级：≤ 95℃）

常见封装 θJA 参考：
| 封装 | θJA（无散热片） | θJA（加散热片） |
|---|---|---|
| TO-220 | 65℃/W | 25~40℃/W |
| TO-263（D2PAK） | 50℃/W | 25~40℃/W |
| SOIC-8 | 120~150℃/W | 不适用 |

## 5. 降低结温的方法

| 方法 | 效果 | 代价 |
|---|---|---|
| 改用同步整流 Buck | 效率 80%→90% | 成本↑，复杂度↑ |
| 降低 Vin | 减小压差 | 输入源受限 |
| 降低负载电流 | 直接减小 P_loss | 应用受限 |
| 加散热片 | θJA 50→25℃/W，T_J→116.25℃ | 成本↑，体积↑ |
| 降低环境温度 | 降低 T_J | 场景受限 |

## 6. 本 Case 应用

LM2596：P_loss=1.25W，θJA=50℃/W，T_A=85℃
T_J = 147.5℃ > 125℃（T_J_max）且 > 95℃（GJB/Z 35 Ⅱ级降额线）→ 超限

解决方案：加散热片（T_J→116.25℃）、改同步整流（P_loss=0.556W，T_J=112.8℃）、降低负载（<0.5A）。
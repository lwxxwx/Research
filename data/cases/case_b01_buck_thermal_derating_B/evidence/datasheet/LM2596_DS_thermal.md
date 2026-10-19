# LM2596 Datasheet 摘录 · Thermal Considerations

> 来源：TI LM2596 Datasheet (SNVS124E)
> 章节：Absolute Maximum Ratings / Thermal Considerations / Efficiency
> 摘录人：PROVISIONAL_AI_ASSISTANT
> 摘录日期：2026-10-19

## 1. Absolute Maximum Ratings

| 参数 | 值 | 单位 |
|---|---|---|
| 最大结温 T_J_max | 125 | ℃ |
| 存储温度 | -65 ~ 150 | ℃ |

## 2. Thermal Considerations（TO-263 封装）

| 参数 | 值 | 单位 | 备注 |
|---|---|---|---|
| θJA（结到环境热阻） | 50 | ℃/W | TO-263 封装，无散热片 |
| θJA（加散热片） | < 25 | ℃/W | 需外加散热片 |

**来源**：TI LM2596 Datasheet (SNVS124E), Section "Thermal Considerations"

## 3. Efficiency（12V→5V/1A）

- 典型效率 η ≈ 80%

**来源**：TI LM2596 Datasheet, Figure "Efficiency vs Load Current"

## 4. 功耗计算示例（12V→5V/1A）

1. P_out = 5V × 1A = 5W
2. P_in = P_out / η = 5W / 0.8 = 6.25W
3. P_loss = P_in - P_out = 1.25W
4. ΔT = P_loss × θJA = 1.25W × 50℃/W = 62.5℃
5. T_J = T_A + ΔT = 85 + 62.5 = 147.5℃
6. 对比 T_J_max = 125℃ → 超限

## 5. 关联缺陷

- **DEF-B01-2**：LM2596 在 85℃ + 1A 下结温 147.5℃，超 T_J_max 和 GJB/Z 35 降额线
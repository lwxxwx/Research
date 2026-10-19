# Sprint 1 Golden Case 覆盖矩阵

> 依据：Sprint 1 融合版 §23.5、前置条件 §4.4
> 最后更新：2026-10-19
> 状态：🟢 A 类 6/6 + B 类 4/4 完成；阶段 B 的 P0 目标全部闭环

## IR 命名约定（重要）

`Net.connected_pins` 的格式是 `{Component.ref}.{Pin.pin_id}`，**只能有一个点**。
因此：

- **`Component.ref`**：不能含点（如 `U1` / `R1` / `C1`）
- **`Pin.pin_id`**：不能含点
  - 对于 8051 的 `P1.0` 等引脚，`pin_id` 用下划线：`P1_0`
  - `Pin.name` 可保留 `P1.0`（显示用）
  - `Pin.number` 可保留 `"1"`（物理脚号）

**正确 / 错误示例**：

| 写法 | 结果 |
|---|---|
| `"U1.P1_0"` | ✅ 正确（1 个点） |
| `"U1.P1.0"` | ❌ 错误（2 个点，违反 Schema 校验） |

## A 类 Case（6/6 完成）

| Case | 主题 | 覆盖规则 | 预期命中 | 专家金标 | 证据来源 | 状态 |
|---|---|---|---|---|---|---|
| case001 | 8051 最小系统 | POWER_001, POWER_002 | 2 条 | ✅ | datasheet + app_note + ref_des | ✅ 已标注 |
| case002 | 8051 + LDO 稳压器（降额违规） | POWER_003 | 1 条 | ✅ | datasheet + ref_des | ✅ 已标注 |
| case003 | 8051 晶振负载电容错误 | CLOCK_001 | 1 条 | ✅ | datasheet + ref_des | ✅ 已标注 |
| case004 | 8051 UART 接口（IO 冲突） | IO_002, IO_003 | 2 条 | ✅ | datasheet + ref_des | ✅ 已标注 |
| case005 | 8051 外扩存储器（使能悬空） | CTRL_001 | 1 条 | ✅ | datasheet + ref_des | ✅ 已标注 |
| case006 | 8051 晶振不完整 + 复位时序 | CLOCK_002, CTRL_002 | 2 条 | ✅ | datasheet + app_note + ref_des | ✅ 已标注 |

## B 类 Case（4/4 完成）

| Case | 主题 | 覆盖规则 | AI-only 候选 | 专家金标 | 必需证据 | 状态 |
|---|---|---|---|---|---|---|
| case_b01_buck_thermal_derating_B | Buck 热降额 | POWER_001 | 热降额裕量不足（AI 发现） | PROVISIONAL | 热降额曲线 | ✅ 已标注 |
| case_b02_reset_rc_B | Reset RC 时序 | 无（纯 AI-only） | RC 时间常数裕量不足 | PROVISIONAL | 复位时序图 | ✅ 已标注 |
| case_b03_io_direction_B | IO 上拉/开漏冲突 | IO_002 | I2C 开漏总线缺上拉 | PROVISIONAL | MCU IO 特性 | ✅ 已标注 |
| case_b04_power_sequence_B | 上电时序 | POWER_001 | 多路电源上电顺序异常 | PROVISIONAL | 时序图 | ✅ 已标注 |

> **B 类 4 个 Case 的状态为 `PROVISIONAL`**（AI 交叉审查占位，待真实硬件专家校准）。`reviewer: PROVISIONAL_AI_ASSISTANT`，`pending_review_by: hardware-expert`。

## 规则覆盖情况（10 / 10 ✅）

| Rule ID | 分类 | 来源 | 关联 Case | 正例 | 反例 | 实现状态 |
|---|---|---|---|---|---|---|
| POWER_001 | power | Sprint 0 | case001, case_b01, case_b04 | ✅ | - | ✅（已修复前缀匹配 + 电容值容差） |
| POWER_002 | timing | Sprint 0 | case001 | ✅ | - | ✅（已通用化 reset_polarity） |
| POWER_003 | power | Sprint 0 预留 | case002 | ✅ | - | ✅（完整实现，温度降额曲线） |
| IO_001 | interface | Sprint 0 | case001, case_b01~b04 | ✅ | - | ✅ |
| IO_002 | interface | Sprint 0 | case004, case_b03 | ✅ | - | ✅ |
| CLOCK_001 | clock | Sprint 1 新增 | case003 | ✅ | case006 | ✅ |
| CLOCK_002 | clock | Sprint 1 新增 | case006 | ✅ | case003 | ✅ |
| IO_003 | interface | Sprint 1 新增 | case004 | ✅ | case_b04 | ✅ |
| CTRL_001 | interface | Sprint 1 新增 | case005 | ✅ | - | ✅ |
| CTRL_002 | timing | Sprint 1 新增 | case006 | ✅ | case_b02 | ✅ |

> **规则语义边界说明**：
> - `IO_002` vs `IO_003`：`IO_002` 检查 IO 间方向冲突；`IO_003` 检查 MCU IO 与外部器件输出（推挽）的冲突
> - `POWER_002` vs `CTRL_002`：`POWER_002` 检查复位 RC 拓扑（路径）；`CTRL_002` 检查复位 RC 时间常数（数值）
> - `CLOCK_001` vs `CLOCK_002`：`CLOCK_001` 检查负载电容值不匹配；`CLOCK_002` 检查晶振电路完整性（缺项）

## 测试覆盖统计

| 测试文件 | 测试数 | 职责 |
|---|---|---|
| `test_rules.py` | 58+ | 规则引擎 + A/B 类规则命中 |
| `test_case_b_validation.py` | 60 | B 类资产完整性 |
| `test_case_naming.py` | 105 | A/B 类命名 + 结构校验 |
| 其他（Sprint 0 回归） | 253+ | - |
| **总计** | **476** | ✅ 全部通过 |

## 进度统计

| 项 | 已完成 | 总数 | 进度 |
|---|---|---|---|
| 规则 YAML | 10 | 10 | ✅ 100% |
| 规则实现 | 10 | 10 | ✅ 100% |
| A 类 Case | 6 | 6 | ✅ 100% |
| B 类 Case | 4 | 4 | ✅ 100% |
| 覆盖矩阵 | 1 | 1 | ✅ 100% |
| 测试 | 476 | - | ✅ 全通过 |

## 覆盖空白与风险

| 空白 | 原因 | 风险 | 处理 |
|---|---|---|---|
| B 类 4 个 Case 为 PROVISIONAL | 需真实硬件专家校准 | AI_NER/EAR 正式验收依赖专家签字 | 待专家校准后改为 CALIBRATED |
| CTRL_001 的悬空缺陷无 net | Benchmark 匹配逻辑要求 net 交集非空 | 可能 FP | 阶段 F 前修正匹配逻辑 |
| POWER_003 的缺陷也无 net | 功耗是器件级属性，不关联网络 | 可能 FP | 阶段 F 前修正匹配逻辑 |
| POWER_003 温度降额曲线为占位值 | 无项目硬件规范 | 规则阈值可能偏差 | 等硬件依据后替换 YAML 参数 |

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-15 | 初版创建 |
| 2026-10-16 | A 类 6/6 完成，规则 10/10 实现 |
| 2026-10-19 | B 类 4/4 完成；`test_case_naming.py` 建立；阶段 B 的 P0 目标全部闭环 |
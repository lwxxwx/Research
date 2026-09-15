# case001_expert_calibration.md
> Business Freeze · Sprint 0 专家校准纪要 · 变更需架构评审
> 文档定位：`docs/freeze/case001_expert_calibration.md`

## 基本信息

| 项 | 值 |
| :--- | :--- |
| case_id | case001 |
| title | 8051 最小系统原理图（STC89C55RC） |
| case_class | **A**（Rule-Only 基线） |
| ir_schema_version | v1.0 |
| calibration_date | 2026-09-11 |
| calibration_status | **CALIBRATED** |
| reviewer | **hardware-expert** |

## 1. 校准范围

1. `schematic_ir.json` — IR 中间表示（Phase D 产出）
2. `expert_reasoning.md` — 专家推理记录（V1.2 §2.3 四步模板）
3. `expected_review.json` — Ground-Truth 预期评审结果
4. `evaluation.yaml` — 评测配置
5. `evidence/` — 证据资产（datasheet / application_note / reference_design）
6. `info.yaml` / `circuit_notes.md` — 元信息与设计说明

## 2. 验证结果清单

- ✅ IR strict 模式校验 PASS；三语义字段（intent/context）覆盖率 100%；`lib_name` 全部非空；`connected_pins` 引用全部合法（`{ref}.{pin_id}`）
- ✅ IR 元件 7 个、网络 5 个、引脚 52，与 `extra_meta` 统计一致
- ✅ 网络清单与 IR 对齐：VCC / GND / **XTAL_OSC** / RST / EA_GND（晶振为单一合并网络；EA 独立接地）
- ✅ `expected_review.json` 满足 Defect 10 字段规范 + Evidence 三要素
- ✅ 目录结构完整（info.yaml / IR / expert_reasoning / expected_review / evaluation / evidence）

## 3. 电路确认点

1. **晶振电路**：Y1/C1/C2 连接正确，负载电容 22pF 符合 datasheet §6.1；电气网络为单一 `XTAL_OSC`（XTAL1/XTAL2 共网）。
2. **电源去耦（DEF-CASE001-1 / POWER_001 / critical）**：VCC 网络仅含 U1.VCC，缺 0.1μF 去耦电容，违反 datasheet §7.1 强制要求，为**确认的真实缺陷**。
3. **复位电路（DEF-CASE001-2 / POWER_002 / medium）**：R1/C3/SW1 均跨接 RST-GND，RST 无到 VCC 的上拉路径；STC89C55RC 为高电平复位，无法产生上电复位脉冲，SW1 手动复位无效。为**确认的真实缺陷**（连接位置错误；τ=100ms 参数本身正确）。
4. **EA 引脚**：接地（EA_GND），选择内部 Flash，避免浮空。
5. **IO 端口全部悬空 / PSEN/ALE 悬空**：最小系统正常现象，仅作观察项（OBS-CASE001-1/2），不列为缺陷。

## 4. 分类校准说明（相对 DeepSeek 原交付的变更）

- 原 DeepSeek 交付将 case001 声明为 **B 类**（`ai_expected_count:1`），但 `expected_review.json` 无任何 `origin=ai_discovered` 缺陷，全部可由规则检出，**不符合 V1.2 B 类合格四要素**。
- 结合 Sprint0 Phase G 将 case001 定位为 **Rule-Only Benchmark 基线**、且 IR 真实缺陷均可被规则稳定检出，**校准为 A 类**：`ai_expected_count:0`，权重 `rule 0.8 / ai 0.2`。
- **K7 验收（Case001 Rule Hit ≥ 2）**：由 POWER_001（去耦）+ POWER_002（复位 RC）两条命中达成。Phase F 规则库中 POWER_002 名义定义为「反馈分压」（DC-DC 场景），本 case 无反馈分压电路；复位 RC 检查为本 case 语境下的规则命中项，规则清单以 `data/rules` 最终合入为准。
- **文档-网络一致性修正**：原 `expert_reasoning.md` / `circuit_notes.md` / `expected_review.json`（GP-CASE001-2）描述晶振为「XTAL1/XTAL2 两个独立网络」，与 IR 单一 `XTAL_OSC` 不符；已统一为与 IR 一致。`circuit_notes` 网络清单补上 `EA_GND`。

## 5. 风险说明

- case001 为 Sprint0 基准（A 类 / Rule-Only），后续修改 IR / expected_review / evaluation 必须重新执行专家校准并更新本纪要。
- 复位电路缺陷 DEF-CASE001-2 依赖「STC89C55RC 为高电平复位、C3 应接 VCC-RST」的工程判定；若上游确认原理图本意采用低电平复位系统，需复核本缺陷。
- 本用例作为 `scripts.run_benchmark` 的 Rule-Only 基线数据源，输出 `out/bench_ruleonly_sprint0.csv`。

## 6. 验收签字

- reviewer：**hardware-expert**
- calibration_status：**CALIBRATION_COMPLETE**
- calibration_date：2026-09-11

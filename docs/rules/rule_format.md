# Rule YAML Format v1.0（Business Freeze · Sprint 0）

> 变更需架构评审；rule_id 不复用，废弃规则用 `enabled: false`。

## 1. 必填字段

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `rule_id` | string | 规则唯一标识，如 `POWER_001` |
| `rule_name` | string | 规则中文名 |
| `version` | string | 规则版本，Sprint1 规则演进使用 |
| `category` | string | 与 `report_defects.category` 对齐：power/clock/interface/memory/thermal/esd/timing/layout/parametric/other |
| `severity` | string | low/medium/high/critical |
| `applicable_condition` | object | 规则匹配条件（见 §3） |
| `check_logic` | object | `{function, params}`，见 §4 |
| `rule_basis` | string | 规则依据（Datasheet/规范章节） |
| `suggestion` | string | 默认建议模板（可被 LLM 重写） |
| `enabled` | bool | **可选**，默认 `true`；关闭后引擎跳过 |

> **注**：本表列 10 个字段（9 必填 + 1 可选）；方案 §18 的"9 字段冻结"口径为不含 `enabled` 的核心 9 字段。

## 2. 可选字段

| 字段 | 类型 | 默认 | 说明 |
| :--- | :--- | :--- | :--- |
| `enabled` | bool | `true` | 关闭后引擎跳过（与 §1 表末行重复，保留以兼容旧引用） |

## 3. `applicable_condition` 算子

| 算子 | 语义 |
| :--- | :--- |
| `scope` | 占位字段，MVP 视为 always |
| `component_type_in` | 匹配 `Component.lib_name` 集合 |
| `component_ref_in` | 匹配 `Component.ref` 集合 |
| `net_name_in` | 匹配 `Net.net_name` 集合 |
| `net_is_power` | 匹配电源网络布尔值 |
| `attribute_exists` | 匹配 `Component.attributes[].key` |
| `any_of` | 任一条件块匹配即适用 |
| `all_of` | 全部条件块匹配才适用 |
| `always: true` | 无条件适用 |

## 4. `check_logic`

```yaml
check_logic:
  function: "circuit_checks.check_decoupling"   # 必须 ∈ registry keys
  params:
    power_net_names: ["VCC"]
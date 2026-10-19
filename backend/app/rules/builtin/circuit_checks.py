"""
app/rules/builtin/circuit_checks.py
Sprint0 Phase F 电路规则内置原子检查函数（含电源 + IO 方向/悬空）。

设计约束（V1.2 §4.3 规则定义/执行引擎分离）：
  - 每个 check 函数只负责"哪里命中了"，不关心 rule_id/severity/suggestion
  - 返回 List[RuleHitItem]；无命中返回空列表
  - 函数签名统一：(schematic_ir, rule_params) -> List[RuleHitItem]

覆盖规则：
  - POWER_001  check_decoupling           电源引脚去耦电容
  - POWER_002  check_reset_rc_topology    高电平 RC 复位拓扑
  - POWER_003  check_derating             器件功率降额（Sprint0 预留，返回空）
  - IO_001     check_io_floating          IO 引脚悬空检查（汇总，severity=low）
  - IO_002     check_io_direction         IO 方向冲突检查（severity=critical）

迁移说明（v1.0.1）：
  原 power_checks.py 的 3 条电源规则已整体迁入本文件；
  registry.py 的 function 键名统一为 "circuit_checks.check_xxx"。
"""
from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.ir.schema import PinDirection, PinType, SchematicIRDocument


# ---------------------------------------------------------------------------
# 命中项中间层：结构化描述"命中位置 + IR 证据引用"
# ---------------------------------------------------------------------------
class RuleHitItem(BaseModel):
    component_refs: List[str] = Field(default_factory=list)
    net_refs: List[str] = Field(default_factory=list)
    message: str
    # 结构化 IR 证据引用，供 Phase G 证据链展示（替代自由 dict 的 extra）
    evidence_ir_refs: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
_PASSIVE_LIB_PREFIXES = ("CAP", "RES", "IND")
_CAP_LIB_PREFIXES = ("CAP",)
_MCU_LIB_NAMES = {"STC89C55RC", "STC89C5x", "STC89C51", "STC89C52", "AT89C5x", "8051", "MCU_8051"}


def _find_component(ir: SchematicIRDocument, ref: str):
    for c in ir.components:
        if c.ref == ref:
            return c
    return None


def _net_map(ir: SchematicIRDocument) -> Dict[str, Any]:
    return {n.net_name: n for n in ir.nets}


def _comp_pin_nets(ir: SchematicIRDocument, comp) -> Dict[str, str]:
    """返回 {pin_id: net_name}，用于快速判断元件各引脚落在哪个网络。"""
    pin_to_net: Dict[str, str] = {}
    for n in ir.nets:
        for pin_ref in n.connected_pins:
            ref, _, pin_id = pin_ref.partition(".")
            if ref == comp.ref:
                pin_to_net[pin_id] = n.net_name
    return pin_to_net


def _is_cap(lib_name: str) -> bool:
    return lib_name.startswith(_CAP_LIB_PREFIXES)


def _is_passive(lib_name: str) -> bool:
    return lib_name.startswith(_PASSIVE_LIB_PREFIXES)


def _has_io_pins(comp) -> bool:
    """元件是否存在 IO 类型引脚（pin_type == io）。

    Sprint0 判定标准：不依赖 MCU 型号白名单，只看元件是否有 IO 引脚；
    这样任何带 IO 的器件（MCU/CPLD/FPGA/扩展器）都能被检查。
    """
    return any(p.pin_type == PinType.IO for p in comp.pins)


def _io_pins(comp) -> List[Any]:
    """取元件所有 pin_type == io 的引脚。"""
    return [p for p in comp.pins if p.pin_type == PinType.IO]


# ---------------------------------------------------------------------------
# POWER_001：电源引脚去耦电容
# ---------------------------------------------------------------------------
def check_decoupling(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（Sprint1 修正版）：
      对每个电源网络，检查是否存在一个"电容值接近期望值"的电容，
      其一端接该电源网络、另一端接 GND。若不存在 → 命中。

    Sprint1 修正（2026-10-19）：
      - 【修正 1】增加电容值检查：220μF 电解不算 0.1μF 陶瓷去耦
      - 【修正 2】支持电源网络名前缀匹配：VCC_5V 以前缀 VCC 匹配
      - 【修正 3】显式名 + 前缀匹配合并

    规则通用化说明（对齐 Sprint1 §3.2）：
      - power_net_names：显式网络名（向后兼容）
      - power_net_names_prefix：前缀匹配（如 VCC 匹配 VCC / VCC_5V / VCC_3V3）
      - 两者合并使用（显式名 + 前缀匹配）
      - 理由：不同项目电源网络命名不同（VCC / VCC_5V / VDD / 3V3），硬编码单一名称不够通用

    rule_params:
      power_net_names: List[str]             显式网络名（如 ["VCC"]）
      power_net_names_prefix: List[str]      前缀匹配（如 ["VCC", "VDD"]）
      target_component_refs: List[str]       命中时优先作为 component_refs 返回
      decouple_cap_value_expected: str       期望电容值（如 "0.1μF"）
      ground_net_names: List[str]            地网络名，默认 ["GND", "VSS"]
      tolerance_ratio: float                 电容值容差比例，默认 0.5（50%）
    """
    hits: List[RuleHitItem] = []

    power_net_names: List[str] = list(rule_params.get("power_net_names", []))
    power_net_prefixes: List[str] = list(rule_params.get("power_net_names_prefix", []))
    target_refs: List[str] = rule_params.get("target_component_refs", [])
    expected_value: str = rule_params.get("decouple_cap_value_expected", "0.1μF")
    ground_net_names: List[str] = rule_params.get("ground_net_names", ["GND", "VSS"])
    tolerance_ratio: float = float(rule_params.get("tolerance_ratio", 0.5))

    # 解析期望电容值
    expected_pf = _parse_capacitance_pf(expected_value)

    nets = _net_map(schematic_ir)

    # 合并电源网络：显式名 + 前缀匹配
    matched_net_names: set = set()
    for name in power_net_names:
        if name in nets:
            matched_net_names.add(name)
    for prefix in power_net_prefixes:
        for net in schematic_ir.nets:
            if net.net_name.startswith(prefix):
                matched_net_names.add(net.net_name)

    for net_name in sorted(matched_net_names):
        net = nets.get(net_name)
        if net is None:
            continue

        involved_refs = sorted({p.split(".")[0] for p in net.connected_pins})

        has_decouple_cap = False
        for comp in schematic_ir.components:
            if not _is_cap(comp.lib_name):
                continue

            # 检查电容值是否接近期望值
            if expected_pf is not None and comp.value:
                cap_pf = _parse_capacitance_pf(comp.value)
                if cap_pf is not None:
                    if abs(cap_pf - expected_pf) / expected_pf > tolerance_ratio:
                        continue  # 偏离太大，不算去耦电容

            pin_nets = set(_comp_pin_nets(schematic_ir, comp).values())
            if net_name in pin_nets and any(g in pin_nets for g in ground_net_names):
                has_decouple_cap = True
                break

        if not has_decouple_cap:
            hits.append(RuleHitItem(
                component_refs=target_refs or involved_refs,
                net_refs=[net_name],
                message=(
                    f"电源网络 {net_name} 缺少 {expected_value} 去耦电容"
                    f"（要求电容值接近 {expected_value}，且一端接 {net_name}、另一端接 GND）"
                ),
                evidence_ir_refs=[
                    f"Net[{net_name}].connected_pins = {net.connected_pins}",
                    f"该网络元件集合 {involved_refs} 中无电容值接近 {expected_value} 且跨接 {net_name}-GND 的电容",
                ],
            ))
    return hits

# ---------------------------------------------------------------------------
# POWER_002：RC 复位拓扑（通用化：高/低电平复位）
# ---------------------------------------------------------------------------
def check_reset_rc_topology(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（通用化）：
      复位网络上必须存在一条通往"复位有效电平"的元件级路径。
      - reset_polarity=high（8051）：路径通向 VCC
      - reset_polarity=low（AVR/ARM）：路径通向 GND

    判定方式：复位网络上某个非 MCU 元件，其引脚同时落在 RST 和 目标网络。
    若不存在 → 命中。

    规则通用化说明（Sprint1）：
      - 原实现硬编码 _MCU_LIB_NAMES 排除 MCU，现改为按 pin_type=reset 排除
      - 新增 reset_polarity 参数，支持高/低电平复位
      - 新增 exclude_component_refs 参数，允许 case 显式排除特定元件

    rule_params:
      reset_net: str                必填，复位网络名（如 "RST"）
      vcc_net: str                  必填，电源网络名（如 "VCC"）
      gnd_net: str                  必填，地网络名（如 "GND"）
      reset_polarity: str           可选，默认 "high"；可选 "high" / "low"
      exclude_component_refs: List[str]  可选，显式排除的元件位号
    """
    hits: List[RuleHitItem] = []

    reset_net_name: str = rule_params["reset_net"]
    vcc_net_name: str = rule_params["vcc_net"]
    gnd_net_name: str = rule_params["gnd_net"]
    reset_polarity: str = rule_params.get("reset_polarity", "high")
    exclude_refs: set = set(rule_params.get("exclude_component_refs", []))

    # 根据极性选择目标网络
    target_net_name = vcc_net_name if reset_polarity == "high" else gnd_net_name

    nets = _net_map(schematic_ir)
    rst_net = nets.get(reset_net_name)
    target_net = nets.get(target_net_name)
    if rst_net is None or target_net is None:
        return hits

    rst_refs = sorted({p.split(".")[0] for p in rst_net.connected_pins})

    # 判定：RST 网络上是否存在一个非 MCU 元件同时连到 RST 和目标网络
    has_target_path = False
    for ref in rst_refs:
        comp = _find_component(schematic_ir, ref)
        if comp is None:
            continue
        # 通用化排除：显式排除 或 含 reset 类型引脚（MCU）
        if ref in exclude_refs:
            continue
        if any(p.pin_type == PinType.RESET for p in comp.pins):
            continue
        pin_nets = set(_comp_pin_nets(schematic_ir, comp).values())
        if reset_net_name in pin_nets and target_net_name in pin_nets:
            has_target_path = True
            break

    if not has_target_path:
        # 定位锚点：RST 上第一个非 MCU 元件
        anchor_refs = [
            r for r in rst_refs
            if (c := _find_component(schematic_ir, r))
            and not any(p.pin_type == PinType.RESET for p in c.pins)
            and r not in exclude_refs
        ]
        polarity_desc = "高电平" if reset_polarity == "high" else "低电平"
        hits.append(RuleHitItem(
            component_refs=anchor_refs or rst_refs,
            net_refs=[reset_net_name, target_net_name],
            message=(
                f"复位网络 {reset_net_name} 没有通往 {target_net_name} 的电气路径；"
                f"{polarity_desc}复位 MCU 无法产生上电复位脉冲"
            ),
            evidence_ir_refs=[
                f"Net[{reset_net_name}].connected_pins = {rst_net.connected_pins}",
                f"RST 网络元件集合 {rst_refs} 均未提供 {reset_net_name}-{target_net_name} 通路",
            ],
        ))
    return hits

# ---------------------------------------------------------------------------
# POWER_003：器件功率降额检查
# ---------------------------------------------------------------------------
def check_derating(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（Sprint1 完整版）：
      对每个带功耗属性的器件，检查：
        实际功耗 ≤ 额定功耗 × 温度降额系数 × 器件降额系数
      不满足 → 命中。

    Sprint1 决策依据（对齐 Sprint 1 融合版 §3.2 / §3.4）：
      - 【通用化设计】
          * 属性字段名通过 rule_params 传入（不硬编码）
          * 温度降额曲线通过 rule_params 传入（不硬编码）
          * 支持任意器件（LDO / 三极管 / 电阻 / MOSFET），不绑定具体型号
      - 【决策 1】LDO 型号：case002 用 AMS1117-5.0（SOT-223，经典 5V LDO，额定 1W）
          * 理由：8051 系统最常见的 LDO 选型；SOT-223 封装功耗上限低，降额违规易演示
      - 【决策 2】电源拓扑：12V → LDO → 5V
          * 理由：压差 7V，负载 500mA → 功耗 3.5W，远超 1W 降额阈值
          * 更真实、更有教学意义
      - 【决策 3】温度降额曲线：保留占位曲线（25/70/85℃ → 1.0/0.8/0.6）
          * 理由：Sprint1 无项目硬件规范，先用通用降额准则（GJB/Z 35 / IPC-9592）
          * 后续替换为项目规范时，只改 YAML，不改代码
      - 【缺参处理】
          * 缺额定/实际功耗属性 → 跳过（不误报）
          * 单位解析失败 → 跳过（后续 Sprint 可改为 NEED_EXPERT_REVIEW）
      - 【温度降额】
          * 曲线按 temp_max_c 分段，线性取最接近的温度点
          * 无曲线 / 无温度属性 → 温度系数为 1.0（不降额）
      - 【每器件一条 hit】
          * 与 CTRL_001（每引脚一条）不同：降额是"器件级"问题
          * 一条 hit 汇总该器件的额定/实际/系数/阈值

    rule_params:
      default_derating_coefficient: float      默认降额系数，默认 0.8
      attr_rated_power: str                    额定功耗属性名
      attr_actual_power: str                   实际功耗属性名
      attr_temperature: str                    工作温度属性名
      attr_derating_coef: str                  器件级降额系数属性名
      temperature_derating_curve: List[Dict]   温度降额曲线
    """
    hits: List[RuleHitItem] = []

    attr_rated = rule_params.get("attr_rated_power", "power_dissipation_rated")
    attr_actual = rule_params.get("attr_actual_power", "power_dissipation_actual")
    attr_temp = rule_params.get("attr_temperature", "operating_temperature")
    attr_coef = rule_params.get("attr_derating_coef", "derating_coefficient")
    default_coef = float(rule_params.get("default_derating_coefficient", 0.8))
    curve = rule_params.get("temperature_derating_curve", []) or []

    for comp in schematic_ir.components:
        attrs = {a.key: a.value for a in (comp.attributes or [])}

        # 缺额定/实际功耗属性 → 跳过（不误报）
        if attr_rated not in attrs or attr_actual not in attrs:
            continue

        try:
            rated = float(attrs[attr_rated])
            actual = float(attrs[attr_actual])
        except (TypeError, ValueError):
            # 单位/格式不兼容 → 跳过（占位：后续可改为 NEED_EXPERT_REVIEW）
            continue

        # 器件级降额系数优先于默认值
        try:
            coef = float(attrs.get(attr_coef, default_coef))
        except (TypeError, ValueError):
            coef = default_coef

        # 温度降额（占位：线性取最接近的温度点）
        temp_coef = 1.0
        if curve and attr_temp in attrs:
            try:
                temp = float(attrs[attr_temp])
                for point in curve:
                    if temp <= point.get("temp_max_c", 999):
                        temp_coef = float(point.get("coefficient", 1.0))
                        break
            except (TypeError, ValueError):
                temp_coef = 1.0

        threshold = rated * coef * temp_coef
        if actual > threshold:
            hits.append(RuleHitItem(
                component_refs=[comp.ref],
                net_refs=[],
                message=(
                    f"{comp.ref}（{comp.lib_name}）实际功耗 {actual}W "
                    f"超过降额阈值 {threshold:.3f}W"
                    f"（额定 {rated}W × 降额系数 {coef} × 温度系数 {temp_coef}）"
                ),
                evidence_ir_refs=[
                    f"Component[{comp.ref}].attributes[{attr_rated}] = {rated}",
                    f"Component[{comp.ref}].attributes[{attr_actual}] = {actual}",
                    f"Component[{comp.ref}].attributes[{attr_coef}] = {coef}",
                    f"温度降额系数 = {temp_coef}（temperature_derating_curve）",
                    f"阈值 = {rated} × {coef} × {temp_coef} = {threshold:.3f}W",
                ],
            ))

    return hits

# ---------------------------------------------------------------------------
# IO_001：IO 引脚悬空检查（汇总输出）
# ---------------------------------------------------------------------------
def check_io_floating(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑：
      遍历所有元件，找出含 pin_type == io 引脚的元件；
      对每个此类元件，统计其未被任何 Net.connected_pins 引用的 IO 引脚；
      若存在悬空 IO，输出一条汇总命中（每个元件一条）。

    rule_params:
      io_pin_types: List[str]     可选，默认 ["io"]；当前仅支持 "io"

    输出（汇总，per-component）：
      每个含悬空 IO 的元件 → 一条 RuleHitItem。
      例：U1 有 32 个 IO 引脚悬空
    """
    hits: List[RuleHitItem] = []

    # 收集所有被网络引用的引脚
    referenced_pins: set = set()
    for n in schematic_ir.nets:
        for pin_ref in n.connected_pins:
            referenced_pins.add(pin_ref)

    for comp in schematic_ir.components:
        if not _has_io_pins(comp):
            continue

        io_pins = _io_pins(comp)
        floating = [
            p for p in io_pins
            if f"{comp.ref}.{p.pin_id}" not in referenced_pins
        ]
        if not floating:
            continue

        pin_ids = [p.pin_id for p in floating]
        hits.append(RuleHitItem(
            component_refs=[comp.ref],
            net_refs=[],
            message=f"{comp.ref} 有 {len(floating)} 个 IO 引脚悬空",
            evidence_ir_refs=[
                f"{comp.ref} 的 IO 引脚（pin_type=io）共 {len(io_pins)} 个",
                f"其中 {len(floating)} 个未出现在 ir.nets[*].connected_pins",
                f"悬空引脚：{', '.join(pin_ids)}",
            ],
        ))

    return hits

# ---------------------------------------------------------------------------
# IO_002：IO 方向冲突检查
# ---------------------------------------------------------------------------
def check_io_direction(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（"输出冲突"检查，V1.3 扩展 BIDIRECTIONAL 分支）：
      对每个含 IO 引脚的网络：
        - 统计"潜在输出引脚"数；
        - 若 ≥ 2 → 命中"输出冲突"。

    "潜在输出引脚"的判定受 rule_params.ignore_bidirectional 控制：
      - true（默认）：只统计 direction == OUTPUT；
      - false：统计 direction == OUTPUT + BIDIRECTIONAL。

    INPUT 引脚永不参与判定（不驱动，不会与其他引脚形成冲突）。

    rule_params:
      ignore_bidirectional: bool   可选，默认 true；
                                   true = 忽略 BIDIRECTIONAL（普通 GPIO/UART 场景）；
                                   false = 将 BIDIRECTIONAL 视为潜在输出（I2C/SPI 双向总线场景）。

    输出（逐条，per-net）：
      每个冲突网络 → 一条 RuleHitItem。

    V1.3 边界澄清（3 条不变式）：
      1. 2 OUTPUT：永远算冲突；
      2. INPUT：永远不算冲突（不驱动）；
      3. BIDIRECTIONAL：按 ignore_bidirectional 参数走。
         当 false 时：
           - 1 OUTPUT + 1 BIDIRECTIONAL → 算冲突；
           - 2 BIDIRECTIONAL → 算冲突。
         当 true 时：上述两种均不算。
    """
    hits: List[RuleHitItem] = []

    ignore_bidirectional = bool(rule_params.get("ignore_bidirectional", True))

    # 建 ref.pin_id → Pin 的索引
    pin_map: Dict[str, Any] = {}
    for comp in schematic_ir.components:
        for p in comp.pins:
            pin_map[f"{comp.ref}.{p.pin_id}"] = p

    for net in schematic_ir.nets:
        net_pins = [pin_map.get(pref) for pref in net.connected_pins]
        net_pins = [p for p in net_pins if p is not None]

        # 仅关心含 IO 引脚的网络
        if not any(p.pin_type == PinType.IO for p in net_pins):
            continue

        # ⚠️ V1.3 修改：ignore_bidirectional=false 时，把 BIDIRECTIONAL 也纳入"潜在输出"
        # 判定语义：
        #   - true（默认）：只统计 OUTPUT 引脚，≥2 即冲突
        #   - false：统计 OUTPUT + BIDIRECTIONAL，≥2 即冲突
        # 理由：双向总线（I2C/SPI）上多个 BIDIRECTIONAL 主控同网 = 潜在总线冲突
        # [原逻辑 - 保留注释，便于对照回滚]
        # outputs = [
        #     p for p in net_pins
        #     if p.direction == PinDirection.OUTPUT
        # ]
        outputs = [
            p for p in net_pins
            if p.direction == PinDirection.OUTPUT
            or (not ignore_bidirectional and p.direction == PinDirection.BIDIRECTIONAL)
        ]
        if len(outputs) >= 2:
            # 命中：组装 RuleHitItem
            involved_refs = sorted({
                pref.split(".")[0] for pref in net.connected_pins
            })
            hits.append(RuleHitItem(
                component_refs=involved_refs,
                net_refs=[net.net_name],
                message=(
                    f"网络 {net.net_name} 上有 {len(outputs)} 个输出引脚相连，"
                    f"可能存在输出冲突"
                ),
                evidence_ir_refs=[
                    f"Net[{net.net_name}].connected_pins = {net.connected_pins}",
                    f"该网络 OUTPUT 引脚数 = {len(outputs)}（≥2）",
                ],
            ))

    return hits

# ---------------------------------------------------------------------------
# CLOCK_001：晶振负载电容配置检查
# ---------------------------------------------------------------------------
def check_crystal_load_cap(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（聚焦"电容值不匹配"，数量不足留给 CLOCK_002）：
      对每个晶振元件（lib_name ∈ crystal_lib_names）：
        1. 找到晶振所在的网络（如 XTAL_OSC）
        2. 找到该网络上的负载电容（lib_name 以 CAP 开头）
        3. 对每个负载电容，解析其 value（如 "22pF"），
           判定是否在 expected_load_cap_pf ± tolerance_pf 范围内
        4. 任一电容超差 → 命中

    Sprint1 决策依据（对齐 Sprint1 融合版 §3.2 / §3.4）：
      - 与 CLOCK_002 明确分工：
          * CLOCK_001（本函数）：聚焦"电容值不匹配"（数量已够，但值错）
          * CLOCK_002：聚焦"电路完整性"（数量不足或缺失）
      - 不检查"两个负载电容值是否对称"：
          * 有些晶振电路故意使用不对称负载电容（如为补偿寄生电容）
          * 强制对称可能误报
          * 只查"每个电容值是否在 expected ± tolerance 内"
      - case003 采用 C1=10pF（偏小）作为典型缺陷，而非 C1=47pF：
          * 负载电容偏小会导致起振困难（增益不足）
          * 是实践中最常见的晶振电路错误
      - 电容值解析：正则提取数字部分，支持 pF / nF / μF 单位
          * IR 的 Component.value 是字符串（如 "22pF"），需要解析
          * 不做单位转换（只支持 pF），因为晶振负载电容几乎都是 pF 级

    rule_params:
      crystal_lib_names: List[str]        晶振 lib_name 前缀，默认 ["CRYSTAL", "XTAL"]
      expected_load_cap_pf: float         期望负载电容值（pF），默认 22
      tolerance_pf: float                 容差（pF），默认 5
      min_load_caps: int                  最少负载电容数量，默认 2（用于触发前提）
      ground_net_names: List[str]         地网络名，默认 ["GND", "VSS"]
    """
    hits: List[RuleHitItem] = []

    crystal_lib_names: List[str] = rule_params.get("crystal_lib_names", ["CRYSTAL", "XTAL"])
    expected_pf: float = float(rule_params.get("expected_load_cap_pf", 22))
    tolerance_pf: float = float(rule_params.get("tolerance_pf", 5))
    min_load_caps: int = int(rule_params.get("min_load_caps", 2))
    ground_net_names: List[str] = rule_params.get("ground_net_names", ["GND", "VSS"])

    nets = _net_map(schematic_ir)

    for crystal in schematic_ir.components:
        if not any(crystal.lib_name.startswith(p) for p in crystal_lib_names):
            continue

        # 找到晶振所在网络（晶振的任一脚落在的 net）
        crystal_nets: List[str] = []
        for net in schematic_ir.nets:
            for pin_ref in net.connected_pins:
                ref, _, _ = pin_ref.partition(".")
                if ref == crystal.ref:
                    crystal_nets.append(net.net_name)
                    break

        if not crystal_nets:
            # 晶振未连接任何网络 → 完整性由 CLOCK_002 负责，跳过
            continue

        # 检查每个晶振网络上的负载电容
        for net_name in crystal_nets:
            net = nets.get(net_name)
            if net is None:
                continue

            # 收集该网络上的电容元件（去重）
            cap_refs = set()
            for pin_ref in net.connected_pins:
                ref, _, _ = pin_ref.partition(".")
                comp = _find_component(schematic_ir, ref)
                if comp and _is_cap(comp.lib_name):
                    cap_refs.add(ref)

            # 数量不足 → 由 CLOCK_002 负责，本规则跳过
            # 这样两条规则分工清晰：CLOCK_001 = 值错；CLOCK_002 = 缺项
            if len(cap_refs) < min_load_caps:
                continue

            # 检查每个电容值
            for cap_ref in cap_refs:
                cap_comp = _find_component(schematic_ir, cap_ref)
                if cap_comp is None or cap_comp.value is None:
                    continue

                # 解析电容值（如 "22pF" → 22.0）
                parsed_pf = _parse_capacitance_pf(cap_comp.value)
                if parsed_pf is None:
                    # 单位不兼容/解析失败 → 跳过（占位：后续可改 NEED_EXPERT_REVIEW）
                    continue

                if abs(parsed_pf - expected_pf) > tolerance_pf:
                    hits.append(RuleHitItem(
                        component_refs=[cap_ref, crystal.ref],
                        net_refs=[net_name],
                        message=(
                            f"晶振 {crystal.ref} 的负载电容 {cap_ref} = {cap_comp.value}，"
                            f"超出期望值 {expected_pf}pF ± {tolerance_pf}pF 容差；"
                            f"负载电容不匹配会导致起振困难或频率偏移"
                        ),
                        evidence_ir_refs=[
                            f"Component[{crystal.ref}].value = {crystal.value}",
                            f"Component[{cap_ref}].value = {cap_comp.value}",
                            f"Net[{net_name}].connected_pins 包含 {cap_ref}",
                            f"期望 {expected_pf}pF ± {tolerance_pf}pF，实际 {parsed_pf}pF",
                        ],
                    ))

    return hits

def _parse_capacitance_pf(value_str: str) -> Optional[float]:
    """
    解析电容值字符串为 pF 数值。

    支持单位：pF（皮法）、nF（纳法）、μF / uF（微法）
    不支持单位 → 返回 None

    示例：
      "22pF"  → 22.0
      "0.1μF" → 100000.0
      "1nF"   → 1000.0
      "abc"   → None
    """
    if not value_str:
        return None

    # 提取数字部分
    import re
    m = re.match(r"^\s*([\d.]+)\s*(pF|nF|μF|uF|PF|NF|UF)\s*$", value_str.strip(), re.IGNORECASE)
    if not m:
        return None

    try:
        num = float(m.group(1))
    except ValueError:
        return None

    unit = m.group(2).lower().replace("μ", "u")
    if unit == "pf":
        return num
    elif unit == "nf":
        return num * 1000.0
    elif unit == "uf":
        return num * 1_000_000.0
    return None

# ---------------------------------------------------------------------------
# CLOCK_002：晶振电路完整性检查
# ---------------------------------------------------------------------------
def check_crystal_circuit_complete(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（Sprint1，与 CLOCK_001 分工互补）：
      对每个晶振元件（lib_name ∈ crystal_lib_names）：
        1. 找到晶振所在网络（如 XTAL_OSC）
        2. 检查 MCU 时钟引脚（pin_type == clock）是否也连接到此网络
        3. 检查该网络上的负载电容数量是否 >= required_load_caps
        4. 任一条件不满足 → 命中

    Sprint1 决策依据（对齐 Sprint 1 融合版 §3.2 / §3.4）：
      - 【与 CLOCK_001 的分工】
          * CLOCK_001（值不匹配）：负载电容数量足够，但某个电容值偏离期望
          * CLOCK_002（完整性/缺项）：负载电容数量不足，或 MCU 时钟引脚未连接到晶振网络
          * 两条规则互补不重叠：
              - CLOCK_001 命中时，CLOCK_002 不应命中（反之亦然）
              - 通过"数量充足性检查"做逻辑隔离
      - 【检查三要素】
          * 要素 1：晶振本体存在（遍历 components 时已保证）
          * 要素 2：MCU 时钟引脚连到同一网络（pin_type == clock）
          * 要素 3：负载电容数量 >= required_load_caps（默认 2）
      - 【通用化】
          * 不绑定具体 MCU 型号，用 pin_type == clock 通用判定
          * 支持任意 MCU / PS：8051 / AVR / STM32 均可
      - 【每晶振一条 hit（逐元件）】
          * 与 CLOCK_001（每电容一条）不同：完整性问题是"整个电路缺项"
          * 一条 hit 汇总该晶振的所有缺失要素

    rule_params:
      crystal_lib_names: List[str]        晶振 lib_name 前缀，默认 ["CRYSTAL", "XTAL"]
      mcu_clock_pin_types: List[str]      MCU 时钟引脚类型，默认 ["clock"]
      required_load_caps: int             最少负载电容数量，默认 2
      ground_net_names: List[str]         地网络名，默认 ["GND", "VSS"]
    """
    hits: List[RuleHitItem] = []

    crystal_lib_names: List[str] = rule_params.get("crystal_lib_names", ["CRYSTAL", "XTAL"])
    required_load_caps: int = int(rule_params.get("required_load_caps", 2))

    nets = _net_map(schematic_ir)

    for crystal in schematic_ir.components:
        if not any(crystal.lib_name.startswith(p) for p in crystal_lib_names):
            continue

        # 找到晶振所在的网络（可能跨多个 net，取全部）
        crystal_nets: set = set()
        for net in schematic_ir.nets:
            for pin_ref in net.connected_pins:
                ref, _, _ = pin_ref.partition(".")
                if ref == crystal.ref:
                    crystal_nets.add(net.net_name)
                    break

        # 场景 1：晶振未连接任何网络 → 命中（最严重的缺项）
        if not crystal_nets:
            hits.append(RuleHitItem(
                component_refs=[crystal.ref],
                net_refs=[],
                message=(
                    f"晶振 {crystal.ref}（value={crystal.value}）未连接到任何网络；"
                    f"晶振电路不完整，MCU 无法获得时钟"
                ),
                evidence_ir_refs=[
                    f"Component[{crystal.ref}].lib_name = {crystal.lib_name}",
                    f"Component[{crystal.ref}] 的引脚未出现在任何 Net.connected_pins 中",
                ],
            ))
            continue

        # 场景 2：晶振已连接，逐个网络检查完整性
        missing_reasons: List[str] = []
        missing_evidences: List[str] = []

        for net_name in crystal_nets:
            net = nets.get(net_name)
            if net is None:
                continue

            # 要素 A：MCU 时钟引脚是否连接到此网络
            has_mcu_clock_pin = False
            for pin_ref in net.connected_pins:
                ref, _, pin_id = pin_ref.partition(".")
                comp = _find_component(schematic_ir, ref)
                if comp is None:
                    continue
                for pin in comp.pins:
                    if pin.pin_id == pin_id and pin.pin_type == PinType.CLOCK:
                        # 排除晶振自身的时钟引脚（晶振的 pin_type 也是 clock）
                        if comp.ref == crystal.ref:
                            continue
                        has_mcu_clock_pin = True
                        break
                if has_mcu_clock_pin:
                    break

            if not has_mcu_clock_pin:
                missing_reasons.append(
                    f"网络 {net_name} 上没有 MCU 时钟引脚（pin_type=clock）"
                )
                missing_evidences.append(
                    f"Net[{net_name}].connected_pins 中无 pin_type=clock 的非晶振元件引脚"
                )

            # 要素 B：负载电容数量
            cap_refs = set()
            for pin_ref in net.connected_pins:
                ref, _, _ = pin_ref.partition(".")
                comp = _find_component(schematic_ir, ref)
                if comp and _is_cap(comp.lib_name):
                    cap_refs.add(ref)

            if len(cap_refs) < required_load_caps:
                missing_reasons.append(
                    f"网络 {net_name} 上的负载电容数 {len(cap_refs)} < 要求 {required_load_caps}"
                )
                missing_evidences.append(
                    f"Net[{net_name}].connected_pins 中电容元件集合 = {sorted(cap_refs)}"
                )

        # 任一要素缺失 → 一条 hit 汇总该晶振的所有问题
        if missing_reasons:
            hits.append(RuleHitItem(
                component_refs=[crystal.ref],
                net_refs=sorted(crystal_nets),
                message=(
                    f"晶振 {crystal.ref} 电路不完整：{'; '.join(missing_reasons)}"
                ),
                evidence_ir_refs=[
                    f"Component[{crystal.ref}].lib_name = {crystal.lib_name}",
                    f"Component[{crystal.ref}] 连接的网络 = {sorted(crystal_nets)}",
                ] + missing_evidences,
            ))

    return hits


# ---------------------------------------------------------------------------
# IO_003：MCU IO 与外部器件输出冲突检查
# ---------------------------------------------------------------------------
def check_io_output_conflict(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（Sprint1 决策 A/A）：
      对每个网络：
        1. 找出网络上的"MCU 输出引脚"（属于 MCU 元件 + direction ∈ {output, bidirectional}）
        2. 找出网络上的"外部器件输出引脚"（不属于 MCU + direction == output）
        3. 若两者都 >= 1 → 命中（推挽对推挽冲突）

    Sprint1 决策依据（对齐 Sprint 1 融合版 §3.2 / §3.4）：
      - 【与 IO_002 的分工】
          * IO_002（check_io_direction）：检查 IO 之间的方向冲突（多个 OUTPUT 同网）
          * IO_003（本函数）：检查 MCU IO 与外部器件输出的冲突（推挽对推挽）
          * 两者互补：
              - 场景 1：U1.P1.0（output）+ U1.P1.1（output）同网 → IO_002 命中
              - 场景 2：U1.P1.0（MCU output）+ U2.OUT（外设 output）同网 → IO_003 命中
      - 【决策 1】判定 MCU 的方式：方案 A（lib_name 前缀白名单）
          * 理由：lib_name 是 IR 的稳定字段，规则引擎核心匹配依据
          * 不依赖 pin_type 特征（有些 MCU 可能缺 reset/clock 引脚）
          * 不依赖显式传 ref 列表（可维护性差）
          * 白名单可扩展（mcu_lib_patterns 参数）
      - 【决策 2】判定"外部器件输出"：方案 A（direction=output 且非 MCU）
          * 理由：output 引脚天然是"驱动方"；external 隐含"非 MCU"
          * 结合 lib_name 白名单排除 MCU 后，剩下的 output 引脚即为"外部器件输出"
      - 【决策 3】MCU IO 视为潜在输出：direction ∈ {output, bidirectional}
          * 理由：MCU 的 GPIO 多为 bidirectional，配置为推挽输出时会与外部器件冲突
          * 这与 IO_002 的 ignore_bidirectional 参数语义一致
      - 【每冲突网络一条 hit】
          * 与 IO_002 的 per-net 粒度一致
          * 便于专家针对该网络反馈

    rule_params:
      mcu_lib_patterns: List[str]        MCU lib_name 前缀白名单，默认常见 MCU
      exclude_net_names: List[str]       排除的网络名（电源/地），默认 ["VCC", "GND", "VSS"]
      output_pin_types: List[str]        关注的引脚类型，默认 ["io", "analog"]
    """
    hits: List[RuleHitItem] = []

    # 决策 1：MCU lib_name 前缀白名单
    mcu_patterns: List[str] = rule_params.get(
        "mcu_lib_patterns",
        ["STC89C5", "AT89C5", "MCU_", "PIC", "AVR", "STM32", "ESP32", "ESP8266"],
    )
    exclude_net_names: List[str] = rule_params.get(
        "exclude_net_names", ["VCC", "GND", "VSS"]
    )

    def _is_mcu(comp) -> bool:
        """判断元件是否为 MCU（lib_name 前缀匹配白名单）。"""
        lib = (comp.lib_name or "").upper()
        return any(lib.startswith(p.upper()) for p in mcu_patterns)

    # 建 ref.pin_id → (comp, pin) 索引
    pin_index: Dict[str, Any] = {}
    for comp in schematic_ir.components:
        for pin in comp.pins:
            pin_index[f"{comp.ref}.{pin.pin_id}"] = (comp, pin)

    for net in schematic_ir.nets:
        # 跳过电源/地网络
        if net.net_name in exclude_net_names:
            continue
        if net.is_power or net.is_ground:
            continue

        # 决策 2 + 决策 3：
        # - MCU 输出引脚：属于 MCU + direction ∈ {output, bidirectional}
        # - 外部器件输出引脚：不属于 MCU + direction == output
        mcu_outputs: List[str] = []
        external_outputs: List[str] = []

        for pin_ref in net.connected_pins:
            entry = pin_index.get(pin_ref)
            if entry is None:
                continue
            comp, pin = entry

            # 只关心 IO / analog 类型引脚（数字/模拟 IO）
            if pin.pin_type not in (PinType.IO, PinType.ANALOG):
                continue

            if _is_mcu(comp):
                if pin.direction in (PinDirection.OUTPUT, PinDirection.BIDIRECTIONAL):
                    mcu_outputs.append(pin_ref)
            else:
                if pin.direction == PinDirection.OUTPUT:
                    external_outputs.append(pin_ref)

        # 两者都 >= 1 → 命中
        if mcu_outputs and external_outputs:
            involved_refs = sorted({
                pref.split(".")[0] for pref in net.connected_pins
            })
            hits.append(RuleHitItem(
                component_refs=involved_refs,
                net_refs=[net.net_name],
                message=(
                    f"网络 {net.net_name} 上存在 MCU 输出引脚 "
                    f"{mcu_outputs} 与外部器件输出引脚 {external_outputs} 直连；"
                    f"推挽对推挽可能造成短路或大电流"
                ),
                evidence_ir_refs=[
                    f"Net[{net.net_name}].connected_pins = {net.connected_pins}",
                    f"MCU 输出引脚 = {mcu_outputs}",
                    f"外部器件输出引脚 = {external_outputs}",
                ],
            ))

    return hits

# ---------------------------------------------------------------------------
# CTRL_001：使能信号连接检查
# ---------------------------------------------------------------------------
def check_enable_pin_connection(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（Sprint1 决策 A/A）：
      对每个元件，遍历其引脚，找出 pin.name 匹配 enable_pin_name_patterns
      （默认 EN/OE/CE/CS/WE）的引脚；
      检查该引脚是否出现在任何 Net.connected_pins 里；
      未出现 → 悬空 → 命中。

    Sprint1 决策依据（对齐 Sprint 1 融合版 §3.2 / §3.4）：
      - 【决策 1】判定粒度：方案 A（每个悬空引脚一条 hit）
          * 理由：定位更精确，便于专家逐条反馈（feedback_item 可挂到具体 defect）
          * 对比方案 B（每元件一条汇总 hit）：定位粗糙，专家无法针对某引脚单独反馈
          * 与 IO_001（汇总式）形成互补：
              - IO_001：IO 引脚悬空，汇总式（low 级提示，量大没必要逐条）
              - CTRL_001：使能脚悬空，逐条式（high 级缺陷，需精确定位）
      - 【决策 2】是否区分上拉/下拉：方案 A（只查"是否悬空"，不查方向）
          * 理由：Sprint1 聚焦"悬空"这一确定性缺陷；上拉/下拉方向判断
            需要更强的语义（如 Datasheet 中 EN 的有效电平），暂不在 Sprint1 范围
          * 参数 pull_required 预留：未来可扩展为"要求上拉/下拉"
          * 悬空时器件行为不确定（CMOS 输入浮空可能振荡、功耗增大），是确定性缺陷
      - 【决策 3】引脚名匹配：支持大小写不敏感 + 前缀匹配
          * "EN" 匹配 "EN" / "ENABLE" / "EN_1" / "~EN"
          * "CE" 匹配 "CE" / "ChipEnable" / "Chip_Enable"
          * 匹配范围覆盖常见命名风格

    rule_params:
      enable_pin_name_patterns: List[str]   引脚名前缀，默认 ["EN", "OE", "CE", "CS", "WE"]
      floating_check: bool                  是否检查悬空，默认 True
      pull_required: bool                   是否要求上拉/下拉，默认 True
                                            （Sprint1 暂不使用，预留）
    """
    hits: List[RuleHitItem] = []

    patterns: List[str] = rule_params.get(
        "enable_pin_name_patterns", ["EN", "OE", "CE", "CS", "WE"]
    )
    floating_check: bool = bool(rule_params.get("floating_check", True))
    # pull_required: bool = bool(rule_params.get("pull_required", True))  # Sprint1 预留

    if not floating_check:
        return hits

    # 构建"被引用引脚"集合（与 IO_001 的建索引方式一致）
    referenced_pins: set = set()
    for n in schematic_ir.nets:
        for pin_ref in n.connected_pins:
            referenced_pins.add(pin_ref)

    # 大小写不敏感的前缀匹配
    upper_patterns = [p.upper() for p in patterns]

    def _matches_enable_pattern(pin_name: str) -> bool:
        """判断引脚名是否匹配任一使能脚模式（大小写不敏感，前缀匹配）。"""
        if not pin_name:
            return False
        # 去除常见前缀符号（如 ~EN / /EN 表示低有效）
        normalized = pin_name.lstrip("~/!").upper()
        return any(normalized.startswith(pat) for pat in upper_patterns)

    for comp in schematic_ir.components:
        for pin in comp.pins:
            if not _matches_enable_pattern(pin.name):
                continue

            pin_ref = f"{comp.ref}.{pin.pin_id}"
            if pin_ref in referenced_pins:
                # 已连接 → 不命中
                continue

            # 悬空 → 命中（方案 A：每引脚一条 hit）
            hits.append(RuleHitItem(
                component_refs=[comp.ref],
                net_refs=[],
                message=(
                    f"{comp.ref} 的使能脚 {pin.name}（pin_id={pin.pin_id}）悬空，"
                    f"器件使能状态不确定"
                ),
                evidence_ir_refs=[
                    f"Component[{comp.ref}].lib_name = {comp.lib_name}",
                    f"Component[{comp.ref}].pins 中存在使能脚 pin_id={pin.pin_id}, name={pin.name}",
                    f"该引脚 {pin_ref} 未出现在任何 Net.connected_pins 中",
                ],
            ))

    return hits

# ---------------------------------------------------------------------------
# CTRL_002：上电复位时序检查
# ---------------------------------------------------------------------------
def check_power_on_sequence(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（Sprint1，与 POWER_002 分工互补）：
      检查复位 RC 时间常数 τ = R × C 是否 ≥ min_reset_high_ms。
      - 找到复位网络上的电阻 R 和电容 C
      - 解析 R 值（如 "10kΩ" → 10000 Ω）
      - 解析 C 值（如 "10μF" → 10e-6 F）
      - τ (ms) = R × C × 1000
      - τ < min_reset_high_ms → 命中

    Sprint1 决策依据（对齐 Sprint 1 融合版 §3.2 / §3.4）：
      - 【与 POWER_002 的分工】
          * POWER_002（拓扑/路径）：RST 网络是否有到 VCC 的元件级路径
          * CTRL_002（时序/数值）：R × C 的时间常数是否满足最小脉宽
          * 两条规则互补：拓扑对但参数错 → 只 CTRL_002 命中
      - 【通用化】
          * 不绑定具体 MCU，min_reset_high_ms 通过 rule_params 传入
          * 支持任意 R/C 值解析（kΩ/MΩ/Ω，pF/nF/μF）
      - 【每复位网络一条 hit】
          * 定位：R 和 C 的 ref
          * 缺失 R 或 C → 跳过（由 POWER_002 负责拓扑检查）
      - 【单位不兼容】
          * 解析失败 → 跳过（占位：后续可改 NEED_EXPERT_REVIEW）

    rule_params:
      reset_net: str                复位网络名，默认 "RST"
      vcc_net: str                  电源网络名，默认 "VCC"
      min_reset_high_ms: float      最小复位脉宽（ms），默认 2
      reset_rc_detection: bool      是否启用 RC 检测，默认 True
    """
    hits: List[RuleHitItem] = []

    reset_net_name: str = rule_params.get("reset_net", "RST")
    vcc_net_name: str = rule_params.get("vcc_net", "VCC")
    min_ms: float = float(rule_params.get("min_reset_high_ms", 2))
    rc_detection: bool = bool(rule_params.get("reset_rc_detection", True))

    if not rc_detection:
        return hits

    nets = _net_map(schematic_ir)
    rst_net = nets.get(reset_net_name)
    if rst_net is None:
        return hits

    # 从 RST 网络上找到 R 和 C
    rst_refs = sorted({p.split(".")[0] for p in rst_net.connected_pins})

    r_refs: List[str] = []
    c_refs: List[str] = []
    for ref in rst_refs:
        comp = _find_component(schematic_ir, ref)
        if comp is None:
            continue
        # 排除 MCU（pin_type=reset 的元件）
        if any(p.pin_type == PinType.RESET for p in comp.pins):
            continue
        # 识别 R / C
        lib_upper = comp.lib_name.upper()
        if lib_upper.startswith("RES"):
            r_refs.append(ref)
        elif lib_upper.startswith("CAP"):
            c_refs.append(ref)

    # 需要至少 1 个 R 和 1 个 C 才能计算 RC
    if not r_refs or not c_refs:
        return hits

    # 取第一个 R 和第一个 C（简化：不处理多 R 多 C 并联）
    r_comp = _find_component(schematic_ir, r_refs[0])
    c_comp = _find_component(schematic_ir, c_refs[0])

    r_ohm = _parse_resistance_ohm(r_comp.value) if r_comp.value else None
    c_farad = _parse_capacitance_farad(c_comp.value) if c_comp.value else None

    # 解析失败 → 跳过（占位：后续可改 NEED_EXPERT_REVIEW）
    if r_ohm is None or c_farad is None:
        return hits

    # τ (ms) = R (Ω) × C (F) × 1000
    tau_ms = r_ohm * c_farad * 1000.0

    if tau_ms < min_ms:
        hits.append(RuleHitItem(
            component_refs=[c_comp.ref, r_comp.ref],
            net_refs=[reset_net_name],
            message=(
                f"复位 RC 时间常数不足：R={r_comp.ref}({r_comp.value}) × "
                f"C={c_comp.ref}({c_comp.value}) → τ={tau_ms:.3f}ms "
                f"< 要求 {min_ms}ms"
            ),
            evidence_ir_refs=[
                f"Component[{r_comp.ref}].value = {r_comp.value} → {r_ohm}Ω",
                f"Component[{c_comp.ref}].value = {c_comp.value} → {c_farad}F",
                f"τ = R × C = {tau_ms:.3f}ms；阈值 min_reset_high_ms = {min_ms}ms",
            ],
        ))

    return hits


def _parse_resistance_ohm(value_str: str) -> Optional[float]:
    """解析电阻值字符串为 Ω。"""
    if not value_str:
        return None
    import re
    s = value_str.strip().replace(" ", "")
    # 支持 "10kΩ" / "10KΩ" / "10k" / "10000" 等
    m = re.match(r"^([\d.]+)\s*([kKmM]?)\s*(?:Ω|ohm|OHM)?$", s)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    prefix = m.group(2).lower()
    if prefix == "k":
        return num * 1000.0
    elif prefix == "m":
        return num * 1_000_000.0
    else:
        return num


def _parse_capacitance_farad(value_str: str) -> Optional[float]:
    """
    解析电容值字符串为法拉（F）。

    支持单位：pF / nF / μF / uF
    """
    pf = _parse_capacitance_pf(value_str)
    if pf is None:
        return None
    return pf * 1e-12

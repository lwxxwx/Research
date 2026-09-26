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

from typing import Any, Dict, List

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
    判定逻辑：
      对 rule_params.power_net_names 指定的每个电源网络，
      检查是否存在一个电容元件，其一端接该电源网络、另一端接 GND。
      若不存在 → 命中。

    rule_params:
      power_net_names: List[str]           必填，目标电源网络名
      target_component_refs: List[str]     可选，命中时优先作为 component_refs 返回
      decouple_cap_value_expected: str     可选，仅用于 message 文案
      ground_net_names: List[str]          可选，默认 ["GND", "VSS"]
    """
    hits: List[RuleHitItem] = []

    power_net_names: List[str] = rule_params.get("power_net_names", [])
    target_refs: List[str] = rule_params.get("target_component_refs", [])
    expected_value: str = rule_params.get("decouple_cap_value_expected", "0.1μF")
    ground_net_names: List[str] = rule_params.get("ground_net_names", ["GND", "VSS"])

    nets = _net_map(schematic_ir)

    for net_name in power_net_names:
        net = nets.get(net_name)
        if net is None:
            continue

        involved_refs = sorted({p.split(".")[0] for p in net.connected_pins})

        has_decouple_cap = False
        for comp in schematic_ir.components:
            if not _is_cap(comp.lib_name):
                continue
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
                    f"（要求电容一端接 {net_name}、另一端接 GND）"
                ),
                evidence_ir_refs=[
                    f"Net[{net_name}].connected_pins = {net.connected_pins}",
                    f"该网络元件集合 {involved_refs} 中无任何电容跨接 {net_name}-GND",
                ],
            ))
    return hits


# ---------------------------------------------------------------------------
# POWER_002：高电平 RC 复位拓扑
# ---------------------------------------------------------------------------
def check_reset_rc_topology(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    判定逻辑（8051 高电平复位）：
      复位网络上必须存在一条通往 VCC 的元件级路径。
      判定方式：复位网络上某个非 MCU 元件，其引脚同时落在 RST 和 VCC 网络。
      若不存在 → 命中。

    rule_params:
      reset_net: str                必填，复位网络名（如 "RST"）
      vcc_net: str                  必填，电源网络名（如 "VCC"）
      gnd_net: str                  可选，地网络名（用于 message）
    """
    hits: List[RuleHitItem] = []

    reset_net_name: str = rule_params["reset_net"]
    vcc_net_name: str = rule_params["vcc_net"]
    gnd_net_name: str = rule_params.get("gnd_net", "GND")

    nets = _net_map(schematic_ir)
    rst_net = nets.get(reset_net_name)
    vcc_net = nets.get(vcc_net_name)
    if rst_net is None or vcc_net is None:
        return hits

    rst_refs = sorted({p.split(".")[0] for p in rst_net.connected_pins})

    # 判定：RST 网络上是否存在一个元件同时连到 RST 和 VCC
    has_vcc_path = False
    vcc_path_via: List[str] = []
    for ref in rst_refs:
        comp = _find_component(schematic_ir, ref)
        if comp is None:
            continue
        if comp.lib_name in _MCU_LIB_NAMES:
            continue
        pin_nets = set(_comp_pin_nets(schematic_ir, comp).values())
        if reset_net_name in pin_nets and vcc_net_name in pin_nets:
            has_vcc_path = True
            vcc_path_via.append(ref)

    if not has_vcc_path:
        # 定位锚点：RST 上第一个非 MCU 元件
        anchor_refs = [
            r for r in rst_refs
            if (c := _find_component(schematic_ir, r)) and c.lib_name not in _MCU_LIB_NAMES
        ]
        hits.append(RuleHitItem(
            component_refs=anchor_refs or rst_refs,
            net_refs=[reset_net_name, vcc_net_name, gnd_net_name],
            message=(
                f"复位网络 {reset_net_name} 没有通往 {vcc_net_name} 的电气路径；"
                f"高电平复位 MCU 无法产生上电复位脉冲"
            ),
            evidence_ir_refs=[
                f"Net[{reset_net_name}].connected_pins = {rst_net.connected_pins}",
                f"RST 网络元件集合 {rst_refs} 均未提供 {reset_net_name}-{vcc_net_name} 通路",
            ],
        ))
    return hits


# ---------------------------------------------------------------------------
# POWER_003：器件功率降额（Sprint0 预留）
# ---------------------------------------------------------------------------
def check_derating(
    schematic_ir: SchematicIRDocument,
    rule_params: Dict[str, Any],
) -> List[RuleHitItem]:
    """
    Sprint0 阶段保留接口，返回空列表。
    待 Sprint1 补：读取 Component.attributes 中的 v_rated / p_rated，
    与工作电压/功耗比较，超过降额阈值则命中。
    """
    return []


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
    判定逻辑（仅判"输出冲突"，Sprint0 决策）：
      对每个含 IO 引脚的网络：
        - 统计 direction == OUTPUT 的引脚数；
        - 若 ≥ 2 → 命中"输出冲突"。
      BIDIRECTIONAL 与 INPUT 不参与判定。

    rule_params:
      ignore_bidirectional: bool   可选，默认 true；true 时忽略 BIDIRECTIONAL

    输出（逐条，per-net）：
      每个冲突网络 → 一条 RuleHitItem。
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

        outputs = [
            p for p in net_pins
            if p.direction == PinDirection.OUTPUT
        ]
        if len(outputs) >= 2:
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
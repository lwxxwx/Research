"""app/rules/registry.py
内置 check 函数显式注册表。
"""
from typing import Callable, Dict

from app.rules.builtin.circuit_checks import (
    check_decoupling,
    check_derating,
    check_io_direction,
    check_io_floating,
    check_reset_rc_topology,
    # ★ Sprint1 新增
    check_crystal_load_cap,
    check_crystal_circuit_complete,
    check_io_output_conflict,
    check_enable_pin_connection,
    check_power_on_sequence,
)

BUILTIN_CHECK_REGISTRY: Dict[str, Callable] = {
    "circuit_checks.check_decoupling": check_decoupling,
    "circuit_checks.check_reset_rc_topology": check_reset_rc_topology,
    "circuit_checks.check_derating": check_derating,
    "circuit_checks.check_io_floating": check_io_floating,
    "circuit_checks.check_io_direction": check_io_direction,
    # ★ Sprint1 新增
    "circuit_checks.check_crystal_load_cap": check_crystal_load_cap,
    "circuit_checks.check_crystal_circuit_complete": check_crystal_circuit_complete,
    "circuit_checks.check_io_output_conflict": check_io_output_conflict,
    "circuit_checks.check_enable_pin_connection": check_enable_pin_connection,
    "circuit_checks.check_power_on_sequence": check_power_on_sequence,
}


def get_builtin_check_func(check_ref: str) -> Callable:
    """按 function 引用取内置 check 函数。未注册则抛 KeyError。"""
    if check_ref not in BUILTIN_CHECK_REGISTRY:
        raise KeyError(
            f"builtin check '{check_ref}' not found in registry. "
            f"registered keys: {sorted(BUILTIN_CHECK_REGISTRY.keys())}"
        )
    return BUILTIN_CHECK_REGISTRY[check_ref]
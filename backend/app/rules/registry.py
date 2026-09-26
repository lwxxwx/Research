"""
app/rules/registry.py
内置 check 函数显式注册表。
设计理由：显式白名单，防止规则 YAML 中 check_logic.function 注入任意代码。

迁移说明（v1.0.1）：
  原 power_checks 模块已合并至 circuit_checks；键名统一为
  "circuit_checks.check_xxx"，与 data/rules/*.yaml 中 function 字段保持一致。
"""
from typing import Callable, Dict

from app.rules.builtin.circuit_checks import (
    check_decoupling,
    check_reset_rc_topology,
    check_derating,
    check_io_floating,
    check_io_direction,
)

# key 与 YAML 中 check_logic.function 字段值一一对应
BUILTIN_CHECK_REGISTRY: Dict[str, Callable] = {
    "circuit_checks.check_decoupling": check_decoupling,
    "circuit_checks.check_reset_rc_topology": check_reset_rc_topology,
    "circuit_checks.check_derating": check_derating,
    "circuit_checks.check_io_floating": check_io_floating,
    "circuit_checks.check_io_direction": check_io_direction,
}


def get_builtin_check_func(check_ref: str) -> Callable:
    """按 function 引用取内置 check 函数。未注册则抛 KeyError。"""
    if check_ref not in BUILTIN_CHECK_REGISTRY:
        raise KeyError(
            f"builtin check '{check_ref}' not found in registry. "
            f"registered keys: {sorted(BUILTIN_CHECK_REGISTRY.keys())}"
        )
    return BUILTIN_CHECK_REGISTRY[check_ref]
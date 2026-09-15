"""
app/rules/registry.py
内置 check 函数显式注册表。
设计理由：显式白名单，防止规则 YAML 中 check_logic.function 注入任意代码。
"""
from typing import Callable, Dict

from app.rules.builtin.power_checks import (
    check_decoupling,
    check_reset_rc_topology,
    check_derating,
)

# key 与 YAML 中 check_logic.function 字段值一一对应
BUILTIN_CHECK_REGISTRY: Dict[str, Callable] = {
    "power_checks.check_decoupling": check_decoupling,
    "power_checks.check_reset_rc_topology": check_reset_rc_topology,
    "power_checks.check_derating": check_derating,
}


def get_builtin_check_func(check_ref: str) -> Callable:
    """按 function 引用取内置 check 函数。未注册则抛 KeyError。"""
    if check_ref not in BUILTIN_CHECK_REGISTRY:
        raise KeyError(
            f"builtin check '{check_ref}' not found in registry. "
            f"registered keys: {sorted(BUILTIN_CHECK_REGISTRY.keys())}"
        )
    return BUILTIN_CHECK_REGISTRY[check_ref]
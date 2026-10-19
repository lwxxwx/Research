"""
A/B 类 Case 命名与结构校验。

职责：
  - 校验目录名 = info.yaml::case_id
  - 校验 A 类命名规则（case<3位数字>）
  - 校验 B 类命名规则（case_b<2位数字>_<描述>_B）
  - 校验 case_class 与目录前缀一致
  - 校验 A/B 类必需文件齐全
  - 校验 B 类 evidence 三件套

与 test_rules.py 的分工：
  - test_rules.py：规则引擎行为（含 A/B 类规则命中）
  - 本文件：Case 目录命名与结构完整性

与 test_case_b_validation.py 的分工：
  - test_case_b_validation.py：B 类资产内容（ai_discovered / 权重 / 10 字段）
  - 本文件：A + B 类目录命名与结构

执行命令：
  docker compose -f infra/docker/docker-compose.yml -f infra/docker/docker-compose.dev.yml `
    exec backend uv run pytest tests/test_case_naming.py -v

⚠️ 本测试不依赖 PostgreSQL、不依赖规则引擎
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml


# ============================================================
# 路径常量
# ============================================================
def _find_project_root(start: pathlib.Path) -> pathlib.Path:
    """向上查找含 data/cases 的目录作为项目根。"""
    for p in [start, *start.parents]:
        if (p / "data" / "cases").is_dir():
            return p
    return start


PROJECT_ROOT: pathlib.Path = _find_project_root(pathlib.Path(__file__).resolve())
CASES_ROOT: pathlib.Path = PROJECT_ROOT / "data" / "cases"


# ============================================================
# A 类 Case 清单（6 个，命名：case<3位数字>）
# ============================================================
A_CASE_IDS: list[str] = [
    "case001",
    "case002",
    "case003",
    "case004",
    "case005",
    "case006",
]

# ============================================================
# B 类 Case 清单（4 个，命名：case_b<2位数字>_<描述>_B）
# ============================================================
B_CASE_IDS: list[str] = [
    "case_b01_buck_thermal_derating_B",
    "case_b02_reset_rc_B",
    "case_b03_io_direction_B",
    "case_b04_power_sequence_B",
]

ALL_CASE_IDS: list[str] = A_CASE_IDS + B_CASE_IDS


# ============================================================
# 命名规则正则
# ============================================================
A_CASE_PATTERN: re.Pattern = re.compile(r"^case\d{3}$")
B_CASE_PATTERN: re.Pattern = re.compile(r"^case_b\d{2}_[a-z0-9_]+_B$")


# ============================================================
# 工具函数
# ============================================================
def _load_yaml(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _skip_if_missing(case_id: str) -> pathlib.Path:
    """Case 目录不存在时 skip 当前测试。"""
    d = CASES_ROOT / case_id
    if not d.is_dir():
        pytest.skip(f"Case 未建成: {case_id}")
    return d


# ============================================================
# 1. 目录命名规则校验
# ============================================================
@pytest.mark.parametrize("case_id", A_CASE_IDS)
def test_a_case_naming_pattern(case_id):
    """A 类目录命名必须符合 `case<3位数字>`。"""
    assert A_CASE_PATTERN.match(case_id), (
        f"A 类目录名 {case_id} 不符合 `case<3位数字>` 规则"
    )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_b_case_naming_pattern(case_id):
    """B 类目录命名必须符合 `case_b<2位数字>_<描述>_B`。"""
    assert B_CASE_PATTERN.match(case_id), (
        f"B 类目录名 {case_id} 不符合 `case_b<2位数字>_<描述>_B` 规则"
    )


def test_all_cases_no_overlap():
    """A 类与 B 类目录名不能有重叠。"""
    overlap = set(A_CASE_IDS) & set(B_CASE_IDS)
    assert not overlap, f"A/B 类目录名重叠: {overlap}"


# ============================================================
# 2. 目录存在性
# ============================================================
@pytest.mark.parametrize("case_id", ALL_CASE_IDS)
def test_case_dir_exists(case_id):
    """所有 A/B 类目录必须存在。"""
    d = CASES_ROOT / case_id
    assert d.is_dir(), f"Case 目录不存在: {d}"


# ============================================================
# 3. 目录名 = info.yaml::case_id
# ============================================================
@pytest.mark.parametrize("case_id", ALL_CASE_IDS)
def test_dir_name_matches_info_case_id(case_id):
    """目录名必须等于 info.yaml::case_id。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")
    assert info.get("case_id") == case_id, (
        f"目录名 {case_id} 与 info.yaml::case_id={info.get('case_id')} 不一致"
    )


# ============================================================
# 4. case_class 与目录前缀一致
# ============================================================
@pytest.mark.parametrize("case_id", A_CASE_IDS)
def test_a_case_class(case_id):
    """A 类 info.yaml::case_class 必须为 A。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")
    assert info.get("case_class") == "A", (
        f"{case_id} case_class 应为 A，实际 {info.get('case_class')}"
    )


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_b_case_class(case_id):
    """B 类 info.yaml::case_class 必须为 B。"""
    d = _skip_if_missing(case_id)
    info = _load_yaml(d / "info.yaml")
    assert info.get("case_class") == "B", (
        f"{case_id} case_class 应为 B，实际 {info.get('case_class')}"
    )


# ============================================================
# 5. A 类必需文件
# ============================================================
A_REQUIRED_FILES = [
    "info.yaml",
    "schematic_ir.json",
    "expected_review.json",
    "evaluation.yaml",
]


@pytest.mark.parametrize("case_id", A_CASE_IDS)
@pytest.mark.parametrize("filename", A_REQUIRED_FILES)
def test_a_case_required_files(case_id, filename):
    """A 类必须包含: info.yaml / schematic_ir.json / expected_review.json / evaluation.yaml。"""
    d = _skip_if_missing(case_id)
    f = d / filename
    assert f.is_file(), f"{case_id} 缺 {filename}"


# ============================================================
# 6. B 类必需文件
# ============================================================
B_REQUIRED_FILES = [
    "info.yaml",
    "schematic_ir.json",
    "expected_review.json",
    "evaluation.yaml",
    "expert_reasoning.md",
]


@pytest.mark.parametrize("case_id", B_CASE_IDS)
@pytest.mark.parametrize("filename", B_REQUIRED_FILES)
def test_b_case_required_files(case_id, filename):
    """B 类必须包含: info.yaml / schematic_ir.json / expected_review.json / evaluation.yaml / expert_reasoning.md。"""
    d = _skip_if_missing(case_id)
    f = d / filename
    assert f.is_file(), f"{case_id} 缺 {filename}"


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_b_case_evidence_dir(case_id):
    """B 类必须有 evidence/ 目录。"""
    d = _skip_if_missing(case_id)
    evidence_dir = d / "evidence"
    assert evidence_dir.is_dir(), f"{case_id} 缺 evidence/ 目录"


@pytest.mark.parametrize("case_id", B_CASE_IDS)
@pytest.mark.parametrize("subdir", ["datasheet", "application_note", "reference_design"])
def test_b_case_evidence_subdirs(case_id, subdir):
    """B 类 evidence/ 必须有 datasheet / application_note / reference_design 三个子目录。"""
    d = _skip_if_missing(case_id)
    sub = d / "evidence" / subdir
    assert sub.is_dir(), f"{case_id} 缺 evidence/{subdir}/"


@pytest.mark.parametrize("case_id", B_CASE_IDS)
def test_b_case_evidence_not_empty(case_id):
    """B 类 evidence/ 下至少有一个 .md 文件。"""
    d = _skip_if_missing(case_id)
    evidence_dir = d / "evidence"
    md_files = list(evidence_dir.rglob("*.md"))
    assert len(md_files) >= 1, f"{case_id} evidence/ 下无 .md 文件"
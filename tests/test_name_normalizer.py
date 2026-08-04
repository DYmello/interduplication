from __future__ import annotations

from src.name_normalizer import (
    classify_serial,
    normalize_name_for_embedding,
    normalize_serial,
)


def test_serial_classification_handles_full_width_parentheses_and_numbers():
    assert classify_serial(" 六 ", "平台") == "platform"
    assert classify_serial("（二）", "系统") == "system"
    assert classify_serial(10.0, "软件") == "software"
    assert classify_serial("（ 2 ）", "模块") == "module"
    assert classify_serial(None, "分类标题") == "category"


def test_name_normalization_is_format_only():
    assert normalize_name_for_embedding("  元数据  完整性评测模块  ") == "元数据完整性评测模块"
    assert normalize_name_for_embedding("ABC  软件") == "abc 软件"
    assert normalize_serial(" （ １２ ） ") == "(12)"


def test_remove_module_suffix_is_opt_in():
    name = "元数据完整性评测模块"
    assert normalize_name_for_embedding(name) == name
    assert normalize_name_for_embedding(name, remove_module_suffix=True) == "元数据完整性评测"

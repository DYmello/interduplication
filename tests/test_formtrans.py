from __future__ import annotations

import pytest
from openpyxl import Workbook

from src.hierarchy_parser import HierarchyError, parse_hierarchy


def sample_worksheet():
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["序号", "应用系统名称"])
    worksheet.append(["一", "平台一"])
    worksheet.append(["（一）", "系统一"])
    worksheet.append([None, "分类标题"])
    worksheet.append([1, "软件一"])
    worksheet.append(["（1）", "模块一"])
    worksheet.append(["（2）", "模块二"])
    worksheet.append([2, "软件二"])
    return worksheet


def test_recognizes_hierarchy_and_excludes_category():
    result = parse_hierarchy(sample_worksheet(), group_id="A")
    assert result.counts == {"platform": 1, "system": 1, "software": 2, "module": 2}
    assert [item.title for item in result.excluded_categories] == ["分类标题"]
    assert result.excluded_categories[0].source_row == 4


def test_software_row_has_empty_module_fields():
    result = parse_hierarchy(sample_worksheet())
    software_rows = [item for item in result.records if not item.module_id]
    assert len(software_rows) == 2
    assert all(item.module_name == "" for item in software_rows)


def test_module_inherits_full_parent_path():
    result = parse_hierarchy(sample_worksheet())
    module = next(item for item in result.records if item.module_id == "M0001")
    assert (
        module.platform_id,
        module.platform_name,
        module.system_id,
        module.system_name,
        module.software_id,
        module.software_name,
    ) == ("P001", "平台一", "S001", "系统一", "SW001", "软件一")


def test_module_without_software_raises_in_strict_mode():
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["序号", "应用系统名称"])
    worksheet.append(["一", "平台一"])
    worksheet.append(["（一）", "系统一"])
    worksheet.append(["（1）", "孤立模块"])
    with pytest.raises(HierarchyError, match="缺少当前软件"):
        parse_hierarchy(worksheet, strict=True)


def test_same_input_produces_deterministic_ids():
    first = parse_hierarchy(sample_worksheet(), group_id="G001")
    second = parse_hierarchy(sample_worksheet(), group_id="G001")
    assert first.records == second.records
    assert first.excluded_categories == second.excluded_categories

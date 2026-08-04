from __future__ import annotations

import pytest

from detectduplication import (
    build_parser,
    parse_target_groups,
    validate_directional_groups,
)


def test_target_groups_are_trimmed_deduplicated_and_ordered():
    assert parse_target_groups(" B,C, B ,D ") == ["B", "C", "D"]
    args = build_parser().parse_args(
        [
            "--input_files",
            "input.xlsx",
            "--comparison_mode",
            "cross",
            "--anchor_group",
            "A",
            "--target_groups",
            "B,C,B,D",
        ]
    )
    assert args.target_groups == ["B", "C", "D"]


def test_target_groups_must_not_parse_to_empty():
    with pytest.raises(ValueError, match="target_groups 解析后不能为空"):
        validate_directional_groups("cross", "A", parse_target_groups(" , "), ["A", "B"])


def test_anchor_group_must_exist():
    with pytest.raises(ValueError, match="anchor_group 不存在.*X"):
        validate_directional_groups("cross", "X", ["B"], ["A", "B"])


def test_unknown_target_groups_are_listed():
    with pytest.raises(ValueError, match="X,Y"):
        validate_directional_groups("cross", "A", ["B", "X", "Y"], ["A", "B"])


def test_target_groups_cannot_contain_anchor_group():
    with pytest.raises(ValueError, match="不能包含 anchor_group"):
        validate_directional_groups("cross", "A", ["B", "A"], ["A", "B"])


@pytest.mark.parametrize(
    ("anchor_group", "target_groups"),
    [("A", None), (None, ["B"])],
)
def test_anchor_and_target_groups_must_be_provided_together(anchor_group, target_groups):
    with pytest.raises(ValueError, match="必须同时提供"):
        validate_directional_groups(
            "cross",
            anchor_group,
            target_groups,
            ["A", "B"],
        )


@pytest.mark.parametrize("comparison_mode", ["within", "all"])
def test_directional_parameters_only_apply_to_cross_mode(comparison_mode):
    with pytest.raises(ValueError, match="只适用于 cross 模式"):
        validate_directional_groups(
            comparison_mode,
            "A",
            ["B"],
            ["A", "B"],
        )


def test_omitting_directional_parameters_preserves_default_scope():
    assert validate_directional_groups("cross", None, None, ["A", "B", "C"]) == (
        None,
        None,
    )

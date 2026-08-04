from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

from openpyxl import load_workbook

from src.embedding_model import BGEEmbeddingModel
from src.excel_schema import INPUT_HEADERS, InputRecord
from src.name_normalizer import clean_display_name
from src.similarity_engine import (
    SimilarityEngine,
    extract_objects,
    sort_results,
)
from src.xlsx_writer import write_similarity_workbook


LOGGER = logging.getLogger("detectduplication")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 BGE-M3 dense embedding 检测软件和模块名称的文字语义相似性"
    )
    parser.add_argument("--input_files", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output_file", type=Path, default=Path("duplication_results.xlsx")
    )
    parser.add_argument("--comparison_mode", choices=["within", "cross", "all"], default="within")
    parser.add_argument(
        "--anchor_group",
        help="定向跨组比较的基准组编号；只适用于 cross 模式",
    )
    parser.add_argument(
        "--target_groups",
        type=parse_target_groups,
        help="定向跨组比较的目标组编号，多个组以逗号分隔；只适用于 cross 模式",
    )
    parser.add_argument("--levels", default="software,module")
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--include_unmatched", action="store_true")
    parser.add_argument("--remove_module_suffix", action="store_true")
    parser.add_argument("--model_name_or_path", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--use_fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GPU 默认启用；可用 --no-use_fp16 显式关闭",
    )
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_levels(value: str) -> list[str]:
    levels = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not levels or any(item not in {"software", "module"} for item in levels):
        raise ValueError("levels 只能由 software,module 组成")
    result: list[str] = []
    for item in levels:
        if item not in result:
            result.append(item)
    return result


def parse_target_groups(value: str | None) -> list[str] | None:
    """Parse a comma-separated group list, preserving first-seen order."""
    if value is None:
        return None
    result: list[str] = []
    for item in value.split(","):
        group_id = item.strip()
        if group_id and group_id not in result:
            result.append(group_id)
    return result


def validate_directional_groups(
    comparison_mode: str,
    anchor_group: str | None,
    target_groups: Sequence[str] | None,
    available_groups: Sequence[str],
) -> tuple[str | None, list[str] | None]:
    """Validate and normalize the optional directed cross-group scope."""
    directional = anchor_group is not None or target_groups is not None
    if not directional:
        return None, None
    if comparison_mode != "cross":
        raise ValueError("anchor_group 和 target_groups 只适用于 cross 模式")
    if (anchor_group is None) != (target_groups is None):
        rak^}�Kh��춻�q�^ucomparison_mode",
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

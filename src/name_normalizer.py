from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal


Level = Literal["platform", "system", "software", "module", "category", "unknown"]

CHINESE_NUMBER = r"[零〇一二三四五六七八九十百千万两]+"
PLATFORM_PATTERN = re.compile(rf"^{CHINESE_NUMBER}$")
SYSTEM_PATTERN = re.compile(rf"^\(({CHINESE_NUMBER})\)$")
SOFTWARE_PATTERN = re.compile(r"^\d+$")
MODULE_PATTERN = re.compile(r"^\((\d+)\)$")
WHITESPACE_PATTERN = re.compile(r"\s+")
CJK_SPACE_PATTERN = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")


def normalize_serial(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return WHITESPACE_PATTERN.sub("", text)


def clean_display_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    return WHITESPACE_PATTERN.sub(" ", text)


def classify_serial(value: Any, name: Any = None) -> Level:
    serial = normalize_serial(value)
    display_name = clean_display_name(name)
    if not serial:
        return "category" if display_name else "unknown"
    if PLATFORM_PATTERN.fullmatch(serial):
        return "platform"
    if SYSTEM_PATTERN.fullmatch(serial):
        return "system"
    if SOFTWARE_PATTERN.fullmatch(serial):
        return "software"
    if MODULE_PATTERN.fullmatch(serial):
        return "module"
    return "unknown"


def normalize_name_for_embedding(
    name: Any,
    *,
    remove_module_suffix: bool = False,
    remove_cjk_spaces: bool = True,
) -> str:
    text = unicodedata.normalize("NFKC", clean_display_name(name))
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if remove_cjk_spaces:
        text = CJK_SPACE_PATTERN.sub("", text)
    if remove_module_suffix and text.endswith("模块"):
        text = text[: -len("模块")].rstrip()
    return text.casefold()

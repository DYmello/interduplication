from __future__ import annotations

from dataclasses import dataclass


INPUT_HEADERS = [
    "组编号",
    "平台编号",
    "平台名称",
    "系统编号",
    "系统名称",
    "软件编号",
    "软件名称",
    "模块编号",
    "模块名称",
]

EXCLUDED_HEADERS = [
    "平台编号",
    "平台名称",
    "系统编号",
    "系统名称",
    "被排除的分类标题",
    "原始行号",
]

SIMILARITY_HEADERS = [
    "序号",
    "比较层级",
    "相似度排名",
    "BGE-M3余弦相似度",
    "相似等级",
    "A组编号",
    "A平台编号",
    "A平台名称",
    "A系统编号",
    "A系统名称",
    "A软件编号",
    "A软件名称",
    "A模块编号",
    "A模块名称",
    "B组编号",
    "B平台编号",
    "B平台名称",
    "B系统编号",
    "B系统名称",
    "B软件编号",
    "B软件名称",
    "B模块编号",
    "B模块名称",
]

FORMTRANS_CONFIG = [
    ("comparison_mode", "within", "默认在同一组内比较"),
    ("comparison_levels", "software,module", "仅执行软件同层、模块同层比较"),
    ("same_level_only", "true", "禁止软件与模块跨层比较"),
    ("similarity_threshold", 0.85, "初始工程阈值，正式使用前应人工校准"),
    ("top_k", 10, "每个对象最多保留的相似候选数量"),
    ("include_unmatched", "false", "是否输出无匹配对象"),
    ("remove_module_suffix", "false", "是否在计算时删除名称末尾的“模块”"),
]


@dataclass(frozen=True)
class InputRecord:
    group_id: str
    platform_id: str
    platform_name: str
    system_id: str
    system_name: str
    software_id: str
    software_name: str
    module_id: str = ""
    module_name: str = ""

    def as_row(self) -> list[object]:
        return [
            self.group_id,
            self.platform_id,
            self.platform_name,
            self.system_id,
            self.system_name,
            self.software_id,
            self.software_name,
            self.module_id,
            self.module_name,
        ]

    @classmethod
    def from_row(cls, values: list[object]) -> "InputRecord":
        return cls(*(str(value).strip() if value is not None else "" for value in values))


@dataclass(frozen=True)
class ExcludedCategory:
    platform_id: str
    platform_name: str
    system_id: str
    system_name: str
    title: str
    source_row: int

    def as_row(self) -> list[object]:
        return [
            self.platform_id,
            self.platform_name,
            self.system_id,
            self.system_name,
            self.title,
            self.source_row,
        ]

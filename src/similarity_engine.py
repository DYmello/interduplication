from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .excel_schema import InputRecord
from .name_normalizer import normalize_name_for_embedding


class Embedder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class NameObject:
    level: str
    group_id: str
    platform_id: str
    platform_name: str
    system_id: str
    system_name: str
    software_id: str
    software_name: str
    module_id: str = ""
    module_name: str = ""

    @property
    def object_id(self) -> str:
        return self.module_id if self.level == "module" else self.software_id

    @property
    def name(self) -> str:
        return self.module_name if self.level == "module" else self.software_name

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.level, self.group_id, self.object_id


@dataclass(frozen=True)
class SimilarityResult:
    level: str
    rank: int
    score: float | None
    similarity_level: str
    object_a: NameObject
    object_b: NameObject | None


def extract_objects(records: Sequence[InputRecord]) -> dict[str, list[NameObject]]:
    software: dict[tuple[str, str], NameObject] = {}
    modules: dict[tuple[str, str], NameObject] = {}
    for record in records:
        if not record.group_id:
            raise ValueError("组编号不能为空")
        if not record.software_id or not record.software_name:
            raise ValueError("软件编号和软件名称不能为空")
        software_object = NameObject(
            "software",
            record.group_id,
            record.platform_id,
            record.platform_name,
            record.system_id,
            record.system_name,
            record.software_id,
            record.software_name,
        )
        software_key = (record.group_id, record.software_id)
        existing_software = software.get(software_key)
        if existing_software is not None and existing_software != software_object:
            raise ValueError(
                f"同一组编号+软件编号对应不同名称或归属路径：{software_key}"
            )
        software[software_key] = software_object

        has_module_id = bool(record.module_id)
        has_module_name = bool(record.module_name)
        if has_module_id != has_module_name:
            raise ValueError(
                f"模块编号和模块名称必须同时为空或同时非空：{record.group_id}/{record.software_id}"
            )
        if has_module_id:
            if not record.platform_id or not record.system_id or not record.software_id:
                raise ValueError(f"模块 {record.module_id} 缺少完整软件归属路径")
            module_object = NameObject(
                "module",
                record.group_id,
                record.platform_id,
                record.platform_name,
                record.system_id,
                record.system_name,
                record.software_id,
                record.software_name,
                record.module_id,
                record.module_name,
            )
            module_key = (record.group_id, record.module_id)
            existing_module = modules.get(module_key)
            if existing_module is not None and existing_module != module_object:
                raise ValueError(
                    f"同一组编号+模块编号对应不同名称或归属路径：{module_key}"
                )
            modules[module_key] = module_object

    return {"software": list(software.values()), "module": list(modules.values())}


def similarity_label(normalized_a: str, normalized_b: str, score: float) -> str:
    if normalized_a == normalized_b:
        return "名称完全相同"
    if score >= 0.90:
        return "极高相似"
    if score >= 0.85:
        return "高相似"
    if score >= 0.75:
        return "中等相似"
    return "低相似"


class SimilarityEngine:
    def __init__(self, embedder: Embedder, logger: logging.Logger | None = None) -> None:
        self.embedder = embedder
        self.logger = logger or logging.getLogger(__name__)

    def compare(
        self,
        objects: Sequence[NameObject],
        *,
        comparison_mode: str,
        similarity_threshold: float,
        top_k: int,
        include_unmatched: bool,
        remove_module_suffix: bool,
        batch_size: int,
        max_length: int,
        block_size: int = 2048,
        anchor_group: str | None = None,
        target_groups: Sequence[str] | None = None,
    ) -> list[SimilarityResult]:
        if comparison_mode not in {"within", "cross", "all"}:
            raise ValueError(f"不支持的比较模式：{comparison_mode}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold 必须位于 [0, 1]")
        if top_k <= 0 or block_size <= 0:
            raise ValueError("top_k 和 block_size 必须为正整数")
        directional = anchor_group is not None or target_groups is not None
        if directional and comparison_mode != "cross":
            raise ValueError("anchor_group 和 target_groups 只适用于 cross 模式")
        if (anchor_group is None) != (target_groups is None):
            raise ValueError("anchor_group 和 target_groups 必须同时提供")
        target_group_set: set[str] = set()
        if directional:
            if not anchor_group:
                raise ValueError("anchor_group 不能为空")
            if not target_groups:
                raise ValueError("target_groups 解析后不能为空")
            target_group_set = set(target_groups)
            if anchor_group in target_group_set:
                raise ValueError("target_groups 不能包含 anchor_group")
        if not objects:
            return []
        levels = {item.level for item in objects}
        if len(levels) != 1:
            raise ValueError("一次 compare 调用只能包含同一层级对象")
        if len(objects) > 10_000:
            self.logger.warning(
                "对象数量为 %d，建议在生产环境评估 FAISS；本次仍使用分块矩阵计算",
                len(objects),
            )

        normalized_names = [
            normalize_name_for_embedding(
                item.name,
                remove_module_suffix=remove_module_suffix and item.level == "module",
            )
            for item in objects
        ]
        if any(not name for name in normalized_names):
            raise ValueError("规范化后名称不能为空")
        embeddings = self.embedder.encode(
            normalized_names,
            batch_size=batch_size,
            max_length=max_length,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(objects):
            raise ValueError("embedding 数量与对象数量不一致")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)
        if not np.isfinite(embeddings).all():
            raise ValueError("归一化向量包含 NaN 或无穷值")

        group_order: dict[str, int] = {}
        for item in objects:
            group_order.setdefault(item.group_id, len(group_order))

        results: list[SimilarityResult] = []
        count = len(objects)
        for block_start in range(0, count, block_size):
            block_end = min(block_start + block_size, count)
            score_block = embeddings[block_start:block_end] @ embeddings.T
            for local_index, object_a in enumerate(objects[block_start:block_end]):
                index_a = block_start + local_index
                if directional and object_a.group_id != anchor_group:
                    continue
                candidates: list[tuple[float, int]] = []
                for index_b, object_b in enumerate(objects):
                    if directional:
                        if not self._directed_cross_pair(
                            index_a,
                            object_a,
                            index_b,
                            object_b,
                            anchor_group,
                            target_group_set,
                        ):
                            continue
                    elif not self._canonical_pair(
                        index_a,
                        object_a,
                        index_b,
                        object_b,
                        comparison_mode,
                        group_order,
                    ):
                        continue
                    score = float(score_block[local_index, index_b])
                    if not math.isfinite(score):
                        raise ValueError("余弦相似度出现非有限值")
                    score = max(-1.0, min(1.0, score))
                    if score + 1e-7 < similarity_threshold:
                        continue
                    candidates.append((score, index_b))
                candidates.sort(
                    key=lambda item: (
                        -item[0],
                        objects[item[1]].group_id,
                        objects[item[1]].object_id,
                    )
                )
                selected = candidates[:top_k]
                if not selected and include_unmatched:
                    results.append(
                        SimilarityResult(
                            object_a.level,
                            0,
                            None,
                            "无匹配",
                            object_a,
                            None,
                        )
                    )
                for rank, (score, index_b) in enumerate(selected, start=1):
                    object_b = objects[index_b]
                    results.append(
                        SimilarityResult(
                            object_a.level,
                            rank,
                            score,
                            similarity_label(
                                normalized_names[index_a],
                                normalized_names[index_b],
                                score,
                            ),
                            object_a,
                            object_b,
                        )
                    )
        return results

    @staticmethod
    def _directed_cross_pair(
        index_a: int,
        object_a: NameObject,
        index_b: int,
        object_b: NameObject,
        anchor_group: str,
        target_groups: set[str],
    ) -> bool:
        """Return true only for anchor-to-target pairs in directed cross mode."""
        return (
            index_a != index_b
            and object_a.group_id == anchor_group
            and object_b.group_id in target_groups
        )

    @staticmethod
    def _canonical_pair(
        index_a: int,
        object_a: NameObject,
        index_b: int,
        object_b: NameObject,
        comparison_mode: str,
        group_order: dict[str, int],
    ) -> bool:
        if index_a == index_b:
            return False
        same_group = object_a.group_id == object_b.group_id
        if comparison_mode == "within" and not same_group:
            return False
        if comparison_mode == "cross" and same_group:
            return False

        if same_group:
            return index_a < index_b
        rank_a = group_order[object_a.group_id]
        rank_b = group_order[object_b.group_id]
        return rank_a < rank_b


def sort_results(results: Sequence[SimilarityResult]) -> list[SimilarityResult]:
    level_order = {"software": 0, "module": 1}
    return sorted(
        results,
        key=lambda item: (
            level_order.get(item.level, 99),
            item.object_a.group_id,
            item.object_a.object_id,
            item.rank,
            -(item.score if item.score is not None else -1.0),
        ),
    )

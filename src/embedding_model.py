from __future__ import annotations

import logging
from typing import Sequence

import numpy as np


class BGEEmbeddingModel:
    """Thin dense-only adapter around FlagEmbedding.BGEM3FlagModel."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        *,
        device: str = "auto",
        use_fp16: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.model_name_or_path = model_name_or_path
        self.device = self._resolve_device(device)
        self.use_fp16 = self.device == "cuda" if use_fp16 is None else bool(use_fp16)
        if self.device == "cpu" and self.use_fp16:
            self.logger.warning("CPU 不支持本程序的 FP16 编码，已自动关闭")
            self.use_fp16 = False

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 FlagEmbedding；请先执行 pip install -r requirements.txt"
            ) from exc

        kwargs = {"use_fp16": self.use_fp16}
        if self.device != "auto":
            kwargs["devices"] = self.device
        self.logger.info(
            "加载 BGE-M3 模型：%s，device=%s，use_fp16=%s",
            model_name_or_path,
            self.device,
            self.use_fp16,
        )
        try:
            self.model = BGEM3FlagModel(model_name_or_path, **kwargs)
        except TypeError as exc:
            if "devices" not in kwargs:
                raise
            self.logger.debug("当前 FlagEmbedding 不接受 devices，尝试 device 参数")
            kwargs["device"] = kwargs.pop("devices")
            try:
                self.model = BGEM3FlagModel(model_name_or_path, **kwargs)
            except TypeError:
                raise exc

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"不支持的 device：{requested}")
        try:
            import torch
        except ImportError:
            if requested == "cuda":
                raise RuntimeError("--device cuda 需要安装支持 CUDA 的 PyTorch")
            return "cpu" if requested == "auto" else requested
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但 PyTorch 未检测到可用 GPU")
        return requested

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        max_length: int = 64,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        output = self.model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        if not isinstance(output, dict) or "dense_vecs" not in output:
            raise RuntimeError("BGE-M3 未返回 dense_vecs")
        embeddings = np.asarray(output["dense_vecs"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"BGE-M3 向量形状异常：{embeddings.shape}，期望首维 {len(texts)}"
            )
        if not np.isfinite(embeddings).all():
            raise RuntimeError("BGE-M3 向量包含 NaN 或无穷值")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)
        return embeddings.astype(np.float32, copy=False)

from __future__ import annotations

import os

import numpy as np
import pytest

from src.embedding_model import BGEEmbeddingModel


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_BGE_INTEGRATION") != "1",
    reason="Set RUN_BGE_INTEGRATION=1 and provide a local/online BGE-M3 model",
)
def test_real_bge_m3_self_similarity():
    model = BGEEmbeddingModel(os.getenv("BGE_M3_MODEL", "BAAI/bge-m3"), device="auto")
    embeddings = model.encode(["元数据完整性评测模块"], batch_size=1, max_length=64)
    assert np.isclose(float(embeddings[0] @ embeddings[0]), 1.0, atol=1e-5)

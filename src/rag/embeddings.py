"""文本向量化：mock（确定性哈希向量，离线可用）与 OpenAI 兼容 Embedding 两种实现。

- mock：对字符 bigram 做带符号哈希累加并 L2 归一化，相似文本余弦更高，无需任何网络/key；
- openai：经 langchain_openai.OpenAIEmbeddings 调用兼容端点（OpenAI / DeepSeek / 通义等），
  失败时自动降级 mock，保证管线不中断。
- 为 Chroma 提供自定义 EmbeddingFunction，阻止 Chroma 默认下载 `all-MiniLM-L6-v2` ONNX 模型。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from ..config import settings

DIM = 256


class Embedder:
    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider or settings.embedding_provider
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        self.base_url = base_url if base_url is not None else settings.embedding_base_url
        self.model = model or settings.embedding_model
        self._real = None

    def _real_client(self):
        if self._real is None:
            from langchain_openai import OpenAIEmbeddings

            self._real = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key or None,
                base_url=self.base_url or None,
            )
        return self._real

    def embed(self, text: str) -> list[float]:
        if self.provider == "openai" and (self.api_key or self.base_url):
            try:
                return self._real_client().embed_query(text)
            except Exception:
                pass  # 失败降级 mock
        return self._mock_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai" and (self.api_key or self.base_url):
            try:
                return self._real_client().embed_documents(texts)
            except Exception:
                pass  # 失败降级逐条 mock
        return [self._mock_embed(t) for t in texts]

    @staticmethod
    def _mock_embed(text: str, dim: int = DIM) -> list[float]:
        vec = [0.0] * dim
        for i in range(0, len(text) - 1, 2):
            gram = text[i : i + 2]
            h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:4], "big")
            idx = h % dim
            sign = 1.0 if (h & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class ChromaEmbeddingFunction:
    """Chroma 自定义 embedding function，复用项目 Embedder，避免 Chroma 自动下载 ONNX 模型。

    Chroma 默认 embedding function 会联网下载 `all-MiniLM-L6-v2/onnx.tar.gz`。
    我们显式传入此 wrapper，让 Chroma 使用项目自己的 mock/OpenAI 向量，
    从而做到「零本地模型下载、零 ONNX 依赖」。
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()

    def __call__(self, input: list[str]) -> list[Any]:  # noqa: A002 — Chroma 协议参数名
        import numpy as np

        return [np.array(v, dtype=np.float32) for v in self.embedder.embed_batch(input)]

    @staticmethod
    def name() -> str:
        """Chroma 用此名称做持久化校验；必须返回稳定字符串，不能是默认的 NotImplemented。"""
        return "custom_embedder"

    @classmethod
    def create(cls, embedder: Embedder | None = None) -> Any:
        """返回 chromadb.api.types.EmbeddingFunction 子类实例，复用项目 Embedder。

        继承顺序必须是 (cls, EF)：确保 Chroma 的 __init_subclass__ 包装到
        我们自己的 __call__，而不是 EmbeddingFunction 的抽象占位方法。
        """
        try:
            from chromadb.api.types import EmbeddingFunction as EF

            class _EF(cls, EF):  # noqa: N801 — 局部匿名子类
                pass

            return _EF(embedder=embedder)
        except Exception:
            return cls(embedder=embedder)

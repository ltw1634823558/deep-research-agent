"""重排（Rerank）：在向量召回的候选集上做精排。

- mock：语义余弦(0.6) + 词面重叠(0.4) 的混合打分，离线可用；
- cohere：调用 Cohere Rerank API（配置 COHERE_API_KEY 后生效），失败时自动降级 mock。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import settings
from .embeddings import Embedder, cosine

logger = logging.getLogger(__name__)


@dataclass
class RankedDoc:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class Reranker:
    def __init__(
        self,
        provider: str | None = None,
        cohere_api_key: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.provider = provider or settings.rerank_provider
        self.cohere_api_key = (
            cohere_api_key if cohere_api_key is not None else settings.cohere_api_key
        )
        self.embedder = embedder or Embedder()

    def rerank(
        self,
        query: str,
        docs: list[RankedDoc],
        top_n: int | None = None,
        *,
        emb_lookup: dict | None = None,
    ) -> list[RankedDoc]:
        """精排候选集。

        `emb_lookup`（可选）：以 `RankedDoc.id` 为键的文档向量缓存，由调用方复用已算好的
        向量，避免 mock 精排重复调用 embedding（openai provider 下即重复 HTTP 请求）。
        """
        if self.provider == "cohere" and self.cohere_api_key:
            try:
                return self._cohere_rerank(query, docs, top_n)
            except Exception as e:  # 降级 mock
                logger.warning(
                    "rerank: Cohere Rerank 调用失败，降级 mock 混合打分：%s", e, exc_info=False
                )
        return self._mock_rerank(query, docs, top_n, emb_lookup=emb_lookup)

    def _mock_rerank(
        self,
        query: str,
        docs: list[RankedDoc],
        top_n: int | None,
        *,
        emb_lookup: dict | None = None,
    ) -> list[RankedDoc]:
        if not docs:
            return []
        q_emb = self.embedder.embed(query)
        q_tokens = set(query.lower().split())
        scored: list[RankedDoc] = []
        for d in docs:
            d_emb = (
                emb_lookup[d.id]
                if emb_lookup is not None and d.id in emb_lookup
                else self.embedder.embed(d.text)
            )
            sem = max(cosine(q_emb, d_emb), 0.0)
            d_tokens = set(d.text.lower().split())
            overlap = len(q_tokens & d_tokens) / (len(q_tokens | d_tokens) or 1)
            hybrid = 0.6 * sem + 0.4 * overlap
            scored.append(
                RankedDoc(id=d.id, text=d.text, score=round(hybrid, 4), metadata=d.metadata)
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n] if top_n else scored

    def _cohere_rerank(
        self, query: str, docs: list[RankedDoc], top_n: int | None
    ) -> list[RankedDoc]:
        import cohere

        client = cohere.Client(self.cohere_api_key)
        resp = client.rerank(
            query=query,
            documents=[d.text for d in docs],
            top_n=top_n or len(docs),
            model="rerank-english-v3.0",
        )
        out: list[RankedDoc] = []
        for r in resp.results:
            d = docs[r.index]
            out.append(
                RankedDoc(
                    id=d.id,
                    text=d.text,
                    score=round(float(r.relevance_score), 4),
                    metadata=d.metadata,
                )
            )
        return out

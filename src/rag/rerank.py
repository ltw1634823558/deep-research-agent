"""重排（Rerank）：在向量召回的候选集上做精排。

- mock：语义余弦(0.6) + 词面重叠(0.4) 的混合打分，离线可用；
- cohere：调用 Cohere Rerank API（配置 COHERE_API_KEY 后生效），失败时自动降级 mock。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings
from .embeddings import Embedder, cosine


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
        self, query: str, docs: list[RankedDoc], top_n: int | None = None
    ) -> list[RankedDoc]:
        if self.provider == "cohere" and self.cohere_api_key:
            try:
                return self._cohere_rerank(query, docs, top_n)
            except Exception:
                pass  # 降级 mock
        return self._mock_rerank(query, docs, top_n)

    def _mock_rerank(self, query: str, docs: list[RankedDoc], top_n: int | None) -> list[RankedDoc]:
        if not docs:
            return []
        q_emb = self.embedder.embed(query)
        q_tokens = set(query.lower().split())
        scored: list[RankedDoc] = []
        for d in docs:
            sem = max(cosine(q_emb, self.embedder.embed(d.text)), 0.0)
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

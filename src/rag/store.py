"""RAG 向量库：向量召回 + Rerank 重排。

后端可切换（settings.rag_backend / RAG_BACKEND）：
- `auto`（默认）：优先尝试 Chroma，离线/受限时自动降级内存余弦召回，保证「零 key 离线可跑」；
- `chroma`：强制使用 Chroma（ephemeral，无需起服务）；初始化失败则告警并降级内存；
- `memory`：纯内存向量召回（无外部依赖，最快、最稳）。

Chroma 在离线环境会尝试连接远端遥测而阻塞，因此用守护线程 + 超时探测包裹其初始化，
超时可安全放弃而不阻塞进程退出。返回的精排上下文喂给 LLM——即真实 RAG 的「稠密召回 + 精排」阶段。
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..state import Source
from .embeddings import ChromaEmbeddingFunction, Embedder, cosine
from .rerank import RankedDoc, Reranker

logger = logging.getLogger(__name__)

_CHROMA_TIMEOUT = 6.0
_CHROMA_PROBE: bool | None = None  # 模块级缓存：避免重复触发离线阻塞


@dataclass
class RAGDoc:
    id: str
    text: str
    source: Source | None = None
    metadata: dict = field(default_factory=dict)


class ChromaRAGStore:
    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        collection: str | None = None,
        backend: str | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker(embedder=self.embedder)
        self._docs: dict[str, RAGDoc] = {}
        self._collection: Any = None
        self._use_chroma = False

        want = backend or settings.rag_backend
        if want != "memory":
            self._use_chroma = self._try_init_chroma(collection, force=(want == "chroma"))

    def _try_init_chroma(self, collection: str | None, force: bool) -> bool:
        global _CHROMA_PROBE
        if _CHROMA_PROBE is False:
            return False  # 已知离线/不可用，直接降级，避免重复阻塞
        box: dict = {}

        def _init() -> None:
            try:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                col = client.get_or_create_collection(
                    name=collection or f"rag_{uuid.uuid4().hex[:8]}",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=ChromaEmbeddingFunction.create(self.embedder),
                )
                # 探针：确认离线环境下不会卡死
                col.add(
                    ids=["__probe__"],
                    documents=["probe"],
                    metadatas=[{"id": "__probe__"}],
                )
                col.query(query_texts=["probe"], n_results=1)
                col.delete(ids=["__probe__"])
                box["col"] = col
            except Exception as e:  # 降级内存向量库
                logger.warning(
                    "chroma: 初始化/探针失败（可能离线或网络受限），将降级内存向量库：%s",
                    e,
                    exc_info=False,
                )

        t = threading.Thread(target=_init, daemon=True)
        t.start()
        t.join(_CHROMA_TIMEOUT)
        if "col" in box:
            self._collection = box["col"]
            _CHROMA_PROBE = True
            return True
        _CHROMA_PROBE = False
        if force:
            import sys

            print(
                "[rag] 警告：Chroma 初始化失败（可能离线/网络受限），已回退内存向量库。",
                file=sys.stderr,
            )
        return False

    @property
    def backend(self) -> str:
        return "chromadb" if self._use_chroma else "in-memory"

    def add_documents(self, docs: list[RAGDoc]) -> None:
        if not docs:
            return
        self._docs.update({d.id: d for d in docs})
        if self._use_chroma:
            self._collection.add(
                ids=[d.id for d in docs],
                documents=[d.text for d in docs],
                metadatas=[{**d.metadata, "id": d.id} for d in docs],
            )

    def retrieve(
        self, query: str, top_k: int | None = None, rerank_top_n: int | None = None
    ) -> list[RankedDoc]:
        k = top_k or settings.rag_top_k
        if not self._docs:
            return []
        emb_lookup: dict[str, list[float]] | None = None
        if self._use_chroma:
            n = min(k, len(self._docs))
            res = self._collection.query(query_texts=[query], n_results=n)
            ids = (res.get("ids") or [[]])[0]
            docs = [self._docs[i] for i in ids if i in self._docs]
        else:
            # 向量只算一次：query 1 次 + 全量文档 1 次批量，随后召回排序与精排共享，
            # 避免「排序 N 次 + 精排 N 次」的重复 embedding 调用。
            all_docs = list(self._docs.values())
            q_emb = self.embedder.embed(query)
            doc_embs = self.embedder.embed_batch([d.text for d in all_docs])
            emb_lookup = {d.id: e for d, e in zip(all_docs, doc_embs, strict=True)}
            ranked = sorted(
                all_docs,
                key=lambda d: cosine(q_emb, emb_lookup[d.id]),
                reverse=True,
            )
            docs = ranked[:k]
        base = [RankedDoc(id=d.id, text=d.text, score=0.0, metadata=d.metadata) for d in docs]
        return self.reranker.rerank(query, base, top_n=rerank_top_n or k, emb_lookup=emb_lookup)

"""RAG 模块测试：离线 mock 嵌入/重排/向量召回，无需任何 key。

注：离线 CI 下 Chroma 会因远端遥测握手阻塞，故测试固定使用内存向量后端
（settings.rag_backend=memory），与「稠密召回 + 重排」逻辑完全等价；
Chroma 后端仅在可联网环境手动开启（RAG_BACKEND=chroma）时生效。
"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ["RAG_BACKEND"] = "memory"

from src.rag.embeddings import Embedder, cosine  # noqa: E402
from src.rag.rerank import RankedDoc, Reranker  # noqa: E402
from src.rag.store import ChromaRAGStore, RAGDoc  # noqa: E402


def test_embedder_mock_deterministic_and_dim():
    e = Embedder(provider="mock")
    v1 = e.embed("hello world")
    v2 = e.embed("hello world")
    assert v1 == v2
    assert len(v1) == 256
    # 相似文本余弦 > 不相似文本余弦
    sim = cosine(e.embed("人工智能 Agent 编排"), e.embed("智能体 Agent 编排"))
    diff = cosine(e.embed("人工智能 Agent 编排"), e.embed("足球比赛比分"))
    assert sim > diff


def test_reranker_mock_orders_relevant_first():
    r = Reranker(provider="mock")
    docs = [
        RankedDoc(id="a", text="关于 Agent 编排与多智能体的内容", score=0.0),
        RankedDoc(id="b", text="完全无关的话题例如烹饪食谱", score=0.0),
    ]
    out = r.rerank("Agent 编排", docs)
    assert out[0].id == "a"


def test_store_retrieve_returns_reranked_top():
    store = ChromaRAGStore()
    docs = [
        RAGDoc(id="1", text="LangGraph 多智能体编排详解", metadata={"source_type": "web"}),
        RAGDoc(id="2", text="今天天气不错适合散步", metadata={"source_type": "web"}),
    ]
    store.add_documents(docs)
    out = store.retrieve("多智能体编排", top_k=2)
    assert out[0].id == "1"
    assert store.backend == "in-memory"


def test_store_empty_query_is_safe():
    store = ChromaRAGStore()
    assert store.retrieve("anything") == []


def test_researcher_uses_rag_without_error():
    from src.graph import build_graph
    from src.state import initial_state

    state = initial_state("RAG 召回测试", 1, mode="web")
    result = build_graph().invoke(state)
    # 管线跑通且至少产出一份来源（说明 researcher 的 RAG 召回阶段未中断）
    assert len(result.get("sources", [])) >= 1
    assert len(result.get("report", "")) > 0

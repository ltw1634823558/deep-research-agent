"""Researcher：对下一个 pending 子主题做联网检索 + 长记忆召回，产出带引用的摘要。"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import get_llm, settings
from ..memory import memory_store
from ..observability import callbacks
from ..rag.store import ChromaRAGStore, RAGDoc
from ..state import Finding, ResearchState, Source
from ..tools.fetch import fetch_url
from ..tools.obsidian import obsidian_search
from ..tools.search import search


def _source_text(s: Source) -> str:
    """按来源类型读取内容：web 走抓取，local 直接取笔记摘要（已是正文）。"""
    if getattr(s, "source_type", "web") == "local":
        return s.snippet or ""
    return fetch_url(s.url)


def researcher_node(state: ResearchState, config: RunnableConfig) -> dict:
    subtopic = next((s for s in state["plan"] if s.status == "pending"), None)
    if subtopic is None:
        return {}

    # 按研究模式聚合来源：web / local / hybrid
    mode = state.get("mode") or settings.research_mode
    sources: list[Source] = []
    if mode in ("web", "hybrid"):
        sources += search(subtopic.question)
    if mode in ("local", "hybrid"):
        # 私域知识库检索，并自动带上双链关联笔记（文档联系）
        sources += obsidian_search(subtopic.question)

    # 长记忆召回：复用历史相关研究，减少重复检索
    prior = memory_store.recall(subtopic.question)
    prior_ctx = "\n".join(prior) if prior else "（无历史记忆）"

    # ===== 真实 RAG 阶段：候选来源 -> 向量化 -> Chroma 召回 -> 重排精排 =====
    store = ChromaRAGStore()
    docs = []
    for idx, s in enumerate(sources[:8]):
        body = _source_text(s)
        text = f"{s.title}\n{s.snippet}\n{body[:800]}"
        docs.append(
            RAGDoc(
                id=f"d{idx}",
                text=text,
                source=s,
                metadata={"source_type": s.source_type, "url": s.url, "title": s.title},
            )
        )
    store.add_documents(docs)
    ranked = store.retrieve(subtopic.question, top_k=settings.rag_top_k)

    # 用重排后的精排上下文喂给 LLM（召回阶段已按相关性排序）
    fetched_parts = []
    for r in ranked:
        tag = f"[{r.metadata.get('source_type', 'web')}] {r.metadata.get('title', '')}"
        fetched_parts.append(f"- {tag}（rerank 得分 {r.score}）: {r.text[:500]}")
    fetched = "\n".join(fetched_parts)

    llm = get_llm()
    prompt = (
        f"你是一个研究执行智能体，请基于检索内容与历史记忆，对子主题做简洁摘要（含要点）。\n"
        f"子主题：{subtopic.question}\n\n"
        f"检索内容（已按相关性重排）：\n{fetched}\n\n"
        f"历史记忆：\n{prior_ctx}"
    )
    summary = llm.invoke(prompt, config={"callbacks": callbacks}).content

    finding = Finding(subtopic_id=subtopic.id, summary=summary, sources=sources)

    # 标记该子主题为已完成
    new_plan = [
        s.model_copy(update={"status": "done"}) if s.id == subtopic.id else s
        for s in state["plan"]
    ]
    return {
        "findings": [finding],
        "plan": new_plan,
        "sources": sources,
        "memory_recall": prior,  # 本轮子主题召回到的历史洞察（dashboard 展示）
        "messages": [AIMessage(content=f"[Researcher] 完成子主题：{subtopic.question}（RAG 后端：{store.backend}）")],
    }

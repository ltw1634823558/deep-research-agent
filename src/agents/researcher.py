"""Researcher：对下一个 pending 子主题做联网检索 + 长记忆召回，产出带引用的摘要。"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import Settings, get_llm, resolve_settings
from ..memory import memory_store
from ..rag.store import ChromaRAGStore, RAGDoc
from ..safety import UNTRUSTED_GUARD, wrap_untrusted
from ..state import Finding, ResearchState, Source
from ..tools.fetch import fetch_url
from ..tools.obsidian import obsidian_search
from ..tools.search import search


def _source_text(s: Source, cfg: "Settings | None" = None) -> str:
    """按来源类型读取内容：web 走抓取，local 直接取笔记摘要（已是正文）。

    cfg 透传按请求配置（如 per-request Tavily key），未传则回落全局单例。
    """
    if getattr(s, "source_type", "web") == "local":
        return s.snippet or ""
    return fetch_url(s.url, settings_obj=cfg)


def researcher_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = resolve_settings(config)
    subtopic = next((s for s in state["plan"] if s.status == "pending"), None)
    if subtopic is None:
        return {}

    # 按研究模式聚合来源：web / local / hybrid
    mode = state.get("mode") or cfg.research_mode
    sources: list[Source] = []
    if mode in ("web", "hybrid"):
        sources += search(subtopic.question, settings_obj=cfg)
    if mode in ("local", "hybrid"):
        # 私域知识库检索，并自动带上双链关联笔记（文档联系）
        sources += obsidian_search(subtopic.question, settings_obj=cfg)

    # 长记忆召回：复用历史相关研究，减少重复检索（cfg 透传 per-request 开关/top_k）
    prior = memory_store.recall(subtopic.question, cfg=cfg)
    # 召回记忆属于外部不可信内容（可能源自更早被污染的网页），用分隔符包裹以阻断指令注入
    prior_ctx = wrap_untrusted("历史记忆", "\n".join(prior)) if prior else "（无历史记忆）"

    # ===== 真实 RAG 阶段：候选来源 -> 向量化 -> Chroma 召回 -> 重排精排 =====
    store: "ChromaRAGStore | None" = None
    used_sources: list[Source] = []
    fetched_block = ""
    try:
        store = ChromaRAGStore(backend=cfg.rag_backend, settings_obj=cfg)
        docs = []
        for idx, s in enumerate(sources[:8]):
            body = _source_text(s, cfg)
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
        ranked = store.retrieve(subtopic.question, top_k=cfg.rag_top_k)

        # 只保留真正喂给 LLM 的重排 top_k 来源（候选集不外泄到 state，避免来源膨胀/无关引用）
        id_to_source = {d.id: d.source for d in docs}
        seen_urls: set[str] = set()
        for r in ranked:
            s = id_to_source.get(r.id)
            if s is None:
                continue
            key = (s.url or "").strip().lower()
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            used_sources.append(s)

        # 用重排后的精排上下文喂给 LLM（召回阶段已按相关性排序）
        fetched_parts = []
        for r in ranked:
            tag = f"[{r.metadata.get('source_type', 'web')}] {r.metadata.get('title', '')}"
            fetched_parts.append(f"- {tag}（rerank 得分 {r.score}）: {r.text[:500]}")
        fetched = "\n".join(fetched_parts)
        # 网页检索内容同样属于不可信外部数据，用分隔符包裹以阻断指令注入
        fetched_block = wrap_untrusted("web检索", fetched)
    finally:
        # 无论构造/抓取/召回/摘要是否抛错，都释放在线 Chroma ephemeral collection（M-4/L-D/L4）
        rag_backend = store.backend if store is not None else "in-memory"
        if store is not None:
            store.close()

    llm = get_llm(cfg)
    prompt = (
        f"你是一个研究执行智能体，请基于检索内容与历史记忆，对子主题做简洁摘要（含要点）。\n"
        f"子主题：{subtopic.question}\n\n"
        f"检索内容（已按相关性重排）：\n{fetched_block}\n\n"
        f"历史记忆：\n{prior_ctx}\n\n"
        f"{UNTRUSTED_GUARD}"
    )
    # 透传节点自身的 RunnableConfig（含本次任务专属 callbacks），
    # 不再用模块级全局 callbacks 覆盖，避免并发任务 trace 串扰。
    summary = llm.invoke(prompt, config=config).content

    finding = Finding(subtopic_id=subtopic.id, summary=summary, sources=used_sources)

    # 标记该子主题为已完成
    new_plan = [
        s.model_copy(update={"status": "done"}) if s.id == subtopic.id else s
        for s in state["plan"]
    ]
    return {
        "findings": [finding],
        "plan": new_plan,
        "sources": used_sources,
        "memory_recall": prior,  # 本轮子主题召回到的历史洞察（dashboard 展示）
        "messages": [AIMessage(content=f"[Researcher] 完成子主题：{subtopic.question}（RAG 后端：{rag_backend}）")],
    }

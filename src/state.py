"""共享状态定义：LangGraph 在节点间传递的 TypedDict，以及业务数据模型。"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class Subtopic(BaseModel):
    id: str
    question: str
    status: str = "pending"  # pending | done


class Source(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    source_type: str = "web"  # web | local（区分联网检索与私域知识库）


class Finding(BaseModel):
    subtopic_id: str
    summary: str
    sources: list[Source] = Field(default_factory=list)


def _add_sources(existing: list[Source] | None, new: list[Source] | None) -> list[Source]:
    """sources 归并 reducer：按归一化 URL 去重，保留首次出现顺序，先到先得。

    替代 `operator.add`：后者在每次累加时重复堆叠同一 URL（跨子主题重复检索到同一网页），
    导致来源列表膨胀、评估分母失真。
    """
    merged: list[Source] = []
    seen: set[str] = set()
    for s in list(existing or []) + list(new or []):
        key = (s.url or "").strip().lower()
        if key:
            if key in seen:
                continue
            seen.add(key)
        merged.append(s)
    return merged


def _add_messages_windowed(
    existing: list[BaseMessage] | None, new: list[BaseMessage] | None
) -> list[BaseMessage]:
    """messages 归并 reducer：先用官方 `add_messages`（按 id 去重/覆盖），再截留最近 N 条。

    N 取 `settings.research_window`。此前该配置项在 `.env.example` 中被公开为可调旋钮，
    但代码里无人读取——等于对使用者撒谎，且 researcher 的迭代回路每个子主题都会追加
    一条消息，长任务下 messages 会无界增长。这里让它真正生效：<=0 表示不限制。

    reducer 由 LangGraph 在图内部调用，拿不到 per-request config，故读全局 settings——
    这是结构性上限，不属于按租户覆写的参数。
    """
    merged = add_messages(existing or [], new or [])
    from .config import settings as _settings  # 延迟导入，避免 state <-> config 循环依赖

    window = getattr(_settings, "research_window", 0) or 0
    if window > 0 and len(merged) > window:
        return list(merged[-window:])
    return merged


class ResearchState(TypedDict):
    query: str
    plan: list[Subtopic]
    findings: Annotated[list[Finding], add]
    analysis: str
    report: str
    sources: Annotated[list[Source], _add_sources]
    iteration: int
    max_iterations: int
    mode: str  # web | local | hybrid（研究模式，决定检索来源）
    messages: Annotated[list[BaseMessage], _add_messages_windowed]
    # 长记忆：本轮各子主题召回到的历史洞察（供 dashboard 展示，累加）
    memory_recall: Annotated[list[str], add]
    # Analyst 自愈回路实际尝试次数（仅真实 LLM 触发）
    analyst_self_heal: int
    # 本次研究写入长记忆的洞察条数
    memory_writes: int


def initial_state(query: str, max_iterations: int, mode: str = "web") -> ResearchState:
    return {
        "query": query,
        "plan": [],
        "findings": [],
        "analysis": "",
        "report": "",
        "sources": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "mode": mode,
        "messages": [],
        "memory_recall": [],
        "analyst_self_heal": 0,
        "memory_writes": 0,
    }

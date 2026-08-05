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


class ResearchState(TypedDict):
    query: str
    plan: list[Subtopic]
    findings: Annotated[list[Finding], add]
    analysis: str
    report: str
    sources: Annotated[list[Source], add]
    iteration: int
    max_iterations: int
    mode: str  # web | local | hybrid（研究模式，决定检索来源）
    messages: Annotated[list[BaseMessage], add_messages]
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

"""LangGraph 编排：Planner -> Researcher(循环) -> Analyst(可回环) -> Writer。

路由逻辑（确定性，可测试、可观测）：
- researcher 之后：还有 pending 子主题 -> 继续 researcher；否则 -> analyst
- analyst 之后：还有 pending 子主题 -> 回到 researcher 补充检索；否则 -> writer
迭代上限由 state.max_iterations 兜底，防止死循环。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.analyst import analyst_node
from .agents.planner import planner_node
from .agents.researcher import researcher_node
from .agents.writer import writer_node
from .state import ResearchState


def _has_pending(state: ResearchState) -> bool:
    return any(s.status == "pending" for s in state.get("plan", []))


def route_after_research(state: ResearchState) -> str:
    return "researcher" if _has_pending(state) else "analyst"


def route_after_analysis(state: ResearchState) -> str:
    return "researcher" if _has_pending(state) else "writer"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst", analyst_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_conditional_edges(
        "researcher",
        route_after_research,
        {"researcher": "researcher", "analyst": "analyst"},
    )
    g.add_conditional_edges(
        "analyst",
        route_after_analysis,
        {"researcher": "researcher", "writer": "writer"},
    )
    g.add_edge("writer", END)
    return g.compile()

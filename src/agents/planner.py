"""Planner：把用户问题拆成若干可独立检索的子主题（任务规划 / 流程工程）。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import get_llm
from ..observability import callbacks
from ..state import ResearchState, Subtopic


def _parse_plan(raw: str, query: str) -> list[Subtopic]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "subtopics" in data:
            data = data["subtopics"]
        if isinstance(data, list):
            return [Subtopic(id=f"s{i + 1}", question=str(d)) for i, d in enumerate(data)]
    except Exception:
        pass
    # 兜底：基于原问题派生 3 个通用子主题
    return [
        Subtopic(id="s1", question=f"背景与定义：{query}"),
        Subtopic(id="s2", question=f"核心技术与方案：{query}"),
        Subtopic(id="s3", question=f"趋势与挑战：{query}"),
    ]


def planner_node(state: ResearchState, config: RunnableConfig) -> dict:
    llm = get_llm()
    prompt = (
        f"你是一个研究规划智能体。请把研究问题拆解为 3-5 个可独立检索的子主题，"
        f"直接返回一个 JSON 字符串数组，不要其他内容。\n研究问题：{state['query']}"
    )
    raw = llm.invoke(prompt, config={"callbacks": callbacks}).content
    plan = _parse_plan(raw, state["query"])
    return {
        "plan": plan,
        "messages": [AIMessage(content=f"[Planner] 拆解出 {len(plan)} 个子主题")],
    }

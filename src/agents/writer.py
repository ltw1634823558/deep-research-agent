"""Writer：综合所有发现与引用，产出最终深度研究报告，并落库到长期记忆。"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import get_llm
from ..memory import memory_store
from ..observability import callbacks
from ..state import ResearchState


def writer_node(state: ResearchState, config: RunnableConfig) -> dict:
    llm = get_llm()
    findings_text = "\n\n".join(
        f"## 子主题 {f.subtopic_id}\n{f.summary}\n" + "来源：" + ", ".join(s.url for s in f.sources)
        for f in state["findings"]
    )
    prompt = (
        "你是一个报告撰写智能体。请基于下列研究发现，撰写一份结构清晰的深度研究报告，"
        "使用 Markdown，并在文中标注引用来源。\n"
        f"研究问题：{state['query']}\n\n研究发现：\n{findings_text}"
    )
    report = llm.invoke(prompt, config={"callbacks": callbacks}).content

    # 落库长期记忆（抽取洞察写入语义索引），返回写入条数供可观测
    n_written = memory_store.save(state["query"], report)

    return {
        "report": report,
        "memory_writes": n_written,
        "messages": [AIMessage(content="[Writer] 报告已生成")],
    }

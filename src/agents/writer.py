"""Writer：综合所有发现与引用，产出最终深度研究报告，并落库到长期记忆。"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import get_llm, resolve_settings
from ..memory import memory_store
from ..safety import UNTRUSTED_GUARD, wrap_untrusted
from ..state import ResearchState


def writer_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = resolve_settings(config)
    llm = get_llm(cfg)
    findings_text = "\n\n".join(
        f"## 子主题 {f.subtopic_id}\n{f.summary}\n" + "来源：" + ", ".join(s.url for s in f.sources)
        for f in state["findings"]
    )
    # 研究发现源自更早被外部网页污染的检索摘要，同样属不可信数据，用分隔符包裹（含转义）以阻断注入
    findings_block = wrap_untrusted("研究发现", findings_text)
    prompt = (
        "你是一个报告撰写智能体。请基于下列研究发现，撰写一份结构清晰的深度研究报告，"
        "使用 Markdown，并在文中标注引用来源。\n"
        f"研究问题：{state['query']}\n\n研究发现：\n{findings_block}\n\n"
        f"{UNTRUSTED_GUARD}"
    )
    # 转发节点自身的 RunnableConfig（含本次任务专属 callbacks），
    # 不再用模块级全局 callbacks 覆盖，避免并发任务 trace 串扰。
    report = llm.invoke(prompt, config=config).content

    # 落库长期记忆（抽取洞察写入语义索引），返回写入条数供可观测
    n_written = memory_store.save(state["query"], report, cfg=cfg)

    return {
        "report": report,
        "memory_writes": n_written,
        "messages": [AIMessage(content="[Writer] 报告已生成")],
    }

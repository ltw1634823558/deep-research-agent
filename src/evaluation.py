"""研究质量评估：任务完成率 / 引用准确率 / 幻觉率（启发式代理指标）。

三项指标均为 [0,1]，便于跨运行对比与长期追踪：
- task_completion_rate：计划子主题全部完成 + 报告非空且达最低长度；
- citation_accuracy：报告显式引用的 URL 中，命中检索来源集合的比例（精度）；
- hallucination_rate：未被报告引用的检索来源占比（「未接地」代理指标）。

配置 LangFuse 后，指标作为 score 写入对应 trace，便于在 LangFuse UI 做看板与回归对比。
注：mock 模式下 LLM 不产出真实引用，指标仅作管线演示；接真实模型后即为真实评估值。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .observability import get_langfuse_client
from .state import ResearchState, Source

URL_RE = re.compile(r"https?://[^\s)\]]+")


@dataclass
class EvalResult:
    task_completion_rate: float
    citation_accuracy: float
    hallucination_rate: float
    citation_recall: float
    sources_total: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(state: ResearchState) -> EvalResult:
    report = (state.get("report") or "").strip()
    sources: list[Source] = state.get("sources") or []
    plan = state.get("plan") or []

    # 1) 任务完成率：计划子主题完成度 + 报告完整性
    total = len(plan)
    done = sum(1 for s in plan if s.status == "done")
    plan_ratio = (done / total) if total else 1.0
    report_ok = 1.0 if len(report) >= 120 else 0.0
    task_completion_rate = round((plan_ratio + report_ok) / 2.0, 3)

    # 2) 引用准确率 / 召回率
    urls_in_report = set(URL_RE.findall(report))
    source_urls = {s.url for s in sources}
    if urls_in_report:
        correct = sum(1 for u in urls_in_report if u in source_urls)
        citation_accuracy = round(correct / len(urls_in_report), 3)
    else:
        # 未显式引用 URL：视为无虚假引用，精度满分（真实模型下会反映真实引用）
        citation_accuracy = 1.0
    referenced = sum(1 for s in sources if (s.url in report or s.title in report))
    citation_recall = round(referenced / len(sources), 3) if sources else 1.0

    # 3) 幻觉率（启发式代理）：未被报告引用的检索来源占比
    grounded = referenced
    hallucination_rate = round(1.0 - (grounded / len(sources)), 3) if sources else 0.0

    result = EvalResult(
        task_completion_rate=task_completion_rate,
        citation_accuracy=citation_accuracy,
        hallucination_rate=hallucination_rate,
        citation_recall=citation_recall,
        sources_total=len(sources),
        details={
            "mode": state.get("mode"),
            "subtopics_total": total,
            "subtopics_done": done,
            "report_len": len(report),
            "urls_in_report": len(urls_in_report),
        },
    )
    _log_to_langfuse(result, state)
    return result


def _log_to_langfuse(result: EvalResult, state: ResearchState) -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        trace = client.trace(
            name="deep-research-eval",
            metadata={"query": state.get("query"), "mode": state.get("mode")},
        )
        for name in ("task_completion_rate", "citation_accuracy", "hallucination_rate"):
            client.score(trace_id=trace.id, name=name, value=getattr(result, name))
        client.flush()
    except Exception:
        pass

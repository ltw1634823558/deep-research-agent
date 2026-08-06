"""评估模块边界测试：空 title 不再虚高引用召回、URL/title 命中判定。

离线：evaluate() 在 langfuse 未配置时直接 return，无网络依赖。
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from src.evaluation import evaluate  # noqa: E402
from src.state import Source, Subtopic, initial_state  # noqa: E402


def _state_with_report(report: str, sources):
    st = initial_state("测试", 2, mode="web")
    st["plan"] = [Subtopic(id="1", question="x", status="done")]
    st["report"] = report
    st["sources"] = sources
    return st


def test_empty_title_does_not_inflate_recall():
    # 来源 title 为空、url 不在报告中 → 不应被算作「被引用」
    st = _state_with_report(
        "这是一份足够长的报告内容用于满足最低长度阈值。" * 10,
        [Source(url="https://missing.com", title="")],
    )
    res = evaluate(st)
    assert res.citation_recall == 0.0
    assert res.hallucination_rate == 1.0


def test_url_match_counts_as_referenced():
    st = _state_with_report(
        "结论见 https://real.com 来源。",
        [Source(url="https://real.com", title="R")],
    )
    res = evaluate(st)
    assert res.citation_recall == 1.0
    assert res.hallucination_rate == 0.0


def test_title_match_counts_as_referenced():
    st = _state_with_report(
        "关键结论在「自动驾驶技术」一节。",
        [Source(url="https://x.com", title="自动驾驶技术")],
    )
    res = evaluate(st)
    assert res.citation_recall == 1.0


def test_empty_sources_yields_full_recall_and_no_hallucination():
    st = _state_with_report("报告内容。" * 20, [])
    res = evaluate(st)
    assert res.citation_recall == 1.0
    assert res.hallucination_rate == 0.0

"""评估模块测试：指标计算、范围、真实引用识别、端到端管线。"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ["RAG_BACKEND"] = "memory"

from src.evaluation import evaluate  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.state import Subtopic, initial_state  # noqa: E402


def test_evaluate_keys_and_ranges():
    st = initial_state("测试", 2, mode="web")
    st["plan"] = [Subtopic(id="1", question="x", status="done")]
    st["report"] = "# 报告\n" + "内容足够长用来满足最低长度阈值。" * 20
    st["sources"] = [
        __import__("src.state", fromlist=["Source"]).Source(url="https://a.com", title="A")
    ]
    res = evaluate(st)
    for k in ("task_completion_rate", "citation_accuracy", "hallucination_rate"):
        assert 0.0 <= getattr(res, k) <= 1.0
    assert res.task_completion_rate == 1.0


def test_evaluate_with_real_citation():
    from src.state import Source

    st = initial_state("测试", 2, mode="web")
    st["plan"] = [Subtopic(id="1", question="x", status="done")]
    st["report"] = "结论见 https://real.com 来源。"
    st["sources"] = [Source(url="https://real.com", title="R")]
    res = evaluate(st)
    assert res.citation_accuracy == 1.0


def test_evaluate_fake_citation_hurts_accuracy():
    from src.state import Source

    st = initial_state("测试", 2, mode="web")
    st["plan"] = [Subtopic(id="1", question="x", status="done")]
    st["report"] = "结论见 https://fake-not-in-sources.com 来源。"
    st["sources"] = [Source(url="https://real.com", title="R")]
    res = evaluate(st)
    assert res.citation_accuracy == 0.0


def test_evaluate_real_pipeline_runs():
    r = build_graph().invoke(initial_state("RAG 评估端到端", 1, mode="web"))
    res = evaluate(r)
    assert 0.0 <= res.task_completion_rate <= 1.0
    assert res.to_dict()["sources_total"] >= 1

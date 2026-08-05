"""语义长记忆 + Analyst 自愈回路 单元测试。

CI 下 conftest 已强制 MEMORY_BACKEND=memory，故语义层走内存索引，快速确定、零网络依赖。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.agents.analyst import _critique_analysis, analyst_node
from src.memory import MemoryStore, SemanticIndex, _extract_insights
from src.state import Finding


def test_semantic_index_memory_ranks_related() -> None:
    idx = SemanticIndex(backend="memory")
    a = "Python 异步编程模型与事件循环"
    b = "深度学习中的梯度下降优化算法"
    idx.add(a, {"kind": "insight"})
    idx.add(b, {"kind": "insight"})
    hits = idx.search("异步 事件循环", top_n=3)
    assert a in hits  # 语义更相关的应被召回
    assert idx.backend == "in-memory"


def test_memory_store_save_then_recall() -> None:
    db = Path(tempfile.gettempdir()) / f"dra_mem_{id(Path)}.db"
    if db.exists():
        db.unlink()
    store = MemoryStore(path=str(db), semantic=SemanticIndex(backend="memory"))
    report = (
        "# 报告标题\n"
        "- 自动驾驶依赖多传感器融合与实时决策\n"
        "- 端到端神经网络正在替代模块化流水线\n"
        "> 引用来源示例\n"
        "```code block```\n"
    )
    n = store.save("自动驾驶技术演进", report)
    assert n >= 2  # 至少写入两条洞察
    hits = store.recall("自动驾驶 传感器融合", k=3)
    assert any("传感器融合" in h for h in hits)


def test_extract_insights_skips_noise() -> None:
    report = "# 标题\n> 引用\n```py\nx=1\n```\n- 要点一很重要\n- 要点二也关键\n"
    ins = _extract_insights("主题", report)
    assert "研究主题：主题" in ins  # 主题锚点
    assert any("要点一" in i for i in ins)
    assert all(not i.startswith((">", "```", "#")) for i in ins if not i.startswith("研究主题"))


def test_critique_analysis_ok() -> None:
    findings = [Finding(subtopic_id="s1", summary="异步编程能提升吞吐")]
    analysis = "经分析，子主题 s1 的发现表明异步编程能够显著提升系统吞吐，结论可靠且与其他维度一致。"
    ok, issues = _critique_analysis(findings, analysis)
    assert ok is True
    assert issues == []


def test_critique_analysis_low_confidence() -> None:
    findings = [Finding(subtopic_id="s1", summary="x")]
    analysis = "目前数据不足，无法确定结论。"
    ok, issues = _critique_analysis(findings, analysis)
    assert ok is False
    assert any("不确定" in i or "置信" in i for i in issues)


def test_analyst_node_mock_no_self_heal() -> None:
    state = {
        "query": "测试问题",
        "plan": [],
        "findings": [Finding(subtopic_id="s1", summary="某发现")],
        "analysis": "",
        "report": "",
        "sources": [],
        "iteration": 0,
        "max_iterations": 3,
        "mode": "web",
        "messages": [],
        "memory_recall": [],
        "analyst_self_heal": 0,
        "memory_writes": 0,
    }
    out = analyst_node(state, config={})
    assert "analysis" in out
    # mock 模式不进入自愈循环
    assert out.get("analyst_self_heal", 0) == 0
    # 不应因 mock 分析缺少「缺口」标记而误触发补充回环
    assert out["iteration"] == 1

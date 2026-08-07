"""状态 reducer 测试：_add_sources 按 URL 去重、保序、空安全；
_add_messages_windowed 落实 research_window 滑动窗口。

替代 operator.add 后，跨子主题重复检索同一网页不再堆叠，评估分母不再失真。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from src.state import Source, _add_messages_windowed, _add_sources


def test_dedup_by_url_keeps_first_occurrence_order():
    a = Source(url="https://a.com", title="A")
    b = Source(url="https://b.com", title="B")
    a_dup = Source(url="https://a.com", title="A-dup")  # 同 url，应被丢弃
    out = _add_sources([a, b], [a_dup])
    assert [s.url for s in out] == ["https://a.com", "https://b.com"]
    assert out[0] is a  # 首次出现保留


def test_case_insensitive_url_normalization():
    a = Source(url="https://A.com/x")
    a_lower = Source(url="https://a.com/x")
    out = _add_sources([a], [a_lower])
    assert len(out) == 1


def test_none_inputs_are_safe():
    assert _add_sources(None, None) == []
    s = Source(url="https://x.com")
    assert _add_sources(None, [s]) == [s]
    assert _add_sources([s], None) == [s]


def test_empty_url_sources_are_preserved():
    # 空 url 不入去重集合，但应保留（避免误丢本地/未知来源）
    s1 = Source(url="")
    s2 = Source(url="https://a.com")
    out = _add_sources([s1], [s2])
    assert out == [s1, s2]


def test_messages_window_keeps_only_recent(monkeypatch):
    """research_window 曾是 .env.example 里公开却无人读取的死旋钮，现已真正生效。"""
    from src import config

    monkeypatch.setattr(config.settings, "research_window", 3)
    existing = [AIMessage(content=f"m{i}") for i in range(5)]
    out = _add_messages_windowed(existing, [AIMessage(content="new")])
    assert [m.content for m in out] == ["m3", "m4", "new"]


def test_messages_window_disabled_when_non_positive(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "research_window", 0)
    existing = [AIMessage(content=f"m{i}") for i in range(5)]
    out = _add_messages_windowed(existing, [AIMessage(content="new")])
    assert len(out) == 6


def test_messages_window_none_inputs_are_safe(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "research_window", 10)
    assert _add_messages_windowed(None, None) == []
    out = _add_messages_windowed(None, [AIMessage(content="x")])
    assert [m.content for m in out] == ["x"]

"""配置测试：resolve_settings 按请求注入与全局回落、Settings 空值强制兜底。

pydantic-settings 迁移后，env 里的空串 / 非法值必须回落默认值，
且节点能通过 RunnableConfig 注入按请求配置，不再永远拿全局那一份。
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from src.config import Settings, resolve_settings, settings  # noqa: E402


def test_resolve_settings_falls_back_to_global():
    assert resolve_settings(None) is settings
    assert resolve_settings({}) is settings
    assert resolve_settings({"configurable": {}}) is settings
    # 非 Settings 类型的注入值应忽略，回落全局
    assert resolve_settings({"configurable": {"settings": "not-a-settings"}}) is settings


def test_resolve_settings_uses_injected_settings():
    custom = Settings(llm_provider="openai", openai_api_key="sk-test", model_name="gpt-4o")
    resolved = resolve_settings({"configurable": {"settings": custom}})
    assert resolved is custom
    assert resolved.llm_provider == "openai"


def test_empty_int_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "")
    monkeypatch.setenv("MAX_RESEARCH_ITERATIONS", "")
    monkeypatch.setenv("MEMORY_TOP_K", "")
    monkeypatch.setenv("ANALYST_SELF_HEAL", "")
    monkeypatch.setenv("RESEARCH_WINDOW", "")
    s = Settings()
    assert s.rag_top_k == 4
    assert s.max_research_iterations == 3
    assert s.memory_top_k == 3
    assert s.analyst_self_heal == 2
    assert s.research_window == 10


def test_invalid_int_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "not-a-number")
    assert Settings().rag_top_k == 4


def test_empty_float_env_falls_back(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "")
    assert Settings().temperature == 0.2
    monkeypatch.setenv("TEMPERATURE", "abc")
    assert Settings().temperature == 0.2


def test_memory_enabled_parsing(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    assert Settings().memory_enabled is False
    monkeypatch.setenv("MEMORY_ENABLED", "YES")
    assert Settings().memory_enabled is True
    monkeypatch.setenv("MEMORY_ENABLED", "1")
    assert Settings().memory_enabled is True

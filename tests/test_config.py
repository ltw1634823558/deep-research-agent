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


def test_llm_cache_keyed_by_credential_fingerprint():
    """M-A 回归：不同 api_key 必须拿到不同 LLM 客户端。

    旧实现缓存键只放 has_key 布尔，同 model/base_url 的两个租户会命中同一客户端，
    租户 B 的请求实际用租户 A 的 key 调上游——计费错记、配额互打。
    """
    from src.config import Settings, get_llm

    base = dict(llm_provider="openai", model_name="gpt-4o-mini", openai_base_url="https://api.example.com/v1")
    a = get_llm(Settings(openai_api_key="KEY-TENANT-A", **base))
    b = get_llm(Settings(openai_api_key="KEY-TENANT-B", **base))
    a_again = get_llm(Settings(openai_api_key="KEY-TENANT-A", **base))

    assert a is not b, "不同 api_key 不得共享同一 LLM 客户端"
    assert a is a_again, "相同凭据应命中缓存，避免重复建连接池"


def test_llm_cache_is_bounded():
    """缓存必须有界，否则多租户下每个 key 的 httpx 连接池永久驻留内存。"""
    from src import config
    from src.config import Settings, get_llm

    for i in range(config._LLM_CACHE_MAXSIZE + 12):
        get_llm(
            Settings(
                llm_provider="openai",
                model_name="gpt-4o-mini",
                openai_base_url="https://api.example.com/v1",
                openai_api_key=f"KEY-{i}",
            )
        )
    assert len(config._LLM_CACHE) <= config._LLM_CACHE_MAXSIZE


def test_credential_fingerprint_does_not_leak_plaintext():
    from src.config import _credential_fingerprint

    fp = _credential_fingerprint("sk-super-secret")
    assert "sk-super-secret" not in fp
    assert fp == _credential_fingerprint("sk-super-secret")
    assert fp != _credential_fingerprint("sk-other")

"""LangFuse 可观测层测试：per-request 凭据隔离 + 不污染进程环境变量。

全程用假客户端替身，不产生任何网络请求。
"""

from __future__ import annotations

import os

import langfuse
import langfuse.langchain as lf_langchain

from src import observability
from src.config import Settings


def test_callbacks_use_per_request_credentials(monkeypatch):
    """M-2 回归：按请求 cfg 的 langfuse 凭据必须真正生效，且不写 os.environ。

    旧实现只读全局 settings 并把凭据写进 os.environ：多租户下请求 A 的 trace 会
    落到请求 B 的项目里（隐私串号），并发写环境变量还会互相覆盖。
    """
    seen: dict = {}

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class _FakeHandler:
        def __init__(self, public_key=None):
            seen["handler_public_key"] = public_key

    monkeypatch.setattr(langfuse, "Langfuse", _FakeLangfuse)
    monkeypatch.setattr(lf_langchain, "CallbackHandler", _FakeHandler)

    # 全局配置为空，证明凭据只可能来自 per-request cfg
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")

    env_before = os.environ.get("LANGFUSE_PUBLIC_KEY")
    cfg = Settings(
        langfuse_public_key="pk-tenant-a",
        langfuse_secret_key="sk-tenant-a",
        langfuse_host="https://tenant-a.example.com",
    )
    handlers = observability.get_callbacks(cfg)

    assert len(handlers) == 1
    assert seen["public_key"] == "pk-tenant-a"
    assert seen["secret_key"] == "sk-tenant-a"
    assert seen["host"] == "https://tenant-a.example.com"
    assert seen["handler_public_key"] == "pk-tenant-a"
    # 关键：进程级环境变量必须原封不动
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == env_before


def test_callbacks_empty_without_credentials(monkeypatch):
    """未配置凭据时回调为空列表，对业务零侵入。"""
    monkeypatch.setattr(observability.settings, "langfuse_public_key", "")
    monkeypatch.setattr(observability.settings, "langfuse_secret_key", "")
    assert observability.get_callbacks() == []
    assert observability.get_langfuse_client() is None
    assert observability.get_callbacks(Settings(langfuse_public_key="", langfuse_secret_key="")) == []

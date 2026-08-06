"""LangFuse 可观测集成：配置 key 后自动追踪每个节点的调用链路与 token 消耗。

不配置则回调为空列表，对业务零侵入。
调用方每次运行都应通过 get_callbacks() 现取一份全新的 handler 列表，
再以 config={'callbacks': get_callbacks()} 注入——共享同一个 handler 会让
并发任务的 trace 相互串扰。
"""

from __future__ import annotations

from .config import settings


def get_callbacks() -> list:
    """每次调用返回一份全新的回调列表（每个任务独立，避免 trace 串扰）。"""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse.callback import CallbackHandler

        return [
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        ]
    except Exception:
        return []


def get_langfuse_client():
    """返回 LangFuse 客户端（用于写入评估 score）；未配置则返回 None。"""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None


# 向后兼容：仍可 `from .observability import callbacks`，但新代码请用 get_callbacks()。
callbacks = get_callbacks()

"""LangFuse 可观测集成：配置 key 后自动追踪每个节点的调用链路与 token 消耗。

不配置则回调为空列表，对业务零侵入。节点调用 LLM 时通过 config={'callbacks': callbacks} 注入。
"""

from __future__ import annotations

from .config import settings


def get_callbacks():
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


callbacks = get_callbacks()

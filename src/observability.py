"""LangFuse 可观测集成：配置 key 后自动追踪每个节点的调用链路与 token 消耗。

不配置则回调为空列表，对业务零侵入。
调用方每次运行都应通过 get_callbacks() 现取一份全新的 handler 列表，
再以 config={'callbacks': get_callbacks(cfg)} 注入——共享同一个 handler 会让
并发任务的 trace 相互串扰。

凭据一律走 `Langfuse(...)` 构造参数显式传入，**不再写 os.environ**：
langfuse v4 内部按 public_key 维护客户端注册表，因此同一份 key 重复构造会复用
资源；而写环境变量是进程级副作用，多租户并发下后一个请求会覆盖前一个请求的凭据，
导致 trace 串到别的项目。
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from .config import Settings, settings

logger = logging.getLogger(__name__)

# langfuse v4 的客户端会常驻一组 OTel batch processor 线程（实测约 4 线程 + 一个 httpx 池），
# 且按 public_key 长期持有、GC 不回收。多租户下「每来一个新租户就 new 一个」等于线程泄漏，
# 几百租户即触顶 `can't start new thread`。故在本层做有界 LRU，淘汰时显式 shutdown。
_LF_CACHE_MAXSIZE = 8
_LF_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_LF_LOCK = threading.Lock()


def _shutdown_client(client) -> None:
    """尽力释放 OTel processor 线程与连接池；失败不影响主流程。"""
    for method in ("flush", "shutdown"):
        try:
            fn = getattr(client, method, None)
            if callable(fn):
                fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangFuse 客户端 %s() 失败（已忽略）：%s", method, exc)


def _client_for(cfg: "Settings | None" = None):
    """按（per-request 优先的）配置注册并返回 LangFuse 客户端；未配置返回 None。"""
    s = cfg or settings
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    key = (s.langfuse_public_key, s.langfuse_secret_key, s.langfuse_host or "")
    with _LF_LOCK:
        cached = _LF_CACHE.get(key)
        if cached is not None:
            _LF_CACHE.move_to_end(key)
            return cached
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host or None,
        )
    except Exception as exc:  # noqa: BLE001
        # 不要静默：host 拼错 / key 非法时运维需要知道「追踪已被关闭」
        logger.warning("LangFuse 客户端初始化失败，追踪已禁用：%s", exc)
        return None

    evicted = []
    with _LF_LOCK:
        existing = _LF_CACHE.get(key)
        if existing is not None:
            _LF_CACHE.move_to_end(key)
            evicted.append(client)  # 并发重复构建，关掉自己这份
            client = existing
        else:
            _LF_CACHE[key] = client
            while len(_LF_CACHE) > _LF_CACHE_MAXSIZE:
                _, old = _LF_CACHE.popitem(last=False)
                evicted.append(old)
    for old in evicted:
        _shutdown_client(old)
    return client


def get_callbacks(cfg: "Settings | None" = None) -> list:
    """每次调用返回一份全新的回调列表（每个任务独立，避免 trace 串扰）。

    `cfg` 为按请求注入的配置；缺省回落全局 settings。必须先注册对应 public_key 的
    客户端，`CallbackHandler(public_key=...)` 才能取到正确项目的凭据。
    """
    s = cfg or settings
    client = _client_for(s)
    if client is None:
        return []
    try:
        # langfuse v4：langchain CallbackHandler 按 public_key 从注册表取客户端
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler(public_key=s.langfuse_public_key)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LangFuse CallbackHandler 创建失败，本次运行不追踪：%s", exc)
        return []


def get_langfuse_client(cfg: "Settings | None" = None):
    """返回 LangFuse 客户端（用于写入评估 score）；未配置则返回 None。"""
    return _client_for(cfg)


def shutdown_all_clients() -> None:
    """进程退出前 flush 并关闭全部 LangFuse 客户端。

    langfuse v4 的 OTel BatchSpanProcessor 是「攒批后异步上报」：默认要等批次
    写满或调度间隔到点才发。进程若直接退出，最后一批 span 会连同刚跑完那次
    研究的完整链路一起丢失（生产表现为「最近几个请求在 LangFuse 里查不到」）。
    必须由应用的 shutdown 钩子显式调用。幂等：调用后缓存清空，再次调用无副作用。
    """
    with _LF_LOCK:
        clients = list(_LF_CACHE.values())
        _LF_CACHE.clear()
    for client in clients:
        _shutdown_client(client)


# 注意：此处不再提供模块级 `callbacks` 变量。
# 它全库无人读取，却会在**导入期**就构造 Langfuse 客户端（多开 4 个 OTel 线程 +
# 一个 httpx 连接池），属于纯粹的副作用。请一律使用 get_callbacks(cfg)。

"""MCP Client：经 stdio 拉起 MCP Server（默认本项目自带的 Tavily MCP 服务），
调用其 tavily_search 工具，并把结果转成项目的 Source 列表。

也可指向任意兼容 MCP 的搜索服务（如官方 `npx -y tavily-mcp`），只需改 mcp_server_command。
任何异常都会降级为 mock，保证检索失败也不阻断研究管线。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from mcp import ClientSession

from ..config import Settings, settings
from ..state import Source

# ExceptionGroup is a builtin only since Python 3.11. The project supports 3.10+,
# so fall back to the backport or a lightweight shim when running on 3.10.
if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup
else:
    try:
        from exceptiongroup import ExceptionGroup
    except ImportError:
        # Fallback shim so the rest of the module works on 3.10 without the dep
        class ExceptionGroup(Exception):
            def __init__(self, message, exceptions):
                super().__init__(message)
                self.exceptions = list(exceptions)

            def __str__(self):
                return f"{super().__str__()} ({len(self.exceptions)} sub-exceptions)"

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_command() -> list[str]:
    """默认拉起本项目自带的 MCP Server；用当前解释器，保证依赖与路径一致。"""
    return [sys.executable, "-m", "src.mcp.server"]


def _resolve_command(settings_obj: "Settings | None" = None) -> list[str]:
    s = settings_obj or settings
    raw = (s.mcp_server_command or "").strip()
    if not raw:
        return _default_command()
    parts = raw.split()
    if parts[0] in ("python", "python3"):
        parts[0] = sys.executable  # 统一到当前 venv 解释器
    return parts


async def _call_tool(command: list[str], query: str, max_results: int, settings_obj: "Settings | None" = None) -> str:
    s = settings_obj or settings
    # 把 per-request 的 key 注入子进程环境，确保 MCP Server 侧按请求凭据生效（M-B）
    mcp_env = {**os.environ}
    if s.tavily_api_key:
        mcp_env["TAVILY_API_KEY"] = s.tavily_api_key
    else:
        # 按请求「无 key」的租户不应继承进程级 TAVILY_API_KEY（L6）。
        # 注意：这里必须置空而不能 pop —— MCP Server 子进程启动时会 load_dotenv()，
        # 若变量缺失，.env 里的 key 会被重新注入；置空则因 override=False 而保留空值，
        # 子进程稳定走 mock 分支，凭据隔离才真正成立。
        mcp_env["TAVILY_API_KEY"] = ""
    params = StdioServerParameters(
        command=command[0],
        args=command[1:],
        cwd=str(PROJECT_ROOT),
        env=mcp_env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(
            "tavily_search", {"query": query, "max_results": max_results}
        )
        # 如果 MCP Server 侧工具执行失败，直接把原始错误抛给上层
        if getattr(result, "isError", False):
            text = ""
            for item in result.content:
                if isinstance(item, TextContent):
                    text = item.text
                    break
            raise RuntimeError(f"MCP tool tavily_search failed: {text}")
        # 安全取首个文本内容（工具返回的是 JSON 字符串），兼容 union 内容类型
        for item in result.content:
            if isinstance(item, TextContent):
                return item.text
        return ""


# MCP 子进程调用整体超时：避免子进程挂起永久占住 job 线程（L-9）
_MCP_JOIN_TIMEOUT = 30.0
# 留给「取消协程 -> 关 stdin -> 终止子进程树」的清理时间；join 超时 = 上者 + 本值
_JOIN_GRACE = 5.0


async def _with_timeout(coro):
    """协程级超时：取消会传播到 `stdio_client` 的 __aexit__，由它终止子进程。

    这一层是超时的**主**防线。`_run_in_thread` 的 join 超时只是兜底——它无法
    杀死已挂起的线程，只有协程被取消才能真正回收子进程与句柄。
    """
    try:
        return await asyncio.wait_for(coro, timeout=_MCP_JOIN_TIMEOUT)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"MCP 检索调用超时（>{_MCP_JOIN_TIMEOUT}s），子进程已终止"
        ) from exc


def _run_in_thread(coro):
    """在独立后台线程里跑一个全新的事件循环，避免「已运行循环中再 asyncio.run」的 RuntimeError。

    仍保持每次调用启动子进程（subprocess-per-call）模型；不实现共享长连接会话。
    带整体超时，防止 MCP 子进程挂起永久阻塞调用方线程。
    """
    result_holder: dict = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            result_holder["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 - 跨线程透传给主线程
            result_holder["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    # join 必须比协程级超时更宽松，否则它总是先到期，_with_timeout 的优雅路径
    # （取消 -> stdio_client 清理 -> 杀子进程）永远拿不到机会，worker 线程被白白遗弃。
    thread.join(_MCP_JOIN_TIMEOUT + _JOIN_GRACE)
    if thread.is_alive():
        raise RuntimeError(f"MCP 检索调用超时（>{_MCP_JOIN_TIMEOUT}s），子进程可能已挂起")
    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["value"]


def _run_coro(coro):
    """循环安全执行协程：若已在事件循环中，则放到后台线程跑；否则用 asyncio.run。

    两条分支都必须包 `_with_timeout`。FastAPI 的同步端点与 `_run_job` 后台线程里
    都**没有** running loop，走的正是 `asyncio.run` 分支；若只给线程分支加超时，
    一次挂起的 MCP 子进程就能永久占住 job 线程，4 个并发即打满线程池导致全站饿死。
    """
    guarded = _with_timeout(coro)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(guarded)
    return _run_in_thread(guarded)


def _unwrap_runtime_error(exc: BaseException) -> BaseException:
    """递归解包 asyncio ExceptionGroup，找到最内层的 RuntimeError。"""
    if isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
        inner = exc.exceptions[0]
        if isinstance(inner, RuntimeError):
            return inner
        if isinstance(inner, ExceptionGroup):
            return _unwrap_runtime_error(inner)
    return exc


def _root_cause_message(exc: BaseException) -> str:
    """递归从 ExceptionGroup / asyncio 包装里提取人类可读的根本原因。"""
    if isinstance(exc, ExceptionGroup):
        causes = [_root_cause_message(sub) for sub in exc.exceptions]
        return "; ".join(c for c in causes if c)
    return f"{type(exc).__name__}: {exc}"


def search_via_mcp(query: str, max_results: int = 5, settings_obj: "Settings | None" = None) -> list[Source]:
    """经 MCP Server 检索，返回 Source 列表；失败直接抛异常，不再静默降级 mock。

    settings_obj 透传按请求配置，使 per-request mcp_server_command 真正生效。
    """
    command = _resolve_command(settings_obj)
    try:
        raw = _run_coro(_call_tool(command, query, max_results, settings_obj))
        data = json.loads(raw)
        return [
            Source(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", "")[:300],
            )
            for r in data
        ]
    except Exception as e:
        # 跨线程的异常已是原始异常；asyncio.run 可能把异常包在 ExceptionGroup 里，解包后若已是 RuntimeError 直接抛出
        inner = _unwrap_runtime_error(e)
        if isinstance(inner, RuntimeError):
            logger.error("MCP 检索失败: %s", inner)
            raise inner from e
        msg = _root_cause_message(e)
        logger.error("MCP 检索失败: %s", msg)
        raise RuntimeError(f"MCP 检索失败: {msg}") from e

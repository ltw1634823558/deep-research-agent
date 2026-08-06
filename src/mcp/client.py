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

from ..config import settings
from ..state import Source

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_command() -> list[str]:
    """默认拉起本项目自带的 MCP Server；用当前解释器，保证依赖与路径一致。"""
    return [sys.executable, "-m", "src.mcp.server"]


def _resolve_command() -> list[str]:
    raw = (settings.mcp_server_command or "").strip()
    if not raw:
        return _default_command()
    parts = raw.split()
    if parts[0] in ("python", "python3"):
        parts[0] = sys.executable  # 统一到当前 venv 解释器
    return parts


async def _call_tool(command: list[str], query: str, max_results: int) -> str:
    params = StdioServerParameters(
        command=command[0],
        args=command[1:],
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
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


def _run_in_thread(coro):
    """在独立后台线程里跑一个全新的事件循环，避免「已运行循环中再 asyncio.run」的 RuntimeError。

    仍保持每次调用启动子进程（subprocess-per-call）模型；不实现共享长连接会话。
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

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["value"]


def _run_coro(coro):
    """循环安全执行协程：若已在事件循环中，则放到后台线程跑；否则用 asyncio.run。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _run_in_thread(coro)


def _unwrap_runtime_error(exc: BaseException) -> BaseException:
    """递归解包 asyncio ExceptionGroup，找到最内层的 RuntimeError。"""
    from builtins import ExceptionGroup

    if isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
        inner = exc.exceptions[0]
        if isinstance(inner, RuntimeError):
            return inner
        if isinstance(inner, ExceptionGroup):
            return _unwrap_runtime_error(inner)
    return exc


def _root_cause_message(exc: BaseException) -> str:
    """递归从 ExceptionGroup / asyncio 包装里提取人类可读的根本原因。"""
    from builtins import ExceptionGroup

    if isinstance(exc, ExceptionGroup):
        causes = [_root_cause_message(sub) for sub in exc.exceptions]
        return "; ".join(c for c in causes if c)
    return f"{type(exc).__name__}: {exc}"


def search_via_mcp(query: str, max_results: int = 5) -> list[Source]:
    """经 MCP Server 检索，返回 Source 列表；失败直接抛异常，不再静默降级 mock。"""
    command = _resolve_command()
    try:
        raw = _run_coro(_call_tool(command, query, max_results))
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

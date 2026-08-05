"""MCP 接入测试：MCP Server 工具 + Agent 经 MCP 协议调用搜索。

全部离线可跑（无 TAVILY_API_KEY 时走 mock 降级），CI 友好。
"""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from src.mcp.client import search_via_mcp
from src.mcp.server import _tavily_search, mcp
from src.state import Source
from src.tools.search import search


def test_mcp_server_tool_in_memory():
    """in-memory 直连 Server 对象，验证工具暴露与返回结构。"""

    async def _run():
        async with Client(mcp) as client:
            res = await client.call_tool(
                "tavily_search", {"query": "Agent 工程化", "max_results": 2}
            )
            return json.loads(res.content[0].text)

    data = asyncio.run(_run())
    assert isinstance(data, list) and len(data) == 2
    assert "url" in data[0] and data[0]["title"].startswith("[mock]")


def test_server_mock_fallback_without_key(monkeypatch):
    """无 key 时 Server 内部降级 mock，保证离线可演示。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    data = _tavily_search("x", 3)
    assert len(data) == 3
    assert data[0]["title"].startswith("[mock]")


def test_search_dispatcher_routes_to_mcp(monkeypatch):
    """search() 在 search_provider=mcp 时路由到 MCP Server。"""
    monkeypatch.setattr("src.tools.search.settings.search_provider", "mcp")
    fixed = [Source(url="u", title="t", snippet="s")]
    monkeypatch.setattr("src.tools.search.search_via_mcp", lambda q, max_results=5: fixed)
    assert search("anything") is fixed


def test_search_dispatcher_routes_to_tavily(monkeypatch):
    """search() 在 search_provider=tavily（默认）时直连 Tavily。"""
    monkeypatch.setattr("src.tools.search.settings.search_provider", "tavily")
    monkeypatch.setattr(
        "src.tools.search.web_search", lambda q, top_k=5: [Source(url="w", title="w")]
    )
    out = search("anything")
    assert out[0].url == "w"


def test_mcp_client_stdio_roundtrip(monkeypatch):
    """真实拉起 MCP Server 子进程（stdio），端到端验证检索链路。

    显式把 TAVILY_API_KEY 置空（而非 delenv）：MCP Server 子进程会自行
    load_dotenv() 重读 .env，若仅 delenv，子进程会从 .env 重新注入 key，
    在「本机/CI 已存在有效 key」的环境下会返回真实结果、破坏 mock 断言。
    置空后 load_dotenv() 默认不覆盖已存在的空值，子进程稳定走 mock 分支。
    """
    monkeypatch.setenv("TAVILY_API_KEY", "")
    out = search_via_mcp("测试查询", max_results=2)
    assert len(out) == 2
    assert all(isinstance(s, Source) for s in out)
    assert out[0].title.startswith("[mock]")

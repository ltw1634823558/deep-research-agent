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
    monkeypatch.setattr("src.tools.search.search_via_mcp", lambda q, max_results=5, settings_obj=None: fixed)
    assert search("anything") is fixed


def test_search_dispatcher_routes_to_tavily(monkeypatch):
    """search() 在 search_provider=tavily（默认）时直连 Tavily。"""
    monkeypatch.setattr("src.tools.search.settings.search_provider", "tavily")
    monkeypatch.setattr(
        "src.tools.search.web_search", lambda q, top_k=5, **kwargs: [Source(url="w", title="w")]
    )
    out = search("anything")
    assert out[0].url == "w"


def test_mcp_client_stdio_roundtrip(monkeypatch):
    """真实拉起 MCP Server 子进程（stdio），端到端验证检索链路。

    凭据来源以 Settings 为准（M-B：per-request key 注入子进程 env），所以这里必须
    把 settings.tavily_api_key 置空，仅 setenv 已无效。客户端会把 TAVILY_API_KEY=""
    显式写进子进程环境；子进程 load_dotenv() 默认 override=False，不会用 .env 覆盖
    已存在的空值，因此稳定走 mock 分支——在本机/CI 已配置真实 key 时也不会联网。
    """
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setattr("src.mcp.client.settings.tavily_api_key", "")
    out = search_via_mcp("测试查询", max_results=2)
    assert len(out) == 2
    assert all(isinstance(s, Source) for s in out)
    assert out[0].title.startswith("[mock]")


def test_mcp_env_isolation_blocks_process_key(monkeypatch):
    """L6 回归：按请求无 key 时，子进程 env 必须被显式置空（而非继承/pop）。

    pop 会让子进程 load_dotenv() 从 .env 重新注入 key，导致租户凭据串用。
    """
    import asyncio

    from src.config import Settings
    from src.mcp import client as mcp_client

    monkeypatch.setenv("TAVILY_API_KEY", "process-level-secret")
    captured: dict = {}

    class _FakeParams:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mcp_client, "StdioServerParameters", _FakeParams)

    def _boom(*a, **kw):
        raise RuntimeError("stop-after-env-build")

    monkeypatch.setattr(mcp_client, "stdio_client", _boom)

    no_key = Settings(tavily_api_key="")
    try:
        asyncio.run(mcp_client._call_tool(["x"], "q", 1, no_key))
    except RuntimeError:
        pass
    assert captured["env"]["TAVILY_API_KEY"] == ""

    with_key = Settings(tavily_api_key="tenant-key")
    try:
        asyncio.run(mcp_client._call_tool(["x"], "q", 1, with_key))
    except RuntimeError:
        pass
    assert captured["env"]["TAVILY_API_KEY"] == "tenant-key"


def test_mcp_timeout_applies_without_running_loop(monkeypatch):
    """H-1 回归：无 running loop（FastAPI 同步端点 / job 后台线程）时也必须超时。

    此前 `_run_coro` 只给线程分支加超时，`asyncio.run` 分支裸跑——一次挂起的 MCP
    子进程即可永久占住 job 线程，4 个并发就能打满线程池让全站饿死。
    """
    import asyncio
    import time

    from src.mcp import client as mcp_client

    monkeypatch.setattr(mcp_client, "_MCP_JOIN_TIMEOUT", 0.3)

    async def _hang():
        await asyncio.sleep(30)

    started = time.monotonic()
    try:
        mcp_client._run_coro(_hang())
        raise AssertionError("挂起的调用必须超时，不应正常返回")
    except RuntimeError as exc:
        assert "超时" in str(exc)
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"超时未生效，耗时 {elapsed:.1f}s"

"""MCP Server：把 Tavily 搜索封装为 MCP 工具，供 Agent 通过 MCP 协议（stdio）调用。

这是进阶路线里「接入 MCP Server」的落地：Agent 不再直接 import 第三方 SDK，
而是通过标准 MCP 协议消费搜索能力——可替换为任意兼容 MCP 的搜索服务（含官方 Tavily MCP）。

运行方式：
    python -m src.mcp.server          # stdio 模式，供 MCP Client 拉起
模块对象 `mcp` 也可被 in-memory 测试直接复用（见 tests/test_mcp.py）。

无 TAVILY_API_KEY 时自动降级 mock 结果，保证离线可跑、可演示、可 CI。
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

# 作为独立子进程拉起时，确保从 .env 读取 TAVILY_API_KEY（server 进程是全新 Python 环境）
load_dotenv()

logger = logging.getLogger(__name__)
mcp = FastMCP("deep-research-tavily")


def _mock_search(query: str, max_results: int) -> list[dict]:
    """离线 mock 检索：仅在没有 TAVILY_API_KEY 时作为兜底，保证离线可跑。"""
    return [
        {
            "url": f"https://example.com/result/{i + 1}",
            "title": f"[mock] 关于「{query}」的检索结果 {i + 1}",
            "content": "这是 MCP 服务离线返回的 mock 摘要（配置 TAVILY_API_KEY 后即为真实联网结果）。",
        }
        for i in range(max_results)
    ]


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """真实走 Tavily API；无 key 时降级为 mock，有 key 时失败直接抛异常不静默。"""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        logger.info("MCP Server: TAVILY_API_KEY 未配置，返回离线 mock 检索结果。")
        return _mock_search(query, max_results)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=max_results)
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": (r.get("content") or "")[:500],
            }
            for r in resp.get("results", [])
        ]
    except Exception as e:
        logger.error(
            "MCP Server: Tavily 联网检索失败（已配置 TAVILY_API_KEY，但调用报错 %s: %s）。"
            "不再静默返回 mock，请检查 key 是否有效或网络是否可达。",
            type(e).__name__, e,
        )
        raise


def _tavily_extract(url: str) -> str:
    """抓取并提取指定 URL 的正文；无 key 时返回 mock 提示，有 key 时失败抛异常。"""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return f"[mock] 无法提取 {url}（配置 TAVILY_API_KEY 后返回真实正文）"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        resp = client.extract(urls=[url])
        results = resp.get("results", [])
        if results:
            return results[0].get("raw_content") or results[0].get("content") or ""
        return ""
    except Exception as e:
        logger.error(
            "MCP Server: Tavily extract 失败（%s: %s）。不再静默返回 mock。",
            type(e).__name__, e,
        )
        raise


@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> str:
    """通过 Tavily 进行联网检索，返回 JSON 字符串列表，每项含 url/title/content。"""
    return json.dumps(_tavily_search(query, max_results), ensure_ascii=False)


@mcp.tool()
def tavily_extract(url: str) -> str:
    """抓取并提取指定 URL 的正文内容。"""
    return _tavily_extract(url)


if __name__ == "__main__":
    # stdio 模式：被 MCP Client 以子进程方式拉起
    mcp.run()

"""联网检索工具（Tool Use / Function Calling 的落点之一）。

对外统一暴露 `search()`：根据 settings.search_provider 路由——
- "tavily"：直连 Tavily（含 mock 降级），保持向后兼容
- "mcp"：经 MCP Server（本项目自带的 Tavily MCP 服务）调用，对上层透明

无论哪种后端，无 key / 异常时都降级 mock，保证离线可跑、可测、不阻断管线。
"""
from __future__ import annotations

import logging

from ..config import settings
from ..mcp.client import search_via_mcp
from ..state import Source

logger = logging.getLogger(__name__)


def _mock_search(query: str, top_k: int) -> list[Source]:
    """离线 mock 检索：仅在没有 TAVILY_API_KEY 时作为兜底。"""
    return [
        Source(
            url=f"https://example.com/result/{i+1}",
            title=f"[mock] 关于「{query}」的检索结果 {i+1}",
            snippet=f"这是离线 mock 检索返回的第 {i+1} 条摘要，配置 TAVILY_API_KEY 后即为真实联网结果。",
        )
        for i in range(top_k)
    ]


def web_search(query: str, top_k: int = 5) -> list[Source]:
    """直连 Tavily；配置了 key 则必须真实联网，失败直接抛异常，不再静默 mock。"""
    key = settings.tavily_api_key.strip()
    if not key:
        logger.info("未配置 TAVILY_API_KEY，使用离线 mock 检索结果（非真实联网）。")
        return _mock_search(query, top_k)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        resp = client.search(query=query, max_results=top_k)
        # 兼容 dict 与 TavilyResponse 对象两种返回形态
        if isinstance(resp, dict):
            results = resp.get("results", [])
        else:
            results = getattr(resp, "results", []) or []
        return [
            Source(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", "")[:300],
            )
            for r in results
        ]
    except Exception as e:
        logger.error(
            "Tavily 联网检索失败（已配置 TAVILY_API_KEY，但调用报错 %s: %s）。"
            "不再静默返回 mock，请检查 key 是否有效 / 网络是否可达。",
            type(e).__name__, e,
        )
        raise


def search(query: str, top_k: int = 5) -> list[Source]:
    """统一检索入口：按 search_provider 路由到 Tavily 直连或 MCP Server。"""
    if settings.search_provider == "mcp":
        return search_via_mcp(query, max_results=top_k)
    return web_search(query, top_k)


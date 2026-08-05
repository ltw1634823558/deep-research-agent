"""网页抓取工具：把检索到的 URL 抓成正文，喂给 LLM 做摘要与引用。

无网络环境下降级为 mock 正文。可替换为 Jina Reader / FireCrawl 等。
"""

from __future__ import annotations

import requests

from ..config import settings


def fetch_url(url: str, timeout: int = 10) -> str:
    # example.com 是项目内置 mock 域名：无论是否配置 key，都直接返回占位正文，
    # 避免「配置了 Tavily key 但离线」时真实联网抓取挂起（mock 检索回退的 URL 即为此域名）。
    if "example.com" in url:
        return f"[mock] 抓取到的网页正文（{url}）：本段为离线占位内容，用于演示工具调用链路。"
    if not settings.tavily_api_key:
        return f"[mock] 抓取到的网页正文（{url}）：本段为离线占位内容（未配置 Tavily key）。"

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "research-agent/1.0"})
        resp.raise_for_status()
        # 极简清洗：真实项目应做 HTML→正文抽取（trafilatura/readsbid）
        text = resp.text
        return text[:4000]
    except Exception:
        return f"[抓取失败] {url}"

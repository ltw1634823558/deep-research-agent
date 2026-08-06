"""网页抓取工具：把检索到的 URL 抓成正文，喂给 LLM 做摘要与引用。

无网络环境下降级为 mock 正文。可替换为 Jina Reader / FireCrawl 等。
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import requests

from ..config import settings

# 内网/本机域名后缀：这些主机名解析后指向内部服务，一律不抓
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal")
_BLOCKED_NAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def _host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _is_mock_host(host: str) -> bool:
    """精确匹配内置 mock 域名，避免 evil-example.com.attacker.net 这类子串绕过。"""
    return host == "example.com" or host.endswith(".example.com")


def _is_internal_host(host: str) -> bool:
    """SSRF 基础防护：环回 / 私网 / 链路本地 / 保留地址 / 本机域名一律拒绝。"""
    if not host:
        return True
    if host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # 普通域名，交给正常抓取流程
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


def fetch_url(url: str, timeout: int = 10) -> str:
    host = _host_of(url)

    # SSRF 防护：内网/本机地址直接放弃，抓取器永不触达内部服务
    if _is_internal_host(host):
        return ""

    # example.com 是项目内置 mock 域名：无论是否配置 key，都直接返回占位正文，
    # 避免「配置了 Tavily key 但离线」时真实联网抓取挂起（mock 检索回退的 URL 即为此域名）。
    if _is_mock_host(host):
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

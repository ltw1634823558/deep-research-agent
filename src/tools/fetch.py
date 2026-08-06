"""网页抓取工具：把检索到的 URL 抓成正文，喂给 LLM 做摘要与引用。

无网络环境下降级为 mock 正文。可替换为 Jina Reader / FireCrawl 等。
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import requests

from ..config import settings

logger = logging.getLogger(__name__)

# 内网/本机域名后缀：这些主机名解析后指向内部服务，一律不抓
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal")
_BLOCKED_NAMES = {"localhost", "ip6-localhost", "ip6-loopback"}
_ALLOWED_SCHEMES = {"http", "https"}


def _host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _is_mock_host(host: str) -> bool:
    """精确匹配内置 mock 域名，避免 evil-example.com.attacker.net 这类子串绕过。"""
    return host == "example.com" or host.endswith(".example.com")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """环回 / 私网 / 链路本地 / 保留 / 组播 / 未指定地址一律视为内网目标。"""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """把主机名解析成「字面量 IP」，兼容十进制/十六进制/八进制等 IPv4 变体。

    2130706433 / 0x7f000001 / 0177.0.0.1 都等价于 127.0.0.1，
    直接丢给 ipaddress.ip_address() 会抛 ValueError，绝不能因此放行。
    返回 None 表示「这不是 IP 字面量」，需要走 DNS 解析校验。
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        # inet_aton 认得 inet_addr 家族的所有历史写法（纯十进制/十六进制/八进制/短点分）
        return ipaddress.ip_address(socket.inet_aton(host))
    except (OSError, UnicodeError, ValueError):
        return None


def _resolve_ips(host: str) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """DNS 解析出全部 A/AAAA 记录，用于识别「域名指向内网」的 SSRF。"""
    ips: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP):
        addr = str(info[4][0]).split("%", 1)[0]  # 去掉 IPv6 的 %scope 后缀
        ips.add(ipaddress.ip_address(addr))
    return ips


def _block_reason(url: str, host: str) -> str | None:
    """返回拒绝原因；None 表示这个目标可以抓。任何异常都 fail-closed（拒绝）。"""
    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return f"协议不在白名单内: {scheme or '(空)'}"
    if not host:
        return "URL 缺少主机名"
    if host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        return "命中内网主机名黑名单"

    literal = _literal_ip(host)
    if literal is not None:
        if _is_blocked_ip(literal):
            return f"目标是内网 IP 字面量: {literal}"
        return None

    try:
        ips = _resolve_ips(host)
    except Exception as exc:  # DNS 失败时宁可不抓，也不能放行
        return f"DNS 解析失败: {exc}"
    if not ips:
        return "DNS 无解析结果"
    blocked = sorted(str(ip) for ip in ips if _is_blocked_ip(ip))
    if blocked:
        return f"域名解析到内网 IP: {', '.join(blocked)}"
    return None


def fetch_url(url: str, timeout: int = 10) -> str:
    host = _host_of(url)

    # 先做廉价的主机名黑名单，再走 mock 短路，避免离线环境下 mock 域名被 DNS 校验误伤
    if not host or host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        logger.warning("SSRF 防护：拒绝抓取 %s（命中内网主机名黑名单）", url)
        return ""

    # example.com 是项目内置 mock 域名：无论是否配置 key，都直接返回占位正文，
    # 避免「配置了 Tavily key 但离线」时真实联网抓取挂起（mock 检索回退的 URL 即为此域名）。
    if _is_mock_host(host):
        return f"[mock] 抓取到的网页正文（{url}）：本段为离线占位内容，用于演示工具调用链路。"
    if not settings.tavily_api_key:
        return f"[mock] 抓取到的网页正文（{url}）：本段为离线占位内容（未配置 Tavily key）。"

    # SSRF 防护：协议白名单 + IP 字面量归一化 + DNS 解析结果全量校验
    reason = _block_reason(url, host)
    if reason is not None:
        logger.warning("SSRF 防护：拒绝抓取 %s（%s）", url, reason)
        return ""

    try:
        # 关键：不跟随重定向。否则 302 到 169.254.169.254 等元数据服务即可绕过上面所有校验。
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "research-agent/1.0"},
            allow_redirects=False,
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            logger.warning(
                "SSRF 防护：拒绝跟随重定向 %s -> %s（HTTP %s）",
                url,
                resp.headers.get("Location", "(无 Location)"),
                resp.status_code,
            )
            return ""
        resp.raise_for_status()
        # 极简清洗：真实项目应做 HTML→正文抽取（trafilatura/readsbid）
        text = resp.text
        return text[:4000]
    except Exception:
        return f"[抓取失败] {url}"

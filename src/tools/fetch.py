"""网页抓取工具：把检索到的 URL 抓成正文，喂给 LLM 做摘要与引用。

无网络环境下降级为 mock 正文。可替换为 Jina Reader / FireCrawl 等。
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urlparse

import requests

from ..config import Settings, settings

logger = logging.getLogger(__name__)

# —— SSRF 防 DNS rebinding：把校验通过的安全 IP 钉死到连接，避免二次解析绕过 ——
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter


def _numeric_timeout(timeout) -> float:
    """urllib3 可能把 timeout 设为哨兵对象，socket 只认数字。"""
    return timeout if isinstance(timeout, (int, float)) else 10.0


def _sock_family(ip_str: str) -> int:
    """根据 IP 字面量判断地址族，连接阶段不再走 getaddrinfo。"""
    try:
        return socket.AF_INET6 if ipaddress.ip_address(ip_str).version == 6 else socket.AF_INET
    except ValueError:
        return socket.AF_INET


def _pinned_socket(conn) -> socket.socket:
    """直接连到已校验的安全 IP：用原始 socket 连接，绝不再调 getaddrinfo，
    否则攻击者可借「连接时二次解析」把域名 rebinding 到内网/元数据服务。

    同时保留 urllib3 原生 `_new_conn` 的行为：套用 `socket_options`（默认含
    TCP_NODELAY）、绑定 `source_address`，并把 timeout 归一为数字（None 会永久阻塞）。
    """
    ip = conn.pinned_ip
    sock = socket.socket(_sock_family(ip), socket.SOCK_STREAM)
    try:
        for opt in getattr(conn, "socket_options", None) or ():
            sock.setsockopt(*opt)
        sock.settimeout(_numeric_timeout(conn.timeout))
        source_address = getattr(conn, "source_address", None)
        if source_address:
            sock.bind(source_address)
        sock.connect((ip, conn.port))
    except Exception:
        sock.close()
        raise
    return sock


class _PinnedHTTPConnection(HTTPConnection):
    pinned_ip = None

    def _new_conn(self):
        if self.pinned_ip is None:
            return super()._new_conn()
        return _pinned_socket(self)


class _PinnedHTTPSConnection(HTTPSConnection):
    pinned_ip = None

    def _new_conn(self):
        if self.pinned_ip is None:
            return super()._new_conn()
        return _pinned_socket(self)


class _PinnedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PinnedHTTPConnection

    def _new_conn(self):
        conn = super()._new_conn()
        conn.pinned_ip = self.pinned_ip
        return conn


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PinnedHTTPSConnection

    def _new_conn(self):
        conn = super()._new_conn()
        conn.pinned_ip = self.pinned_ip
        return conn


class _PinnedPoolManager(PoolManager):
    """把 pinned_ip 注入每个连接池：requests 2.32+ 直接走 poolmanager.connection_from_host，
    不经由 adapter.get_connection，所以必须在这一层注入；同时剥离自定义键以免污染 PoolKey。"""

    def __init__(self, *args, pinned_ip=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinned_ip = pinned_ip
        # urllib3 2.x 用 pool_classes_by_scheme 选连接池类型
        self.pool_classes_by_scheme = {
            "http": _PinnedHTTPConnectionPool,
            "https": _PinnedHTTPSConnectionPool,
        }

    def connection_from_host(self, host, port=None, scheme="http", pool_kwargs=None):
        pool_kwargs = dict(pool_kwargs or {})
        pool_kwargs["pinned_ip"] = self.pinned_ip
        return super().connection_from_host(host, port, scheme, pool_kwargs)

    def connection_from_context(self, request_context):
        request_context = dict(request_context)
        pinned_ip = request_context.pop("pinned_ip", None)
        pool = super().connection_from_context(request_context)
        pool.pinned_ip = pinned_ip
        return pool


class PinnedIPAdapter(HTTPAdapter):
    """把请求钉死到指定 IP：连接用 pinned_ip，但 Host/SNI 仍用真实域名。

    校验阶段解析出的安全 IP 即最终连接目标，攻击者无法用 TTL=0 的域名在
    校验后二次解析到内网/云元数据服务（DNS rebinding TOCTOU）。
    """

    def __init__(self, pinned_ip: str | None = None, **kwargs):
        self.pinned_ip = pinned_ip
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        self.poolmanager = _PinnedPoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            pinned_ip=self.pinned_ip,
            **pool_kwargs,
        )

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
    """默认拒绝一切「非公网」地址，而不是逐条枚举已知内网段。

    枚举式黑名单必然漏网：`100.64.0.0/10`（CGNAT 共享地址空间）的
    `is_private` / `is_reserved` 全为 False，旧实现会直接放行——在运营商
    CGNAT 或共享出口的部署里，攻击者可借此打到内网基础设施。
    `is_global` 是白名单语义（RFC 6890 特殊用途地址一律为 False），
    再叠加显式条件作为跨版本兜底。
    """
    return (
        not ip.is_global
        or ip.is_loopback
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


def _safe_ips(host: str) -> "set | None":
    """解析并校验主机名/IP 的安全集合：返回可抓的 IP 集合；任一结果为内网/保留 IP、
    或解析失败，返回 None（拒）。供 SSRF 校验与「钉 IP 防 rebinding」共用，只解析一次。
    """
    literal = _literal_ip(host)
    if literal is not None:
        return None if _is_blocked_ip(literal) else {literal}
    try:
        ips = _resolve_ips(host)
    except Exception:
        return None
    if not ips:
        return None
    if any(_is_blocked_ip(ip) for ip in ips):
        return None
    return ips


def _validate(url: str, host: str) -> "tuple[str | None, set | None]":
    """SSRF 校验：返回 (拒绝原因, 安全 IP 集合)。

    reason 非 None 即拒绝；safe 为 None 表示无安全 IP。校验与「钉 IP」共用同一次
    DNS 解析结果，避免二次解析引入 rebinding 窗口。
    """
    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return f"协议不在白名单内: {scheme or '(空)'}", None
    if not host:
        return "URL 缺少主机名", None
    if host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        return "命中内网主机名黑名单", None
    safe = _safe_ips(host)
    if safe is None:
        return "主机解析到内网/保留 IP 或解析失败", None
    return None, safe


def _block_reason(url: str, host: str) -> str | None:
    """返回拒绝原因；None 表示这个目标可以抓。保持历史签名供测试与外部调用。"""
    return _validate(url, host)[0]


def fetch_url(url: str, timeout: int = 10, settings_obj: Optional["Settings"] = None) -> str:
    s = settings_obj or settings
    host = _host_of(url)

    # 先做廉价的主机名黑名单，再走 mock 短路，避免离线环境下 mock 域名被 DNS 校验误伤
    if not host or host in _BLOCKED_NAMES or host.endswith(_BLOCKED_SUFFIXES):
        logger.warning("SSRF 防护：拒绝抓取 %s（命中内网主机名黑名单）", url)
        return ""

    # example.com 是项目内置 mock 域名：无论是否配置 key，都直接返回占位正文，
    # 避免「配置了 Tavily key 但离线」时真实联网抓取挂起（mock 检索回退的 URL 即为此域名）。
    if _is_mock_host(host):
        return f"[mock] 抓取到的网页正文（{url}）：本段为离线占位内容，用于演示工具调用链路。"
    if not s.tavily_api_key:
        return f"[mock] 抓取到的网页正文（{url}）：本段为离线占位内容（未配置 Tavily key）。"

    # SSRF 校验（含钉 IP 所需安全 IP 集合），只解析一次
    reason, safe = _validate(url, host)
    if reason is not None:
        logger.warning("SSRF 防护：拒绝抓取 %s（%s）", url, reason)
        return ""

    # 校验通过：钉死到一个安全 IP，防止二次 DNS 解析（rebinding）绕过校验（M-3）
    pinned = next((ip for ip in safe if ip.version == 4), next(iter(safe)))

    try:
        # 关键：不跟随重定向。否则 302 到 169.254.169.254 等元数据服务即可绕过上面所有校验。
        session = requests.Session()
        # 整个 scheme 都走钉 IP 适配器：按 host 前缀挂载会被带 userinfo / IPv6 方括号的
        # URL 绕过（requests 用 url.lower().startswith(prefix) 匹配），从而重新走默认适配器的
        # getaddrinfo，重开 rebinding 窗口。每次请求都是新建 session，故 scheme 级挂载无副作用。
        adapter = PinnedIPAdapter(pinned_ip=str(pinned))
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        # 禁止走环境代理：否则 requests 会另建普通 ProxyManager，绕过钉 IP 适配器。
        session.trust_env = False
        resp = session.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "research-agent/1.0"},
            allow_redirects=False,
            stream=True,
            proxies={},
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            logger.warning(
                "SSRF 防护：拒绝跟随重定向 %s -> %s（HTTP %s）",
                url,
                resp.headers.get("Location", "(无 Location)"),
                resp.status_code,
            )
            resp.close()
            return ""
        resp.raise_for_status()

        # 预检大小：避免超大/恶意响应撑爆内存（M-2）
        MAX_BYTES = 512 * 1024
        cl = resp.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES:
            logger.warning("资源防护：拒绝抓取 %s（响应过大 %s 字节）", url, cl)
            resp.close()
            return f"[抓取失败] {url}（响应过大）"
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            chunks.append(chunk)
            if total > MAX_BYTES:
                logger.warning("资源防护：截断过大响应 %s（> %s 字节）", url, MAX_BYTES)
                break
        resp.close()
        text = b"".join(chunks).decode("utf-8", errors="ignore")
        return text[:4000]
    except Exception:
        return f"[抓取失败] {url}"

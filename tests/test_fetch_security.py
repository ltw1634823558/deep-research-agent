"""fetch 工具 SSRF 防护测试：内网 IP 字面量、十进制绕过归一、协议白名单、
域名解析到内网、fetch_url 端到端拒绝。全程离线（无真实网络请求）。
"""
from __future__ import annotations

import ipaddress
import socket

from src.tools import fetch as fetch_mod
from src.tools.fetch import fetch_url


def test_is_blocked_ip_classifies_private_ranges():
    blocked = [
        "127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254",
        "0.0.0.0", "172.16.0.1", "::1", "fe80::1",
    ]
    for ip in blocked:
        assert fetch_mod._is_blocked_ip(ipaddress.ip_address(ip)) is True, ip
    assert fetch_mod._is_blocked_ip(ipaddress.ip_address("8.8.8.8")) is False
    assert fetch_mod._is_blocked_ip(ipaddress.ip_address("1.1.1.1")) is False


def test_is_blocked_ip_covers_non_global_special_ranges():
    """枚举式黑名单的漏网段：这些地址 is_private / is_reserved 均为 False。

    100.64.0.0/10 是运营商 CGNAT 共享地址空间——部署在共享出口环境时，
    放行它等于把内网基础设施暴露给 SSRF。必须由 `not is_global` 兜住。
    """
    for ip in ["100.64.0.1", "100.127.255.254", "198.18.0.5", "192.0.0.8", "192.0.2.5"]:
        addr = ipaddress.ip_address(ip)
        assert fetch_mod._is_blocked_ip(addr) is True, f"{ip} 应被拒绝（is_global={addr.is_global}）"


def test_decimal_ip_normalization_cannot_bypass():
    # 2130706433 == 127.0.0.1，历史写法不能绕过检测
    assert fetch_mod._literal_ip("2130706433") == ipaddress.ip_address("127.0.0.1")
    assert fetch_mod._literal_ip("0x7f000001") == ipaddress.ip_address("127.0.0.1")
    assert fetch_mod._block_reason("http://2130706433/", "2130706433") is not None
    # 非 IP 字面量返回 None（需 DNS 校验）
    assert fetch_mod._literal_ip("example.com") is None


def test_scheme_allowlist_rejects_non_http():
    reason = fetch_mod._block_reason("ftp://example.com/x", "example.com")
    assert reason is not None
    assert "协议" in reason


def test_block_reason_rejects_domain_resolving_to_private(monkeypatch):
    def fake_getaddrinfo(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]
    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", fake_getaddrinfo)
    reason = fetch_mod._block_reason("http://internal.corp/", "internal.corp")
    assert reason is not None
    assert "内网" in reason


def test_block_reason_fails_closed_on_dns_error(monkeypatch):
    def fake_getaddrinfo(host, port, **kw):
        raise socket.gaierror("boom")
    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", fake_getaddrinfo)
    # 非字面量 IP 且 DNS 失败 → fail-closed 拒绝
    assert fetch_mod._block_reason("http://unknown.invalid/", "unknown.invalid") is not None


def test_fetch_url_refuses_blocked_hostnames():
    # localhost 等命中内网主机名黑名单，先短路拒绝，不依赖 key
    assert fetch_url("http://localhost/secret") == ""
    assert fetch_url("http://ip6-localhost/x") == ""


def test_fetch_url_refuses_private_ip_literal(monkeypatch):
    # 需配置 key 让 fetch_url 越过 mock 短路、走到 SSRF 校验分支
    monkeypatch.setattr(fetch_mod.settings, "tavily_api_key", "dummy-key")
    assert fetch_url("http://10.0.0.5/") == ""
    assert fetch_url("http://169.254.169.254/latest/meta-data") == ""


def test_fetch_url_refuses_domain_resolving_to_private(monkeypatch):
    monkeypatch.setattr(fetch_mod.settings, "tavily_api_key", "dummy-key")

    def fake_getaddrinfo(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.5", 0))]
    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", fake_getaddrinfo)
    assert fetch_url("http://internal.corp/") == ""


def test_fetch_pins_ip_prevents_rebinding(monkeypatch):
    """DNS rebinding：校验用解析返回公网 IP，连接时若再解析会返回内网 IP。

    钉 IP 后连接目标必须是校验阶段解析到的安全 IP，且连接阶段不得再走 getaddrinfo
    （否则攻击者可用 TTL=0 的域名二次解析到内网/元数据服务）。
    """
    public_ip = "93.184.216.34"
    internal_ip = "10.0.0.99"
    calls = {"n": 0}

    def fake_getaddrinfo(host, port, *args, **kw):
        calls["n"] += 1
        ip = public_ip if calls["n"] == 1 else internal_ip
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(fetch_mod.settings, "tavily_api_key", "dummy-key")

    connected_to = {}

    # _new_conn 现在用原始 socket 直接连接钉死的 IP（不走 getaddrinfo/create_connection），
    # 因此监控 socket.socket 的 connect 来确认连接目标。
    class _FakeSocket:
        def settimeout(self, t):
            pass

        def setsockopt(self, *a, **k):
            pass

        def close(self):
            pass

        def connect(self, addr):
            connected_to["addr"] = addr
            raise ConnectionError("simulated")  # 中断真实联网，验证连接目标即可

    monkeypatch.setattr(fetch_mod.socket, "socket", lambda *a, **k: _FakeSocket())

    # 用非 mock 域名（.example.com 会被 _is_mock_host 短路到 mock 正文，绕开网络路径）
    result = fetch_url("http://rebind.attacker.net/page")
    # 连接目标必须是校验阶段解析到的公网 IP，而非第二次解析的内网 IP
    assert connected_to.get("addr") == (public_ip, 80)
    # 连接阶段不得再次解析 DNS（getaddrinfo 只在 _validate 阶段被调用一次）
    assert calls["n"] == 1, "连接阶段不应再触发 DNS 解析"
    # fetch 应走抓取失败兜底分支（连接被我们模拟中断）
    assert result.startswith("[抓取失败]")


def test_fetch_userinfo_url_still_pinned(monkeypatch):
    """H-A 回归：带 userinfo 的 URL 曾落到默认适配器、重走 getaddrinfo 打开 rebinding 窗口。

    改为 scheme 级挂载后，带 userinfo 的 URL 仍须钉死到校验过的公网 IP。
    """
    public_ip = "93.184.216.34"
    internal_ip = "10.0.0.99"
    calls = {"n": 0}

    def fake_getaddrinfo(host, port, *args, **kw):
        calls["n"] += 1
        ip = public_ip if calls["n"] == 1 else internal_ip
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(fetch_mod.settings, "tavily_api_key", "dummy-key")

    connected_to = {}

    class _FakeSocket:
        def settimeout(self, t):
            pass

        def setsockopt(self, *a, **k):
            pass

        def close(self):
            pass

        def connect(self, addr):
            connected_to["addr"] = addr
            raise ConnectionError("simulated")

    monkeypatch.setattr(fetch_mod.socket, "socket", lambda *a, **k: _FakeSocket())

    result = fetch_url("http://user:pw@rebind.attacker.net/page")
    assert connected_to.get("addr") == (public_ip, 80)
    assert calls["n"] == 1
    assert result.startswith("[抓取失败]")


def test_fetch_rejects_redirect(monkeypatch):
    """M-2：禁止跟随重定向，否则 302 到元数据服务可绕过 SSRF 校验。"""
    import requests as _requests

    monkeypatch.setattr(
        fetch_mod.socket,
        "getaddrinfo",
        lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setattr(fetch_mod.settings, "tavily_api_key", "dummy-key")

    fake_resp = _requests.Response()
    fake_resp.status_code = 302
    fake_resp.headers["Location"] = "http://169.254.169.254/latest/meta-data"
    fake_resp.url = "http://rebind.attacker.net/page"
    fake_resp.close = lambda: None  # 裸 Response.close() 会因 _connection 为 None 抛错

    def _fake_get(self, *a, **k):
        return fake_resp

    monkeypatch.setattr(_requests.Session, "get", _fake_get)

    result = fetch_url("http://rebind.attacker.net/page")
    # 重定向被拒绝，不跟随、返回空
    assert result == ""


def test_fetch_truncates_oversized_response(monkeypatch):
    """资源防护：超 512KB 响应须截断，避免撑爆内存（M-2）。"""
    import requests as _requests

    monkeypatch.setattr(
        fetch_mod.socket,
        "getaddrinfo",
        lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setattr(fetch_mod.settings, "tavily_api_key", "dummy-key")

    big = b"x" * (600 * 1024)
    fake_resp = _requests.Response()
    fake_resp.status_code = 200
    fake_resp.headers = {}

    def _iter(chunk_size=8192):
        yield big

    fake_resp.iter_content = _iter
    fake_resp.raise_for_status = lambda: None
    fake_resp.close = lambda: None

    def _fake_get(self, *a, **k):
        return fake_resp

    monkeypatch.setattr(_requests.Session, "get", _fake_get)

    result = fetch_url("http://rebind.attacker.net/page")
    # 超过 512KB 应被截断到 ~4000 字符上限
    assert len(result) <= 5000
    assert result.startswith("xxxx")

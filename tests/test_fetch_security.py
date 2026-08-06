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

"""T-0160 C-5: the opt-in real-network rows (docs/tasks/T-0160.md). The only module that touches the real network.

Runs only with ``ASTRA_BROWSER_TESTS=1 GEARVIS_WEB_NETWORK=1`` (``make browser-web-network``); never in the fast suite
or ``make check``. The real ``BrowserToolExecutor`` in web mode over a real ``StartPageServer``, with the proxy's
default resolver; its connector is the default (``asyncio.open_connection`` to the checked IP literal, A-5) wrapped
only to count the addresses it is handed, so a row can say how many CONNECTs were tunnelled and to where.

Regime: CPU, headless Chromium 1248 (Playwright 1.64) at 1280x800, sandbox on (H0), real network, n = 1 per row, by
hand only. A row that cannot reach ``example.com`` fails naming the network, not the code.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import ipaddress
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome

pytestmark = [pytest.mark.browser]

GATED = (
    os.environ.get("ASTRA_BROWSER_TESTS") == "1" and os.environ.get("GEARVIS_WEB_NETWORK") == "1"
)
W, H = 1280, 800
ADDRESS_FIELD = (
    "input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=checkbox])"
)


@pytest.fixture(autouse=True)
def _network_gate() -> None:
    if not GATED:
        pytest.skip(
            "ASTRA_BROWSER_TESTS=1 GEARVIS_WEB_NETWORK=1 not both set: the C-5 rows use the real network"
        )
    importlib.import_module("playwright.async_api")


class Counted:
    """The default connector, counting the IP literals it is handed (A-5 (f): never a hostname)."""

    def __init__(self) -> None:
        self.ips: list[str] = []

    async def __call__(self, ip: str, port: int) -> Any:
        ipaddress.ip_address(ip)  # A-5 (f): an IP literal, or this raises
        self.ips.append(ip)
        return await asyncio.open_connection(ip, port)


@contextlib.asynccontextmanager
async def web() -> AsyncIterator[tuple[Any, str, Counted]]:
    from local_astra.computer.executor import BrowserToolExecutor
    from local_astra.pages.server import StartPageServer

    pages = StartPageServer()
    pages.start()
    counted = Counted()
    ex = BrowserToolExecutor(
        now_ms=lambda: int(time.time() * 1000),
        viewport=(W, H),
        headless=True,
        web=True,
        start_origin=pages.origin,
        allowed_origins=frozenset({pages.origin}),
        proxy_connect=counted,
    )
    try:
        await asyncio.wait_for(ex.open(pages.origin + "/web.html"), 60)
        yield ex, pages.origin, counted
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 30)
        pages.stop()
    non_public = [ip for ip in counted.ips if not ipaddress.ip_address(ip).is_global]
    assert non_public == [], (
        f"T-0160 C-5: the connector was handed non-public addresses {non_public}"
    )


def _units(px: float, size: int) -> int:
    c = max(0, min(999, round(px * 999 / (size - 1))))
    assert abs(to_pixels(c, size) - px) <= 1.0
    return c


async def type_address(ex: Any, address: str) -> ToolOutcome:
    box = await ex.driver.page.locator(ADDRESS_FIELD).first.bounding_box(timeout=10_000)
    assert box is not None, "premise: /web.html has its address field"
    f = await asyncio.wait_for(ex.frame(), 30)
    g = FrameGeometry(phash=f.phash, source=f.source, width=f.width, height=f.height)
    call = ToolInvocation(
        call_id=f"n{time.monotonic_ns()}",
        turn_id="t1",
        name="type_at",
        arguments={
            "x": _units(box["x"] + box["width"] / 2, W),
            "y": _units(box["y"] + box["height"] / 2, H),
            "text": address,
            "press_enter": True,
        },
        source="draft",
        frame=g,
    )
    return await asyncio.wait_for(ex.execute(call), 60)


async def settle_url(ex: Any, prefix: str, timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ex.driver.page.url.startswith(prefix):
            return str(ex.driver.page.url)
        await asyncio.sleep(0.2)
    return str(ex.driver.page.url)


def _blocked_hosts(ex: Any) -> list[str]:
    return [str(r.get("host")) for r in list(ex.egress_log)]


async def test_example_com_through_web_html() -> None:
    """C-5: ``/web.html`` -> ``example.com`` shows the page; that action's ``resolved["hosts"]`` holds example.com."""
    async with web() as (ex, _origin, counted):
        out = await type_address(ex, "example.com")
        url = await settle_url(ex, "https://example.com")
        assert url.startswith("https://example.com"), f"C-5: reached {url} (network down? {out})"
        title = await ex.driver.page.title()
        assert "Example Domain" in title, title
        hosts = list(out.resolved.get("hosts") or [])
        assert "example.com" in hosts, f"T-0160 B-6/C-5: resolved hosts {out.resolved}"
        assert counted.ips, "premise: the proxy tunnelled at least one public address"


@pytest.mark.parametrize(
    ("address", "host"),
    [
        ("https://localtest.me/", "localtest.me"),  # a public name resolving to 127.0.0.1
        ("https://169.254.169.254/", "169.254.169.254"),  # the metadata endpoint
        ("https://2130706433/", None),  # 127.0.0.1 in single-decimal form
    ],
    ids=["localtest_me", "metadata", "decimal_loopback"],
)
async def test_non_public_targets_blocked_and_logged(address: str, host: str | None) -> None:
    """C-5: each is blocked (route or proxy) and logged in the egress log; the connector never sees it."""
    async with web() as (ex, _origin, counted):
        before = len(counted.ips)
        await type_address(ex, address)
        await asyncio.sleep(2.0)
        url = str(ex.driver.page.url)
        assert not url.startswith(address.rstrip("/")), f"C-5: {address} was reached ({url})"
        assert len(counted.ips) == before, (
            f"C-5: the connector was used for {address}: {counted.ips[before:]}"
        )
        blocked = _blocked_hosts(ex)
        assert blocked, f"T-0160 B-6/C-5: {address} is in the egress log; log {list(ex.egress_log)}"
        if host is not None:
            assert host in blocked, blocked
        for r in list(ex.egress_log):
            assert set(r) <= {"host", "scheme"}, f"B-6: {{host, scheme}} only; {r}"


async def test_idle_web_html_makes_no_connect() -> None:
    """C-5: idle on ``/web.html`` for 5 s, the proxy tunnels 0 public CONNECTs (no phone-home)."""
    async with web() as (_ex, _origin, counted):
        before = list(counted.ips)
        await asyncio.sleep(5.0)
        assert counted.ips == before == [], f"C-5: CONNECTs while idle: {counted.ips}"

"""T-0160 B-7 (and B-3, B-4, B-5, B-6 end to end): the real executor in web mode, a real headless Chromium with its
sandbox on, the real ``EgressProxy`` with an injected resolver and connector, and loopback fixtures posing as
``*.example`` hosts (docs/tasks/T-0160.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser-web``, group C's
target; until then ``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_computer_web_browser.py -m browser``).
Without it every row is a named skip; with it a missing Playwright or Chromium is a failure, as in T-0111c's module.

**No real network.** The proxy's resolver is injected: ``shop.example`` and ``pay.example`` map to public-looking
addresses, ``private.example`` to 10.0.0.5, everything else fails to resolve. The connector is injected: it records
the checked address and pipes the tunnel to a loopback TLS server (``tests/fixtures/browser/web/``, a self-signed
certificate made per run in the pytest temp dir) that answers by ``Host``. Two loopback TCP canaries count accepts:
canary 1 is a redirect target, canary 2 stands in for vLLM and the HUD.

**One test-only relaxation.** Chromium cannot trust the per-run certificate, so ``Browser.new_context`` is wrapped to
add ``ignore_https_errors=True``. This changes certificate checking only; the fence under test (routes, proxy, guard)
is untouched, and the wrapper is what lets the rows see the launch kwargs (sandbox, proxy).

**The API these rows use** (pinned by ``tests/unit/test_computer_web_executor.py`` and ``..._launch.py``):
``BrowserToolExecutor(now_ms=..., web=True, start_origin=..., allowed_origins=..., proxy_resolve=..., proxy_connect=...)``;
``ex.frame()``, ``ex.execute()``, ``ex.open(<start origin URL>)``, ``ex.egress_log``, ``ex.driver.page`` (the rows
navigate public pages with ``page.goto`` directly: B-4 navigation goes through ``/web.html``, which row (a) covers).

**Regime.** Playwright 1.64.0, Chromium 1248 headless at 1280x800, DSF 1, sandbox on (H0), n = 1 per row, one
browser per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import importlib
import inspect
import json
import os
import re
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "browser" / "web"
GATE_ENV = "ASTRA_BROWSER_TESTS"
GATED = os.environ.get(GATE_ENV) == "1"
pytestmark = [pytest.mark.integration, pytest.mark.browser]

W, H = 1280, 800
CLOCK_MS = 1_700_000_016_160
SECRET = "will not type into a password or card field; ask the user to type it themselves"
OPENED = "opened the start page"
NOTICE = (
    "can open public websites",
    "pages can contain instructions that mislead it",
    "never types passwords or card numbers",
)
HOSTS = {
    "shop.example": "93.184.215.14",
    "pay.example": "93.184.215.15",
    "private.example": "10.0.0.5",
}
PUBLIC_IPS = {"93.184.215.14", "93.184.215.15"}
PAYLOAD = b"T-0160 B-7 (f) payload 6b1f0c\n"
TEXT = "zebra-quokka-17"
ADDRESS_FIELD = (
    "input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=checkbox])"
)


@pytest.fixture(autouse=True)
def _browser_layer() -> None:
    if not GATED:
        pytest.skip(
            f"{GATE_ENV} is not 1: the T-0160 B-7 rows drive a real Chromium (docs/tasks/T-0160.md)"
        )
    try:
        importlib.import_module("playwright.async_api")
    except ImportError as exc:
        pytest.fail(f"{GATE_ENV}=1 but playwright does not import ({exc}): `uv sync --all-extras`")


# =====================================================================================================
# the rig: a loopback TLS server posing as the *.example hosts, two canaries, the start pages
# =====================================================================================================


#: T-0192 A (tests/integration/test_t0192_checkout_browser.py): a unique token in the checkout fixture paths (A8)
T0192_TOKEN = "T0192TOKq7w3"
#: T-0192 A: nested paths of the ``t0192_*.html`` fixtures (MANIFEST.md); ``(302, target)`` is a redirect hop
T0192_ROUTES: dict[str, str | tuple[int, str]] = {
    #: R-A4 (3db7e131): T-0160's card-field page at a non-checkout path, so its B-7 (c) rows test the card guard
    "/card_form.html": "checkout.html",
    "/t0192/dp/kettle": "t0192_product.html",
    "/cart.html": "t0192_cart.html",
    "/checkout/entry": (
        302,
        "/signin.html?openid.return_to=https%3A%2F%2Fshop.example%2Fcheckout%2Fentry",
    ),
    "/signin.html": "t0192_signin.html",
    "/checkout/entry2": (302, f"/checkout/{T0192_TOKEN}/pay"),
    f"/checkout/{T0192_TOKEN}/pay": "t0192_checkout.html",
    "/checkout-faq.html": "t0192_checkout_faq.html",
    "/t0192/framed.html": "t0192_framed.html",
    f"/checkout/{T0192_TOKEN}/frame": "t0192_frame.html",
    #: review R-A: a cart form POST whose redirect chain has the checkout only in the middle
    "/t0192/cart-form.html": "t0192_cart_form.html",
    "/gp/cart/submit": (302, "/checkout/entry?proceedToCheckout=1"),
}


class _Fixture(http.server.BaseHTTPRequestHandler):
    hits: list[tuple[str, str]]

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        u = urllib.parse.urlsplit(self.path)
        self.hits.append((host, u.path))
        route = T0192_ROUTES.get(u.path)
        if isinstance(route, tuple):
            self._send(route[0], b"", extra={"Location": route[1]})
            return
        if isinstance(route, str):
            self._send(200, (FIXTURES / route).read_bytes(), "text/html; charset=utf-8")
            return
        if u.path == "/redirect":
            self._send(302, b"", extra={"Location": urllib.parse.parse_qs(u.query)["to"][0]})
            return
        if u.path == "/file.bin":
            self._send(
                200,
                PAYLOAD,
                "application/octet-stream",
                {"Content-Disposition": 'attachment; filename="t0160-payload.bin"'},
            )
            return
        name = "index.html" if u.path in ("", "/") else u.path.lstrip("/")
        f = FIXTURES / name
        if "/" in name or not f.is_file() or f.suffix != ".html":
            self._send(404, b"not found")
            return
        self._send(200, f.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        # read the request body before answering (T-0192 review R-A: a form POST), so closing the connection with
        # unread input cannot reset it before the client has the response
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.do_GET()

    def _send(
        self,
        status: int,
        body: bytes,
        ctype: str = "text/plain",
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


class _QuietTLSServer(http.server.ThreadingHTTPServer):
    """Chromium probes the self-signed certificate before ``ignore_https_errors`` applies (seen in the rig probe as
    ``SSLV3_ALERT_CERTIFICATE_UNKNOWN``); those aborted handshakes are not row failures, so they are not printed."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class Canary:
    """A loopback TCP listener that counts accepts (premise-checked once at start)."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.accepts = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            self.accepts += 1
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._t.join(2)
        self.sock.close()


class Rig:
    def __init__(
        self, tls_port: int, hits: list[tuple[str, str]], start: str, c1: Canary, c2: Canary
    ) -> None:
        self.tls_port, self.hits, self.start, self.c1, self.c2 = tls_port, hits, start, c1, c2


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    if not GATED:
        pytest.skip(f"{GATE_ENV} is not 1")
    d = tmp_path_factory.mktemp("t0160-tls")
    key, crt = d / "key.pem", d / "crt.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
         "-days", "2", "-subj", "/CN=shop.example",
         "-addext", "subjectAltName=DNS:shop.example,DNS:pay.example,DNS:private.example"],
        check=True, capture_output=True,
    )  # fmt: skip
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    hits: list[tuple[str, str]] = []
    handler = type("_Bound", (_Fixture,), {"hits": hits})
    srv = _QuietTLSServer(("127.0.0.1", 0), handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    from local_astra.pages.server import StartPageServer

    pages = StartPageServer()
    pages.start()
    c1, c2 = Canary(), Canary()
    for c in (c1, c2):  # premise: a canary counts a connect
        socket.create_connection(("127.0.0.1", c.port), 1).close()
        deadline = time.monotonic() + 2
        while c.accepts < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert c.accepts == 1, "premise: the canary does not count accepts"
        c.accepts = 0
    try:
        yield Rig(srv.server_address[1], hits, pages.origin, c1, c2)
    finally:
        c1.close()
        c2.close()
        pages.stop()
        srv.shutdown()
        srv.server_close()


class Net:
    """The proxy's injected seams: a resolver over ``HOSTS`` and a connector to the loopback TLS server."""

    def __init__(self, tls_port: int) -> None:
        self.tls_port = tls_port
        self.resolves: list[str] = []
        self.connects: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
        self.resolves.append(host)
        ip = HOSTS.get(host.lower().rstrip("."))
        if ip is None:
            raise socket.gaierror(socket.EAI_NONAME, "not a fixture host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    async def connect(
        self, ip: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.connects.append((ip, port))
        return await asyncio.open_connection("127.0.0.1", self.tls_port)

    def assert_public_only(self) -> None:
        bad = [c for c in self.connects if c[0] not in PUBLIC_IPS or c[1] != 443]
        assert bad == [], f"the connector got a non-public or non-443 target: {bad}"


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The test-only certificate relaxation (module docstring), and a record of ``chromium.launch``'s kwargs."""
    from playwright.async_api import Browser, BrowserType

    seen: list[dict[str, Any]] = []
    orig_ctx, orig_launch = Browser.new_context, BrowserType.launch

    async def new_context(self: Any, **kw: Any) -> Any:
        kw["ignore_https_errors"] = True
        return await orig_ctx(self, **kw)

    async def launch(self: Any, **kw: Any) -> Any:
        seen.append(dict(kw))
        return await orig_launch(self, **kw)

    monkeypatch.setattr(Browser, "new_context", new_context)
    monkeypatch.setattr(BrowserType, "launch", launch)
    return seen


@contextlib.asynccontextmanager
async def web_executor(rig: Rig, net: Net) -> AsyncIterator[Any]:
    from local_astra.computer.executor import BrowserToolExecutor

    params = inspect.signature(BrowserToolExecutor.__init__).parameters
    missing = [
        k for k in ("web", "start_origin", "proxy_resolve", "proxy_connect") if k not in params
    ]
    assert not missing, f"T-0160 B-1/B-7: BrowserToolExecutor has no keyword(s) {missing} yet"
    ex = BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=(W, H),
        web=True,
        start_origin=rig.start,
        allowed_origins=frozenset({rig.start}),
        proxy_resolve=net.resolve,
        proxy_connect=net.connect,
    )
    try:
        yield ex
    finally:
        await asyncio.wait_for(ex.close(), 30)
        net.assert_public_only()


# =====================================================================================================
# helpers
# =====================================================================================================

_N = iter(range(1_000_000))


def inv(name: str, frame: Frame | None = None, **arguments: Any) -> ToolInvocation:
    g = (
        FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
        if frame is not None
        else None
    )
    return ToolInvocation(
        call_id=f"w{next(_N)}",
        turn_id="t1",
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


def units(px: float, size: int) -> int:
    """The 0-999 coordinate nearest ``px`` (targeting only; checked to land within 1 px)."""
    c = max(0, min(999, round(px * 999 / (size - 1))))
    assert abs(to_pixels(c, size) - px) <= 1.0, (px, size, c)
    return c


async def execute(ex: Any, i: ToolInvocation, timeout: float = 30.0) -> ToolOutcome:
    out = await asyncio.wait_for(ex.execute(i), timeout)
    assert isinstance(out, ToolOutcome) and out.call_id == i.call_id, out
    return out


async def goto(ex: Any, url: str) -> None:
    """Harness navigation of the active page (through the context routes and the proxy, like any page load)."""
    await asyncio.wait_for(ex.frame(), 30)
    with contextlib.suppress(
        Exception
    ):  # a refused navigation raises; the rows check what happened
        await ex.driver.page.goto(url, wait_until="domcontentloaded", timeout=10_000)


async def center(ex: Any, selector: str, frame_selector: str | None = None) -> tuple[int, int]:
    page = ex.driver.page
    loc = (page.frame_locator(frame_selector) if frame_selector else page).locator(selector)
    box = await loc.bounding_box(timeout=10_000)
    assert box is not None, f"{selector} is not on the page"
    return units(box["x"] + box["width"] / 2, W), units(box["y"] + box["height"] / 2, H)


async def type_into(
    ex: Any, selector: str, text: str, *, enter: bool = False, frame_selector: str | None = None
) -> ToolOutcome:
    x, y = await center(ex, selector, frame_selector)
    f = await asyncio.wait_for(ex.frame(), 30)
    return await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=enter))


async def click(ex: Any, selector: str) -> ToolOutcome:
    x, y = await center(ex, selector)
    f = await asyncio.wait_for(ex.frame(), 30)
    return await execute(ex, inv("click_at", f, x=x, y=y))


async def value_of(ex: Any, selector: str, frame_selector: str | None = None) -> str:
    page = ex.driver.page
    loc = (page.frame_locator(frame_selector) if frame_selector else page).locator(selector)
    return str(await loc.input_value(timeout=10_000))


async def wait_url(ex: Any, want: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        url = ex.driver.page.url
        if url == want:
            return url
        await asyncio.sleep(0.1)
    return str(ex.driver.page.url)


def hosts_of(*outs: ToolOutcome) -> set[str]:
    return {h for o in outs for h in (o.resolved.get("hosts") or [])}


def logged_hosts(ex: Any) -> set[str]:
    return {r.get("host", "") for r in ex.egress_log}


def _artifact_files() -> set[Path]:
    tmp = Path(tempfile.gettempdir())
    return {p for d in tmp.glob("playwright-artifacts-*") for p in d.rglob("*") if p.is_file()}


# =====================================================================================================
# B-1 (sandbox and proxy at launch), B-5 (/web.html), B-7 (a) navigation
# =====================================================================================================


async def test_launch_is_sandboxed_and_proxied(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-1: the real launch has the sandbox on and the proxy with the explicit ``<-loopback>`` bypass."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await asyncio.wait_for(ex.frame(), 30)
        assert len(launches) == 1, launches
        assert launches[0].get("chromium_sandbox") is True, launches[0]
        proxy = launches[0].get("proxy") or {}
        assert str(proxy.get("server", "")).startswith("http://127.0.0.1:"), proxy
        assert proxy.get("bypass") == "<-loopback>", proxy


async def test_web_html_shape(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-5 / C-7: ``/web.html`` has one labelled, autofocused web-address field and a Go button, and carries the
    notice's first three clauses verbatim."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await asyncio.wait_for(ex.open(f"{rig.start}/web.html"), 30)
        page = ex.driver.page
        fields = page.locator(ADDRESS_FIELD)
        assert await fields.count() == 1, "B-5: /web.html must have exactly one web-address field"
        facts = await page.evaluate(
            f"""() => {{ const i = document.querySelector({json.dumps(ADDRESS_FIELD)});
                return {{ focused: document.activeElement === i,
                         labelled: (i.labels && i.labels.length > 0) || !!i.getAttribute('aria-label'),
                         go: [...document.querySelectorAll('button, input[type=submit]')]
                               .some(b => /\\bgo\\b/i.test(b.textContent || b.value || '')),
                         text: document.body.innerText }}; }}"""
        )
        assert facts["focused"], "B-5: the address field is not autofocused"
        assert facts["labelled"], "B-5: the address field has no label"
        assert facts["go"], "B-5: no Go button"
        for clause in NOTICE:
            assert clause in facts["text"], f"C-7: /web.html lacks {clause!r}"


@pytest.mark.parametrize(
    "typed, outcome",
    [
        ("http://shop.example", "https://shop.example/"),
        ("localhost", "refused-no-dot"),
        ("ftp://shop.example", "refused"),
        ("shop .example", "refused"),
        ("javascript:alert(1)", "refused"),
    ],
)
async def test_web_html_address_rules(
    rig: Rig, launches: list[dict[str, Any]], typed: str, outcome: str
) -> None:
    """B-5 JS: ``http://`` is upgraded to ``https://``; another scheme, a space, or no dot is refused visibly and
    nothing is resolved (the no-dot refusal says "Type a web address, such as example.com": T-0230 A1 re-pinned
    T-0160's "Type a web address like en.wikipedia.org")."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await asyncio.wait_for(ex.open(f"{rig.start}/web.html"), 30)
        page = ex.driver.page
        await page.fill(ADDRESS_FIELD, typed)
        await page.press(ADDRESS_FIELD, "Enter")
        if outcome.startswith("https://"):
            assert await wait_url(ex, outcome) == outcome
            return
        await asyncio.sleep(1.0)
        assert page.url == f"{rig.start}/web.html", page.url
        assert net.resolves == [], net.resolves
        if outcome == "refused-no-dot":
            body = await page.inner_text("body")
            assert "Type a web address, such as example.com" in body, body
            assert "en.wikipedia.org" not in body, body


async def test_t0230_web_html_shows_no_address(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """T-0230 A1 (adversarial: the model reads pixels, so a greyed example address is an address): at rest the
    rendered ``/web.html``'s visible text and the address input's placeholder hold no host-shaped string, the
    placeholder is "Type a web address", the field is empty and no error shows; typing ``nodot`` and Enter shows
    "Type a web address, such as example.com", stays on ``/web.html`` and resolves nothing."""
    host = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+", re.IGNORECASE)
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await asyncio.wait_for(ex.open(f"{rig.start}/web.html"), 30)
        page = ex.driver.page
        facts = await page.evaluate(
            f"""() => {{ const i = document.querySelector({json.dumps(ADDRESS_FIELD)});
                return {{ placeholder: i.getAttribute('placeholder'), value: i.value,
                         error: document.getElementById('error').innerText,
                         text: document.body.innerText }}; }}"""
        )
        assert facts["placeholder"] == "Type a web address", facts
        assert not host.search(facts["placeholder"] or ""), facts
        assert facts["value"] == "", facts
        assert facts["error"].strip() == "", facts
        assert not host.findall(facts["text"]), (
            f"T-0230 A1: an address on the page at rest: {facts['text']!r}"
        )
        await page.fill(ADDRESS_FIELD, "nodot")
        await page.press(ADDRESS_FIELD, "Enter")
        await asyncio.sleep(1.0)
        assert page.url == f"{rig.start}/web.html", page.url
        assert net.resolves == [], net.resolves
        err = await page.inner_text("#error")
        assert err == "Type a web address, such as example.com", err


async def test_a_navigation_through_web_html(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-7 (a) / B-6: ``type_at`` "shop.example" with Enter on ``/web.html`` reaches the fixture page through the
    proxy (one resolve of the name, connects to the checked address only), and the host shows in ``resolved["hosts"]``
    of that action or the next."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await asyncio.wait_for(ex.open(f"{rig.start}/web.html"), 30)
        out = await type_into(ex, ADDRESS_FIELD, "shop.example", enter=True)
        assert out.status == "executed", out
        assert await wait_url(ex, "https://shop.example/") == "https://shop.example/"
        assert "Shop fixture" in await ex.driver.page.inner_text("body")
        nxt = await execute(ex, inv("scroll", None, direction="down"))
        assert "shop.example" in hosts_of(out, nxt), (out.resolved, nxt.resolved)
        assert ("shop.example", "/") in rig.hits
        assert "shop.example" in net.resolves and ("93.184.215.14", 443) in net.connects
        for o in (out, nxt):
            assert all("/" not in h for h in o.resolved.get("hosts") or []), o.resolved


# =====================================================================================================
# B-7 (b), (c): passwords and cards
# =====================================================================================================


async def test_b_password_and_username(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-7 (b) / B-3: ``type_at`` on the password field is ``failed`` (SECRET_FIELD_REASON) and the field stays empty;
    on the username field it is ``executed``; a card number typed into the username field is refused and not typed."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/login.html")
        out = await type_into(ex, "#password", TEXT, enter=True)
        assert (out.status, out.reason) == ("failed", SECRET), out
        assert out.resolved.get("secret_field") == "password", out.resolved
        assert TEXT not in json.dumps(out.resolved, default=str)
        assert await value_of(ex, "#password") == ""
        out = await type_into(ex, "#username", "james")
        assert out.status == "executed", out
        assert await value_of(ex, "#username") == "james"
        out = await type_into(ex, "#username", "4111 1111 1111 1111")
        assert out.status == "refused", out
        assert await value_of(ex, "#username") == "james"


@pytest.mark.parametrize(
    "page, selector, frame_selector",
    [
        pytest.param("card_form.html", "#cc", None, id="autocomplete-cc-number"),
        pytest.param("card_form.html", "#card", "#payframe", id="cross-origin-iframe"),
        pytest.param("flip.html", "#note", None, id="flips-to-password-on-focus"),
    ],
)
async def test_c_card_fields(
    rig: Rig, launches: list[dict[str, Any]], page: str, selector: str, frame_selector: str | None
) -> None:
    """B-7 (c), adversarial: a ``cc-number`` field, a card field in a cross-origin (``pay.example``) iframe, and a
    field that becomes ``type=password`` when focused are all ``failed`` with nothing typed; the value stays empty."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"https://shop.example/{page}")
        out = await type_into(ex, selector, TEXT, enter=True, frame_selector=frame_selector)
        assert (out.status, out.reason) == ("failed", SECRET), out
        assert out.resolved.get("secret_field") in ("password", "card"), out.resolved
        assert await value_of(ex, selector, frame_selector) == ""


async def frame_by_url(ex: Any, suffix: str, timeout: float = 10.0) -> Any:
    """The active page's frame whose URL ends with ``suffix`` (an ``<object>``/``<embed>`` browsing context has no
    ``frame_locator``); waits for it to load."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for fr in ex.driver.page.frames:
            if fr is not ex.driver.page.main_frame and fr.url.endswith(suffix):
                return fr
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"premise: no frame loaded {suffix}: {[f.url for f in ex.driver.page.frames]}"
    )


async def type_in_frame(ex: Any, fr: Any, selector: str, text: str) -> ToolOutcome:
    box = await fr.locator(selector).bounding_box(timeout=10_000)
    assert box is not None, f"{selector} is not in {fr.url}"
    x, y = units(box["x"] + box["width"] / 2, W), units(box["y"] + box["height"] / 2, H)
    f = await asyncio.wait_for(ex.frame(), 30)
    return await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=True))


def assert_secret(out: ToolOutcome, kind: str) -> None:
    assert (out.status, out.reason) == ("failed", SECRET), out
    assert out.resolved.get("secret_field") == kind, out.resolved
    assert TEXT not in json.dumps(out.resolved, default=str)


@pytest.mark.parametrize(
    "page, selector, kind",
    [
        pytest.param("shadow_open.html", "#spw", "password", id="open-shadow-password"),
        pytest.param("shadow_nested.html", "#ncc", "card", id="nested-open-shadow-card"),
        pytest.param("label_card.html", "#f1", "card", id="label-only-card"),
    ],
)
async def test_c_secret_fields_found_by_walk_and_label(
    rig: Rig, launches: list[dict[str, Any]], page: str, selector: str, kind: str
) -> None:
    """B-3 (b)/(c) / B-7 (b)-(c), adversarial (reviewer A): a password field inside an open shadow root (the walk
    from the host into the shadow root) and a card field named only by its ``<label>`` are ``failed`` with
    SECRET_FIELD_REASON and the right kind; the value stays empty."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"https://shop.example/{page}")
        out = await type_into(ex, selector, TEXT, enter=True)
        assert_secret(out, kind)
        assert await value_of(ex, selector) == ""


async def test_c_card_in_the_second_of_two_iframes(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """B-3 (b) / B-7 (c), adversarial (reviewer A: the guard reads the first child frame, not the focused one): the
    card field is in the second of two iframes; the first holds an ordinary field. ``failed`` as card, value empty;
    the plain field in the first iframe still types (the guard is not too wide)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/two_frames.html")
        out = await type_into(ex, "#card", TEXT, enter=True, frame_selector="#second")
        assert_secret(out, "card")
        assert await value_of(ex, "#card", "#second") == ""
        out = await type_into(ex, "#plain", "otter", frame_selector="#first")
        assert out.status == "executed", out
        assert await value_of(ex, "#plain", "#first") == "otter"


async def test_b_password_in_an_iframe_inside_a_shadow_root(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """B-3 (b), adversarial (reviewer B): an iframe inside an open shadow root holds a password field; the walk goes
    host, shadow root, iframe, frame. ``failed`` as a password, value empty."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/shadow_iframe.html")
        fr = await frame_by_url(ex, "/pw_inner.html")
        out = await type_in_frame(ex, fr, "#ipw", TEXT)
        assert_secret(out, "password")
        assert await fr.locator("#ipw").input_value(timeout=10_000) == ""


async def test_contenteditable_types_and_tabindex_div_is_focus_unknown(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """B-3 (d) both ways (orchestrator, e73b2b56): in web mode a contenteditable div is a field and types; a plain
    focusable ``div[tabindex]`` is not a field the guard can classify, so it is ``failed`` with FOCUS_UNKNOWN_REASON
    and receives no keystroke."""
    from local_astra.computer import executor as executor_mod

    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/editable.html")
        out = await type_into(ex, "#ed", "otter")
        assert out.status == "executed", out
        assert await ex.driver.page.inner_text("#ed") == "otter"
        out = await type_into(ex, "#tab", TEXT, enter=True)
        assert await ex.driver.page.evaluate("document.activeElement.id") == "tab", (
            "premise: the click focused the div"
        )
        assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
        assert await ex.driver.page.evaluate("window.__tabKeys") == 0


@pytest.mark.parametrize(
    "page", [pytest.param("object_pw.html", id="object"), pytest.param("embed_pw.html", id="embed")]
)
async def test_b_password_inside_object_or_embed(
    rig: Rig, launches: list[dict[str, Any]], page: str
) -> None:
    """B-3 (b) / B-7 (b), adversarial (reviewer A): a password field inside ``<object type="text/html">`` or
    ``<embed type="text/html">`` (nested browsing contexts that are not iframes) is ``failed`` as a password, value
    empty. *Reading:* B-3 (b)'s "frames" include these."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"https://shop.example/{page}")
        fr = await frame_by_url(ex, "/pw_inner.html")
        out = await type_in_frame(ex, fr, "#ipw", TEXT)
        assert_secret(out, "password")
        assert await fr.locator("#ipw").input_value(timeout=10_000) == ""


async def test_b_password_in_iframe_after_navigating_between_iframe_pages(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """B-3 (b), adversarial (reviewer A: stale frames): in one browser, type into an iframe field on one page (it
    types), navigate to another page whose iframe sits in the same place and holds a password field, and type there:
    ``failed`` as a password (not FOCUS_UNKNOWN from a frame left over from the first page), value empty."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/frames_a.html")
        out = await type_into(ex, "#plain", "otter", frame_selector="#fa")
        assert out.status == "executed", out
        await goto(ex, "https://shop.example/frames_b.html")
        out = await type_into(ex, "#ipw", TEXT, enter=True, frame_selector="#fb")
        assert_secret(out, "password")
        assert await value_of(ex, "#ipw", "#fb") == ""


async def test_b_closed_shadow_password_is_focus_unknown(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """B-3 (d), adversarial (reviewer A): a password field inside a closed shadow root cannot be read from outside
    (the focus stops at the host), so in web mode it is ``failed`` with FOCUS_UNKNOWN_REASON, not typed into as the
    host element. *Reading:* B-3 (b) walks open shadow roots only; an unreadable focus is "could not be checked"."""
    from local_astra.computer import executor as executor_mod

    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/shadow_closed.html")
        out = await type_into(ex, "#host", TEXT, enter=True)
        assert await ex.driver.page.evaluate("window.__closedInput.matches(':focus')"), (
            "premise: the click did not focus the field inside the closed shadow root"
        )
        assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
        assert TEXT not in json.dumps(out.resolved, default=str)
        assert await ex.driver.page.evaluate("window.__closedInput.value") == ""


# =====================================================================================================
# B-7 (d), (e): redirects and subresources to loopback or private addresses
# =====================================================================================================


async def test_d_redirects_to_loopback_and_private(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """B-7 (d), adversarial: a public-posing page's 302 to ``http://127.0.0.1:<canary>/`` and to a name resolving to
    10.0.0.5 never reaches the canary or the connector (the connector only ever sees the two public addresses), and
    both are in the egress log."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        to_canary = urllib.parse.quote(f"http://127.0.0.1:{rig.c1.port}/", safe="")
        await goto(ex, f"https://shop.example/redirect?to={to_canary}")
        await goto(
            ex,
            f"https://shop.example/redirect?to={urllib.parse.quote('https://private.example/', safe='')}",
        )
        await asyncio.sleep(1.0)
        assert ("shop.example", "/redirect") in rig.hits, "premise: the redirecting page was served"
        assert rig.c1.accepts == 0, (
            f"B-7 (d): the loopback canary was reached {rig.c1.accepts} time(s)"
        )
        net.assert_public_only()
        assert "private.example" in net.resolves, "premise: the redirect target reached the proxy"
        logged = logged_hosts(ex)
        assert {"127.0.0.1", "private.example"} <= logged, ex.egress_log


async def test_e_subresources_to_a_loopback_port(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-7 (e), adversarial: fetch, img, iframe, WebSocket (ws and wss) and ``location=`` from a public-posing page to
    a second loopback port (vLLM, the HUD): 0 accepts; the attempts are logged host and scheme only."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"https://shop.example/subresources.html?c={rig.c2.port}")
        await asyncio.sleep(3.0)
        assert ("shop.example", "/subresources.html") in rig.hits, "premise: the page was served"
        assert rig.c2.accepts == 0, (
            f"B-7 (e): the loopback port was reached {rig.c2.accepts} time(s)"
        )
        assert "127.0.0.1" in logged_hosts(ex), ex.egress_log
        assert all(set(r) == {"host", "scheme"} for r in ex.egress_log), ex.egress_log
        assert str(rig.c2.port) not in json.dumps(ex.egress_log), "B-6: a port or path is logged"


# =====================================================================================================
# B-7 (f): files
# =====================================================================================================


async def test_f_download_and_upload_blocked(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-7 (f): a download link is reported under ``blocked: download`` and saves nothing; a file-input click is
    reported under ``blocked: file_chooser`` and sets no file."""
    net = Net(rig.tls_port)
    before = _artifact_files()
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/download.html")
        out = await click(ex, "#dl")
        assert out.status == "executed", out
        await asyncio.sleep(1.0)
        later = await execute(ex, inv("scroll", None, direction="down"))
        kinds = [b.get("kind") for o in (out, later) for b in o.resolved.get("blocked") or []]
        assert "download" in kinds, (out.resolved, later.resolved)
        saved = [p for p in _artifact_files() - before if p.read_bytes() == PAYLOAD]
        assert saved == [], f"the download was saved: {saved}"
        assert not (Path.home() / "Downloads" / "t0160-payload.bin").exists()

        await goto(ex, "https://shop.example/upload.html")
        out = await click(ex, "#file")
        assert out.status == "executed", out
        assert {"kind": "file_chooser"} in (out.resolved.get("blocked") or []), out.resolved
        await asyncio.sleep(0.5)
        page = ex.driver.page
        assert await page.evaluate("document.getElementById('file').files.length") == 0
        assert await page.evaluate("window.__changes") == 0


# =====================================================================================================
# B-7 (g), (h): navigation keys, no phone-home
# =====================================================================================================


async def test_g_ctrl_t_from_a_public_page(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-7 (g) / B-4: ``Ctrl+T`` on a public-posing page shows ``/web.html``; ``Ctrl+V`` is refused."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, "https://shop.example/")
        assert ex.driver.page.url == "https://shop.example/", "premise: the public page loaded"
        out = await execute(ex, inv("press_key", None, combo="Ctrl+T"))
        assert (out.status, out.action) == ("executed", OPENED), out
        assert await wait_url(ex, f"{rig.start}/web.html") == f"{rig.start}/web.html"
        out = await execute(ex, inv("press_key", None, combo="Ctrl+V"))
        assert (out.status, out.reason) == ("refused", "pasting is off on real websites"), out


# =====================================================================================================
# T-0160 C amendment (reviewer A): clipboard writes blocked in web mode; loopback launch-BASE partner
# =====================================================================================================

CLIP_MARKERS = ("T0160-CLIP-EXEC-7c1e", "T0160-CLIP-API-7c1e")  # copy.html's two writes


async def _clipboard(page: Any, origin: str) -> str:
    """The clipboard, read on ``page`` (already on ``origin``) with ``clipboard-read`` granted to the test for that
    origin only (the readback; the runtime itself grants nothing)."""
    await page.context.grant_permissions(["clipboard-read"], origin=origin)
    return str(await page.evaluate("navigator.clipboard.readText()"))


async def _copy_then_read(ex: Any, origin: str) -> tuple[str, str, Any]:
    await goto(ex, f"{origin}/copy.html")
    assert ex.driver.page.url == f"{origin}/copy.html", (
        f"premise: the copy page loaded ({ex.driver.page.url})"
    )
    before = await _clipboard(ex.driver.page, origin)
    assert before not in CLIP_MARKERS, (
        f"premise: the clipboard does not already hold a marker ({before!r})"
    )
    out = await click(ex, "#copy")
    assert out.status == "executed", out
    await asyncio.sleep(0.5)
    state = await ex.driver.page.evaluate("window.__state")
    return before, await _clipboard(ex.driver.page, origin), state


async def test_clipboard_writes_blocked_in_web_mode(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """Amendment (reviewer A): in web mode a public page that copies on a model click (``execCommand('copy')`` and
    ``navigator.clipboard.writeText``) leaves the clipboard as it was (headed Chromium's clipboard is the user's)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        before, after, state = await _copy_then_read(ex, "https://shop.example")
        assert after == before and after not in CLIP_MARKERS, (
            f"T-0160 C amendment: a public page wrote the clipboard in web mode: {after!r} (page state {state})"
        )


async def test_clipboard_partner_loopback_launch_is_base(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """The partner (the readback can see a write): with ``web=False`` the same page on a loopback origin, launched
    as BASE, does write the clipboard on the same click (loopback unchanged by the amendment)."""
    from local_astra.computer.executor import BrowserToolExecutor

    origin = f"https://127.0.0.1:{rig.tls_port}"
    ex = BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=(W, H),
        web=False,
        allowed_origins=frozenset({rig.start, origin}),
    )
    try:
        before, after, state = await _copy_then_read(ex, origin)
        assert after in CLIP_MARKERS and after != before, (
            f"premise/partner: the loopback launch writes the clipboard: {after!r} (page state {state})"
        )
        assert not launches[-1].get("proxy"), (
            f"the partner is a BASE launch (no proxy); {launches[-1]}"
        )
    finally:
        await asyncio.wait_for(ex.close(), 30)


async def test_h_no_phone_home(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """B-7 (h): idle on ``/web.html`` for 5 s the proxy sees nothing: no resolve, no connect, no refusal (P1: only the
    start-origin tunnel, which none of these count). Before that a page phoning home in the background (beacon.html)
    reaches neither the canary nor a private address, even as it is left (``pagehide``)."""
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await goto(ex, f"https://shop.example/beacon.html?c={rig.c2.port}")
        await asyncio.sleep(2.0)
        out = await execute(ex, inv("press_key", None, combo="Ctrl+T"))
        assert out.status == "executed", out
        await asyncio.sleep(1.0)
        assert rig.c2.accepts == 0, (
            f"the beacon reached the loopback canary {rig.c2.accepts} time(s)"
        )
        net.assert_public_only()
        assert ex.driver.page.url == f"{rig.start}/web.html", ex.driver.page.url
        snap = (len(net.resolves), len(net.connects), len(ex.egress_log), len(rig.hits))
        await asyncio.sleep(5.0)
        now = (len(net.resolves), len(net.connects), len(ex.egress_log), len(rig.hits))
        assert now == snap, (
            f"B-7 (h): the idle browser phoned home: resolves/connects/refusals/hits {snap} -> {now}; "
            f"{net.resolves[snap[0] :]} {ex.egress_log[snap[2] :]}"
        )

"""T-0206 group A end to end: the submit stop in web mode with the real executor, a real headless Chromium (sandbox
on), the real ``EgressProxy`` with T-0160's injected resolver and connector, and loopback fixtures posing as
``https://shop.example`` / ``https://pay.example`` (docs/tasks/T-0206.md, A2-A9, "Tests to write",
``tests/integration/test_t0206_submit_browser.py``).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser`` once the
orchestrator adds the module). T-0160's gate, ``Net`` and certificate relaxation (``launches``) are imported from
``tests/integration/test_computer_web_browser.py``; this module runs its own TLS server over
``tests/fixtures/t0206/`` (``MANIFEST.md``), which records every ``POST /sent`` and ``GET /results`` so a row can tell
whether a form was sent.

**Readings (the test-author's, reported):**

- The overlay row pins the known gap as the Implementation notes say: hit-testing returns the transparent overlay, the
  read finds no send control (or fails open), the click executes and the form *is* sent.
- A Space typed into a text field of a filled form is pinned as executed (a space in the value): A4 "Space applies when
  the focused element is itself the control".
- The A9 bar (``A9_BAR_MS = 5``, orchestrator ruling 2f491800: 1.5x the implementer's measured 0.53-0.60 ms median,
  rounded up to 5 ms; T-0204 B-R3 precedent). The measured quantity is the driver read itself (``submit_facts`` at the
  click point, the only work A3 adds to an executed click), n = 100 each at two points of cost.html: a paragraph
  (no control: the short path) and the form's "Send" button with Name and Email filled (the full path with fields);
  the same bar holds for both.
- The D7 rows run on ``local_astra/pages/scenarios/trees/quote.html`` (T-0205 B, merged on main at 1bce14d5), served
  as T-0205's rig does (``StartPageServer(static_dir=<scenarios>)`` as the start origin); they skip on a branch that
  does not hold the file.

**Regime.** Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, no real network, n = 1 per row (A9:
n = 100, load < 2), one browser per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import os
import ssl
import statistics
import subprocess
import threading
import time
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import fence
from local_astra.contracts.toolcall import ToolOutcome
from tests.integration import test_computer_web_browser as web_rig
from tests.integration.test_computer_web_browser import (
    CLOCK_MS,
    H,
    Net,
    W,
    center,
    execute,
    inv,
    value_of,
)

_browser_layer = web_rig._browser_layer
launches = web_rig.launches

pytestmark = [pytest.mark.integration, pytest.mark.browser]

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "t0206"
SHOP = "https://shop.example"
#: A3, verbatim
SUBMIT_REASON = "this button sends the form; the user presses it themselves"
NOTE_WHAT = "live click: submit check"
#: A8: in token.html's pre-filled value and button label
TOKEN = "T0206TOKv7q2"
#: contact.html's hidden ``ref`` value
HIDDEN_VALUE = "T0206VALh1dd"
#: A9 (pinned from measurement after implementation; module docstring)
#: ruling 2f491800 (docs/tasks/T-0206.md Status): 1.5x the measured 0.53-0.60 ms median, rounded up to 5 ms
A9_BAR_MS: float | None = 5.0
A1_CONTROL_KEYS = {"tag", "type", "role", "label", "href_nav", "structural"}
A1_FIELD_KEYS = {
    "tag",
    "type",
    "autocomplete",
    "name",
    "id",
    "placeholder",
    "label",
    "filled",
    "visible",
}


# ============================================================================================ the rig


class _Fixture(http.server.BaseHTTPRequestHandler):
    hits: list[tuple[str, str, str]]

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _serve(self, method: str) -> None:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        u = urllib.parse.urlsplit(self.path)
        self.hits.append((method, host, u.path))
        name = {"/sent": "sent.html", "/results": "results.html", "/": "contact.html"}.get(
            u.path, u.path.lstrip("/")
        )
        f = FIXTURES / name
        if "/" in name or not f.is_file() or f.suffix != ".html":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, f.read_bytes(), "text/html; charset=utf-8")

    def do_GET(self) -> None:
        self._serve("GET")

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._serve("POST")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _Quiet(http.server.ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class Rig:
    def __init__(self, tls_port: int, hits: list[tuple[str, str, str]], start: str) -> None:
        self.tls_port, self.hits, self.start = tls_port, hits, start

    def sent(self) -> list[tuple[str, str, str]]:
        return [h for h in self.hits if h[2] == "/sent"]

    def searched(self) -> list[tuple[str, str, str]]:
        return [h for h in self.hits if h[2] == "/results"]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    if not web_rig.GATED:
        pytest.skip(f"{web_rig.GATE_ENV} is not 1")
    d = tmp_path_factory.mktemp("t0206-tls")
    key, crt = d / "key.pem", d / "crt.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
         "-days", "2", "-subj", "/CN=shop.example",
         "-addext", "subjectAltName=DNS:shop.example,DNS:pay.example"],
        check=True, capture_output=True,
    )  # fmt: skip
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    hits: list[tuple[str, str, str]] = []
    srv = _Quiet(("127.0.0.1", 0), type("_Bound", (_Fixture,), {"hits": hits}))
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    from local_astra.pages.server import StartPageServer

    pages = StartPageServer()
    pages.start()
    try:
        yield Rig(srv.server_address[1], hits, pages.origin)
    finally:
        pages.stop()
        srv.shutdown()
        srv.server_close()


@contextlib.asynccontextmanager
async def web(
    rig: Rig, notes: list[dict[str, Any]] | None = None, start: str | None = None
) -> AsyncIterator[Any]:
    from local_astra.computer.executor import BrowserToolExecutor

    net = Net(rig.tls_port)
    ex = BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=(W, H),
        web=True,
        start_origin=start or rig.start,
        allowed_origins=frozenset({start or rig.start}),
        proxy_resolve=net.resolve,
        proxy_connect=net.connect,
    )
    if notes is not None:
        ex.on_note = notes.append
    try:
        yield ex
    finally:
        await asyncio.wait_for(ex.close(), 30)
        net.assert_public_only()


async def page(ex: Any, name: str) -> None:
    await asyncio.wait_for(ex.frame(), 30)
    await ex.driver.page.goto(f"{SHOP}/{name}", wait_until="load", timeout=15_000)
    assert ex.driver.page.url == f"{SHOP}/{name}", ex.driver.page.url
    await asyncio.wait_for(ex.frame(), 30)


async def click(ex: Any, selector: str, frame_selector: str | None = None) -> ToolOutcome:
    x, y = await center(ex, selector, frame_selector)
    f = await asyncio.wait_for(ex.frame(), 30)
    return await execute(ex, inv("click_at", f, x=x, y=y))


async def type_(
    ex: Any, selector: str, text: str, *, enter: bool = False, frame_selector: str | None = None
) -> ToolOutcome:
    x, y = await center(ex, selector, frame_selector)
    f = await asyncio.wait_for(ex.frame(), 30)
    out = await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=enter))
    return out


async def key(ex: Any, combo: str) -> ToolOutcome:
    await asyncio.wait_for(ex.frame(), 30)
    return await execute(ex, inv("press_key", None, combo=combo))


async def fill(ex: Any, selector: str, text: str, frame_selector: str | None = None) -> None:
    """Fill a field through the executor (``type_at``, no Enter): the focus click reads the field itself (no send
    control), so this executes."""
    out = await type_(ex, selector, text, frame_selector=frame_selector)
    assert out.status == "executed", out


def need(ex: Any) -> None:
    """A2: the real driver's class has both reads (a named failure while they are absent)."""
    for name in ("submit_facts", "submit_facts_focused"):
        assert getattr(type(ex.driver), name, None) is not None, (
            f"T-0206 A2: PlaywrightDriver has no {name} yet"
        )


def reason() -> str:
    value = getattr(executor_mod, "SUBMIT_REASON", None)
    assert value is not None, "T-0206 A3: executor.SUBMIT_REASON does not exist yet"
    assert value == SUBMIT_REASON, value
    return str(value)


def assert_refused(out: ToolOutcome) -> None:
    assert (out.status, out.reason) == ("refused", reason()), out


async def settle_net(seconds: float = 0.8) -> None:
    """Give a (wrongly) sent form time to reach the server before asserting it did not."""
    await asyncio.sleep(seconds)


# ============================================================================================ A2: the driver reads


async def test_a2_reads_shape_and_no_value(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A2: ``submit_facts(x, y)`` at contact.html's "Send message" returns A1's keys: the control is the button
    (``structural``, not a link), the scope is the form, every field has exactly A1's keys and booleans for
    ``filled``/``visible`` (the hidden ``ref`` input is not visible, the filled Name is filled, the empty Email is not),
    and no field value appears anywhere in the facts. ``submit_facts_focused()`` with the Name field focused returns
    facts of the same shape."""
    async with web(rig) as ex:
        await page(ex, "contact.html")
        await fill(ex, "#name", "Ada Lovelace")
        p = ex.driver.page
        box = await p.locator("#go").bounding_box()
        assert box is not None
        need(ex)
        got = await ex.driver.submit_facts(
            box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        )
        assert isinstance(got, dict), got
        assert set(got["control"]) == A1_CONTROL_KEYS, got["control"]
        c = got["control"]
        assert (c["tag"], c["href_nav"], c["structural"]) == ("button", False, True), c
        assert " ".join(str(c["label"]).split()) == "Send message", c
        assert got["scope"] == "form" and got["search_scope"] is False, got
        assert 1 <= len(got["fields"]) <= 64
        by = {f["name"]: f for f in got["fields"]}
        for f in got["fields"]:
            assert set(f) == A1_FIELD_KEYS, f
            assert type(f["filled"]) is bool and type(f["visible"]) is bool, f
        assert by["name"]["filled"] is True and by["name"]["visible"] is True, by["name"]
        assert by["email"]["filled"] is False, by["email"]
        if (
            "ref" in by
        ):  # whether hidden inputs are listed at all is the driver's choice; listed, never visible
            assert by["ref"]["visible"] is False, by["ref"]
        blob = json.dumps(got)
        assert "Ada" not in blob and HIDDEN_VALUE not in blob, (
            f"A2: a field value left the page: {blob}"
        )
        assert fence.irreversible_submit(got) is True
        await p.focus("#name")
        foc = await ex.driver.submit_facts_focused()
        assert isinstance(foc, dict) and set(foc["control"]) == A1_CONTROL_KEYS, foc
        assert "Ada" not in json.dumps(foc) and HIDDEN_VALUE not in json.dumps(foc)


async def test_a2_reads_change_no_page_state(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A2: both reads change no page state: the focused element, the scroll position, the values and a log of focus,
    blur, pointer, mouse and key events are the same before and after."""
    js_arm = """() => { window.__ev = []; for (const t of ["focus","blur","focusin","focusout","pointerdown",
      "pointerup","mousedown","mouseup","click","keydown","keyup","input","change","submit","mouseover","pointermove"])
      document.addEventListener(t, e => window.__ev.push(t), true); }"""
    js_state = """() => ({ev: window.__ev.slice(), active: document.activeElement && document.activeElement.id,
      y: window.scrollY, values: [...document.querySelectorAll("input,textarea")].map(e => e.value)})"""
    async with web(rig) as ex:
        await page(ex, "contact.html")
        await fill(ex, "#phone", "0131 496 0000")
        p = ex.driver.page
        need(ex)
        await p.evaluate(js_arm)
        before = await p.evaluate(js_state)
        box = await p.locator("#go").bounding_box()
        assert box is not None
        await ex.driver.submit_facts(box["x"] + 5, box["y"] + 5)
        await ex.driver.submit_facts_focused()
        after = await p.evaluate(js_state)
        assert after == before, (before, after)
        assert before["active"] == "phone", before


# ============================================================================================ A5: stays open


SEARCHES = [
    pytest.param("google.html", "input[name=q]", id="google"),
    pytest.param("amazon.html", "#twotabsearchtextbox", id="amazon"),
    pytest.param("wikipedia.html", "input[name=search]", id="wikipedia"),
]


@pytest.mark.parametrize(("name", "box"), SEARCHES)
async def test_a5_search_enter_submits(
    rig: Rig, launches: list[dict[str, Any]], name: str, box: str
) -> None:
    """A5 (adversarial: a search box's submit is routine): ``type_at`` with Enter in a Google-, Amazon- or
    Wikipedia-shaped search box executes with no ``submit_withheld``, and the search is sent."""
    async with web(rig) as ex:
        await page(ex, name)
        n = len(rig.searched())
        out = await type_(ex, box, "kettle descaler", enter=True)
        assert out.status == "executed", out
        assert "submit_withheld" not in out.resolved, out.resolved
        await settle_net()
        assert len(rig.searched()) == n + 1, rig.hits[-3:]


@pytest.mark.parametrize(("name", "box"), SEARCHES)
async def test_a5_search_button_click(
    rig: Rig, launches: list[dict[str, Any]], name: str, box: str
) -> None:
    """A5: a click on the search form's button after typing executes and sends the search."""
    async with web(rig) as ex:
        await page(ex, name)
        await fill(ex, box, "kettle descaler")
        n = len(rig.searched())
        out = await click(ex, "#go")
        assert out.status == "executed", out
        await settle_net()
        assert len(rig.searched()) == n + 1, rig.hits[-3:]


OPEN_CLICKS = [
    pytest.param(
        "stores.html",
        [("#address", "1 High Street, Edinburgh")],
        "#go",
        "/results",
        id="d6-find-stores",
    ),
    pytest.param(
        "flights.html",
        [("#origin", "Edinburgh"), ("#destination", "Lisbon")],
        "#go",
        "/results",
        id="d9-search",
    ),
    pytest.param("signin.html", [("#email", "ada@example.org")], "#go", "/sent", id="sign-in"),
    pytest.param("cart.html", [], "#go", "/sent", id="add-to-cart"),
    pytest.param("cta.html", [], "#go", "/quote.html", id="cta-link"),
    pytest.param(
        "cta.html",
        [("#newsletter", "ada@example.org")],
        "#go",
        "/quote.html",
        id="cta-link-email-filled",
    ),
]


@pytest.mark.parametrize(("name", "fills", "target", "lands"), OPEN_CLICKS)
async def test_a5_open_clicks(
    rig: Rig,
    launches: list[dict[str, Any]],
    name: str,
    fills: list[tuple[str, str]],
    target: str,
    lands: str,
) -> None:
    """A5: the store locator with a street address, the flight search with origin and destination, a sign-in with an
    email, add-to-cart, a "Get a free quote" link (with and without a newsletter email filled: rule (a)): executed,
    and the request is made."""
    async with web(rig) as ex:
        await page(ex, name)
        for sel, text in fills:
            await fill(ex, sel, text)
        n = len([h for h in rig.hits if h[2] == lands])
        out = await click(ex, target)
        assert out.status == "executed", out
        await settle_net()
        assert len([h for h in rig.hits if h[2] == lands]) == n + 1, rig.hits[-3:]


async def test_a5_start_page_address_enter(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A5 (census, too-wide side): the start page ``/web.html``'s address box is named ``address`` (personal by A1's
    word list) and its button is "Go" (a search label): ``type_at`` + Enter there executes, is not withheld and
    navigates, as T-0160/T-0189/T-0191's rows rely on."""
    async with web(rig) as ex:
        await asyncio.wait_for(ex.frame(), 30)
        await ex.driver.page.goto(f"{rig.start}/web.html", wait_until="load", timeout=15_000)
        await asyncio.wait_for(ex.frame(), 30)
        out = await type_(ex, "#address", "shop.example/results.html", enter=True)
        assert out.status == "executed", out
        assert "submit_withheld" not in out.resolved, out.resolved
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ex.driver.page.url.startswith(SHOP):
            await asyncio.sleep(0.1)
        assert ex.driver.page.url.startswith(f"{SHOP}/results"), ex.driver.page.url


async def test_a5_filter_and_cookie_banner(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A5: a filter checkbox and a cookie banner's "Accept all" (an email filled on the page) execute and act."""
    async with web(rig) as ex:
        await page(ex, "filters.html")
        await fill(ex, "#email", "ada@example.org")
        out = await click(ex, "#instock")
        assert out.status == "executed", out
        assert await ex.driver.page.is_checked("#instock")
        out = await click(ex, "#accept")
        assert out.status == "executed", out
        assert await ex.driver.page.locator("#banner").count() == 0


# ============================================================================================ A6: refused


async def test_a6_contact_send_refused(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6/A3: contact.html with Name filled: the click on "Send message" is refused with SUBMIT_REASON, nothing is
    sent, the form keeps its value."""
    async with web(rig) as ex:
        await page(ex, "contact.html")
        await fill(ex, "#name", "Ada Lovelace")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.sent()) == n, rig.hits[-3:]
        assert await value_of(ex, "#name") == "Ada Lovelace"


async def test_a6_quote_description_only(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6: quote.html (T-0205 D7 shape) with only the Description (a textarea) filled: "Request quote" refused."""
    async with web(rig) as ex:
        await page(ex, "quote.html")
        await fill(ex, "#description", "Two oaks to fell")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.sent()) == n


async def test_a6_quote_enter_in_phone_withheld(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6/A4: Enter in D7's Phone field (Name filled): ``type_at`` + Enter types the phone number, Enter is withheld
    (``submit_withheld`` True, executed, the action of ``press_enter=False``), nothing sent; then ``press_key Enter``
    with the focus still in Phone is refused."""
    async with web(rig) as ex:
        await page(ex, "quote.html")
        await fill(ex, "#name", "Ada Lovelace")
        n = len(rig.sent())
        out = await type_(ex, "#phone", "0131 496 0000", enter=True)
        assert out.status == "executed", out
        assert out.resolved.get("submit_withheld") is True, out.resolved
        assert out.action.endswith("typed 13 characters"), out.action
        await settle_net()
        assert len(rig.sent()) == n
        assert await value_of(ex, "#phone") == "0131 496 0000"
        assert_refused(await key(ex, "Enter"))
        await settle_net()
        assert len(rig.sent()) == n


async def test_a6_quote_enter_target_only_withheld(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """A4: the target field counts as filled when the text is non-empty: Enter after typing the Name into an otherwise
    empty D7 form is withheld."""
    async with web(rig) as ex:
        await page(ex, "quote.html")
        n = len(rig.sent())
        out = await type_(ex, "#name", "Ada Lovelace", enter=True)
        assert out.status == "executed", out
        assert out.resolved.get("submit_withheld") is True, out.resolved
        await settle_net()
        assert len(rig.sent()) == n


async def test_a6_space_on_focused_submit(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6/A4: Space on a focused "Send message" (Name filled) is refused and nothing is sent; Space in the Name field of
    the same form executes (a space is typed: the focused element is not the control)."""
    async with web(rig) as ex:
        await page(ex, "contact.html")
        await fill(ex, "#name", "Ada")
        n = len(rig.sent())
        out = await key(ex, "Space")
        assert out.status == "executed", out
        assert await value_of(ex, "#name") == "Ada ", "premise: Space typed into the Name field"
        await ex.driver.page.focus("#go")
        assert_refused(await key(ex, "Space"))
        await settle_net()
        assert len(rig.sent()) == n


async def test_a6_textarea_enter_is_a_newline(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A4 (too-wide side): ``type_at`` + Enter in contact.html's Message textarea (Name filled) presses Enter: a newline
    in the value, no ``submit_withheld``, nothing sent."""
    async with web(rig) as ex:
        await page(ex, "contact.html")
        await fill(ex, "#name", "Ada")
        n = len(rig.sent())
        out = await type_(ex, "#message", "Hello", enter=True)
        assert out.status == "executed", out
        assert "submit_withheld" not in out.resolved, out.resolved
        assert await value_of(ex, "#message") == "Hello\n"
        await settle_net()
        assert len(rig.sent()) == n


async def test_a6_empty_label_structural(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6: an empty-label structural submit with the email filled is refused."""
    async with web(rig) as ex:
        await page(ex, "empty_label.html")
        await fill(ex, "#email", "ada@example.org")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.sent()) == n


@pytest.mark.parametrize("name", ["spa.html", "shadow.html"])
async def test_a6_formless_div_send(rig: Rig, launches: list[dict[str, Any]], name: str) -> None:
    """A6/A2: a formless SPA ``div[role=button]`` "Send message" with the textarea filled is refused (also inside an
    open shadow root); its click handler never runs."""
    async with web(rig) as ex:
        await page(ex, name)
        await fill(ex, "#message", "Please call me back")
        sel = "#go" if name == "spa.html" else "#host >> #go"
        assert_refused(await click(ex, sel))
        assert await ex.driver.page.evaluate("window.__sent") == 0


async def test_spa_quote_under_header_search_refused(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """Ruling on reading 6 (safety): spa_quote.html, a formless quote form whose fields and div "Send" sit directly
    under ``<body>`` beside the header's empty site-search ``q`` box: with Name, Email and Message filled, the click on
    "Send" is refused and the handler never runs. Partner: typing a query into ``q`` with Enter still executes."""
    async with web(rig) as ex:
        await page(ex, "spa_quote.html")
        await fill(ex, "#name", "Ada Lovelace")
        await fill(ex, "#email", "ada@example.org")
        await fill(ex, "#message", "Two oaks to fell")
        assert_refused(await click(ex, "#go"))
        assert await ex.driver.page.evaluate("window.__sent") == 0
        out = await type_(ex, "#q", "opening hours", enter=True)
        assert out.status == "executed", out
        assert "submit_withheld" not in out.resolved, (
            out.resolved
        )  # no form: Enter has no form to send (A4)


async def test_a6_spa_div_executes_when_empty(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A1 (c) partner: the same div with nothing filled executes and its handler runs."""
    async with web(rig) as ex:
        await page(ex, "spa.html")
        out = await click(ex, "#go")
        assert out.status == "executed", out
        assert await ex.driver.page.evaluate("window.__sent") == 1


async def test_a6_cross_origin_iframe_send(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6/A2: a cross-origin (pay.example) iframe contact form on shop.example: "Send" refused after the email is
    typed; nothing is posted."""
    async with web(rig) as ex:
        await page(ex, "framed.html")
        await web_rig.frame_by_url(ex, "/iframe_contact.html")
        await fill(ex, "#email", "ada@example.org", frame_selector="#fr")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go", frame_selector="#fr"))
        await settle_net()
        assert len(rig.sent()) == n, rig.hits[-3:]


async def test_overlay_forwarding_click_known_gap(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """Adversarial, known gap (Implementation notes): a transparent overlay over "Send" that calls ``requestSubmit()``
    hides the button from hit-testing; the click executes and the form is sent. Pinned so the gap is recorded."""
    async with web(rig) as ex:
        await page(ex, "overlay.html")
        await fill(ex, "#email", "ada@example.org")
        p = ex.driver.page
        box = await p.locator("#go").bounding_box()
        assert box is not None
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        top = await p.evaluate("([x, y]) => document.elementFromPoint(x, y).id", [cx, cy])
        assert top == "overlay", f"premise: the overlay is on top of the button ({top})"
        need(ex)
        got = await ex.driver.submit_facts(cx, cy)
        assert got is None or fence.irreversible_submit(got) is False, got
        n = len(rig.sent())
        out = await click(ex, "#go")
        assert out.status == "executed", out
        await settle_net()
        assert len(rig.sent()) == n + 1, rig.hits[-3:]


# ============================================================================================ review rulings R-1..R-4


async def click_point(ex: Any, x_css: float, y_css: float) -> ToolOutcome:
    f = await asyncio.wait_for(ex.frame(), 30)
    x, y = web_rig.units(x_css, W), web_rig.units(y_css, H)
    return await execute(ex, inv("click_at", f, x=x, y=y))


@pytest.mark.parametrize("name", ["slotted.html", "slotted_span.html"])
async def test_r1_slotted_web_component_button(
    rig: Rig, launches: list[dict[str, Any]], name: str
) -> None:
    """R-1: ``<x-btn>Send message</x-btn>`` whose shadow ``<button><slot></slot></button>`` calls ``requestSubmit()``, in
    a form with email and message filled: a click on the slotted label, a click on the inner button's padding and
    Space on the focused inner button are all refused; nothing is sent."""
    async with web(rig) as ex:
        await page(ex, name)
        await fill(ex, "#email", "ada@example.org")
        await fill(ex, "#message", "Hello")
        p = ex.driver.page
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))  # the centre: the slotted text
        inner = await p.locator("#inner").bounding_box()
        assert inner is not None
        assert_refused(await click_point(ex, inner["x"] + 8, inner["y"] + 8))  # the padding
        await p.focus("#inner")
        assert_refused(await key(ex, "Space"))
        await settle_net()
        assert len(rig.sent()) == n, rig.hits[-3:]


async def test_r2_whole_page_form(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """R-2: a whole-page ``<form>`` (WebForms shape) with an empty header ``q`` and a quote form inside: Name, Email and
    Message filled, "Send" is refused; the same page with only ``q`` filled sends."""
    async with web(rig) as ex:
        await page(ex, "webforms.html")
        for sel, text in (
            ("#name", "Ada"),
            ("#email", "ada@example.org"),
            ("#message", "Two oaks"),
        ):
            await fill(ex, sel, text)
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.sent()) == n
        await page(ex, "webforms.html")
        await fill(ex, "#q", "opening hours")
        out = await click(ex, "#go")
        assert out.status == "executed", out
        await settle_net()
        assert len(rig.sent()) == n + 1, rig.hits[-3:]


async def test_r3_store_locator(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """R-3: a store locator with postcode and street address filled and an icon-only submit: the click executes and
    searches, and so does Enter in the postcode (not withheld); with the email also filled the click is refused."""
    async with web(rig) as ex:
        await page(ex, "locator.html")
        await fill(ex, "#postcode", "EH1 1AA")
        await fill(ex, "#address", "1 High Street")
        n = len(rig.searched())
        out = await click(ex, "#go")
        assert out.status == "executed", out
        await settle_net()
        assert len(rig.searched()) == n + 1, rig.hits[-3:]
        await page(ex, "locator.html")
        out = await type_(ex, "#postcode", "EH1 1AA", enter=True)
        assert out.status == "executed" and "submit_withheld" not in out.resolved, out
        await settle_net()
        assert len(rig.searched()) == n + 2, rig.hits[-3:]
        await page(ex, "locator.html")
        await fill(ex, "#postcode", "EH1 1AA")
        await fill(ex, "#email", "ada@example.org")
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.searched()) == n + 2


async def _select_filled(ex: Any) -> dict[str, bool]:
    box = await ex.driver.page.locator("#go").bounding_box()
    assert box is not None
    got = await ex.driver.submit_facts(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    assert isinstance(got, dict), got
    return {f["name"]: f["filled"] for f in got["fields"]}


@pytest.mark.parametrize(("select", "choose"), [("#pt", "landline"), ("#mt", "pm")])
async def test_r4_select_without_selected_attribute(
    rig: Rig, launches: list[dict[str, Any]], select: str, choose: str
) -> None:
    """R-4 variant: a personal ``phone_type`` select with no ``selected`` option (its first option is the default) and a
    ``multiple`` ``phone_times`` select with nothing selected are not filled at their defaults, and "Submit" executes
    and sends; once one is changed it counts as filled and "Submit" is refused."""
    async with web(rig) as ex:
        await page(ex, "select_noattr.html")
        need(ex)
        assert await _select_filled(ex) == {"phone_type": False, "phone_times": False}
        await ex.driver.page.select_option(select, choose)
        got = await _select_filled(ex)
        changed = "phone_type" if select == "#pt" else "phone_times"
        assert got[changed] is True, got
        assert_refused(await click(ex, "#go"))
        await page(ex, "select_noattr.html")
        n = len(rig.sent())
        out = await click(ex, "#go")
        assert out.status == "executed", out
        await settle_net()
        assert len(rig.sent()) == n + 1, rig.hits[-3:]


async def test_r4_select_default_not_filled(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """R-4: a ``<select>`` at its default-selected option is not ``filled``; changed, it is. At the defaults "Submit"
    executes and sends; with ``contact_phone_type`` (personal by name) changed, it is refused."""
    async with web(rig) as ex:
        await page(ex, "select.html")
        need(ex)
        p = ex.driver.page
        box = await p.locator("#go").bounding_box()
        assert box is not None
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        got = await ex.driver.submit_facts(cx, cy)
        by = {f["name"]: f for f in got["fields"]}
        assert by["billing_address_country"]["filled"] is False, by
        assert by["contact_phone_type"]["filled"] is False, by
        await p.select_option("#phonetype", "mobile")
        await p.select_option("#country", "FR")
        got = await ex.driver.submit_facts(cx, cy)
        by = {f["name"]: f for f in got["fields"]}
        assert by["billing_address_country"]["filled"] is True, by
        assert by["contact_phone_type"]["filled"] is True, by
        assert_refused(await click(ex, "#go"))
        await page(ex, "select.html")
        n = len(rig.sent())
        out = await click(ex, "#go")
        assert out.status == "executed", out
        await settle_net()
        assert len(rig.sent()) == n + 1, rig.hits[-3:]


# ============================================================================================ reviewer gaps


async def test_r4_gap_formless_header_search_with_footer_email(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """Reviewer gap r4: a formless header search (``q`` filled, div "Submit") on a page whose footer newsletter email is
    filled executes: the scope is the header's, whose filled inputs are all search-like."""
    async with web(rig) as ex:
        await page(ex, "header_footer.html")
        await fill(ex, "#q", "red shoes")
        await fill(ex, "#nl", "ada@example.org")
        out = await click(ex, "#go")
        assert out.status == "executed", out
        assert await ex.driver.page.evaluate("window.__sent") == 1


async def test_r12_hash_link_send(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """Reviewer gap r12: ``<a href="#">Send</a>`` in a form with email and message filled is refused; nothing sent."""
    async with web(rig) as ex:
        await page(ex, "hashlink.html")
        await fill(ex, "#email", "ada@example.org")
        await fill(ex, "#message", "Hello")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.sent()) == n


async def test_r8_offset_cross_origin_iframe(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """Reviewer gap r8: the cross-origin iframe contact form offset (420, 260) px from the corner: "Send" refused (the
    point is translated into the frame)."""
    async with web(rig) as ex:
        await page(ex, "framed_offset.html")
        await web_rig.frame_by_url(ex, "/iframe_contact.html")
        await fill(ex, "#email", "ada@example.org", frame_selector="#fr")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go", frame_selector="#fr"))
        await settle_net()
        assert len(rig.sent()) == n, rig.hits[-3:]


async def test_r7_enter_without_default_button_withheld(
    rig: Rig, launches: list[dict[str, Any]]
) -> None:
    """Reviewer gap r7 (ruled: withheld): Enter in a personal form with one field and no button (HTML implicit
    submission would send it) is withheld; nothing sent."""
    async with web(rig) as ex:
        await page(ex, "nobutton.html")
        n = len(rig.sent())
        out = await type_(ex, "#email", "ada@example.org", enter=True)
        assert out.status == "executed", out
        assert out.resolved.get("submit_withheld") is True, out.resolved
        await settle_net()
        assert len(rig.sent()) == n


async def test_r2_gap_hidden_q_does_not_exempt(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """Reviewer gap r2: a hidden ``q`` with a value does not make a contact form a search: "Submit" refused."""
    async with web(rig) as ex:
        await page(ex, "hidden_q.html")
        await fill(ex, "#email", "ada@example.org")
        n = len(rig.sent())
        assert_refused(await click(ex, "#go"))
        await settle_net()
        assert len(rig.sent()) == n


# ============================================================================================ A8: privacy


async def test_a8_token_nowhere(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A8: token.html holds a unique token in a pre-filled value and in the button label; through a refused click and a
    refused Enter it is in no outcome, no note, no ``egress_log`` entry and no ``take_events()`` entry."""
    notes: list[dict[str, Any]] = []
    async with web(rig, notes) as ex:
        await page(ex, "token.html")
        outs = [await click(ex, "#go")]
        assert outs[0].reason == reason(), outs[0]
        await ex.driver.page.focus("#name")
        outs.append(await key(ex, "Enter"))
        assert [o.reason for o in outs] == [reason(), reason()], outs
        events = ex.driver.take_events()
        blob = (
            repr(outs)
            + json.dumps(notes, default=str)
            + json.dumps(ex.egress_log)
            + json.dumps(events)
        )
        assert TOKEN not in blob, f"A8: the token left the page: {blob}"


# ============================================================================================ A7: non-web, S11


async def test_a7_s11_non_web_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A7 (active: holds at BASE): T-0138 S11 ("send the booking", non-web) through the real executor still succeeds,
    and its early-Enter trap still submits; neither read is ever called (spies on the real driver's class)."""
    from local_astra.computer import runtime
    from tests.integration import test_t0138_suite_browser as suite

    called: list[str] = []
    for name in ("submit_facts", "submit_facts_focused"):
        orig = getattr(runtime.PlaywrightDriver, name, None)

        def spy(self: Any, *a: Any, _n: str = name, _o: Any = orig) -> Any:
            called.append(_n)
            if _o is None:
                raise AssertionError(f"A7: {_n} called outside web mode")
            return _o(self, *a)

        monkeypatch.setattr(runtime.PlaywrightDriver, name, spy, raising=False)
    async with suite.opened("S11") as r:
        assert r.ex.web is False
        voiced = await suite.golden("S11", r)
        assert all(o.status == "executed" for o in r.outcomes), r.outcomes
        v = suite.judge("S11", await r.state(), voiced)
        assert v.success is True, suite.verdict_dict(v)
    async with suite.opened("S11") as r:
        await suite._trap_s11_enter(r)
        assert all(o.status == "executed" for o in r.outcomes), r.outcomes
    assert called == [], called


# ============================================================================================ D7 (T-0205 B)


D7_PAGE = REPO / "local_astra" / "pages" / "scenarios" / "trees" / "quote.html"


@contextlib.asynccontextmanager
async def d7(rig: Rig) -> AsyncIterator[Any]:
    """T-0205's D7 page through the real executor in web mode, its scenario server as the start origin."""
    if not D7_PAGE.is_file():
        pytest.skip(
            "T-0205 B is not on this branch: local_astra/pages/scenarios/trees/quote.html is absent"
        )
    from local_astra.pages.server import StartPageServer

    srv = StartPageServer(static_dir=D7_PAGE.parent.parent)
    srv.start()
    try:
        async with web(rig, start=srv.origin) as ex:
            await asyncio.wait_for(ex.frame(), 30)
            await ex.driver.page.goto(
                f"{srv.origin}/trees/quote.html", wait_until="load", timeout=15_000
            )
            await asyncio.wait_for(ex.frame(), 30)
            yield ex
    finally:
        srv.stop()


async def d7_sent(ex: Any) -> bool:
    """D7 replaces the form with its confirmation on a submit (scenarios.js): shown means sent."""
    return bool(
        await ex.driver.page.evaluate(
            "() => !document.getElementById('confirmation').classList.contains('gone')"
        )
    )


D7_BUTTON = "button[data-scn=request-quote]"


async def test_d7_request_quote_refused(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6 on T-0205's D7 page (read-only): "Request quote" after the Name is typed is refused; nothing sent."""
    async with d7(rig) as ex:
        await fill(ex, "#q-name", "Ada Lovelace")
        assert_refused(await click(ex, D7_BUTTON))
        assert await d7_sent(ex) is False


async def test_d7_enter_in_phone_withheld(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A6/A4 on D7: with the Name filled, ``type_at`` + Enter in Phone is withheld (executed, ``submit_withheld``), the
    phone number is typed and nothing is sent."""
    async with d7(rig) as ex:
        await fill(ex, "#q-name", "Ada Lovelace")
        out = await type_(ex, "#q-phone", "512 555 0100", enter=True)
        assert out.status == "executed", out
        assert out.resolved.get("submit_withheld") is True, out.resolved
        assert await value_of(ex, "#q-phone") == "512 555 0100"
        assert await d7_sent(ex) is False


async def test_d7_address_only_is_open(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """R-3 on D7: with only the Address (a location field) filled, "Request quote" executes and the form is sent (the
    confirmation shows)."""
    async with d7(rig) as ex:
        await fill(ex, "#q-address", "1 Congress Ave")
        out = await click(ex, D7_BUTTON)
        assert out.status == "executed", out
        await settle_net(0.3)
        assert await d7_sent(ex) is True


# ============================================================================================ A9: cost


async def _median_read_ms(ex: Any, selector: str, n: int = 100) -> tuple[float, float, Any]:
    """Median and p90 (ms) of ``n`` ``submit_facts`` reads at the centre of ``selector`` (one warm-up read first)."""
    box = await ex.driver.page.locator(selector).bounding_box()
    assert box is not None, selector
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    got = await ex.driver.submit_facts(cx, cy)
    ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        await ex.driver.submit_facts(cx, cy)
        ms.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ms), statistics.quantiles(ms, n=10)[-1], got


async def test_a9_cost(rig: Rig, launches: list[dict[str, Any]]) -> None:
    """A9: the added latency of an executed web-mode ``click_at`` on cost.html (one form) is the driver read at the
    click point; n = 100 reads at the paragraph (no control) and n = 100 at the form's "Send" with Name and Email filled
    (the full read: control, form scope, fields); median and p90 printed; both medians are at most ``A9_BAR_MS``
    (module docstring). Also: one executed click with its read, and the Send click refused."""
    load = os.getloadavg()[0]
    async with web(rig) as ex:
        await page(ex, "cost.html")
        need(ex)
        out = await click(ex, "#para")
        assert out.status == "executed", out
        await fill(ex, "#name", "Ada Lovelace")
        await fill(ex, "#email", "ada@example.org")
        med_p, p90_p, _ = await _median_read_ms(ex, "#para")
        med_s, p90_s, got = await _median_read_ms(ex, "#go")
        assert isinstance(got, dict) and fence.irreversible_submit(got) is True, (
            f"premise: the Send read is the full path: {got}"
        )
        assert_refused(await click(ex, "#go"))
    msg = (
        f"A9: submit_facts paragraph median {med_p:.2f} ms p90 {p90_p:.2f} ms; "
        f"Send button median {med_s:.2f} ms p90 {p90_s:.2f} ms (n=100 each, load {load:.2f}, bar {A9_BAR_MS} ms)"
    )
    print(msg)
    assert A9_BAR_MS is not None, msg
    assert med_p <= A9_BAR_MS and med_s <= A9_BAR_MS, msg

"""T-0192 A2-A8 at the seam: the checkout refusal and arrival in ``BrowserToolExecutor`` over scripted fake drivers, no
browser (docs/tasks/T-0192.md, "Tests to write", group A, rows (1)-(7)).

**The fake.** ``UrlsMixin`` over T-0111c's ``FakeDriver`` (and T-0160's ``FieldDriver``, T-0151's ``PageDriver``)
adds A2's ``main_frame_urls()``: the browser keeps a store of main-frame navigation request URLs and a current URL;
``visit(*urls)`` appends request URLs (redirect hops first) and makes the last one current; ``push_state(url)`` only
changes the current URL (no request); ``visit_on(name, *urls)`` / ``push_on(name, url)`` do the same during the next
``name`` seam call. ``main_frame_urls()`` returns the store then the current URL, and drains the store. Its ``goto``
also visits the URL (the start page). ``read = "raise"`` makes the read raise ``DriverBoom`` with the current URL
in its message (A8: an exception message must not carry it out). The read is recorded in ``calls`` but is not a seam
call (no ``arm_raise`` count), so screenshots equal the plain fake's.

**Expected values by hand** (viewport 1280x800): ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``; a
``scroll down`` is ``dy = 800 * 3 // 4 = 600``. Byte-identity rows compare against the plain fakes (BASE) with one
fixed phash salt.

**Readings (the test-author's, reported):** A3's "after ``ensure()`` and its restart/page-serial checks" is pinned
as an order (a page-serial refusal comes first and leaves the store unread); after ``reset_page()`` from a checkout
the next action executes (B5: Restart is a way out); a raising read with ``Ctrl+L``, a raising or non-``str`` post-
settle read, and ``page_overview``'s own arrival are not pinned.

Regime: CPU, fake drivers, no browser, no socket, 1280x800, n = 1, deterministic.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.toolcall import ToolOutcome
from tests.unit.test_computer_executor import DriverBoom, FakeDriver, inv, own_frame
from tests.unit.test_computer_web_executor import FieldDriver
from tests.unit.test_t0151_executor import PageDriver, register

CLOCK_MS = 1_700_000_019_200
START = "http://127.0.0.1:8123"
WEB_HTML = f"{START}/web.html"
SALT = 0x0FED_CBA9_8765_4321
#: A3, verbatim
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"
OPENED = "opened the start page"
TOKEN = "T0192TOKu5x9"
SHOP = "https://shop.example"
CART = f"{SHOP}/cart"
CHECKOUT = f"{SHOP}/checkout/{TOKEN}/pay"
HOP = f"{SHOP}/checkout/entry?{TOKEN}=1"
SIGNIN = f"{SHOP}/ap/signin?openid.return_to=https%3A%2F%2Fshop.example%2Fcheckout%2F{TOKEN}"
#: a cart form's submit (no checkout), the first request of the middle-hop chain
SUBMIT = f"{SHOP}/gp/cart/view.html?ref_=cart_submit"
#: the three ways a read can name a checkout: the current URL, a redirect hop only, the host label
AT_CHECKOUT = [
    pytest.param((CHECKOUT,), id="current-url"),
    pytest.param((HOP, SIGNIN), id="redirect-hop-only"),
    #: review R-A (urls[:1]): Amazon's chain, cart POST -> /checkout/entry -> /ap/signin; the checkout is a middle hop
    pytest.param((SUBMIT, HOP, SIGNIN), id="middle-hop"),
    pytest.param(("https://checkout.stripe.com/c/x",), id="host-label"),
]
#: A5/A6: pages where nothing is refused and nothing is reported
OPEN_PAGES = [
    f"{SHOP}/cart",
    f"{SHOP}/gp/cart/view.html",
    f"{SHOP}/cart/smart-wagon",
    f"{SHOP}/gp/aws/cart/add.html",
    f"{SHOP}/gp/buyagain",
    f"{SHOP}/checkout-faq",
    f"{SHOP}/payment-methods",
    f"{SHOP}/s?k=checkout",
    SIGNIN,
    f"{SHOP}/#checkout",
    "https://checkout-help.example/",
    "https://shop.checkout.example/",
    "https://www.google.com/sorry/index?continue=x",
    "https://www.amazon.com/errors/validateCaptcha",
    "https://www.amazon.com/ap/cvf/request",
]
TEXT_FIELD = {"tag": "input", "type": "text", "name": "q"}


class UrlsMixin:
    """A2's ``main_frame_urls`` on top of a fake driver (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]
    events: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.store: list[str] = []
        self.url = f"{SHOP}/"
        self.read = "ok"
        self.reads: list[tuple[str, ...]] = []
        self.taken: list[dict[str, Any]] = []
        self._sched: list[tuple[str, tuple[str, ...], bool]] = []

    # -- test controls
    def visit(self, *urls: str) -> None:
        self.store.extend(urls)
        self.url = urls[-1]

    def push_state(self, url: str) -> None:
        self.url = url

    def visit_on(self, name: str, *urls: str) -> None:
        self._sched.append((name, urls, False))

    def push_on(self, name: str, url: str) -> None:
        self._sched.append((name, (url,), True))

    def _enter(self, name: str, *args: Any) -> None:
        super()._enter(name, *args)  # type: ignore[misc]
        for s in list(self._sched):
            if s[0] == name:
                self._sched.remove(s)
                if s[2]:
                    self.push_state(s[1][0])
                else:
                    self.visit(*s[1])

    async def goto(self, url: str) -> None:
        await super().goto(url)  # type: ignore[misc]
        self.visit(url)

    def take_events(self) -> list[dict[str, Any]]:
        got: list[dict[str, Any]] = super().take_events()  # type: ignore[misc]
        self.taken.extend(got)
        return got

    # -- the capability
    def main_frame_urls(self) -> tuple[str, ...]:
        self.calls.append(("main_frame_urls", ()))
        if self.read == "raise":
            raise DriverBoom(f"synthetic read failure at {self.url}")
        out = (*self.store, self.url)
        self.store = []
        self.reads.append(out)
        return out


class UrlsDriver(UrlsMixin, FakeDriver):
    pass


class UrlsFieldDriver(UrlsMixin, FieldDriver):
    pass


class UrlsPageDriver(UrlsMixin, PageDriver):
    pass


def reason() -> str:
    value = getattr(executor_mod, "CHECKOUT_REASON", None)
    assert value is not None, "T-0192 A3: executor.CHECKOUT_REASON does not exist yet"
    assert value == CHECKOUT_REASON, value
    return str(value)


def build(driver: Any, *, web: bool = True, **kw: Any) -> Any:
    if web:
        kw.setdefault("start_origin", START)
    return executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=driver.viewport, driver=driver, web=web, **kw
    )


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: SALT)


def dump(out: ToolOutcome) -> str:
    return json.dumps([out.status, out.action, out.reason, out.resolved])


async def act(ex: Any, tool: str, args: dict[str, Any]) -> ToolOutcome:
    frame = await own_frame(ex) if tool in ("click_at", "type_at") else None
    out = await ex.execute(inv(tool, frame, **args))
    assert isinstance(out, ToolOutcome)
    return out


def assert_refused(out: ToolOutcome, d: Any, mark: int, tool: str) -> None:
    """A3: refused with CHECKOUT_REASON; nothing dispatched or captured after ``mark``."""
    r = reason()
    assert (out.status, out.reason) == ("refused", r), out
    assert out.action == f"refused {tool}: {r}", out.action
    assert out.resolved == {}, out.resolved
    assert d.dispatched(mark) == [], f"A3: dispatched on a checkout page: {d.dispatched(mark)}"
    after = names(d, mark)
    for name in ("screenshot", "full_page_screenshot", "layout_metrics", "focused_field"):
        assert name not in after, f"A3: {name} ran on a checkout page: {after}"


#: (tool, arguments): every page action A3 refuses on a checkout page
REFUSED = [
    pytest.param("click_at", {"x": 500, "y": 500}, id="click_at"),
    pytest.param(
        "type_at", {"x": 500, "y": 500, "text": "kettle", "press_enter": True}, id="type_at"
    ),
    pytest.param("scroll", {"direction": "down"}, id="scroll"),
    pytest.param("press_key", {"combo": "Enter"}, id="press-Enter"),
    pytest.param("press_key", {"combo": "Tab"}, id="press-Tab"),
    pytest.param("press_key", {"combo": "Ctrl+A"}, id="press-Ctrl+A"),
]


# =====================================================================================================
# (1) A3: refusal on a checkout page
# =====================================================================================================


@pytest.mark.parametrize("urls", AT_CHECKOUT)
@pytest.mark.parametrize(("tool", "args"), REFUSED)
async def test_refused_on_checkout(tool: str, args: dict[str, Any], urls: tuple[str, ...]) -> None:
    """(1) A3: on a checkout page (by the current URL, by a redirect hop alone, by the host label) every page action
    is refused with CHECKOUT_REASON; nothing is dispatched or captured; the read comes after ``ensure()``."""
    d = UrlsFieldDriver(TEXT_FIELD)
    ex = build(d)
    frame = await own_frame(ex) if tool in ("click_at", "type_at") else None
    d.visit(*urls)
    mark = len(d.calls)
    out = await ex.execute(inv(tool, frame, **args))
    assert_refused(out, d, mark, tool)
    seq = names(d, mark)
    assert "main_frame_urls" in seq and "ensure" in seq, seq
    assert seq.index("ensure") < seq.index("main_frame_urls"), f"A3: read before ensure(): {seq}"


async def test_page_overview_refused_on_checkout() -> None:
    """(1) A3: ``page_overview`` on a checkout page is refused; no metrics read, no whole-page capture."""
    d = UrlsPageDriver(content_height=2400.0)
    ex = build(d)
    await own_frame(ex)
    d.visit(CHECKOUT)
    mark = len(d.calls)
    out = await ex.execute(inv("page_overview"))
    assert_refused(out, d, mark, "page_overview")
    assert d.full_page_calls == [], d.full_page_calls


async def test_overview_frame_click_refused_on_checkout() -> None:
    """(1) A3: the T-0151 overview-frame path: an overview registered on the cart, then the page reaches a checkout
    between two actions; a ``click_at`` on the overview frame is refused, no scroll, no click."""
    d = UrlsPageDriver(content_height=1220.0)
    ex = build(d, echo="model")
    d.visit(CART)
    await own_frame(ex)
    ov = await register(ex)
    d.visit(CHECKOUT)
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=500, y=900))
    assert_refused(out, d, mark, "click_at")


async def test_read_after_the_page_serial_check() -> None:
    """(1) A3's order: a popup adopted in ``ensure()`` is refused with PAGE_CHANGED_REASON first and the store is not
    read; the checkout redirect hop it recorded then refuses the next call."""
    d = UrlsDriver()
    ex = build(d)
    await own_frame(ex)
    d.open_popup()
    d.visit(HOP, SIGNIN)
    out = await ex.execute(inv("press_key", None, combo="Enter"))
    assert (out.status, out.reason) == ("refused", executor_mod.PAGE_CHANGED_REASON), out
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo="Enter"))
    assert_refused(out, d, mark, "press_key")


# =====================================================================================================
# (2) A3's exceptions on a checkout page; A5 on open pages
# =====================================================================================================


@pytest.mark.parametrize("combo", ["Ctrl+L", "Ctrl+T"])
async def test_start_page_keys_execute_on_checkout(combo: str) -> None:
    """(2) A3: on a checkout page ``Ctrl+L``/``Ctrl+T`` execute (``goto`` the start page) and carry no ``checkout``
    (A4 reads after the settle: the start page). The checkout was read and refused nothing; the next action on the
    start page executes."""
    d = UrlsDriver()
    ex = build(d)
    await own_frame(ex)
    d.visit(HOP, CHECKOUT)
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert (out.status, out.action) == ("executed", OPENED), out
    assert ("goto", (WEB_HTML,)) in d.dispatched(mark), d.dispatched(mark)
    assert "checkout" not in out.resolved, out.resolved
    assert any(CHECKOUT in r for r in d.reads), f"premise: the checkout was read: {d.reads}"
    nxt = await act(ex, "click_at", {"x": 500, "y": 500})
    assert nxt.status == "executed" and "checkout" not in nxt.resolved, nxt


async def test_google_search_on_checkout_is_the_dry_run() -> None:
    """(2) A3: ``google_search`` on a checkout page is the dry run, exactly as ``DryRunToolExecutor`` describes it;
    the discriminating partner: a click on the same page is refused."""
    d = UrlsDriver()
    ex = build(d)
    await own_frame(ex)
    d.visit(CHECKOUT)
    call = inv("google_search", None, query="kettle reviews")
    want = await DryRunToolExecutor().execute(call)
    got = await ex.execute(call)
    assert got.model_dump() == want.model_dump(), (got, want)
    assert (await act(ex, "click_at", {"x": 500, "y": 500})).reason == reason()


SCRIPT = [
    ("click_at", {"x": 500, "y": 500}),
    ("type_at", {"x": 500, "y": 500, "text": "kettle", "press_enter": True}),
    ("press_key", {"combo": "Enter"}),
    ("scroll", {"direction": "down"}),
    ("press_key", {"combo": "Tab"}),
    ("press_key", {"combo": "Ctrl+L"}),
]


async def run_script(d: Any, page: str | None = None, **kw: Any) -> list[ToolOutcome]:
    ex = build(d, **kw)
    outs = []
    for tool, args in SCRIPT:
        if page is not None and isinstance(d, UrlsMixin):
            d.visit(page)
        outs.append(await act(ex, tool, args))
    return outs


@pytest.mark.parametrize("page", OPEN_PAGES)
async def test_open_pages_execute_as_base(monkeypatch: pytest.MonkeyPatch, page: str) -> None:
    """(2) A5/A6 (active: holds at BASE): on the add-to-cart path, CAPTCHA and verification walls every tool
    executes, and status, action string, reason and ``resolved`` are byte-identical to the plain fake's (no
    ``checkout`` key, never ``checkout: False``, no URL)."""
    fixed_salt(monkeypatch)
    base = await run_script(FieldDriver(TEXT_FIELD))
    got = await run_script(UrlsFieldDriver(TEXT_FIELD), page)
    assert [o.status for o in got] == ["executed"] * len(SCRIPT), got
    assert [dump(o) for o in got] == [dump(o) for o in base]


@pytest.mark.parametrize("page", [CART, f"{SHOP}/gp/buyagain"])
async def test_open_pages_overview_as_base(monkeypatch: pytest.MonkeyPatch, page: str) -> None:
    """(2) A5 (active): ``page_overview`` and an overview-frame click on an open page are byte-identical to BASE."""
    fixed_salt(monkeypatch)
    outs = []
    for d in (PageDriver(content_height=1220.0), UrlsPageDriver(content_height=1220.0)):
        ex = build(d, echo="model")
        if isinstance(d, UrlsMixin):
            d.visit(page)
        await own_frame(ex)
        ov = await register(ex)
        outs.append(dump(await ex.execute(inv("click_at", ov, x=500, y=900))))
    assert json.loads(outs[1])[0] == "executed", outs[1]
    assert outs[1] == outs[0]


# =====================================================================================================
# (3) A4: arrival, from the post-settle read only
# =====================================================================================================

ARRIVALS = [
    pytest.param("click_at", {"x": 500, "y": 500}, "mouse_up", id="click_at"),
    pytest.param(
        "type_at",
        {"x": 500, "y": 500, "text": "kettle", "press_enter": True},
        "key_down",
        id="type-enter",
    ),
    pytest.param("press_key", {"combo": "Enter"}, "key_down", id="press_key"),
    pytest.param("scroll", {"direction": "down"}, "scroll_by", id="scroll"),
    pytest.param("click_at", {"x": 500, "y": 500}, "settle", id="during-settle"),
]


@pytest.mark.parametrize(("tool", "args", "during"), ARRIVALS)
async def test_arrival_sets_checkout(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict[str, Any], during: str
) -> None:
    """(3) A4: an executed action that reaches a checkout carries ``resolved["checkout"] is True``; every other key
    is BASE's (same salt, same scenes), so no URL, path or host rides along. The read is after the last settle and
    before the post-action screenshot."""
    fixed_salt(monkeypatch)
    base_d = FieldDriver(TEXT_FIELD)
    base = await act(build(base_d), tool, args)
    d = UrlsFieldDriver(TEXT_FIELD)
    ex = build(d)
    d.visit(CART)
    await own_frame(ex)
    mark = len(d.calls)
    d.visit_on(during, CHECKOUT)
    out = await act(ex, tool, args)
    assert out.status == "executed", out
    assert out.resolved.get("checkout") is True, (
        f"T-0192 A4: an arrival carries no resolved['checkout'] = True: {out.resolved}"
    )
    rest = {k: v for k, v in out.resolved.items() if k != "checkout"}
    assert json.dumps([out.action, rest]) == json.dumps([base.action, base.resolved])
    seq = names(d, mark)
    last_read = len(seq) - 1 - seq[::-1].index("main_frame_urls")
    last_shot = len(seq) - 1 - seq[::-1].index("screenshot")
    last_settle = max(i for i, n in enumerate(seq[:last_shot]) if n == "settle")
    assert last_settle < last_read < last_shot, f"A4: read out of place: {seq}"


@pytest.mark.parametrize(
    "how",
    [
        pytest.param("hop", id="redirect-hop-only"),
        pytest.param("middle", id="middle-hop"),
        pytest.param("push", id="pushState"),
        pytest.param("host", id="host-label"),
    ],
)
async def test_arrival_shapes(how: str) -> None:
    """(3) A4: a checkout passed only as a redirect hop (the landing is the sign-in page; or a middle hop, after the
    cart submit and before the sign-in page: review R-A, kills ``urls[:1]`` and ``urls[-1:]``), a same-document
    ``pushState`` to ``/checkout`` (no request), a checkout host: ``checkout: True``."""
    d = UrlsDriver()
    ex = build(d)
    d.visit(CART)
    await own_frame(ex)
    if how == "hop":
        d.visit_on("mouse_up", HOP, SIGNIN)
    elif how == "middle":
        d.visit_on("mouse_up", SUBMIT, HOP, SIGNIN)
    elif how == "push":
        d.push_on("mouse_up", f"{SHOP}/checkout")
    else:
        d.visit_on("mouse_up", "https://pay.example/rxo")
    out = await act(ex, "click_at", {"x": 500, "y": 500})
    assert out.status == "executed" and out.resolved.get("checkout") is True, out


async def test_arrival_on_overview_frame_click() -> None:
    """(3) A4: the overview-frame click path reads after its settle too."""
    d = UrlsPageDriver(content_height=1220.0)
    ex = build(d, echo="model")
    d.visit(CART)
    await own_frame(ex)
    ov = await register(ex)
    d.visit_on("mouse_up", CHECKOUT)
    out = await ex.execute(inv("click_at", ov, x=500, y=900))
    assert out.status == "executed" and out.resolved.get("checkout") is True, out


async def test_navigation_after_the_capture_counts_next_time() -> None:
    """(3) A4 placement, both sides: a checkout reached during the post-action screenshot is not this action's
    arrival (no key) and refuses the next call (A3); an arrival is not reported twice (the read drained)."""
    d = UrlsDriver()
    ex = build(d)
    d.visit(CART)
    await own_frame(ex)
    d.visit_on("screenshot", HOP, SIGNIN)
    out = await ex.execute(inv("press_key", None, combo="Enter"))
    assert out.status == "executed" and "checkout" not in out.resolved, out
    mark = len(d.calls)
    nxt = await ex.execute(inv("press_key", None, combo="Enter"))
    assert_refused(nxt, d, mark, "press_key")
    third = await ex.execute(inv("press_key", None, combo="Enter"))
    assert third.status == "executed" and "checkout" not in third.resolved, third


async def test_arrival_then_refusal() -> None:
    """(3)/(1): after an arrival on a checkout page the next action is refused (the current URL is a checkout)."""
    d = UrlsDriver()
    ex = build(d)
    d.visit(CART)
    await own_frame(ex)
    d.visit_on("mouse_up", CHECKOUT)
    out = await act(ex, "click_at", {"x": 500, "y": 500})
    assert out.resolved.get("checkout") is True, out
    frame = await own_frame(ex)
    mark = len(d.calls)
    nxt = await ex.execute(inv("click_at", frame, x=500, y=500))
    assert_refused(nxt, d, mark, "click_at")


# =====================================================================================================
# (4) the read raising: fail closed
# =====================================================================================================


@pytest.mark.parametrize(("tool", "args"), REFUSED[:4])
async def test_read_raising_refuses(tool: str, args: dict[str, Any]) -> None:
    """(4) A3: when ``main_frame_urls`` raises, the call is refused with CHECKOUT_REASON, nothing dispatched; the
    exception's message (which carries the page URL) is not in the outcome (A8)."""
    d = UrlsFieldDriver(TEXT_FIELD)
    ex = build(d)
    d.visit(CHECKOUT)
    frame = await own_frame(ex) if tool in ("click_at", "type_at") else None
    d.read = "raise"
    mark = len(d.calls)
    out = await ex.execute(inv(tool, frame, **args))
    assert_refused(out, d, mark, tool)
    assert TOKEN not in repr(out) and "synthetic" not in repr(out), out


# =====================================================================================================
# (5) A7: unchanged outside web mode; class lookup; the real driver
# =====================================================================================================


async def test_loopback_never_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """(5) A7 (active): with ``web=False`` a driver with ``main_frame_urls`` on a checkout URL is never read, and the
    call sequence and outcomes equal the plain fake's (T-0138's S13 starts on ``/checkout.html``)."""
    fixed_salt(monkeypatch)
    base_d = FakeDriver()
    base = await run_script(base_d, web=False)
    d = UrlsDriver()
    got = await run_script(d, "http://127.0.0.1:8123/checkout.html", web=False)
    assert "main_frame_urls" not in names(d), names(d)
    assert [dump(o) for o in got] == [dump(o) for o in base]
    assert d.calls == base_d.calls


async def test_web_driver_without_the_method_is_base() -> None:
    """(5) A7 (active): a web-mode driver class without ``main_frame_urls`` runs BASE's path (an instance attribute of
    that name is not the capability: looked up on the class, as ``focused_field``)."""
    d = FieldDriver(TEXT_FIELD)
    d.__dict__["main_frame_urls"] = lambda: (CHECKOUT,)
    outs = await run_script(d)
    assert [o.status for o in outs] == ["executed"] * len(SCRIPT), outs
    assert all("checkout" not in o.resolved for o in outs)


def test_real_driver_has_the_seam() -> None:
    """(7) A2/A7: ``PlaywrightDriver.main_frame_urls`` exists on the class, is synchronous and takes no argument."""
    fn = getattr(runtime.PlaywrightDriver, "main_frame_urls", None)
    assert fn is not None, "T-0192 A2: PlaywrightDriver has no main_frame_urls yet"
    assert not inspect.iscoroutinefunction(fn), "A2: main_frame_urls is synchronous"
    assert list(inspect.signature(fn).parameters) == ["self"]


@pytest.mark.parametrize("name", ["tools_v1.json", "tools_v1_overview.json"])
def test_tool_schemas_unchanged(name: str) -> None:
    """A5 (active): the model-facing tool schemas are byte-identical to a9a11add (sha256 of ``git show
    a9a11add:local_astra/toolcall/<name>``, taken by the test-author on 2026-10-29)."""
    import hashlib
    from pathlib import Path

    want = {
        "tools_v1.json": "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707",
        "tools_v1_overview.json": "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc",
    }[name]
    path = Path(executor_mod.__file__).resolve().parents[1] / "toolcall" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == want


# =====================================================================================================
# (6) A8: no URL leaves the executor; Restart
# =====================================================================================================


async def test_token_nowhere() -> None:
    """(6) A8: through a hop arrival, a refusal, a raising read and Ctrl+L, the checkout path's token is in no
    outcome (``repr``), no ``take_events()`` entry and no ``egress_log`` entry."""
    d = UrlsFieldDriver(TEXT_FIELD)
    ex = build(d)
    d.visit(CART)
    await own_frame(ex)
    d.events.append({"kind": "dialog", "type": "alert"})
    d.visit_on("mouse_up", HOP, CHECKOUT)
    outs = [await act(ex, "click_at", {"x": 500, "y": 500})]
    outs.append(await act(ex, "type_at", {"x": 500, "y": 500, "text": "x", "press_enter": False}))
    d.read = "raise"
    outs.append(await ex.execute(inv("scroll", None, direction="down")))
    d.read = "ok"
    outs.append(await ex.execute(inv("press_key", None, combo="Ctrl+L")))
    assert [o.status for o in outs] == ["executed", "refused", "refused", "executed"], outs
    assert outs[0].resolved.get("checkout") is True, outs[0]
    blob = repr(outs) + json.dumps(d.taken) + json.dumps(ex.egress_log)
    for needle in (TOKEN, "/checkout", "shop.example"):
        assert needle not in blob, f"A8: {needle!r} left the executor: {blob}"


async def test_restart_from_checkout() -> None:
    """(6) A8 / B5 (active: holds at BASE, guards against a too-wide guard): ``reset_page()`` from a checkout page goes
    to ``/web.html`` (never refused), and the next action there executes (module docstring reading)."""
    d = UrlsDriver()
    ex = build(d)
    await own_frame(ex)
    d.visit(HOP, CHECKOUT)
    mark = len(d.calls)
    await ex.reset_page()
    assert ("goto", (WEB_HTML,)) in d.dispatched(mark), d.dispatched(mark)
    out = await act(ex, "click_at", {"x": 500, "y": 500})
    assert out.status == "executed" and "checkout" not in out.resolved, out
